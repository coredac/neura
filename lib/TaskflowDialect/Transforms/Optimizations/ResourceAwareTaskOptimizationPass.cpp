//===- ResourceAwareTaskOptimizationPass.cpp - Pipeline Balance & Fusion --===//
// This pass performs two-phase optimization on the task graph:
// 1. Utilization Fusion: merges independent (no-edge) tasks, selecting pairs
//    that minimize |trip_count_a - trip_count_b| for balanced utilization.
// 2. Pipeline Balance: allocates extra CGRAs to critical-path bottleneck tasks.
//    More CGRAs combine tile arrays into larger arrays for mapping, potentially
//    lowering compiled_ii.  Latency model: II * (trip_count - 1) + steps.
//
// Targets a 4x4 CGRA grid (16 CGRAs total).  Each task may use up to 4 CGRAs.
// Supported per-task shapes: rect (1×1..4×1/1×4/2×2), L (3 or 4 CGRAs), T (4
// CGRAs). Compiled_ii must come from the downstream pipeline (asserts on
// failure).
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <set>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

constexpr int kCgraGridRows = 4;
constexpr int kCgraGridCols = 4;
constexpr int kTotalCGRAs = kCgraGridRows * kCgraGridCols; // 16
constexpr int kMaxBalanceIterations = 100;
constexpr int kMaxCgrasPerTask = 4; // Max CGRAs allocatable to a single task.

// Sentinel value: 0 means "not yet profiled". After profileTask() runs,
// both steps and ii MUST be > 0. An assert fires if profiling fails.
constexpr int64_t kUnprofiled = 0;

//===----------------------------------------------------------------------===//
// CGRA Shape Utilities
//===----------------------------------------------------------------------===//

// Represents a CGRA allocation shape on the grid.
//
// For rectangular shapes: rows × cols == cgra_count, and `cgra_positions`
// is empty (all cells in the bounding box are used).
//
// For non-rectangular shapes (L, T): `cgra_positions` stores the explicit
// (col, row) coordinates of the occupied CGRAs.  `rows`/`cols` give the
// bounding box so that tile-level x_tiles/y_tiles can be computed.
struct CgraShape {
  int rows;            // Bounding-box CGRA rows.
  int cols;            // Bounding-box CGRA columns.
  bool is_rectangular; // True if all cells in the bbox are used.
  // Explicit CGRA positions for non-rectangular shapes.
  // Each pair is (col, row) in CGRA coordinates.  Empty for rectangles.
  SmallVector<std::pair<int, int>> cgra_positions;

  int area() const { return rows * cols; }

  // Returns a human-readable description for log messages only (not IR).
  std::string describe(int cgra_count) const {
    std::string s = std::to_string(rows) + "x" + std::to_string(cols);
    if (!is_rectangular) {
      s += "(non-rect, " + std::to_string(cgra_count) + " CGRAs:";
      for (auto &[c, r] : cgra_positions)
        s += " (" + std::to_string(c) + "," + std::to_string(r) + ")";
      s += ")";
    }
    return s;
  }

  // Returns the shape string written into the IR cgra_shape attribute.
  // For rectangular shapes: "NxM" (e.g. "2x2").
  // For non-rectangular shapes: "NxM[(c0,r0)(c1,r1)...]" listing only the
  // occupied CGRA positions so that downstream passes can reconstruct the
  // exact valid tile set for multi-CGRA mapping.
  std::string irAttr() const {
    std::string s = std::to_string(rows) + "x" + std::to_string(cols);
    if (!is_rectangular && !cgra_positions.empty()) {
      s += "[";
      for (auto &[c, r] : cgra_positions)
        s += "(" + std::to_string(c) + "," + std::to_string(r) + ")";
      s += "]";
    }
    return s;
  }
};

// Returns all valid rectangular shapes for `cgra_count` CGRAs.
static SmallVector<CgraShape> getRectangularShapes(int cgra_count) {
  SmallVector<CgraShape> shapes;
  for (int r = 1; r <= kCgraGridRows; ++r) {
    for (int c = 1; c <= kCgraGridCols; ++c) {
      if (r * c == cgra_count) {
        shapes.push_back(
            {r, c, /*is_rectangular=*/true, /*cgra_positions=*/{}});
      }
    }
  }
  return shapes;
}

// Returns true if `cgra_count` CGRAs can fit on the grid and does not
// exceed the per-task limit.
static bool canFitOnGrid(int cgra_count) {
  return cgra_count >= 1 && cgra_count <= kMaxCgrasPerTask;
}

// Returns the set of non-rectangular shapes for `cgra_count` CGRAs.
// Currently defined for cgra_count == 3 (L-shape) and cgra_count == 4
// (L-shape and T-shape variants).  Each shape's coordinates are chosen
// so the bounding box is as small as possible.
static SmallVector<CgraShape> getNonRectangularShapes(int cgra_count) {
  SmallVector<CgraShape> shapes;

  if (cgra_count == 3) {
    // L-shape 3 CGRAs: (0,0)(1,0)(0,1) — bbox 2×2
    shapes.push_back({2, 2, false, {{0, 0}, {1, 0}, {0, 1}}});
  }

  if (cgra_count == 4) {
    // T-shape: three in a row + one below centre
    //   (0,0)(1,0)(2,0)(1,1)  — bbox 2×3
    shapes.push_back({2, 3, false, {{0, 0}, {1, 0}, {2, 0}, {1, 1}}});

    // L-shape: three in a column + one offset
    //   (0,0)(0,1)(0,2)(1,2)  — bbox 3×2
    shapes.push_back({3, 2, false, {{0, 0}, {0, 1}, {0, 2}, {1, 2}}});
  }

  return shapes;
}

// Picks the best shape for display/profiling.
// We prefer shapes with the most compact physical layout (smallest maximum
// distance between nodes) to minimize communication latency. In cases of
// identical bounding box area, we prefer more square-like bounds over long
// rectangles.
//
// TODO: This function only picks a localized shape for an idealized single task
// mapping. Global placement and conflict resolution across multiple tasks is
// legitimately deferred to downstream map-on-cgra pass, as speculative
// profiling assumes unconstrained placement.
static CgraShape pickBestShape(int cgra_count) {
  // For cgra_count == 3, the 2x2 L-shape has a smaller maximum physical routing
  // distance (dist=2) compared to a 1x3 rectangle (dist=3), despite having a
  // larger bounding box. We explicitly prefer the more compact L-shape here for
  // better speculative latency.
  if (cgra_count == 3) {
    auto non_rect_shapes = getNonRectangularShapes(3);
    if (!non_rect_shapes.empty()) {
      return non_rect_shapes.front();
    }
  }

  SmallVector<CgraShape> candidates = getRectangularShapes(cgra_count);
  for (const auto &s : getNonRectangularShapes(cgra_count)) {
    candidates.push_back(s);
  }

  if (!candidates.empty()) {
    return *std::min_element(candidates.begin(), candidates.end(),
                             [](const CgraShape &a, const CgraShape &b) {
                               int area_a = a.area();
                               int area_b = b.area();
                               if (area_a != area_b)
                                 return area_a < area_b;
                               return std::abs(a.rows - a.cols) <
                                      std::abs(b.rows - b.cols);
                             });
  }

  // Fallback: smallest bounding box (should not be reached for 1..4 CGRAs).
  CgraShape best = {kCgraGridRows, kCgraGridCols, false, {}};
  for (int r = 1; r <= kCgraGridRows; ++r) {
    for (int c = 1; c <= kCgraGridCols; ++c) {
      if (r * c >= cgra_count && r * c < best.area()) {
        best = {r, c, false, {}};
      }
    }
  }
  return best;
}

//===----------------------------------------------------------------------===//
// Task Dependency Graph
//===----------------------------------------------------------------------===//

struct TaskGraphNode {
  size_t id;
  TaskflowTaskOp op;
  int64_t trip_count = 1;
  int64_t steps = kUnprofiled;
  int64_t ii = kUnprofiled;
  int cgra_count = 1;
  CgraShape shape = {1, 1, true, {}};

  // Dependency edges (both SSA and memory).
  SmallVector<TaskGraphNode *> predecessors;
  SmallVector<TaskGraphNode *> successors;

  TaskGraphNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}

  // Returns estimated task latency using the pipelined execution model:
  //   latency = II * (trip_count - 1) + steps.
  int64_t estimatedLatency() const { return ii * (trip_count - 1) + steps; }
};

class TaskDependencyGraph {
public:
  SmallVector<std::unique_ptr<TaskGraphNode>> nodes;
  DenseMap<Operation *, TaskGraphNode *> op_to_node;

  void build(func::FuncOp func, bool skip_mapper = false) {
    // 1. Creates TaskGraphNodes.
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskGraphNode>(task_id++, task);

      // If the task already has profiling attributes (e.g., from fusion),
      // skip expensive speculative lowering and use those directly.
      bool has_precomputed =
          task->hasAttr("compiled_ii") && task->hasAttr("steps");
      if (!has_precomputed) {
        // Speculative lowering to Neura to get real metrics.
        profileTask(node.get(), task, skip_mapper);
      }

      // Reads existing trip_count attribute if set by fusion.
      if (auto attr = task->getAttrOfType<IntegerAttr>("trip_count")) {
        node->trip_count = attr.getInt();
      } else {
        node->trip_count = computeTripCount(task);
      }

      // Overrides with explicit attributes if present.
      if (auto attr = task->getAttrOfType<IntegerAttr>("steps")) {
        node->steps = attr.getInt();
      }
      if (auto attr = task->getAttrOfType<IntegerAttr>("compiled_ii")) {
        node->ii = attr.getInt();
      }
      if (auto attr = task->getAttrOfType<IntegerAttr>("cgra_count")) {
        node->cgra_count = attr.getInt();
      }

      op_to_node[task] = node.get();
      nodes.push_back(std::move(node));
    });

    // 2. Builds SSA edges (value dependencies between tasks).
    for (auto &consumer : nodes) {
      for (Value operand : consumer->op.getValueInputs()) {
        if (auto producer_op = operand.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get());
          }
        }
      }
    }

    // 3. Builds memory edges.
    for (auto &consumer : nodes) {
      // RAW: producer wrote a memref that this task reads.
      for (Value memref : consumer->op.getDependencyReadIn()) {
        if (auto producer_op = memref.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get());
          }
        }
      }
      // WAW/WAR: producer wrote or read a memref that this task writes.
      for (Value memref : consumer->op.getDependencyWriteIn()) {
        if (auto producer_op = memref.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get());
          }
        }
      }
    }

    llvm::errs() << "TaskDependencyGraph: " << nodes.size() << " tasks\n";
    for (auto &n : nodes) {
      llvm::errs() << "  Task " << n->id << " (" << n->op.getTaskName().str()
                   << "): trip_count=" << n->trip_count << ", ii=" << n->ii
                   << ", steps=" << n->steps
                   << ", preds=" << n->predecessors.size()
                   << ", succs=" << n->successors.size() << "\n";
    }
  }

  // Returns true if there is any (direct or transitive) dependency from
  // source_node to dest_node.
  bool hasDependency(TaskGraphNode *source_node,
                     TaskGraphNode *dest_node) const {
    if (source_node == dest_node)
      return true;
    DenseSet<TaskGraphNode *> visited;
    SmallVector<TaskGraphNode *> worklist;
    worklist.push_back(source_node);
    while (!worklist.empty()) {
      auto *current = worklist.pop_back_val();
      if (current == dest_node)
        return true;
      if (!visited.insert(current).second)
        continue;
      for (auto *succ : current->successors) {
        worklist.push_back(succ);
      }
    }
    return false;
  }

  // Returns true if a and b are completely independent (no path in either
  // direction).
  bool areIndependent(TaskGraphNode *a, TaskGraphNode *b) const {
    return !hasDependency(a, b) && !hasDependency(b, a);
  }

  // Returns total CGRAs allocated.
  int getTotalAllocatedCGRAs() const {
    int total = 0;
    for (auto &node : nodes) {
      total += node->cgra_count;
    }
    return total;
  }

  // Profiles a single TaskflowTaskOp: clones the task, wraps the kernel in a
  // standalone func, and runs InsertDataMov + MapToAcceleratorPass to obtain
  // ii.  skip_mapper: use only ResMII/RecMII analytical estimates.
  // When skip_mapper=true, only ResMII/RecMII analytical estimates are used
  // (no MapToAcceleratorPass). This is safe for speculative balance checks
  // where the mapper may backtrack indefinitely on larger tile arrays.
  void profileTask(TaskGraphNode *node, TaskflowTaskOp task,
                   bool skip_mapper = false) {
    MLIRContext *ctx = task.getContext();
    OpBuilder builder(ctx);
    Location loc = task.getLoc();

    auto parent_func = task->getParentOfType<func::FuncOp>();
    assert(parent_func &&
           "[profileTask] FATAL: task has no parent func::FuncOp. "
           "compiled_ii must come from downstream pipeline.");

    // Verifies exactly one neura.kernel per task (post-lowering invariant).
    neura::KernelOp the_kernel;
    task.walk([&](neura::KernelOp k) {
      assert(!the_kernel && "task has more than one neura.kernel op");
      the_kernel = k;
    });
    assert(the_kernel && "task has no neura.kernel op");

    // Clones the task into a temporary module so we don't mutate the real IR.
    auto tmp_mod = ModuleOp::create(loc);
    neura::KernelOp cloned_kernel;
    {
      OpBuilder b(tmp_mod.getBodyRegion());
      IRMapping mapping;
      Operation *cloned_task = b.clone(*task.getOperation(), mapping);
      cast<TaskflowTaskOp>(cloned_task).walk([&](neura::KernelOp k) {
        cloned_kernel = k;
      });
    }

    // Computes tile dimensions for the target CGRA shape.
    int per_cgra_cols = neura::getArchitecture().getPerCgraColumns();
    int per_cgra_rows = neura::getArchitecture().getPerCgraRows();
    int x_tiles = node->shape.cols * per_cgra_cols;
    int y_tiles = node->shape.rows * per_cgra_rows;
    std::string valid_tiles;
    if (!node->shape.is_rectangular) {
      // Enumerates individual tile coordinates for non-rectangular shapes
      // so the mapper knows exactly which tiles are valid.
      llvm::raw_string_ostream os(valid_tiles);
      for (auto &[cgra_c, cgra_r] : node->shape.cgra_positions) {
        for (int tr = 0; tr < per_cgra_rows; ++tr) {
          for (int tc = 0; tc < per_cgra_cols; ++tc) {
            if (!os.str().empty())
              os << ",";
            os << (cgra_c * per_cgra_cols + tc) << "_"
               << (cgra_r * per_cgra_rows + tr);
          }
        }
      }
    }

    // Runs Neura pipeline on the kernel to get compiled_ii and steps.
    auto phase2_module = ModuleOp::create(loc);
    int compiled_ii = 0;
    int cp_depth = 1;

    if (succeeded(runNeuraPipelineOnKernel(
            ctx, cloned_kernel, phase2_module, compiled_ii, cp_depth, x_tiles,
            y_tiles, valid_tiles, skip_mapper))) {
      llvm::errs() << "[profileTask] kernel in " << task.getTaskName()
                   << ": compiled_ii=" << compiled_ii
                   << ", cp_depth=" << cp_depth << "\n";
    } else {
      llvm::errs() << "[profileTask] Phase 2 failed for kernel in "
                   << task.getTaskName() << ", extracting partial\n";
      extractMetricsFromPartialIR(phase2_module, compiled_ii, cp_depth, x_tiles,
                                  y_tiles);
    }
    phase2_module.erase();

    assert(compiled_ii > 0 &&
           "[profileTask] FATAL: compiled_ii is 0 after downstream pipeline.");
    node->ii = compiled_ii;
    node->steps = std::max(cp_depth, 1);

    llvm::errs() << "[profileTask] " << task.getTaskName()
                 << ": compiled_ii=" << node->ii << ", steps=" << node->steps
                 << "\n";

    // Erases the temporary module.
    tmp_mod.erase();
  }

private:
  llvm::DenseSet<std::pair<TaskGraphNode *, TaskGraphNode *>> edge_set;

  void addEdge(TaskGraphNode *from, TaskGraphNode *to) {
    auto key = std::make_pair(from, to);
    if (edge_set.insert(key).second) {
      from->successors.push_back(to);
      to->predecessors.push_back(from);
    }
  }

  // Wraps a neura.kernel into a standalone func in dst_module, runs
  // InsertDataMov + mapper, and returns compiled_ii / cp_depth.
  // x_tiles/y_tiles: multi-CGRA tile grid dimensions.
  // valid_tiles: explicit tile list for non-rectangular shapes (empty = full).
  // skip_mapper: skip MapToAcceleratorPass, use ResMII/RecMII only.
  LogicalResult runNeuraPipelineOnKernel(
      MLIRContext *ctx, neura::KernelOp kernel, ModuleOp dst_module,
      int &compiled_ii, int &cp_depth, int x_tiles = 0, int y_tiles = 0,
      const std::string &valid_tiles = "", bool skip_mapper = false) {
    Location loc = kernel.getLoc();
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(dst_module.getBody());

    // Builds function signature: all kernel inputs + iter_args as arguments.
    Region &kernel_body = kernel.getBody();
    if (kernel_body.empty())
      return failure();

    Block &entry = kernel_body.front();
    SmallVector<Type> arg_types;
    for (BlockArgument arg : entry.getArguments())
      arg_types.push_back(arg.getType());

    // Result types from the kernel op.
    SmallVector<Type> result_types(kernel.getResultTypes());

    auto func_type = builder.getFunctionType(arg_types, result_types);
    auto wrapper_func =
        builder.create<func::FuncOp>(loc, "__speculative_kernel__", func_type);

    // Tags as neura accelerator — all downstream passes check this.
    wrapper_func->setAttr("accelerator", builder.getStringAttr("neura"));

    // Clones the entire kernel region (all blocks) into the func body.
    Region &func_region = wrapper_func.getBody();
    IRMapping mapping;
    kernel_body.cloneInto(&func_region, mapping);

    // The cloned region now contains a copy of every block from the kernel.
    // Walks through and replaces neura.yield terminators with func.return.
    for (Block &block : func_region) {
      if (auto yield = dyn_cast<neura::YieldOp>(block.getTerminator())) {
        builder.setInsertionPoint(yield);
        SmallVector<Value> return_vals;
        for (Value v : yield.getResults()) {
          return_vals.push_back(v);
        }
        builder.create<func::ReturnOp>(loc, return_vals);
        yield.erase();
      }
    }

    // The kernel body is already in neura dataflow IR (all lowering passes
    // completed before this pass).  Only InsertDataMov is needed before mapper.
    PassManager pm(ctx);
    pm.enableVerifier(false);

    // InsertDataMov: wraps operands with neura.data_mov for the mapper.
    pm.addPass(neura::createInsertDataMovPass());

    if (failed(pm.run(dst_module))) {
      // Pre-mapper pipeline failed — extract best-effort metrics from partial
      // Neura IR using ResMII/RecMII analysis with the correct multi-CGRA arch.
      extractMetricsFromPartialIR(dst_module, compiled_ii, cp_depth, x_tiles,
                                  y_tiles);
      return failure();
    }

    // Computes ResMII/RecMII as analytical lower-bound (fallback when mapper
    // is skipped or fails).  Uses a custom arch sized to the actual tile array.
    {
      std::unique_ptr<neura::Architecture> custom_arch;
      const neura::Architecture *arch_ptr = &neura::getArchitecture();
      if (x_tiles > 0 && y_tiles > 0) {
        custom_arch =
            neura::getArchitecture().cloneWithNewDimensions(y_tiles, x_tiles);
        arch_ptr = custom_arch.get();
      }
      const neura::Architecture &architecture = *arch_ptr;

      dst_module.walk([&](func::FuncOp fn) {
        if (!fn->hasAttr("accelerator"))
          return;
        Region &region = fn.getBody();
        if (region.empty())
          return;
        int res_mii = neura::calculateResMii(region, architecture);
        auto cycles = neura::collectRecurrenceCycles(region);
        int rec_mii = 1;
        for (auto &cycle : cycles)
          rec_mii = std::max(rec_mii, cycle.length);
        compiled_ii = std::max({compiled_ii, res_mii, rec_mii});
        // Derives cp_depth from ALAP (As-Late-As-Possible) scheduling levels.
        std::set<Operation *> critical_ops;
        for (auto &cycle : cycles)
          for (Operation *op : cycle.operations)
            critical_ops.insert(op);
        auto sorted_ops = neura::getTopologicallySortedOps(region);
        if (!sorted_ops.empty()) {
          auto level_buckets =
              neura::getOpsInAlapLevels(sorted_ops, critical_ops);
          cp_depth = std::max(cp_depth, (int)level_buckets.size());
        }
        llvm::errs() << "[profileTask] analytical fallback: res_mii=" << res_mii
                     << " rec_mii=" << rec_mii
                     << " tiles=" << architecture.getNumTiles() << "\n";
      });
    }

    // Optionally run MapToAcceleratorPass to get the true compiled_ii.
    //
    // Guards:
    //   1. skip_mapper=true: caller explicitly requests analytical-only (e.g.
    //      speculative balance probes where the mapper may loop indefinitely).
    //   2. All non-Reserve operand producers must be DataMovOp (mapper crashes
    //      otherwise on unsupported ops like arith.minimumf).
    //   3. Kernel must be small enough (<= kMapperOpLimit ops) to avoid
    //      exponential backtracking blowup during speculative profiling.
    //
    // If any guard fires, the ResMII/RecMII values computed above serve as
    // the analytical lower-bound estimate (under-estimates true II on smaller
    // arrays, but are safe and instant).
    if (skip_mapper) {
      llvm::errs() << "[profileTask] Skipping mapper (analytical-only mode). "
                   << "Using analytical compiled_ii=" << compiled_ii << "\n";
      return success();
    }

    constexpr int kMapperOpLimit = 150;
    bool all_data_movs_ok = true;
    int total_mapped_ops = 0;
    dst_module.walk([&](func::FuncOp fn) {
      if (!fn->hasAttr("accelerator"))
        return;
      fn.walk([&](Operation *op) {
        if (isa<func::ReturnOp>(op))
          return;
        total_mapped_ops++;
        if (isa<neura::ReserveOp, neura::DataMovOp, neura::CtrlMovOp>(op))
          return;
        for (Value operand : op->getOperands()) {
          Operation *producer = operand.getDefiningOp();
          if (!producer)
            continue;
          if (!isa<neura::DataMovOp, neura::ReserveOp, neura::PhiStartOp,
                   neura::GrantOnceOp, neura::GrantPredicateOp, neura::YieldOp,
                   neura::KernelOp>(producer))
            all_data_movs_ok = false;
        }
      });
    });

    llvm::errs() << "[profileTask] mapper guard: total_ops=" << total_mapped_ops
                 << " all_data_movs=" << all_data_movs_ok
                 << " limit=" << kMapperOpLimit << "\n";

    if (all_data_movs_ok && total_mapped_ops <= kMapperOpLimit) {
      // Runs MapToAcceleratorPass in a fresh pass manager on the
      // already-lowered dst_module (pre-mapper pipeline already ran above).
      // Passes the correct tile dimensions so the mapper uses the right array.
      PassManager pm2(ctx);
      pm2.enableVerifier(false);
      if (x_tiles > 0 && y_tiles > 0) {
        neura::MapToAcceleratorOptions map_options;
        map_options.x_tiles = x_tiles;
        map_options.y_tiles = y_tiles;
        map_options.valid_tiles = valid_tiles;
        pm2.addPass(neura::createMapToAcceleratorPass(map_options));
      } else {
        pm2.addPass(neura::createMapToAcceleratorPass());
      }

      if (succeeded(pm2.run(dst_module))) {
        // Reads true compiled_ii from mapping_info; overrides analytical
        // estimate.
        dst_module.walk([&](func::FuncOp fn) {
          if (!fn->hasAttr("accelerator"))
            return;
          if (auto mapping_info = fn->getAttrOfType<DictionaryAttr>(
                  neura::attr::kMappingInfo)) {
            if (auto ii_attr =
                    mapping_info.getAs<IntegerAttr>(neura::attr::kCompiledII)) {
              compiled_ii = (int)ii_attr.getInt(); // authoritative value
              llvm::errs() << "[profileTask] mapper returned real II="
                           << compiled_ii << "\n";
            }
          }
        });
        return success();
      }
      // Mapper failed for all II values — keep ResMII/RecMII from above.
      llvm::errs() << "[profileTask] WARNING: MapToAcceleratorPass failed, "
                   << "keeping analytical fallback compiled_ii=" << compiled_ii
                   << "\n";
    } else {
      llvm::errs() << "[profileTask] Skipping mapper (too large or DataMov "
                   << "check failed). Using analytical compiled_ii="
                   << compiled_ii << "\n";
    }

    // Fallback already computed via ResMII/RecMII above; nothing more to do.
    return success();
  }

  // Extracts ResMII/RecMII from partially-lowered IR when the full pipeline
  // fails.  Uses custom arch sized to x_tiles × y_tiles if provided.
  void extractMetricsFromPartialIR(ModuleOp tmp_module, int &out_ii,
                                   int &out_cp_depth, int x_tiles = 0,
                                   int y_tiles = 0) {
    // Builds architecture: uses custom tile dimensions if provided.
    std::unique_ptr<neura::Architecture> custom_arch;
    const neura::Architecture *arch_ptr = &neura::getArchitecture();
    if (x_tiles > 0 && y_tiles > 0) {
      custom_arch =
          neura::getArchitecture().cloneWithNewDimensions(y_tiles, x_tiles);
      arch_ptr = custom_arch.get();
    }
    const neura::Architecture &architecture = *arch_ptr;

    int res_mii = 1;
    int rec_mii = 1;
    int cp_depth = 1;

    // Tries func-level analysis on partially-lowered funcs.
    tmp_module.walk([&](func::FuncOp fn) {
      if (!fn->hasAttr("accelerator"))
        return;
      Region &region = fn.getBody();
      if (region.empty())
        return;

      int local_res = neura::calculateResMii(region, architecture);
      res_mii = std::max(res_mii, local_res);

      auto cycles = neura::collectRecurrenceCycles(region);
      std::set<Operation *> critical_ops;
      for (auto &cycle : cycles) {
        rec_mii = std::max(rec_mii, (int)cycle.length);
        for (Operation *op : cycle.operations)
          critical_ops.insert(op);
      }

      auto sorted_ops = neura::getTopologicallySortedOps(region);
      if (!sorted_ops.empty()) {
        auto level_buckets =
            neura::getOpsInAlapLevels(sorted_ops, critical_ops);
        cp_depth = std::max(cp_depth, (int)level_buckets.size());
      }
    });

    out_ii = std::max(res_mii, rec_mii);
    out_cp_depth = std::max(cp_depth, 1);

    llvm::errs() << "[profileTask] (partial) ii=" << out_ii
                 << " (res_mii=" << res_mii << ", rec_mii=" << rec_mii
                 << "), steps=" << out_cp_depth << "\n";
  }

  // Computes total trip count for a task.
  //
  // The trip count is extracted from the taskflow.counter chain in the task
  // body. Each counter has lower_bound, upper_bound, and step attributes.
  // The trip count of a single counter is:
  //   ceil((upper_bound - lower_bound) / step)
  //
  // Counters form chains (root -> relay -> leaf). The trip count of a chain
  // is the product of each counter's individual trip count.
  //
  // Multiple independent counter chains execute concurrently on the CGRA,
  // so the total trip count is max(chain_product) across chains.
  static int64_t computeTripCount(TaskflowTaskOp task) {
    // Collects all taskflow.counter ops in the task body.
    SmallVector<TaskflowCounterOp> counters;
    for (Operation &op : task.getBody().front()) {
      if (auto counter = dyn_cast<TaskflowCounterOp>(&op))
        counters.push_back(counter);
    }

    if (counters.empty()) {
      // Defensive fallback: try neura.counter ops inside kernels.
      int64_t total = 1;
      task.walk([&](neura::KernelOp kernel) {
        int64_t kernel_product = 1;
        kernel.walk([&](neura::CounterOp counter_op) {
          auto getConst = [](Value val) -> std::optional<int64_t> {
            if (auto cst = val.getDefiningOp<neura::ConstantOp>()) {
              return cast<IntegerAttr>(cst.getValueAttr()).getInt();
            }
            return std::nullopt;
          };
          auto lb = getConst(counter_op.getLowerBound());
          auto ub = getConst(counter_op.getUpperBound());
          auto st = getConst(counter_op.getStep());
          if (lb && ub && st && *st > 0) {
            int64_t range = *ub - *lb;
            int64_t step = *st;
            int64_t tc = (range + step - 1) / step;
            if (tc > 0) {
              kernel_product *= tc;
            }
          }
          // Dynamic bounds: conservatively treated as trip_count=1 (unchanged
          // default)
        });
        total = std::max(total, kernel_product);
      });
      return (total > 0) ? total : 1;
    }

    // Builds counter chains from taskflow.counter ops.
    // A root counter has no parent_index. A relay/leaf counter has a
    // parent_index that is the result of another counter.
    // Finds root counters (no parent).
    SmallVector<TaskflowCounterOp> roots;
    for (auto counter : counters) {
      if (!counter.getParentIndex())
        roots.push_back(counter);
    }

    // Builds a map from parent counter result -> child counters.
    DenseMap<Value, SmallVector<TaskflowCounterOp>> parent_to_children;
    for (auto counter : counters) {
      if (auto parent = counter.getParentIndex())
        parent_to_children[parent].push_back(counter);
    }

    auto getConstantIndex = [](Value val) -> int64_t {
      if (auto cst = val.getDefiningOp<arith::ConstantIndexOp>()) {
        return cst.value();
      } else {
        // Unexpected dynamic bounds.
        // TODO: support dynamic bounds if needed.
        assert(false && "Expected constant index for counter parent_index");
      }
      return 0;
    };
    // Computes trip count for a single counter.
    auto counterTripCount =
        [&getConstantIndex](TaskflowCounterOp counter) -> int64_t {
      int64_t lb = getConstantIndex(counter.getLowerBound());
      int64_t ub = getConstantIndex(counter.getUpperBound());
      int64_t step = getConstantIndex(counter.getStep());
      if (step <= 0) {
        return 1;
      }
      int64_t range = ub - lb;
      return (range > 0) ? ((range + step - 1) / step) : 1;
    };

    // DFS from each root, accumulating the product along the chain.
    // Independent chains are concurrent -> take max across chains.
    int64_t total = 1;
    for (auto root : roots) {
      // Follows chain: root -> children -> grandchildren ...
      // Chain product = product of all counters in this chain.
      int64_t chain_product = 1;
      SmallVector<TaskflowCounterOp> worklist;
      worklist.push_back(root);
      while (!worklist.empty()) {
        auto cur = worklist.pop_back_val();
        chain_product *= counterTripCount(cur);
        auto it = parent_to_children.find(cur.getCounterIndex());
        if (it != parent_to_children.end()) {
          for (auto child : it->second)
            worklist.push_back(child);
        }
      }
      total = std::max(total, chain_product);
    }

    return (total > 0) ? total : 1;
  }
};

//===----------------------------------------------------------------------===//
// Pipeline Balancer
//===----------------------------------------------------------------------===//
// Identifies critical-path bottlenecks and allocates extra CGRAs.

class PipelineBalancer {
public:
  using ProfileFn = std::function<void(TaskGraphNode *, TaskflowTaskOp)>;

  // Runs pipeline balance on the graph.
  //
  // For each iteration, speculatively increments the bottleneck task's
  // cgra_count by 1 and re-profiles it via profile_fn. If the new estimated
  // latency is lower, the change is accepted; otherwise it is reverted and
  // the node is marked saturated (no further CGRA additions help it).
  //
  // This avoids blindly assigning more CGRAs without checking whether the
  // larger array actually produces a better compiled_ii.
  //
  // Returns true if any changes were accepted.
  bool balance(TaskDependencyGraph &graph, ProfileFn profile_fn) {
    bool changed = false;
    // Tracks nodes for which adding one more CGRA did not reduce latency.
    // These are skipped in subsequent iterations.
    llvm::DenseSet<TaskGraphNode *> saturated_nodes;

    for (int iter = 0; iter < kMaxBalanceIterations; ++iter) {
      int total_cgras = graph.getTotalAllocatedCGRAs();
      if (total_cgras >= kTotalCGRAs) {
        break;
      }

      // Recomputes critical path each iteration (may shift after rebalance).
      TaskGraphNode *bottleneck = findBottleneck(graph, saturated_nodes);
      if (!bottleneck) {
        break;
      }

      int old_cgra_count = bottleneck->cgra_count;
      int new_cgra_count = old_cgra_count + 1;

      // Check if incrementing cgra_count is feasible on the 4×4 grid.
      // TODO: This currently only checks the capacity (total CGRA count).
      // Ideally, we should invoke a global placement pass (aka
      // MapTaskOnCgraPass) here to verify if the speculatively increased CGRA
      // count and its proposed shape actually fit on the 4x4 grid alongside
      // other previously allocated tasks.
      //
      // Currently, MapTaskOnCgraPass does not support multi-CGRA task
      // placement. Once it does, we should call it here; if global placement
      // fails for the "best" shape, we should backtrack and try alternative
      // shapes before saturating the node.
      if (!canFitOnGrid(new_cgra_count)) {
        saturated_nodes.insert(bottleneck);
        continue;
      }

      // Saves state for potential rollback.
      int64_t old_latency = bottleneck->estimatedLatency();
      int64_t old_ii = bottleneck->ii;
      int64_t old_steps = bottleneck->steps;
      CgraShape old_shape = bottleneck->shape;

      // Speculatively applies the new CGRA count and re-profiles.
      bottleneck->cgra_count = new_cgra_count;
      bottleneck->shape = pickBestShape(new_cgra_count);

      llvm::errs() << "  Balance: trying Task " << bottleneck->id << " ("
                   << bottleneck->op.getTaskName().str()
                   << ") cgra_count=" << old_cgra_count << "->"
                   << new_cgra_count
                   << ", shape=" << bottleneck->shape.describe(new_cgra_count)
                   << ", tile_array="
                   << (bottleneck->shape.rows *
                       neura::getArchitecture().getPerCgraRows())
                   << "x"
                   << (bottleneck->shape.cols *
                       neura::getArchitecture().getPerCgraColumns())
                   << ", old_ii=" << old_ii << ", old_lat=" << old_latency
                   << "\n";

      profile_fn(bottleneck, bottleneck->op);

      int64_t new_latency = bottleneck->estimatedLatency();

      if (new_latency < old_latency) {
        // Accepted: the larger array produces a measurably better latency.
        changed = true;
        llvm::errs() << "  Balance: ACCEPTED Task " << bottleneck->id << " ("
                     << bottleneck->op.getTaskName().str()
                     << ") cgra_count=" << new_cgra_count << ", ii=" << old_ii
                     << "->" << bottleneck->ii << ", lat=" << old_latency
                     << "->" << new_latency
                     << ", total_cgras=" << graph.getTotalAllocatedCGRAs()
                     << "\n";
      } else {
        // Rejected: no latency improvement — roll back and mark saturated.
        llvm::errs() << "  Balance: REJECTED Task " << bottleneck->id
                     << " (ii=" << bottleneck->ii << ", lat=" << new_latency
                     << " >= old_lat=" << old_latency << "). Reverting.\n";
        bottleneck->cgra_count = old_cgra_count;
        bottleneck->shape = old_shape;
        bottleneck->ii = old_ii;
        bottleneck->steps = old_steps;
        saturated_nodes.insert(bottleneck);
      }
    }

    return changed;
  }

private:
  // Computes the weighted critical path length from a given node to any sink.
  int64_t computeCriticalPathFrom(TaskGraphNode *node,
                                  DenseMap<TaskGraphNode *, int64_t> &cache) {
    auto it = cache.find(node);
    if (it != cache.end()) {
      return it->second;
    }

    int64_t max_successor_path = 0;
    for (auto *succ : node->successors) {
      max_successor_path =
          std::max(max_successor_path, computeCriticalPathFrom(succ, cache));
    }

    int64_t path = node->estimatedLatency() + max_successor_path;
    cache[node] = path;
    return path;
  }

  // Computes the longest path from any source to the given node
  // (depth_from_source). Uses dynamic programming with memoization.
  int64_t computeDepthFromSource(TaskGraphNode *node,
                                 DenseMap<TaskGraphNode *, int64_t> &cache) {
    auto it = cache.find(node);
    if (it != cache.end()) {
      return it->second;
    }

    int64_t max_predecessor_depth = 0;
    for (auto *pred : node->predecessors) {
      max_predecessor_depth =
          std::max(max_predecessor_depth, computeDepthFromSource(pred, cache));
    }

    // depth_from_source(node) = max(depth_from_source(pred) for all preds)
    //                           + node's own latency.
    int64_t depth = max_predecessor_depth + node->estimatedLatency();
    cache[node] = depth;
    return depth;
  }

  // Finds the bottleneck node on the critical path using full slack analysis.
  //
  // For each node, slack is defined as:
  //   slack(node) = global_critical_path
  //                 - depth_from_source(node)
  //                 - depth_to_sink(node)
  //                 + node->estimatedLatency()
  //
  // where depth_from_source includes the node's own latency, and
  // depth_to_sink (computeCriticalPathFrom) also includes the node's own
  // latency, so we add it back once to avoid double-counting.
  //
  // A node is on the critical path iff slack == 0.
  // Among critical-path nodes, the one with highest individual latency
  // is the bottleneck (reducing its latency most benefits the pipeline).
  TaskGraphNode *
  findBottleneck(TaskDependencyGraph &graph,
                 const llvm::DenseSet<TaskGraphNode *> &ignored) {
    llvm::DenseMap<TaskGraphNode *, int64_t> to_sink_cache;
    llvm::DenseMap<TaskGraphNode *, int64_t> from_source_cache;

    // Computes depth_to_sink for all nodes (via computeCriticalPathFrom).
    int64_t global_critical_path = 0;
    for (auto &node : graph.nodes) {
      int64_t cp = computeCriticalPathFrom(node.get(), to_sink_cache);
      global_critical_path = std::max(global_critical_path, cp);
    }

    // Computes depth_from_source for all nodes.
    for (auto &node : graph.nodes) {
      computeDepthFromSource(node.get(), from_source_cache);
    }

    // Finds the critical-path node with highest individual latency.
    TaskGraphNode *bottleneck = nullptr;
    int64_t max_latency = -1;

    for (auto &node : graph.nodes) {
      if (ignored.count(node.get()))
        continue;
      if (node->cgra_count >= node->trip_count)
        continue;
      // Per-task CGRA limit: no point trying to add more.
      if (node->cgra_count >= kMaxCgrasPerTask)
        continue;

      int64_t depth_from = from_source_cache[node.get()];
      int64_t depth_to = to_sink_cache[node.get()];

      // slack = global_cp - depth_from - depth_to + node_latency
      // (because both depth_from and depth_to include node's own latency).
      int64_t slack = global_critical_path - depth_from - depth_to +
                      node->estimatedLatency();

      if (slack != 0)
        continue; // Not on the critical path.

      if (node->estimatedLatency() > max_latency) {
        max_latency = node->estimatedLatency();
        bottleneck = node.get();
      }
    }
    return bottleneck;
  }
};

//===----------------------------------------------------------------------===//
// Utilization Fusion
//===----------------------------------------------------------------------===//
// Merges independent tasks (no edge in either direction) into a single task
// to reduce total CGRA count.  Fusion candidates are chosen to minimize
// |trip_count_a - trip_count_b| for balanced utilization.

class UtilizationFuser {
public:
  using ProfileFn = std::function<void(TaskGraphNode *, TaskflowTaskOp)>;

  // Runs utilization fusion. Returns true if any fusions occurred.
  // Only performs ONE fusion per call — the caller should rebuild the graph
  // and call again if more fusions are desired.
  bool fuse(func::FuncOp func, TaskDependencyGraph &graph,
            ProfileFn profile_fn) {
    auto pair = findBestFusionCandidate(graph);
    if (!pair) {
      return false;
    }

    auto [node_a, node_b] = *pair;

    llvm::errs() << "  Fuse: Task " << node_a->id << " ("
                 << node_a->op.getTaskName().str() << ") + Task " << node_b->id
                 << " (" << node_b->op.getTaskName().str() << ")\n";

    return performFusion(func, node_a, node_b, graph, profile_fn);
  }

private:
  // Finds the best pair of independent tasks to fuse.
  // Selects the pair with the most balanced trip_count (minimizes
  // |trip_count_a - trip_count_b|) to avoid wasting computation when
  // the fused task executes both loop nests concurrently on the shared array.
  std::optional<std::pair<TaskGraphNode *, TaskGraphNode *>>
  findBestFusionCandidate(TaskDependencyGraph &graph) {
    TaskGraphNode *best_a = nullptr;
    TaskGraphNode *best_b = nullptr;
    int64_t best_cost = INT64_MAX;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
      for (size_t j = i + 1; j < graph.nodes.size(); ++j) {
        auto *a = graph.nodes[i].get();
        auto *b = graph.nodes[j].get();

        if (!graph.areIndependent(a, b)) {
          continue;
        }

        // Fusion requires single-block task bodies (counter-mode tasks).
        if (!a->op.getBody().hasOneBlock() || !b->op.getBody().hasOneBlock()) {
          continue;
        }

        // Legality: checks no intermediate task depends on a or b.
        if (!canSafelyFuse(a, b, graph)) {
          continue;
        }

        // Utilization metric: minimize |trip_count_a - trip_count_b|.
        // Balanced trip counts mean less wasted computation when fused
        // tasks execute concurrently on the shared tile array.
        int64_t cost = std::abs(a->trip_count - b->trip_count);
        if (cost < best_cost) {
          best_cost = cost;
          best_a = a;
          best_b = b;
        }
      }
    }

    if (!best_a || !best_b) {
      return std::nullopt;
    }
    return std::make_pair(best_a, best_b);
  }

  // Checks whether fusing tasks a and b is safe w.r.t. dominance.
  // Returns false if any other task positioned between a and b in the IR
  // has a dependency (edge) on either a or b — because moving the fused
  // task would break that intermediate dependency.
  bool canSafelyFuse(TaskGraphNode *a, TaskGraphNode *b,
                     TaskDependencyGraph &graph) {
    auto *task_a = a->op.getOperation();
    auto *task_b = b->op.getOperation();

    if (task_a->getBlock() != task_b->getBlock())
      return false;

    // Ensures task_a is before task_b.
    if (!task_a->isBeforeInBlock(task_b)) {
      std::swap(task_a, task_b);
      std::swap(a, b);
    }

    // Check: no other task between a and b should have an edge from/to a or b.
    for (auto &node : graph.nodes) {
      if (node.get() == a || node.get() == b)
        continue;

      auto *other_op = node->op.getOperation();
      if (other_op->getBlock() != task_a->getBlock())
        continue;

      // Is this node between task_a and task_b?
      if (task_a->isBeforeInBlock(other_op) &&
          other_op->isBeforeInBlock(task_b)) {
        // Checks if this intermediate task has any dependency on a or b.
        if (!graph.areIndependent(a, node.get()) ||
            !graph.areIndependent(b, node.get())) {
          return false;
        }
      }
    }
    return true;
  }

  // Performs IR-level fusion of two independent tasks.
  //
  // DFG-Level Fusion:
  //   Since this pass runs post-lowering, each task body is single-block
  //   containing counter ops, one neura.kernel op, and a taskflow.yield.
  //   Fusion concatenates both DFGs into a single neura.kernel (they are
  //   independent, so just placed side-by-side).  The fused task is then
  //   profiled through InsertDataMov + mapper to get accurate compiled_ii.
  bool performFusion(func::FuncOp func, TaskGraphNode *node_a,
                     TaskGraphNode *node_b, TaskDependencyGraph &graph,
                     ProfileFn profile_fn) {
    auto task_a = node_a->op;
    auto task_b = node_b->op;

    // Safety: both tasks must be in the same block.
    if (task_a->getBlock() != task_b->getBlock()) {
      llvm::errs() << "  [Fuse] Skipping: tasks in different blocks\n";
      return false;
    }

    // Safety: fusion requires single-block task bodies.
    if (!task_a.getBody().hasOneBlock() || !task_b.getBody().hasOneBlock()) {
      llvm::errs() << "  [Fuse] Skipping: multi-block task body\n";
      return false;
    }

    // Ensures task_a comes before task_b in the IR for correct dominance.
    if (!task_a->isBeforeInBlock(task_b)) {
      std::swap(task_a, task_b);
      std::swap(node_a, node_b);
    }

    llvm::errs() << "  [Fuse] Merging " << task_a.getTaskName() << " + "
                 << task_b.getTaskName() << "\n";

    // Computes the correct insertion point: must be after all operands of
    // both tasks are defined, but before any consumer of either task's
    // results. We find the latest-positioned operand definition and insert
    // right after it.
    Operation *latest_def = task_a.getOperation();
    auto updateLatest = [&](ValueRange operands) {
      for (Value v : operands) {
        if (auto *def_op = v.getDefiningOp()) {
          if (def_op->getBlock() == task_a->getBlock() &&
              latest_def->isBeforeInBlock(def_op)) {
            latest_def = def_op;
          }
        }
      }
    };
    updateLatest(task_a.getDependencyReadIn());
    updateLatest(task_a.getDependencyWriteIn());
    updateLatest(task_a.getValueInputs());
    updateLatest(task_b.getDependencyReadIn());
    updateLatest(task_b.getDependencyWriteIn());
    updateLatest(task_b.getValueInputs());

    // Inserts right after the latest operand definition.
    OpBuilder builder(latest_def->getBlock(),
                      std::next(Block::iterator(latest_def)));

    // Step 1: Builds merged operand lists.
    SmallVector<Value> merged_read_memrefs;
    SmallVector<Value> merged_write_memrefs;
    SmallVector<Value> merged_value_inputs;
    SmallVector<Value> merged_original_read_memrefs;
    SmallVector<Value> merged_original_write_memrefs;

    // Deduplicates values when merging operand lists from both tasks.
    auto addUnique = [](SmallVector<Value> &target, ValueRange source) {
      for (Value v : source) {
        if (llvm::find(target, v) == target.end()) {
          target.push_back(v);
        }
      }
    };

    addUnique(merged_read_memrefs, task_a.getDependencyReadIn());
    addUnique(merged_read_memrefs, task_b.getDependencyReadIn());
    addUnique(merged_write_memrefs, task_a.getDependencyWriteIn());
    addUnique(merged_write_memrefs, task_b.getDependencyWriteIn());
    addUnique(merged_value_inputs, task_a.getValueInputs());
    addUnique(merged_value_inputs, task_b.getValueInputs());
    addUnique(merged_original_read_memrefs, task_a.getOriginalReadMemrefs());
    addUnique(merged_original_read_memrefs, task_b.getOriginalReadMemrefs());
    addUnique(merged_original_write_memrefs, task_a.getOriginalWriteMemrefs());
    addUnique(merged_original_write_memrefs, task_b.getOriginalWriteMemrefs());

    // Step 2: Builds result types.
    SmallVector<Type> read_output_types;
    for (Value v : merged_read_memrefs) {
      read_output_types.push_back(v.getType());
    }
    SmallVector<Type> write_output_types;
    for (Value v : merged_write_memrefs) {
      write_output_types.push_back(v.getType());
    }
    SmallVector<Type> value_output_types;
    for (Value v : task_a.getValueOutputs()) {
      value_output_types.push_back(v.getType());
    }
    for (Value v : task_b.getValueOutputs()) {
      value_output_types.push_back(v.getType());
    }

    // Step 3: Creates fused task name.
    std::string fused_name = task_a.getTaskName().str() + "_" +
                             task_b.getTaskName().str() + "_utilfused";

    // Step 4: Creates the fused task op.
    auto fused_task = builder.create<TaskflowTaskOp>(
        task_a.getLoc(), read_output_types, write_output_types,
        value_output_types, merged_read_memrefs, merged_write_memrefs,
        merged_value_inputs, fused_name, merged_original_read_memrefs,
        merged_original_write_memrefs);

    // ================================================================
    // Region-Level Fusion (single-block task bodies)
    // ================================================================

    // Step 5: Clones both task regions into the fused task body.
    // Maps source task's block args to fused task's block args.
    auto buildTaskArgMapping = [&](TaskflowTaskOp orig_task,
                                   Region &fused_region, IRMapping &mapping) {
      Block &src_entry = orig_task.getBody().front();
      unsigned src_idx = 0;
      unsigned read_count = orig_task.getDependencyReadIn().size();
      unsigned write_count = orig_task.getDependencyWriteIn().size();

      for (unsigned i = 0; i < read_count; ++i) {
        Value orig_memref = orig_task.getDependencyReadIn()[i];
        auto it = llvm::find(merged_read_memrefs, orig_memref);
        assert(it != merged_read_memrefs.end());
        unsigned fused_idx = std::distance(merged_read_memrefs.begin(), it);
        mapping.map(src_entry.getArgument(src_idx + i),
                    fused_region.front().getArgument(fused_idx));
      }
      src_idx += read_count;

      for (unsigned i = 0; i < write_count; ++i) {
        Value orig_memref = orig_task.getDependencyWriteIn()[i];
        auto it = llvm::find(merged_write_memrefs, orig_memref);
        assert(it != merged_write_memrefs.end());
        unsigned fused_idx = merged_read_memrefs.size() +
                             std::distance(merged_write_memrefs.begin(), it);
        mapping.map(src_entry.getArgument(src_idx + i),
                    fused_region.front().getArgument(fused_idx));
      }
      src_idx += write_count;

      for (unsigned i = 0; i < orig_task.getValueInputs().size(); ++i) {
        Value orig_val = orig_task.getValueInputs()[i];
        auto it = llvm::find(merged_value_inputs, orig_val);
        assert(it != merged_value_inputs.end());
        unsigned fused_idx = merged_read_memrefs.size() +
                             merged_write_memrefs.size() +
                             std::distance(merged_value_inputs.begin(), it);
        mapping.map(src_entry.getArgument(src_idx + i),
                    fused_region.front().getArgument(fused_idx));
      }
    };

    // Creates the fused task's entry block with merged block args.
    Block *entry_block = new Block();
    fused_task.getBody().push_back(entry_block);
    for (Value v : merged_read_memrefs)
      entry_block->addArgument(v.getType(), fused_task.getLoc());
    for (Value v : merged_write_memrefs)
      entry_block->addArgument(v.getType(), fused_task.getLoc());
    for (Value v : merged_value_inputs)
      entry_block->addArgument(v.getType(), fused_task.getLoc());

    // Clones non-yield ops from task_a and task_b into fused entry block.
    IRMapping mapping_a;
    buildTaskArgMapping(task_a, fused_task.getBody(), mapping_a);

    IRMapping mapping_b;
    buildTaskArgMapping(task_b, fused_task.getBody(), mapping_b);

    // Clones all non-yield ops from task_a's body into the fused entry block.
    {
      OpBuilder ob = OpBuilder::atBlockEnd(entry_block);
      for (Operation &op : task_a.getBody().front()) {
        if (isa<TaskflowYieldOp>(&op))
          continue;
        ob.clone(op, mapping_a);
      }
    }

    // Clones all non-yield ops from task_b's body into the fused entry block.
    {
      OpBuilder ob = OpBuilder::atBlockEnd(entry_block);
      for (Operation &op : task_b.getBody().front()) {
        if (isa<TaskflowYieldOp>(&op))
          continue;
        ob.clone(op, mapping_b);
      }
    }

    // Identifies the two cloned kernels in the fused entry block.
    neura::KernelOp cloned_kernel_a, cloned_kernel_b;
    {
      SmallVector<neura::KernelOp> fused_kernels;
      fused_task.walk([&](neura::KernelOp k) { fused_kernels.push_back(k); });
      assert(fused_kernels.size() == 2 &&
             "[performFusion] expected exactly 2 cloned kernels");
      cloned_kernel_a = fused_kernels[0];
      cloned_kernel_b = fused_kernels[1];
    }

    // Merges the two cloned kernels into one fused kernel.
    SmallVector<Value> merged_kernel_inputs;
    auto addKernelInputs = [&](neura::KernelOp kernel) {
      for (Value inp : kernel.getInputs()) {
        if (llvm::find(merged_kernel_inputs, inp) ==
            merged_kernel_inputs.end()) {
          merged_kernel_inputs.push_back(inp);
        }
      }
    };
    addKernelInputs(cloned_kernel_a);
    addKernelInputs(cloned_kernel_b);

    // Concatenates iter_args from both kernels (kernel_a first, then kernel_b).
    SmallVector<Value> merged_iter_args;
    for (Value v : cloned_kernel_a.getIterArgsInit())
      merged_iter_args.push_back(v);
    for (Value v : cloned_kernel_b.getIterArgsInit())
      merged_iter_args.push_back(v);

    // Concatenates result types from both kernels.
    SmallVector<Type> merged_kernel_results;
    for (Type t : cloned_kernel_a.getResultTypes())
      merged_kernel_results.push_back(t);
    for (Type t : cloned_kernel_b.getResultTypes())
      merged_kernel_results.push_back(t);

    // Creates the fused kernel op right before cloned_kernel_a.
    OpBuilder fused_kb(cloned_kernel_a);
    auto fused_kernel = fused_kb.create<neura::KernelOp>(
        task_a.getLoc(), merged_kernel_results, merged_kernel_inputs,
        merged_iter_args,
        /*cgra_id=*/nullptr, /*kernel_name=*/nullptr,
        /*accelerator=*/builder.getStringAttr("neura"));
    fused_kernel->setAttr("dataflow_mode", builder.getStringAttr("predicate"));

    // Builds kernel entry block and block-arg mappings.
    Region &fused_kernel_region = fused_kernel.getBody();
    Block *kernel_body = builder.createBlock(&fused_kernel_region);
    for (Value v : merged_kernel_inputs)
      kernel_body->addArgument(v.getType(), task_a.getLoc());
    for (Value v : merged_iter_args)
      kernel_body->addArgument(v.getType(), task_a.getLoc());

    // Maps each original kernel's block args to the fused kernel's block args.
    // iter_offset tracks where this kernel's iter_args start in the merged
    // list.
    auto buildKernelArgMapping = [&](neura::KernelOp kernel,
                                     unsigned iter_offset) -> IRMapping {
      IRMapping km;
      Block &src_entry = kernel.getBody().front();
      unsigned src_idx = 0;

      // Maps kernel input args.
      for (Value inp : kernel.getInputs()) {
        auto it = llvm::find(merged_kernel_inputs, inp);
        assert(it != merged_kernel_inputs.end());
        unsigned fused_idx = std::distance(merged_kernel_inputs.begin(), it);
        km.map(src_entry.getArgument(src_idx),
               kernel_body->getArgument(fused_idx));
        src_idx++;
      }

      // Maps iter_args.
      for (unsigned i = 0; i < kernel.getIterArgsInit().size(); ++i) {
        km.map(src_entry.getArgument(src_idx + i),
               kernel_body->getArgument(merged_kernel_inputs.size() +
                                        iter_offset + i));
      }

      return km;
    };

    IRMapping kernel_mapping_a = buildKernelArgMapping(cloned_kernel_a, 0);
    IRMapping kernel_mapping_b = buildKernelArgMapping(
        cloned_kernel_b, cloned_kernel_a.getIterArgsInit().size());

    // Clones DFG ops from both kernels and creates the combined neura.yield.
    {
      OpBuilder kb = OpBuilder::atBlockEnd(kernel_body);
      for (auto &op : cloned_kernel_a.getBody().front().getOperations()) {
        if (isa<neura::YieldOp>(&op))
          continue;
        kb.clone(op, kernel_mapping_a);
      }
      for (auto &op : cloned_kernel_b.getBody().front().getOperations()) {
        if (isa<neura::YieldOp>(&op))
          continue;
        kb.clone(op, kernel_mapping_b);
      }

      // Collects yield operands from both kernels' original yields.
      SmallVector<Value> merged_iter_args_next;
      SmallVector<Value> merged_results;
      if (auto yield_a = dyn_cast<neura::YieldOp>(
              cloned_kernel_a.getBody().front().getTerminator())) {
        for (Value v : yield_a.getIterArgsNext())
          merged_iter_args_next.push_back(kernel_mapping_a.lookupOrDefault(v));
        for (Value v : yield_a.getResults())
          merged_results.push_back(kernel_mapping_a.lookupOrDefault(v));
      }
      if (auto yield_b = dyn_cast<neura::YieldOp>(
              cloned_kernel_b.getBody().front().getTerminator())) {
        for (Value v : yield_b.getIterArgsNext())
          merged_iter_args_next.push_back(kernel_mapping_b.lookupOrDefault(v));
        for (Value v : yield_b.getResults())
          merged_results.push_back(kernel_mapping_b.lookupOrDefault(v));
      }

      // Creates the combined neura.yield and preserves yield_type from
      // kernel_a.
      auto fused_yield = kb.create<neura::YieldOp>(
          task_a.getLoc(), merged_iter_args_next, merged_results);
      if (auto yield_a = dyn_cast<neura::YieldOp>(
              cloned_kernel_a.getBody().front().getTerminator())) {
        if (auto attr = yield_a->getAttr("yield_type"))
          fused_yield->setAttr("yield_type", attr);
      }
    }

    // Replaces uses of cloned kernels with fused kernel results, then erases.
    {
      unsigned result_idx = 0;
      for (unsigned i = 0; i < cloned_kernel_a.getNumResults(); ++i) {
        cloned_kernel_a.getResult(i).replaceAllUsesWith(
            fused_kernel.getResult(result_idx++));
      }
      for (unsigned i = 0; i < cloned_kernel_b.getNumResults(); ++i) {
        cloned_kernel_b.getResult(i).replaceAllUsesWith(
            fused_kernel.getResult(result_idx++));
      }
      cloned_kernel_a.erase();
      cloned_kernel_b.erase();
    }

    // Builds and inserts the merged taskflow.yield.
    {
      // Read outputs pass through the entry block's read-memref args.
      SmallVector<Value> yield_reads;
      for (size_t i = 0; i < merged_read_memrefs.size(); ++i) {
        yield_reads.push_back(entry_block->getArgument(i));
      }

      // Writes outputs pass through the entry block's write-memref args.
      SmallVector<Value> yield_writes;
      for (size_t i = 0; i < merged_write_memrefs.size(); ++i) {
        yield_writes.push_back(
            entry_block->getArgument(merged_read_memrefs.size() + i));
      }

      // Value outputs come from the fused kernel's results.
      SmallVector<Value> yield_values;
      unsigned val_idx = 0;
      for (unsigned i = 0; i < task_a.getValueOutputs().size(); ++i)
        yield_values.push_back(fused_kernel.getResult(val_idx++));
      for (unsigned i = 0; i < task_b.getValueOutputs().size(); ++i)
        yield_values.push_back(fused_kernel.getResult(val_idx++));

      // Erases auto-inserted yield and creates the merged one.
      if (!entry_block->empty()) {
        if (auto existing_yield =
                dyn_cast<TaskflowYieldOp>(entry_block->back())) {
          existing_yield.erase();
        }
      }
      OpBuilder tb = OpBuilder::atBlockEnd(entry_block);
      tb.create<TaskflowYieldOp>(fused_task.getLoc(), yield_reads, yield_writes,
                                 yield_values);
    }

    // Step 6: Sets fused trip_count (max of both independent tasks).
    int64_t fused_trip = std::max(node_a->trip_count, node_b->trip_count);
    fused_task->setAttr("trip_count",
                        OpBuilder(fused_task).getI64IntegerAttr(fused_trip));

    // Profiles the fused task to obtain its compiled_ii and steps.
    {
      TaskGraphNode fused_node(/*id=*/0, fused_task);
      fused_node.trip_count = fused_trip;
      profile_fn(&fused_node, fused_task);
      fused_task->setAttr(
          "steps", OpBuilder(fused_task).getI64IntegerAttr(fused_node.steps));
      fused_task->setAttr(
          "compiled_ii",
          OpBuilder(fused_task).getI64IntegerAttr(fused_node.ii));
    }

    // Step 7: Replaces uses of original tasks' results.
    // Value outputs are ordered: task_a's value outputs first, then task_b's.
    unsigned val_offset_a = 0;
    unsigned val_offset_b = task_a.getValueOutputs().size();
    replaceTaskResults(task_a, fused_task, merged_read_memrefs,
                       merged_write_memrefs, val_offset_a);
    replaceTaskResults(task_b, fused_task, merged_read_memrefs,
                       merged_write_memrefs, val_offset_b);

    // Step 8: Erases original tasks.
    // Verifies no remaining uses before erasing.
    auto verifyNoUses = [](TaskflowTaskOp task, StringRef label) {
      for (Value result : task->getResults()) {
        if (!result.use_empty()) {
          llvm::errs() << "[performFusion] ERROR: " << label << " result #"
                       << cast<OpResult>(result).getResultNumber()
                       << " still has uses:\n";
          for (auto &use : result.getUses()) {
            llvm::errs() << "  used by: ";
            use.getOwner()->print(llvm::errs());
            llvm::errs() << "\n";
          }
        }
      }
    };
    verifyNoUses(task_a, "task_a");
    verifyNoUses(task_b, "task_b");
    task_a.erase();
    task_b.erase();

    return true;
  }

  // Finds the index of a value in a list.
  unsigned findOperandIndex(const SmallVector<Value> &list, Value v) {
    for (unsigned i = 0; i < list.size(); ++i) {
      if (list[i] == v)
        return i;
    }
    llvm_unreachable("Value not found in operand list");
  }

  // Replaces results of an original task with corresponding results from the
  // fused task. Handles both write outputs (memrefs) and value outputs
  // (reductions, iter_args).
  void replaceTaskResults(TaskflowTaskOp orig_task, TaskflowTaskOp fused_task,
                          const SmallVector<Value> &merged_read_memrefs,
                          const SmallVector<Value> &merged_write_memrefs,
                          unsigned value_output_offset) {
    // Read outputs: maps by matching the original read memref to its
    // position in the merged read memrefs list.
    for (unsigned i = 0; i < orig_task.getDependencyReadOut().size(); ++i) {
      Value orig_result = orig_task.getDependencyReadOut()[i];
      Value orig_read = orig_task.getDependencyReadIn()[i];
      unsigned fused_idx = findOperandIndex(merged_read_memrefs, orig_read);
      orig_result.replaceAllUsesWith(
          fused_task.getDependencyReadOut()[fused_idx]);
    }
    // Writes outputs: maps by matching the original write memref to its
    // position in the merged write memrefs list.
    for (unsigned i = 0; i < orig_task.getDependencyWriteOut().size(); ++i) {
      Value orig_result = orig_task.getDependencyWriteOut()[i];
      Value orig_write = orig_task.getDependencyWriteIn()[i];
      unsigned fused_idx = findOperandIndex(merged_write_memrefs, orig_write);
      orig_result.replaceAllUsesWith(
          fused_task.getDependencyWriteOut()[fused_idx]);
    }
    // Value outputs: each original task's value_output[i] maps to
    // fused_task.getValueOutputs()[value_output_offset + i].
    for (unsigned i = 0; i < orig_task.getValueOutputs().size(); ++i) {
      Value orig_val = orig_task.getValueOutputs()[i];
      orig_val.replaceAllUsesWith(
          fused_task.getValueOutputs()[value_output_offset + i]);
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ResourceAwareTaskOptimizationPass
    : public PassWrapper<ResourceAwareTaskOptimizationPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ResourceAwareTaskOptimizationPass)

  ResourceAwareTaskOptimizationPass() = default;
  ResourceAwareTaskOptimizationPass(
      const ResourceAwareTaskOptimizationPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "resource-aware-task-optimization";
  }

  StringRef getDescription() const override {
    return "Balances pipeline latency and fuses independent tasks for CGRA "
           "utilization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<neura::NeuraDialect>();
    registry.insert<taskflow::TaskflowDialect>();
  }

  // Estimation mode for profiling task II / steps.
  //   "compiled" (default): runs the full Neura lowering + mapping pipeline
  //       to obtain accurate compiled_ii and steps from MapToAcceleratorPass.
  //   "analytical": uses only ResMII / RecMII analytical estimates without
  //       running the mapper.  Much faster but less accurate — useful for
  //       rapid design-space exploration or when the mapper is unavailable.
  Option<std::string> estimationMode{
      *this, "estimation-mode",
      llvm::cl::desc(
          "Profiling estimation mode: 'compiled' (default) runs the full "
          "Neura lowering + mapping pipeline for accurate II/steps; "
          "'analytical' uses only ResMII/RecMII analytical estimates "
          "(faster but less accurate)."),
      llvm::cl::init("compiled")};

  // Controls whether the balance phase skips the mapper during speculative
  // profiling.  Default is true (analytical-only) for speed — the mapper can
  // backtrack indefinitely on larger tile arrays.  Set to false to run the
  // real mapper during balance probes for accurate compiled_ii at the cost
  // of longer compile times.
  Option<bool> balanceSkipMapper{
      *this, "balance-skip-mapper",
      llvm::cl::desc(
          "Whether balance probes skip the mapper and use only analytical "
          "ResMII/RecMII estimates (default: true).  Set to false for "
          "accurate compiled_ii during balance at the cost of compile time."),
      llvm::cl::init(true)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    bool use_analytical = (estimationMode.getValue() == "analytical");

    llvm::errs() << "=== ResourceAwareTaskOptimization on " << func.getName()
                 << " (estimation-mode=" << estimationMode.getValue()
                 << ") ===\n";

    constexpr int kMaxOuterIterations = 10;

    for (int outer = 0; outer < kMaxOuterIterations; ++outer) {
      // Rebuilds graph from current IR state.
      TaskDependencyGraph graph;
      graph.build(func, use_analytical);

      if (graph.nodes.empty()) {
        return;
      }

      int num_tasks = graph.nodes.size();

      // Asserts that initial tasks fit in the grid.
      assert(num_tasks <= kTotalCGRAs &&
             "Number of tasks exceeds 4x4 CGRA grid capacity! "
             "Reduce task count via streaming fusion or increase grid size.");

      llvm::errs() << "[ResourceAware] Iteration " << outer << ": " << num_tasks
                   << " tasks\n";
      for (auto &node : graph.nodes) {
        llvm::errs() << "  Task " << node->id << " (" << node->op.getTaskName()
                     << "): trip_count=" << node->trip_count
                     << ", cgra_count=" << node->cgra_count
                     << ", est_latency=" << node->estimatedLatency() << "\n";
      }

      // Phase 1: Utilization Fusion.
      // Fuses independent tasks to free up CGRA budget for balance.
      UtilizationFuser fuser;
      // Exposes TaskDependencyGraph::profileTask to UtilizationFuser via a
      // lambda so fused tasks get real profiling.  In analytical mode, the
      // mapper is skipped entirely (only ResMII/RecMII estimates are used).
      auto profile_fn = [&graph, use_analytical](TaskGraphNode *node,
                                                 TaskflowTaskOp task) {
        graph.profileTask(node, task, /*skip_mapper=*/use_analytical);
      };
      bool fuse_changed = fuser.fuse(func, graph, profile_fn);

      llvm::errs() << "[ResourceAware] After fusion: total_cgras="
                   << graph.getTotalAllocatedCGRAs() << "\n";

      // Rebuilds graph after fusion (tasks may have been erased/created).
      if (fuse_changed) {
        graph = TaskDependencyGraph();
        graph.build(func, use_analytical);
      }

      // Phase 2: Latency-Aware Pipeline Balance.
      // Balance probes always use analytical-only profiling (skip_mapper=true)
      // to avoid exponential backtracking blowup during speculative probing.
      // The balance-skip-mapper flag now only controls whether a final
      // verification mapper run is performed after convergence (see below).
      auto balance_profile_fn = [&graph](TaskGraphNode *node,
                                         TaskflowTaskOp task) {
        graph.profileTask(node, task, /*skip_mapper=*/true);
      };
      PipelineBalancer balancer;
      bool balance_changed = balancer.balance(graph, balance_profile_fn);

      // Writes back attributes so the next iteration sees them.
      if (balance_changed || fuse_changed) {
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          node->op->setAttr("cgra_count",
                            b.getI32IntegerAttr(node->cgra_count));
          if (node->ii != kUnprofiled) {
            node->op->setAttr("compiled_ii", b.getI32IntegerAttr(node->ii));
          }
          if (node->steps != kUnprofiled) {
            node->op->setAttr("steps", b.getI32IntegerAttr(node->steps));
          }
          if (node->trip_count > 0) {
            node->op->setAttr("trip_count",
                              b.getI32IntegerAttr(node->trip_count));
          }
          if (balance_changed && node->cgra_count > 1) {
            llvm::errs() << "  [Balance] " << node->op.getTaskName()
                         << " -> cgra_count=" << node->cgra_count
                         << ", est_latency=" << node->estimatedLatency()
                         << "\n";
          }
        }
      }

      llvm::errs() << "[ResourceAware] After balance: total_cgras="
                   << graph.getTotalAllocatedCGRAs() << "\n";

      if (!balance_changed && !fuse_changed) {
        // Converged — optionally re-profile with the real mapper for accuracy.
        // When balance-skip-mapper=false, runs the mapper once per task that
        // had its cgra_count changed, to get authoritative compiled_ii/steps.
        bool balance_skip_mapper =
            use_analytical || balanceSkipMapper.getValue();
        if (!balance_skip_mapper) {
          llvm::errs() << "[ResourceAware] Running final mapper verification "
                       << "for converged allocation...\n";
          for (auto &node : graph.nodes) {
            // Re-profile with the real mapper to get accurate II/steps.
            node->shape = pickBestShape(node->cgra_count);
            graph.profileTask(node.get(), node->op,
                              /*skip_mapper=*/false);
          }
        }

        // Writes ALL attributes (cgra_count, ii, steps) to IR for every task.
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          node->shape = pickBestShape(node->cgra_count);
          node->op->setAttr("cgra_count",
                            b.getI32IntegerAttr(node->cgra_count));
          node->op->setAttr("compiled_ii", b.getI32IntegerAttr(node->ii));
          node->op->setAttr("steps", b.getI32IntegerAttr(node->steps));
          node->op->setAttr("trip_count",
                            b.getI32IntegerAttr(node->trip_count));
          // Writes cgra_shape attribute: simple "NxM" bounding-box string.
          // The detailed occupancy diagram is printed in the summary below.
          std::string shape_str = node->shape.irAttr();
          node->op->setAttr("cgra_shape", b.getStringAttr(shape_str));
        }
        break;
      }
    }

    // Performs final validation and tile occupation summary with visual 4x4
    // grid.
    {
      TaskDependencyGraph final_graph;
      final_graph.build(func, use_analytical);
      int final_total = final_graph.getTotalAllocatedCGRAs();

      // Assigns each task a single character label for the combined grid.
      // Tasks are labelled '0','1','2',... ; free cells shown as '.'.
      // grid[row][col] == -1 means free.
      std::vector<std::vector<int>> combined_grid(
          kCgraGridRows, std::vector<int>(kCgraGridCols, -1));

      // Packs tasks onto the grid left-to-right, top-to-bottom.
      int next_col = 0, next_row = 0;
      int task_idx = 0;

      llvm::errs() << "\n=== Tile Occupation Summary (4x" << kCgraGridCols
                   << " CGRA Grid) ===\n";

      for (auto &node : final_graph.nodes) {
        auto shape = pickBestShape(node->cgra_count);
        int tile_rows = shape.rows * neura::getArchitecture().getPerCgraRows();
        int tile_cols =
            shape.cols * neura::getArchitecture().getPerCgraColumns();

        // Per-task grid (shape.rows x shape.cols bbox, filled up to
        // cgra_count).
        llvm::errs() << "\n  [" << task_idx << "] " << node->op.getTaskName()
                     << "  cgra_count=" << node->cgra_count
                     << "  shape=" << shape.describe(node->cgra_count)
                     << "  tile_array=" << tile_rows << "x" << tile_cols
                     << "  ii=" << node->ii << "  steps=" << node->steps
                     << "  trip_count=" << node->trip_count << "\n";

        // Draws a per-task bounding-box grid (shape.rows x shape.cols).
        int remaining = node->cgra_count;
        llvm::errs() << "      +";
        for (int c = 0; c < shape.cols; ++c)
          llvm::errs() << "---+";
        llvm::errs() << "\n";
        for (int r = 0; r < shape.rows; ++r) {
          llvm::errs() << "      |";
          for (int c = 0; c < shape.cols; ++c) {
            if (remaining > 0) {
              llvm::errs() << " # |";
              --remaining;
            } else {
              llvm::errs() << "   |";
            }
          }
          llvm::errs() << "\n";
          llvm::errs() << "      +";
          for (int c = 0; c < shape.cols; ++c)
            llvm::errs() << "---+";
          llvm::errs() << "\n";
        }

        // Places onto combined grid (pack sequentially).
        int placed = 0;
        for (int r = next_row; r < kCgraGridRows && placed < node->cgra_count;
             ++r) {
          for (int c = (r == next_row ? next_col : 0);
               c < kCgraGridCols && placed < node->cgra_count; ++c) {
            combined_grid[r][c] = task_idx;
            next_row = r;
            next_col = c + 1;
            if (next_col >= kCgraGridCols) {
              next_col = 0;
              next_row = r + 1;
            }
            ++placed;
          }
        }
        ++task_idx;
      }

      // Prints combined 4xN grid.
      llvm::errs() << "\n  Combined 4x" << kCgraGridCols << " Grid"
                   << " (" << final_total << "/" << kTotalCGRAs << " used):\n";
      llvm::errs() << "  +";
      for (int c = 0; c < kCgraGridCols; ++c)
        llvm::errs() << "---+";
      llvm::errs() << "\n";
      for (int r = 0; r < kCgraGridRows; ++r) {
        llvm::errs() << "  |";
        for (int c = 0; c < kCgraGridCols; ++c) {
          int t = combined_grid[r][c];
          if (t < 0)
            llvm::errs() << " . |";
          else
            llvm::errs() << " " << (char)('0' + t) << " |";
        }
        llvm::errs() << "\n";
        llvm::errs() << "  +";
        for (int c = 0; c < kCgraGridCols; ++c)
          llvm::errs() << "---+";
        llvm::errs() << "\n";
      }
      llvm::errs() << "  (" << (kTotalCGRAs - final_total) << " free)\n";
      llvm::errs() << "================================================\n";

      llvm::errs() << "[ResourceAware] Final: " << final_graph.nodes.size()
                   << " tasks, " << final_total << " CGRAs\n";
      assert(final_total <= kTotalCGRAs &&
             "Total CGRA allocation exceeds 4x4 grid after optimization!");
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
mlir::taskflow::createResourceAwareTaskOptimizationPass() {
  return std::make_unique<ResourceAwareTaskOptimizationPass>();
}
