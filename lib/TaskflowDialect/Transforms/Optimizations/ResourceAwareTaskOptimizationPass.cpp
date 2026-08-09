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

#include "TaskflowDialect/TaskFusionUtil.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/analytical_cost_model.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
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
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
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
// Upper bound on context waves when temporal reuse is enabled: the allocation
// may ask for at most this many grids' worth of CGRAs.
constexpr int kMaxTemporalWaves = 4;

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

//===----------------------------------------------------------------------===//
// Topology term: mean tile-to-tile hop distance.
//
// The analytical MII family is a set of ceil(demand/capacity) throughput bounds
// and is therefore SHAPE-BLIND: `ResMII = ceil(ops/tiles)` sees only the tile
// count, and `RouteMII = moves/links` is invariant under transposing the array.
// Two arrays of equal area score identically even when the real mapper does not
// (e.g. fir on 8x4 -> II 6 vs 4x8 -> II 5).
//
// `avg_hop` is the mean pairwise shortest-path distance over the real link
// graph (BFS, works for any shape including non-rectangular ones and the
// asymmetric memory-FU tile_overrides). It is the topology feature that turns
// the sound floor into a shape-aware *prediction*:
//
//   predicted_cost = LB + max(0, coef * avg_hop * cp_depth - 1)
//
// The floor `LB` is left untouched — it stays sound for proofs and pruning; the
// prediction is used only for ranking/selection.
//===----------------------------------------------------------------------===//
//
// Two variants:
//   * `mem_weighted=false` — mean over ALL ordered tile pairs. Cheap, but it is
//     invariant under transposing the array (W x H and H x W have the same mean
//     distance), so it cannot separate 8x4 from 4x8.
//   * `mem_weighted=true`  — mean distance from every tile to its NEAREST
//     memory-capable tile. Live values enter and leave the fabric through the
//     memory FUs, and those sit on an asymmetric subset of tiles
//     (`tile_overrides` puts them on the left column and top row), so this
//     variant IS orientation-sensitive and separates 8x4 from 4x8.
static double averageHop(const neura::Architecture &arch,
                         bool mem_weighted = false) {
  std::vector<neura::Tile *> tiles = arch.getAllTiles();
  if (tiles.size() < 2)
    return 0.0;
  // Adjacency over the real link graph.
  llvm::DenseMap<int, SmallVector<int, 4>> adj;
  llvm::DenseSet<int> ids;
  for (neura::Tile *tile : tiles)
    ids.insert(tile->getId());
  for (neura::Link *link : arch.getAllLinks()) {
    neura::Tile *src = link->getSrcTile(), *dst = link->getDstTile();
    if (!src || !dst)
      continue;
    if (ids.contains(src->getId()) && ids.contains(dst->getId()))
      adj[src->getId()].push_back(dst->getId());
  }
  // Memory-capable tiles: the sources/sinks of every live value.
  llvm::DenseSet<int> mem_tiles;
  if (mem_weighted) {
    for (neura::Tile *tile : tiles)
      if (tile->canSupportOperation(neura::ILoadIndexed) ||
          tile->canSupportOperation(neura::IStoreIndexed) ||
          tile->canSupportOperation(neura::ILoad) ||
          tile->canSupportOperation(neura::IStore))
        mem_tiles.insert(tile->getId());
    // No memory FU anywhere (or all of them): the weighting carries no signal.
    if (mem_tiles.empty() || mem_tiles.size() == tiles.size())
      mem_weighted = false;
  }

  auto bfs = [&](int src, llvm::DenseMap<int, int> &dist) {
    std::deque<int> q;
    dist[src] = 0;
    q.push_back(src);
    while (!q.empty()) {
      int u = q.front();
      q.pop_front();
      for (int v : adj[u])
        if (!dist.count(v)) {
          dist[v] = dist[u] + 1;
          q.push_back(v);
        }
    }
  };

  if (mem_weighted) {
    // Mean over tiles of the distance to the nearest memory tile.
    long long total = 0;
    int counted = 0;
    for (neura::Tile *tile : tiles) {
      llvm::DenseMap<int, int> dist;
      bfs(tile->getId(), dist);
      int best = INT_MAX;
      for (int m : mem_tiles) {
        auto it = dist.find(m);
        if (it != dist.end())
          best = std::min(best, it->second);
      }
      if (best != INT_MAX) {
        total += best;
        ++counted;
      }
    }
    return counted ? (double)total / (double)counted : 0.0;
  }

  long long total = 0, pairs = 0;
  for (neura::Tile *src_tile : tiles) {
    llvm::DenseMap<int, int> dist;
    bfs(src_tile->getId(), dist);
    for (auto &[id, d] : dist) {
      total += d;
      ++pairs;
    }
  }
  return pairs ? (double)total / (double)pairs : 0.0;
}

// Mean latency of getting one message across the whole fabric: the mean
// tile-to-tile hop distance times the per-link latency. This is the physical
// floor on what a single inter-task transfer costs, and it is what makes the
// per-message term of the communication model a derived quantity rather than a
// fitted one -- partitioning multiplies the message COUNT, and this is the
// price of each.
//
// SCOPE. `getArchitecture()` only materialises ONE CGRA's tile mesh
// (`Architecture::initializeTiles` is called with the *per-CGRA* dimensions);
// the multi-CGRA grid exists only as two integers. But an inter-task message
// travels BETWEEN tasks, and distinct tasks sit on distinct CGRAs -- measuring
// the hop count on the per-CGRA mesh answers the wrong question. The rest of
// the compiler stitches the CGRA rectangle into one flat tile array
// (MapToAcceleratorPass clones the architecture to `shape.rows * perCgraRows` x
// `shape.cols * perCgraCols` and routes on that), so the fabric an inter-task
// message actually crosses is the full
// `multiCgraRows*perCgraRows` x `multiCgraCols*perCgraCols` mesh. We build that
// same flat array here.
//
// On architecture_4x4.yaml this is the difference between one 4x4 CGRA
// (mean hop 2.5) and the real 16x16 fabric (mean hop 10.625) -- a 4.25x error
// in the price of every message.
// Mean of |i - (j + offset)| over i in [0, n1), j in [0, n2). One axis of the
// mean Manhattan distance between two tile blocks whose origins are `offset`
// apart on that axis.
static double meanAbsDiff(int a_extent, int b_extent, int offset) {
  if (a_extent <= 0 || b_extent <= 0)
    return 0.0;
  long long total = 0;
  for (int a_pos = 0; a_pos < a_extent; ++a_pos)
    for (int b_pos = 0; b_pos < b_extent; ++b_pos)
      total += std::abs(a_pos - (b_pos + offset));
  return (double)total / ((double)a_extent * (double)b_extent);
}

// The rectangle a packer gives `n` CGRAs: the most square factorisation.
static std::pair<int, int> blockShape(int cgra_count) {
  cgra_count = std::max(1, cgra_count);
  int rows = 1;
  for (int candidate = 1; candidate * candidate <= cgra_count; ++candidate)
    if (cgra_count % candidate == 0)
      rows = candidate;
  return {rows, cgra_count / rows};
}

// `out_concurrency`, when non-null, receives how many transfers the fabric can
// carry at once: a message in flight holds about `mean_hops` links, and there
// are `num_links` of them.
static double fabricMessageLatency(const neura::Architecture &arch,
                                   double *out_concurrency = nullptr,
                                   double *out_link_latency = nullptr) {
  // Mean over all links, not the first one: link_overrides may make them
  // differ, and one arbitrary link is not the fabric's latency.
  auto meanLatency = [](const std::vector<neura::Link *> &links) {
    if (links.empty())
      return 1.0;
    long long total = 0;
    for (neura::Link *link : links)
      total += std::max(1, link->getLatency());
    return (double)total / (double)links.size();
  };

  const int fabric_rows = arch.getMultiCgraRows() * arch.getPerCgraRows();
  const int fabric_cols = arch.getMultiCgraColumns() * arch.getPerCgraColumns();
  double hops, link_latency;
  size_t num_links;
  if (fabric_rows > arch.getPerCgraRows() ||
      fabric_cols > arch.getPerCgraColumns()) {
    std::unique_ptr<neura::Architecture> fabric =
        arch.cloneWithNewDimensions(fabric_rows, fabric_cols);
    hops = averageHop(*fabric, /*mem_weighted=*/false);
    std::vector<neura::Link *> links = fabric->getAllLinks();
    link_latency = meanLatency(links);
    num_links = links.size();
  } else {
    // Single-CGRA architecture: the per-CGRA mesh IS the fabric.
    hops = averageHop(arch, /*mem_weighted=*/false);
    std::vector<neura::Link *> links = arch.getAllLinks();
    link_latency = meanLatency(links);
    num_links = links.size();
  }
  if (out_concurrency)
    *out_concurrency = std::max(1.0, (double)num_links / std::max(1.0, hops));
  if (out_link_latency)
    *out_link_latency = link_latency;
  return hops * link_latency;
}

// Returns the shape-aware predicted II used for ranking (never as a bound).
//
// REAL-VALUED on purpose. The integer form below rounds the topology term to
// zero for every shape whenever `coef * avg_hop * cp_depth < 1.5`, which at the
// calibrated coef=0.05 is the common case: all shapes of a given cgra_count
// then tie at LB and the "search" silently degenerates to picking the first
// candidate (which happens to be the geometric pick, so the knob was inert).
// Ranking uses this continuous value.
static double predictedCost(int64_t lb, double avg_hop, int64_t cp_depth,
                            double coef) {
  return (double)lb + std::max(0.0, coef * avg_hop * (double)cp_depth - 1.0);
}

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

// Parses a "NxM" cgra_shape attribute back into a CgraShape. Returns nullopt
// for the non-rectangular "NxM[(c,r)...]" form and for anything whose area does
// not match `cgra_count` (a stale attribute from an earlier cgra_count).
static std::optional<CgraShape> parseRectShape(StringRef s, int cgra_count) {
  if (s.contains('['))
    return std::nullopt;
  auto x = s.find('x');
  if (x == StringRef::npos)
    return std::nullopt;
  int r = 0, c = 0;
  if (s.take_front(x).getAsInteger(10, r) ||
      s.drop_front(x + 1).getAsInteger(10, c))
    return std::nullopt;
  if (r <= 0 || c <= 0 || r * c != cgra_count)
    return std::nullopt;
  return CgraShape{r, c, true, {}};
}

// Picks the best shape for display/profiling.
// We prefer shapes with the most compact physical layout (smallest maximum
// distance between nodes) to minimize communication latency. In cases of
// identical bounding box area, we prefer more square-like bounds over long
// rectangles.
//
// TODO: This function only picks a localized shape for an idealized single task
// mapping. Global placement and conflict resolution across multiple tasks is
// legitimately deferred to the downstream orchestration pass, as speculative
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
// Loop Partitioning (tiling)
//===----------------------------------------------------------------------===//
//
// Materialises a tiling decision into the IR. Tiling CUTS THE LOOP BOUND: a
// task whose root counter runs [lb, ub) becomes `factor` tasks running
// [lb + i*chunk, min(ub, lb + (i+1)*chunk)). The loop BODY is copied verbatim,
// so the DFG, its op count, its recurrence and therefore its II and steps are
// unchanged — only the trip count shrinks. This is what distinguishes tiling
// from replication: replicas run ONE configuration on several tile arrays at
// once with runtime-supplied bounds, whereas tiles are distinct tasks with
// distinct compile-time bounds that the schedule is free to serialise.
//
// The tiles are chained through the memref dependence-state SSA because that is
// the only well-formed encoding of a task's `will_writes -> done_writes`
// version chain. That chain is an ORDERING artefact, not a data dependence: the
// partitioned dimension is dependence-free by `dlp_replicable`, so the
// scheduler drops intra-group edges (see TaskDependencyGraph::makespan).
class TaskTiler {
public:
  // One cuttable level of a task's loop nest.
  struct PartitionLevel {
    TaskflowCounterOp counter;
    int64_t lb, ub, step, trip;
    int counter_id;
    // Splitting a loop is ALWAYS legal -- it is just loop splitting. What the
    // store-index test decides is whether the resulting tiles may run
    // CONCURRENTLY. histogram accumulates into hist[0,b], an address that does
    // not depend on the counter, so its tiles must stay ordered; but ordered
    // tiles are still smaller tasks that pack into schedule gaps, which is
    // worth having. Conflating the two forbade the cut outright.
    bool parallel = true;
  };

  //===--------------------------------------------------------------------===//
  // Partition legality
  //===--------------------------------------------------------------------===//
  //
  // `dlp_replicable` from ClassifyTaskAndCounterPass only asks whether the task
  // has a hyperblock and no loop-carried SSA dependence (no iter_args). It is
  // MEMORY-BLIND: a reduction that accumulates through a memref carries its
  // dependence in memory, not in SSA, so it is tagged replicable even though
  // splitting the loop makes the partitions race.
  //
  //   histogram:  store hist[0, b]      -- address independent of the counter
  //               -> every partition writes the same cells. ILLEGAL to split.
  //   gemm:       store C[i, j]         -- address indexed by both counters
  //               -> partitions touch disjoint cells. Legal to split on i or j.
  //
  // A counter dimension may therefore be partitioned only if EVERY store in the
  // task is indexed by it. Returns the set of counter_ids that satisfy this.
  static llvm::DenseSet<int> partitionableDims(TaskflowTaskOp task) {
    llvm::DenseSet<int> partitionable;
    // All counter ids present in the task.
    llvm::DenseSet<int> all_counter_ids;
    task.walk([&](neura::CounterOp counter) { all_counter_ids.insert((int)counter.getCounterId()); });
    if (all_counter_ids.empty())
      return partitionable;

    // Counter ids reachable backwards from a value through the dataflow.
    auto reaching = [](Value index_value) {
      llvm::DenseSet<int> ids;
      llvm::SmallVector<Value, 8> worklist{index_value};
      llvm::DenseSet<Value> seen;
      while (!worklist.empty()) {
        Value cur = worklist.pop_back_val();
        if (!seen.insert(cur).second)
          continue;
        Operation *def_op = cur.getDefiningOp();
        if (!def_op)
          continue;
        if (auto counter = dyn_cast<neura::CounterOp>(def_op)) {
          ids.insert((int)counter.getCounterId());
          continue;
        }
        for (Value operand : def_op->getOperands())
          worklist.push_back(operand);
      }
      return ids;
    };

    partitionable = all_counter_ids;
    bool saw_store = false;
    task.walk([&](Operation *op) {
      llvm::SmallVector<Value, 4> indices;
      if (auto store = dyn_cast<neura::StoreIndexedOp>(op)) {
        llvm::append_range(indices, store.getIndices());
      } else if (auto store = dyn_cast<neura::StoreOp>(op)) {
        if (store.getAddr())
          indices.push_back(store.getAddr());
      } else {
        return;
      }
      saw_store = true;
      llvm::DenseSet<int> dims_indexing_store;
      for (Value idx : indices)
        for (int id : reaching(idx))
          dims_indexing_store.insert(id);
      // Keep only dimensions that index THIS store too.
      llvm::DenseSet<int> still_partitionable;
      for (int id : partitionable)
        if (dims_indexing_store.contains(id))
          still_partitionable.insert(id);
      partitionable = std::move(still_partitionable);
    });
    // A task with no stores produces nothing observable; leave every dimension
    // available rather than inventing a restriction.
    (void)saw_store;
    return partitionable;
  }

  // True if ANY counter dimension of the task may legally be partitioned.
  // Replication splits one dimension; which one is a free choice, so the task
  // is replicable as long as at least one dimension indexes every store.
  static bool rootDimIsPartitionable(TaskflowTaskOp task) {
    return replicaDim(task).has_value();
  }

  // The dimension replication should split: the outermost legal level, so each
  // replica gets the largest contiguous chunk.
  static std::optional<PartitionLevel> replicaDim(TaskflowTaskOp task) {
    for (const PartitionLevel &l : partitionLevels(task))
      if (l.parallel)
        return l; // Replicas always run at once, so only a parallel-safe
                  // dimension can carry them.
    return std::nullopt;
  }

  // Returns the root taskflow.counter of `task` if the task can be partitioned
  // at compile time: exactly one root, constant bounds, positive step.
  static TaskflowCounterOp getPartitionableRoot(TaskflowTaskOp task) {
    if (!task.getBody().hasOneBlock())
      return nullptr;
    TaskflowCounterOp root;
    unsigned n_roots = 0;
    for (Operation &op : task.getBody().front()) {
      auto counter = dyn_cast<TaskflowCounterOp>(&op);
      if (!counter)
        continue;
      if (counter.getParentIndex())
        continue;
      ++n_roots;
      root = counter;
    }
    if (n_roots != 1 || !root)
      return nullptr;
    auto isConst = [](Value v) {
      return v.getDefiningOp<arith::ConstantIndexOp>() != nullptr;
    };
    if (!isConst(root.getLowerBound()) || !isConst(root.getUpperBound()) ||
        !isConst(root.getStep()))
      return nullptr;
    if (root.getStep().getDefiningOp<arith::ConstantIndexOp>().value() <= 0)
      return nullptr;
    return root;
  }

  //===--------------------------------------------------------------------===//
  // The partition space of a loop NEST
  //===--------------------------------------------------------------------===//
  //
  // Cutting only the root counter caps the task count at the outer trip count:
  // an 8x8 gemm can then become at most 8 tasks, however much grid is free.
  // Real loop tiling cuts the nest, so the partition space is the product of
  // every constant-bound, legally-partitionable level along the root chain.
  // Returns the chain of cuttable counter levels, outermost first. A level is
  // cuttable when its bounds are compile-time constants and its dimension
  // indexes every store (see partitionableDims).
  static SmallVector<PartitionLevel> partitionLevels(TaskflowTaskOp task) {
    SmallVector<PartitionLevel> levels;
    if (!task.getBody().hasOneBlock())
      return levels;
    SmallVector<TaskflowCounterOp> counters;
    for (Operation &op : task.getBody().front())
      if (auto c = dyn_cast<TaskflowCounterOp>(&op))
        counters.push_back(c);
    if (counters.empty())
      return levels;

    TaskflowCounterOp root;
    unsigned n_roots = 0;
    DenseMap<Value, TaskflowCounterOp> child_of;
    for (auto c : counters) {
      if (!c.getParentIndex()) {
        ++n_roots;
        root = c;
      } else {
        // A chain is only well-defined when each parent has one child.
        if (child_of.count(c.getParentIndex()))
          return levels;
        child_of[c.getParentIndex()] = c;
      }
    }
    if (n_roots != 1 || !root)
      return levels;

    llvm::DenseSet<int> legal = partitionableDims(task);
    auto cst = [](Value v) -> std::optional<int64_t> {
      if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
        return c.value();
      return std::nullopt;
    };
    for (TaskflowCounterOp cur = root; cur;
         cur = child_of.lookup(cur.getCounterIndex())) {
      auto lb = cst(cur.getLowerBound()), ub = cst(cur.getUpperBound()),
           st = cst(cur.getStep());
      int id = cur.getCounterId() ? *cur.getCounterId() : 0;
      // Collects EVERY cuttable level, skipping the ones that are not, rather
      // than stopping at the first. Cutting an inner level while leaving an
      // outer one whole is still a valid partition: each tile runs the full
      // outer loops over a sub-range of the inner one, the sub-ranges are
      // disjoint and their union is the original space. multi-nested stores to
      // out[k] with k the LEAF counter, so the leaf is the only legal
      // dimension -- refusing to look past the root would forbid the one cut
      // that is actually sound.
      if (!lb || !ub || !st || *st <= 0 || *ub <= *lb)
        continue;
      int64_t trip = llvm::divideCeil(*ub - *lb, *st);
      if (trip <= 1)
        continue;
      levels.push_back({cur, *lb, *ub, *st, trip, id, legal.contains(id)});
    }
    return levels;
  }

  // Total number of partitions the nest can be cut into.
  static int64_t partitionSpace(TaskflowTaskOp task) {
    int64_t space = 1;
    for (const PartitionLevel &l : partitionLevels(task))
      space *= l.trip;
    return space;
  }

  // The subset of that space replicas may use: concurrent partitions need a
  // dimension that indexes every store.
  static int64_t parallelPartitionSpace(TaskflowTaskOp task) {
    int64_t space = 1;
    for (const PartitionLevel &l : partitionLevels(task))
      if (l.parallel)
        space *= l.trip;
    return space;
  }

  // Splits `factor` across the nest levels, outermost first, so that the
  // product of the per-level factors is exactly `factor` and each divides its
  // level's trip count. Returns an empty vector when `factor` cannot be
  // realised (e.g. a prime larger than any single level's trip count).
  static SmallVector<int64_t>
  splitFactorAcrossLevels(ArrayRef<PartitionLevel> levels, int64_t factor) {
    SmallVector<int64_t> per_level(levels.size(), 1);
    int64_t remaining = factor;
    for (size_t i = 0; i < levels.size() && remaining > 1; ++i) {
      // Largest divisor of `remaining` that also divides this level's trip.
      int64_t best = 1;
      for (int64_t d = 1; d <= levels[i].trip; ++d)
        if (remaining % d == 0 && levels[i].trip % d == 0)
          best = d;
      per_level[i] = best;
      remaining /= best;
    }
    if (remaining != 1)
      return {};
    return per_level;
  }

  // Trip count of the root dimension alone (the one being cut).
  static int64_t rootTripCount(TaskflowCounterOp root) {
    int64_t lb =
        root.getLowerBound().getDefiningOp<arith::ConstantIndexOp>().value();
    int64_t ub =
        root.getUpperBound().getDefiningOp<arith::ConstantIndexOp>().value();
    int64_t st = root.getStep().getDefiningOp<arith::ConstantIndexOp>().value();
    if (st <= 0 || ub <= lb)
      return 1;
    return llvm::divideCeil(ub - lb, st);
  }

  // Maps each task result index onto the operand index it is a new version of.
  // Yields are sparse and may yield non-block-arg values; when any result
  // cannot be traced back to an operand the task is not chainable and tiling is
  // refused (returns false).
  static bool resultToOperand(TaskflowTaskOp task,
                              SmallVectorImpl<int> &result_to_operand) {
    auto yield =
        dyn_cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
    if (!yield)
      return false;
    SmallVector<Value> yielded;
    llvm::append_range(yielded, yield.getDoneReads());
    llvm::append_range(yielded, yield.getDoneWrites());
    llvm::append_range(yielded, yield.getValueResults());
    if (yielded.size() != task->getNumResults())
      return false;
    result_to_operand.assign(yielded.size(), -1);
    for (unsigned i = 0; i < yielded.size(); ++i) {
      auto ba = dyn_cast<BlockArgument>(yielded[i]);
      if (!ba || ba.getOwner() != &task.getBody().front())
        return false;
      // Block args are laid out [will_reads..., will_writes...,
      // value_inputs...] which is exactly the leading operand order of the op.
      result_to_operand[i] = ba.getArgNumber();
    }
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Replication (AMOEBA MoreReplicas)
  //===--------------------------------------------------------------------===//
  //
  // Replication does NOT clone the task. One configuration is loaded onto
  // `replicas` tile arrays and each is handed a disjoint sub-range of the root
  // counter at launch. The compiler's job is therefore to emit the PARTITION
  // TABLE, not new IR:
  //
  //   dlp_partition_dim   : counter_id of the dimension being split
  //   dlp_partition_mode  : "static" (constant_bound root -> concrete ranges)
  //                         "runtime" (symbol_bound root -> runtime fills them)
  //   dlp_partition       : flat [lb0, ub0, lb1, ub1, ...] for the static case
  //   dlp_partition_count : number of replicas (both cases)
  //
  // Returns the root counter if the task has exactly one, whatever its
  // dynamism; replication only needs the dimension to be dependence-free, which
  // `dlp_replicable` already asserts.
  static TaskflowCounterOp getReplicableRoot(TaskflowTaskOp task) {
    if (!task.getBody().hasOneBlock())
      return nullptr;
    TaskflowCounterOp root;
    unsigned n_roots = 0;
    for (Operation &op : task.getBody().front()) {
      auto counter = dyn_cast<TaskflowCounterOp>(&op);
      if (!counter || counter.getParentIndex())
        continue;
      ++n_roots;
      root = counter;
    }
    return (n_roots == 1) ? root : nullptr;
  }

  static bool canReplicate(TaskflowTaskOp task, int replicas) {
    if (replicas <= 1)
      return false;
    auto dlp = task->getAttrOfType<BoolAttr>("dlp_replicable");
    if (!dlp || !dlp.getValue())
      return false;
    TaskflowCounterOp root = getReplicableRoot(task);
    if (!root)
      return false;
    // A constant-bound root must have enough iterations to go round.
    if (getPartitionableRoot(task) && rootTripCount(root) < replicas)
      return false;
    return true;
  }

  // Writes the partition table onto the task. Returns false when nothing was
  // emitted, which means the `replicas` decision cannot be honoured.
  static bool emitPartitionConfig(TaskflowTaskOp task, int replicas) {
    OpBuilder b(task);
    if (replicas <= 1) {
      task->removeAttr("dlp_partition");
      task->removeAttr("dlp_partition_dim");
      task->removeAttr("dlp_partition_mode");
      task->removeAttr("dlp_partition_count");
      return true;
    }
    // Replication splits the OUTERMOST LEGAL dimension, which is not always
    // the root: a task that stores to out[k] can only be split on k.
    std::optional<PartitionLevel> dimlevel = replicaDim(task);
    TaskflowCounterOp root =
        dimlevel ? dimlevel->counter : getReplicableRoot(task);
    if (!root)
      return false;
    int dim = root.getCounterId() ? *root.getCounterId() : 0;
    task->setAttr("dlp_partition_dim", b.getI32IntegerAttr(dim));
    task->setAttr("dlp_partition_count", b.getI32IntegerAttr(replicas));

    if (dimlevel) {
      // constant_bound: the ranges are known now, so bake them in.
      const int64_t lb = dimlevel->lb;
      const int64_t ub = dimlevel->ub;
      const int64_t st = dimlevel->step;
      const int64_t n = dimlevel->trip;
      const int64_t chunk = llvm::divideCeil(n, (int64_t)replicas);
      SmallVector<int64_t> table;
      for (int r = 0; r < replicas; ++r) {
        int64_t rlb = lb + (int64_t)r * chunk * st;
        if (rlb >= ub)
          break;
        table.push_back(rlb);
        table.push_back(std::min(ub, lb + (int64_t)(r + 1) * chunk * st));
      }
      if (table.empty())
        return false;
      task->setAttr("dlp_partition", b.getDenseI64ArrayAttr(table));
      task->setAttr("dlp_partition_mode", b.getStringAttr("static"));
      task->setAttr("dlp_partition_count",
                    b.getI32IntegerAttr((int)table.size() / 2));
    } else {
      // symbol_bound / dynamic: the bounds are not known until launch, so the
      // configuration ships with a request for an even split and the runtime
      // supplies the actual endpoints. This is exactly why replication needs no
      // task cloning.
      task->setAttr("dlp_partition_mode", b.getStringAttr("runtime"));
      task->removeAttr("dlp_partition");
    }
    return true;
  }

  // True if `task` can be cut into `factor` tiles across its loop NEST.
  static bool canTile(TaskflowTaskOp task, int factor) {
    if (factor <= 1)
      return false;
    // No dlp_replicable gate: cutting a loop into contiguous sub-ranges is
    // sound for any loop. Whether the pieces may overlap in time is a separate
    // question, answered per level by PartitionLevel::parallel and recorded on
    // the IR as `tile_parallel`.
    SmallVector<PartitionLevel> levels = partitionLevels(task);
    if (levels.empty())
      return false;
    if (splitFactorAcrossLevels(levels, factor).empty())
      return false;
    SmallVector<int> map;
    return resultToOperand(task, map);
  }

  // Would a `factor`-way cut of `task` produce tiles that may overlap in time?
  //
  // The same rule `tile()` records on the IR as `tile_parallel`, answered
  // before the cut is made. Scoring needs it: a cut on a dimension that does
  // not index every store is legal but ORDERED, and modelling those tiles as
  // concurrent prices an N-way cut at latency/N when it costs latency.
  static bool tilingIsParallel(TaskflowTaskOp task, int factor) {
    SmallVector<PartitionLevel> levels = partitionLevels(task);
    SmallVector<int64_t> per_level = splitFactorAcrossLevels(levels, factor);
    if (per_level.empty())
      return false;
    for (size_t level = 0; level < levels.size(); ++level)
      if (per_level[level] > 1 && !levels[level].parallel)
        return false;
    return true;
  }

  // Cuts `task` into `factor` tasks. Returns the new tasks in iteration order,
  // or an empty vector if the task could not be partitioned. `ii`/`steps` are
  // copied onto every tile so the rebuilt graph does not re-profile identical
  // bodies.
  static SmallVector<TaskflowTaskOp> tile(TaskflowTaskOp task, int factor,
                                          int group_id, int64_t ii,
                                          int64_t steps, int cgra_count,
                                          int replicas) {
    SmallVector<TaskflowTaskOp> result;
    if (!canTile(task, factor))
      return result;

    // Splits the requested factor across the nest levels, outermost first, so
    // an 8x8 nest asked for 16 tiles becomes 8 cuts on i and 2 on j rather
    // than being capped at the outer trip count.
    SmallVector<PartitionLevel> levels = partitionLevels(task);
    SmallVector<int64_t> per_level = splitFactorAcrossLevels(levels, factor);
    if (per_level.empty())
      return result;

    SmallVector<int> result_to_operand;
    if (!resultToOperand(task, result_to_operand))
      return result;

    // Enumerates the cut index of every level: a mixed-radix counter over
    // per_level, giving exactly `factor` rectangular sub-ranges of the nest.
    OpBuilder builder(task);
    std::string base = task.getTaskName().str();
    SmallVector<Value> incoming(task->getOperands().begin(),
                                task->getOperands().end());

    Operation *insert_after = task.getOperation();
    int emitted = 0;
    for (int t = 0; t < factor; ++t) {
      // Decodes tile index t into a per-level cut index.
      SmallVector<int64_t> cut(levels.size(), 0);
      int64_t rest = t;
      for (int i = (int)levels.size() - 1; i >= 0; --i) {
        cut[i] = rest % per_level[i];
        rest /= per_level[i];
      }

      // Sub-range of every cut level.
      SmallVector<std::pair<int64_t, int64_t>> ranges(levels.size());
      bool empty_tile = false;
      for (size_t i = 0; i < levels.size(); ++i) {
        const PartitionLevel &l = levels[i];
        int64_t chunk = llvm::divideCeil(l.trip, per_level[i]);
        int64_t rlb = l.lb + cut[i] * chunk * l.step;
        int64_t rub = std::min(l.ub, l.lb + (cut[i] + 1) * chunk * l.step);
        if (rlb >= rub) {
          empty_tile = true;
          break;
        }
        ranges[i] = {rlb, rub};
      }
      if (empty_tile)
        continue; // Remainder ran out on this level.

      builder.setInsertionPointAfter(insert_after);
      IRMapping mapping;
      auto clone =
          cast<TaskflowTaskOp>(builder.clone(*task.getOperation(), mapping));
      clone->setOperands(incoming);
      clone.setTaskName(base + "_tile" + std::to_string(t));

      // Rewrites every cut level's bounds inside the clone, matched by
      // counter_id (positions are preserved by the clone).
      SmallVector<PartitionLevel> clone_levels = partitionLevels(clone);
      SmallVector<int64_t> flat_range;
      for (size_t i = 0; i < levels.size(); ++i) {
        if (per_level[i] <= 1)
          continue;
        TaskflowCounterOp target;
        for (const PartitionLevel &cl : clone_levels)
          if (cl.counter_id == levels[i].counter_id)
            target = cl.counter;
        if (!target)
          continue;
        OpBuilder clone_builder(target);
        auto new_lb = clone_builder.create<arith::ConstantIndexOp>(
            target.getLoc(), ranges[i].first);
        auto new_ub = clone_builder.create<arith::ConstantIndexOp>(
            target.getLoc(), ranges[i].second);
        target.getLowerBoundMutable().assign(new_lb);
        target.getUpperBoundMutable().assign(new_ub);

        // Mirrors the cut onto the matching neura.counter inside the kernel.
        // At this point the bounds live in discardable attributes
        // (`fold-constant` folded the SSA constants away), so patch both forms.
        const int id = levels[i].counter_id;
        const int64_t rlb = ranges[i].first, rub = ranges[i].second;
        clone.walk([&](neura::CounterOp nc) {
          if ((int)nc.getCounterId() != id)
            return;
          OpBuilder counter_builder(nc);
          if (nc->hasAttr("lower_bound_value"))
            nc->setAttr("lower_bound_value", counter_builder.getIndexAttr(rlb));
          if (nc->hasAttr("upper_bound_value"))
            nc->setAttr("upper_bound_value", counter_builder.getIndexAttr(rub));
          if (Value v = nc.getLowerBound())
            if (v.getDefiningOp<arith::ConstantIndexOp>()) {
              auto c = counter_builder.create<arith::ConstantIndexOp>(nc.getLoc(), rlb);
              nc.getLowerBoundMutable().assign(c);
            }
          if (Value v = nc.getUpperBound())
            if (v.getDefiningOp<arith::ConstantIndexOp>()) {
              auto c = counter_builder.create<arith::ConstantIndexOp>(nc.getLoc(), rub);
              nc.getUpperBoundMutable().assign(c);
            }
        });
        flat_range.push_back(id);
        flat_range.push_back(rlb);
        flat_range.push_back(rub);
      }

      // Records the decision on the IR and carries the (unchanged) profile so
      // the rebuilt graph does not re-lower identical bodies.
      OpBuilder attr_builder(clone);
      // A group is order-free only if EVERY level it cut is parallel-safe.
      bool group_parallel = true;
      for (size_t i = 0; i < levels.size(); ++i)
        if (per_level[i] > 1 && !levels[i].parallel)
          group_parallel = false;
      clone->setAttr("tile_parallel", attr_builder.getBoolAttr(group_parallel));
      clone->setAttr("tile_group", attr_builder.getI32IntegerAttr(group_id));
      clone->setAttr("tile_index", attr_builder.getI32IntegerAttr(t));
      clone->setAttr("tile_count", attr_builder.getI32IntegerAttr(factor));
      clone->setAttr("tile_range", attr_builder.getDenseI64ArrayAttr(flat_range));
      clone->setAttr("tiling", attr_builder.getI32IntegerAttr(1));
      clone->setAttr("cgra_count", attr_builder.getI32IntegerAttr(cgra_count));
      clone->setAttr("replicas", attr_builder.getI32IntegerAttr(replicas));
      if (ii > 0)
        clone->setAttr("compiled_ii", attr_builder.getI32IntegerAttr((int)ii));
      if (steps > 0) {
        SmallVector<NamedAttribute, 1> pa;
        pa.push_back(
            NamedAttribute(StringAttr::get(attr_builder.getContext(), "duration"),
                           attr_builder.getI32IntegerAttr((int)steps)));
        clone->setAttr("profile_info",
                       DictionaryAttr::get(attr_builder.getContext(), pa));
      }
      clone->removeAttr("trip_count"); // Recomputed from the new bounds.

      // Chains the dependence state: this tile consumes the previous tile's
      // versions of the memrefs it also touches.
      for (unsigned r = 0; r < result_to_operand.size(); ++r) {
        int operand_idx = result_to_operand[r];
        if (operand_idx >= 0 && (unsigned)operand_idx < incoming.size())
          incoming[operand_idx] = clone->getResult(r);
      }

      insert_after = clone.getOperation();
      result.push_back(clone);
      ++emitted;
    }

    if (emitted == 0)
      return result;

    // The last tile carries the final versions: every downstream consumer of
    // the original task now reads from it.
    TaskflowTaskOp last = result.back();
    for (unsigned r = 0; r < task->getNumResults(); ++r)
      task->getResult(r).replaceAllUsesWith(last->getResult(r));
    task->erase();
    return result;
  }
};

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

  // --- Data-level-parallelism decision variables. Both require a partitionable
  // counter dimension, i.e. the `dlp_replicable` tag from
  // ClassifyTaskAndCounterPass. They are DIFFERENT IR transformations:
  //
  // replicas (AMOEBA `MoreReplicas`): the SAME configuration is loaded onto
  //   `replicas` tile arrays, each given a disjoint sub-range of the root
  //   counter at launch time. No task is cloned — the compiler emits a
  //   partition table (`dlp_partition`) and the runtime fills the counter
  //   bounds. Costs `cgra_count` CGRAs per replica, all live SIMULTANEOUSLY.
  //   II and steps are unchanged; the iteration space is divided.
  //
  // tiling (loop partitioning): the root loop bound is CUT at compile time, so
  //   one task becomes `tiling` separate tasks each with a smaller
  //   constant_bound counter. II and steps are unchanged; each tile keeps one
  //   tile array of `cgra_count`, and the tiles need not be resident at the
  //   same time — that is the point: finer granularity lets the orchestrator
  //   squeeze them into schedule bubbles.
  int replicas = 1;
  int tiling = 1;
  bool dlp_replicable = false;

  // Iterations of the ROOT counter alone — the only dimension either move can
  // split. `trip_count` is the product over the whole nest, so using it as the
  // budget lets the search hand out more partitions than there are outer
  // iterations (e.g. 4 tiles x 2 replicas of a 4-iteration root).
  int64_t root_trip = 1;
  // Partitions available to TILING, which may cut any constant-bound level;
  // root_trip is the subset available to REPLICAS, which need a parallel-safe
  // dimension. histogram has split_space 64 and root_trip 1.
  int64_t split_space = 1;

  // Set when `dlp_replicable` was overridden because no counter dimension
  // indexes every store, i.e. the partitions would race in memory.
  bool partition_illegal = false;

  // Set once tiling has been MATERIALIZED into the IR: the node is one tile
  // cut from the task `tile_group`. Tiles of one group are chained through the
  // memref dependence-state SSA (that is the only well-formed encoding), but
  // the partitioned dimension is dependence-free by `dlp_replicable`, so the
  // scheduler treats intra-group edges as unordered.
  int tile_group = -1;
  // False when the tiles must stay in order (the cut dimension does not index
  // every store, e.g. a reduction through memory). Ordered tiles still help --
  // they are smaller and pack better -- but they may not overlap in time.
  bool tile_parallel = true;
  // Which counter dimension this group was cut on (-1 = not a tile). Two groups
  // cut on the same dimension exchange 1:1; on different dimensions they
  // shuffle all-to-all.
  int tile_dim = -1;

  // Facts about the task BODY, which no shape changes: the operation count and
  // the recurrence bound. One profile yields them, and every candidate shape's
  // lower bound then follows in closed form as
  // max(ceil(n_ops / tiles(shape)), rec_mii) -- which is what lets the shape
  // search rank without profiling.
  //
  // Neither DLP move rescales them either: cutting a loop bound or handing a
  // replica a sub-range leaves the loop body, and therefore its op count and
  // its recurrence, bit-identical.
  int64_t n_ops = 0;
  int64_t rec_mii = 1;

  // Shape-aware PREDICTED II (LB + hop term). Used only for ranking candidate
  // shapes; `ii` remains the sound floor used for latency and for pruning.
  //
  // Neither DLP move rescales the underlying profile. Cutting a loop bound
  // (tiling) or handing a replica a sub-range (replicas) leaves the loop BODY
  // -- and therefore the DFG, its op count and its recurrence -- bit-identical;
  // only the trip count changes. An earlier version of this model scaled ResMII
  // by the tiling factor, which is unroll-by-T semantics, not partitioning.
  double avg_hop = 0.0;
  double predicted_cost = 0.0;

  // Dependency edges (both SSA and memory).
  SmallVector<TaskGraphNode *> predecessors;
  SmallVector<TaskGraphNode *> successors;

  TaskGraphNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}

  // Width of ONE schedulable item, in cells: a tile array of `cgra_count` per
  // replica. Tiles are *not* counted here — each tile is its own item of this
  // width, so the packer places them independently.
  int allocatedCgras() const { return cgra_count * std::max(1, replicas); }

  // Replicas run at the same instant, so they need room at the same instant.
  //
  // The temporal budget does not apply to them. `allow-temporal` raises the
  // area budget to `kTotalCGRAs * kMaxTemporalWaves` because a task can be
  // scheduled in a later context wave, and a TILED task can use that: its tiles
  // become separate tasks the orchestrator may place in different contexts. A
  // replicated task cannot. The orchestrator pins every replica to the first
  // replica's start time, drops the ones that do not fit, and warns that
  // `est_latency` still assumes the count that was asked for.
  //
  // Charging replicas against the temporal budget let the search buy 64
  // concurrent replicas of a one-CGRA task on a 16-cell grid. The orchestrator
  // placed 16 and warned; the interval analysis read the attribute and reported
  // one wave. gemm came back at 4100 against the 16400 the pass's own objective
  // computed for the same allocation, and `grid_utilisation` printed 4.000.
  bool replicasFitGrid() const {
    return (int64_t)cgra_count * std::max(1, replicas) <= kTotalCGRAs;
  }

  static bool replicasFitGrid(int cgra_count, int replicas) {
    return (int64_t)cgra_count * std::max(1, replicas) <= kTotalCGRAs;
  }

  // Total cells this node's work claims, tiles included.
  //
  // A pending cut (`tiling = t` before materialisation) is t future tasks of
  // `cgra_count * replicas` cells each, so the node's real footprint is t times
  // its item width. Materialisation makes those tiles separate nodes and stamps
  // `tiling = 1` on each of them, so the same expression stays correct
  // afterwards -- which is what lets the search and the rewritten IR agree.
  //
  // This is NOT `allocatedCgras()`. Charging the item width as the footprint
  // let the greedy balancer overspend the grid by exactly the tiling factor:
  // on bicg it walked to tiling=4 across 20 tasks, believed it held 40 CGRAs,
  // and materialised 80 tasks holding 160 -- against a 64-cell budget. The
  // exact allocator already charged c*k*t, so it was solving the real problem
  // while the greedy solved an unconstrained one, and the "greedy beats exact"
  // readings on bicg/gemv were budget violations rather than better decisions.
  int footprintCgras() const { return allocatedCgras() * std::max(1, tiling); }

  // The initiation interval is invariant under both moves: the loop body is
  // untouched, so the mapped DFG and its II do not change.
  int64_t effectiveII() const { return ii; }

  // Iterations executed by ONE tile array: the trip count split across the
  // replicas and, if tiling has not yet been materialised into separate tasks,
  // across the prospective tiles as well.
  int64_t effectiveTripCount() const {
    int64_t divisor = (int64_t)std::max(1, replicas) * std::max(1, tiling);
    return std::max<int64_t>(1, llvm::divideCeil(trip_count, divisor));
  }

  // Returns estimated task latency using the pipelined execution model:
  //   latency = II * (trip_count - 1) + steps.
  int64_t estimatedLatency() const {
    return effectiveII() * (effectiveTripCount() - 1) + steps;
  }

  // Wall-clock work this node contributes when its tiles are NOT yet
  // materialised: `tiling` back-to-back tile executions if they cannot overlap.
  int64_t serialTiledLatency() const {
    return estimatedLatency() * std::max(1, tiling);
  }
};

// One legal allocation of a single task: `c` CGRAs per tile array, `k` replicas
// and a `t`-way loop cut, costing `cost` cells for `lat` cycles.
struct TaskConfig {
  int cgra_count;
  int replicas;
  int tiling;
  // Cells this configuration occupies: cgra_count * replicas * tiling.
  int cells;
  int64_t latency;
};

class TaskDependencyGraph {
public:
  SmallVector<std::unique_ptr<TaskGraphNode>> nodes;
  DenseMap<Operation *, TaskGraphNode *> op_to_node;

  // When true, the analytical fallback in runNeuraPipelineOnKernel uses the full
  // parametric cost model (computeAnalyticalII: max of ResMII/RecMII/MemMII/
  // RouteMII/RegMII/IssueMII) instead of only ResMII/RecMII. Set for
  // estimation-mode=cost-model-analytical. The mapper is never invoked in this
  // mode (skip_mapper is also forced true).
  bool use_full_cost_model = false;

  // Take the cost model's shape-aware PREDICTION rather than its sound floor.
  // The floor is what `estimation-mode=cost-model-analytical` has always
  // served, and against the exact CP-SAT mapper it under-predicts exactly where
  // routing binds (`plain_gemm` 4x4: floor 1, true 2). Off by default so the
  // two can be measured against each other on the same binary.
  bool use_predicted_ii = false;

  // Verify the shortlist with the exact CP-SAT mapper instead of the heuristic
  // backtracker. See `solveWithCpsat` for why this is the verifier the design
  // asks for; the members carry the interpreter, the script and the per-II time
  // budget so no path here hard-codes a machine-specific location.
  bool verifier_is_cpsat = false;
  std::string cpsat_python;
  std::string cpsat_script;
  int cpsat_seconds = 30;
  bool cpsat_minimize_routing = true;

  // Mirrors the pass options: choose shapes by predicted II (LB + hop) rather
  // than by pickBestShape's geometric rule, and the topology-term coefficient.
  bool search_shape = false;
  double hop_coef = 0.05;
  // Weight the topology term by distance to the nearest memory FU instead of
  // the all-pairs mean; makes it orientation-sensitive (8x4 vs 4x8).
  bool mem_weighted_hop = false;
  // Shortlist depth for the shape ranking; see the pass option of the same
  // name, which also governs the fusion-pair and allocation shortlists.
  int verify_top_k = 4;
  // Op-count ceiling above which `profileTask` serves the analytical estimate
  // instead of running the mapper. See the guard itself for why it stopped
  // being a constant.
  int mapper_op_limit = 150;
  // Communication weight and temporal reuse. comm is a proxy
  // (Sum vol/BW over inter-task edges, scaled by how far the producer and
  // consumer are spread across replicas) -- the pass has no placement, so this
  // is a spread cost, not a routed distance. `allow_temporal` lets the total
  // CGRA demand exceed the grid and charges the resulting context waves, which
  // is how orchestrate time-multiplexes the surplus.
  double comm_weight = 0.0;
  // Fabric geometry, for pricing a message before anything is placed.
  int per_cgra_rows = 4;
  int per_cgra_cols = 4;
  double comm_link_latency = 1.0;
  // Fixed cost of one inter-task message, in cycles. Partitioning multiplies
  // the message COUNT (N*M between an N-way and an M-way group), so this is
  // the term that makes tiling pay for the traffic it creates.
  double comm_msg_cost = 0.0;
  // Set when comm_msg_cost was derived from the architecture rather than given
  // on the command line, so the log can say which it was.
  bool comm_msg_cost_derived = false;
  // How many inter-task transfers the fabric can carry at once. A message in
  // flight occupies about `mean_hops` links, and the fabric has `num_links` of
  // them, so `num_links / mean_hops` transfers fit concurrently. Derived from
  // the architecture alongside comm_msg_cost; 1 means "assume no concurrency".
  double comm_concurrency = 1.0;
  bool allow_temporal = false;
  // Score decisions with the resource-constrained schedule length instead of
  // the shipped `max task latency` interval. Off by default so the shipped
  // numbers stay reproducible.
  bool use_makespan = false;
  // Score with the steady-state pipeline interval instead.
  bool use_pipeline = false;
  // Upper bound on the loop-partitioning factor the search may propose.

  // Bandwidth used by the comm proxy (matches the Category-B harness).
  static constexpr int kCommBW = 8;

  // Returns the number of context waves the current allocation needs.
  int wavesNeeded() const {
    if (!allow_temporal)
      return 1;
    int total = getTotalAllocatedCGRAs();
    return std::max(1, (int)llvm::divideCeil(total, kTotalCGRAs));
  }

  // Communication proxy over inter-task edges. Producer volume is its trip
  // count; an edge costs more when either side is spread over more replicas,
  // because every replica pair has to exchange. Fused tasks contribute nothing
  // (they are a single node and the edge disappears).
  // Communication cost, with the overhead that PARTITIONING introduces.
  //
  // The previous form divided the volume by the tiling factor, which made
  // tiling look like it REDUCED communication. It does not. Splitting a task
  // does not move less data -- it moves the same data in more, smaller pieces:
  //
  //   * an edge between an N-way group and an M-way group is not one transfer
  //     but up to N*M, because any producer tile may hold a slice any consumer
  //     tile needs. Total volume is conserved; the message COUNT is not.
  //   * an ordered group hands its memref state down the chain, one message per
  //     link, which a single untiled task would not have paid at all.
  //   * replicas add the same way: each of the k arrays exchanges separately.
  //
  // So the cost has two parts: a bandwidth term (volume / BW, conserved) and a
  // per-message term (count * comm_msg_cost) that partitioning multiplies.
  // comm_msg_cost defaults to 0, so the shipped behaviour is unchanged until it
  // is set -- the constant is a property of the fabric, not something to invent
  // here.
  // Cost of ONE producer->consumer edge, in cycles.
  //
  //   bandwidth   volume / BW           -- CONSERVED under partitioning
  //   messages    count * msg_cost      -- MULTIPLIED by partitioning
  //
  // Three things this gets right that the earlier proxy did not:
  //
  //  * volume is the element count of the memref that carries the dependence,
  //    not the producer's trip count. A 524288-iteration task may hand over one
  //    scalar; charging it 524288 made every long loop look
  //    communication-bound.
  //  * volume is NOT multiplied by the replica spread. Splitting a transfer k
  //    ways moves the same bytes; only the message count grows. The old form
  //    multiplied the volume AND (once the message term existed) the count,
  //    charging the spread twice.
  //  * an N-way group feeding an M-way group is N*M messages only when the two
  //    cut DIFFERENT dimensions (a genuine shuffle). When they cut the same
  //    dimension the partitions line up and it is max(N,M) transfers. The cut
  //    dimension is on the IR as `tile_range`, so this is checked, not assumed.
  double edgeCost(const TaskGraphNode *producer,
                  const TaskGraphNode *consumer) const {
    const bool intra = producer->tile_group >= 0 &&
                       producer->tile_group == consumer->tile_group;
    if (intra && producer->tile_parallel)
      return 0.0; // disjoint slices; the SSA chain is bookkeeping
    double volume = edgeVolume(producer, consumer);
    if (intra)
      // Ordered group: the running state is handed down the chain. One message
      // per link, which an untiled task would not have paid at all.
      return volume / (double)kCommBW + messageCost(producer, consumer);
    const int spread =
        std::max(1, producer->replicas) + std::max(1, consumer->replicas) - 1;
    const double messages =
        (double)groupMessages(producer, consumer) * (double)spread;
    return volume / (double)kCommBW +
           messages * messageCost(producer, consumer);
  }

  // Cycles for ONE message on this edge: how far it travels, times the per-link
  // latency.
  //
  // The distance is a property of the EDGE, not of the fabric. A flat
  // fabric-wide mean (the previous form) cannot be right, because the quantity
  // moves with the decision being priced -- measured against the real placement
  // on 10 multi-task programs, the realised mean hop rises from 7.3 to 9.9 when
  // tiling spreads the tasks out, while a constant by construction says 10.625
  // for both. Both endpoints' footprints are known here (`allocatedCgras()`),
  // so the distance can be derived: give each side the squarest CGRA rectangle
  // a packer would, abut them the way that keeps the bounding box compact, and
  // average over the tile pairs. That predictor tracks the real placement to
  // 2.6 hops mean absolute error against the constant's 4.2, and it is still
  // parameter-free.
  //
  // A non-negative `comm_msg_cost` (set on the command line) overrides it with
  // a flat constant, which is what makes the term sweepable.
  double messageCost(const TaskGraphNode *producer,
                     const TaskGraphNode *consumer) const {
    if (comm_msg_cost >= 0.0)
      return comm_msg_cost;
    return blockPairHops(producer->allocatedCgras(),
                         consumer->allocatedCgras()) *
           comm_link_latency;
  }

  // Mean tile-to-tile distance between two compactly packed CGRA blocks.
  double blockPairHops(int a_cgras, int b_cgras) const {
    auto [a_cgra_rows, a_cgra_cols] = blockShape(a_cgras);
    auto [b_cgra_rows, b_cgra_cols] = blockShape(b_cgras);
    const int a_tile_rows = a_cgra_rows * per_cgra_rows;
    const int a_tile_cols = a_cgra_cols * per_cgra_cols;
    const int b_tile_rows = b_cgra_rows * per_cgra_rows;
    const int b_tile_cols = b_cgra_cols * per_cgra_cols;
    const double abut_by_col = meanAbsDiff(a_tile_rows, b_tile_rows, 0) +
                               meanAbsDiff(a_tile_cols, b_tile_cols, a_tile_cols);
    const double abut_by_row =
        meanAbsDiff(a_tile_rows, b_tile_rows, a_tile_rows) +
        meanAbsDiff(a_tile_cols, b_tile_cols, 0);
    return std::min(abut_by_col, abut_by_row);
  }

  // Elements crossing the edge: the carrier memref when its shape is static,
  // otherwise the producer's trip count as a last-resort proxy.
  double edgeVolume(const TaskGraphNode *producer,
                    const TaskGraphNode *consumer) const {
    auto recorded = edge_volume.find(
        std::make_pair(const_cast<TaskGraphNode *>(producer),
                       const_cast<TaskGraphNode *>(consumer)));
    if (recorded != edge_volume.end() && recorded->second > 0.0)
      return recorded->second;
    return (double)std::max<int64_t>(1, producer->trip_count);
  }

  // Number of distinct transfers between the two sides' partitions.
  int groupMessages(const TaskGraphNode *producer,
                       const TaskGraphNode *consumer) const {
    auto members = [&](const TaskGraphNode *n) {
      if (n->tile_group < 0)
        return std::max(1, n->tiling); // not yet materialised
      int k = 0;
      for (auto &m : nodes)
        if (m->tile_group == n->tile_group)
          ++k;
      return std::max(1, k);
    };
    const int producer_parts = members(producer);
    const int consumer_parts = members(consumer);
    if (producer_parts == 1 || consumer_parts == 1)
      return std::max(producer_parts, consumer_parts);
    // The cut dimension has to be known the SAME way before and after the cut,
    // or the search scores one branch and the rewritten IR scores the other --
    // which is exactly what made predicted and materialised diverge. After
    // materialisation it is on the IR (`tile_range`); before, it is whatever
    // splitFactorAcrossLevels is going to choose for this factor.
    const int producer_dim = partitionDim(producer);
    const int consumer_dim = partitionDim(consumer);
    if (producer_dim >= 0 && producer_dim == consumer_dim)
      return std::max(producer_parts, consumer_parts); // aligned, 1:1
    return producer_parts * consumer_parts; // different dims: a real shuffle
  }

  // Dimension a node is (or will be) cut on. -1 when it is not partitioned.
  int partitionDim(const TaskGraphNode *n) const {
    if (n->tile_dim >= 0)
      return n->tile_dim; // materialised: read back from tile_range
    if (n->tiling <= 1)
      return -1;
    auto levels = TaskTiler::partitionLevels(n->op);
    auto per = TaskTiler::splitFactorAcrossLevels(levels, n->tiling);
    if (per.empty())
      return -1;
    for (size_t i = 0; i < levels.size(); ++i)
      if (per[i] > 1)
        return levels[i].counter_id; // the outermost level actually cut
    return -1;
  }

  // Total communication, counted once per LOGICAL producer->consumer relation.
  //
  // Materialising a cut multiplies the EDGES as well as the messages: two
  // groups of N and M tiles that touch the same memref produce N*M graph edges,
  // and charging N*M messages on each of them squares the cost. The search then
  // scores 42 edges and the rewritten IR scores 207, which is precisely how far
  // predicted and materialised drifted apart (55.7k vs 259.5k) while every
  // other term of the objective matched exactly. Group pairs are therefore
  // deduplicated, and edgeCost accounts for the fan-out within the pair.
  double commCost() const {
    auto groupKey = [](const TaskGraphNode *n) -> const void * {
      if (n->tile_group >= 0)
        return (const void *)(intptr_t)(n->tile_group + 1);
      return (const void *)n;
    };
    llvm::DenseSet<std::pair<const void *, const void *>> counted;
    double total = 0.0;
    for (auto &n : nodes)
      for (auto *succ : n->successors) {
        const void *kp = groupKey(n.get()), *ks = groupKey(succ);
        if (kp == ks) {
          // Intra-group: an ordered chain still pays per link, a parallel one
          // pays nothing. Charged per edge, not per pair.
          total += edgeCost(n.get(), succ);
          continue;
        }
        if (!counted.insert({kp, ks}).second)
          continue;
        total += edgeCost(n.get(), succ);
      }
    return total;
  }

  // Same quantity, attributed to one edge so the schedule can hide it behind
  // other work instead of always paying it. Sharing edgeCost keeps the two
  // formulations from drifting apart -- they previously disagreed about whether
  // tiling raised or lowered communication.
  double commDelay(const TaskGraphNode *producer,
                   const TaskGraphNode *consumer) const {
    if (comm_weight <= 0.0)
      return 0.0;
    return comm_weight * edgeCost(producer, consumer);
  }

  // Communication delay charged on an edge INSIDE the modelled schedule.
  //
  // Zero in pipeline mode, and that is not an oversight. The reported number is
  // what `--analyze-task-pipeline-interval` computes, and its graph carries no
  // communication term at all: its edges are weighted by the producer's
  // duration and nothing else. A model that lengthens the same path by a
  // transfer time is therefore not an estimate of the reported quantity, and
  // the search that trusts it optimises something else. Measured on three
  // kernels, dropping the term makes the model exact where it was not:
  //
  //     cnn_tiled  57121 -> 50320 predicted, 50320 measured
  //     ffn        40236 -> 30952 predicted, 30952 measured
  //     gemv        1825 ->  1308 predicted,  1308 measured
  //
  // and gemv's allocation improves from 1512 to 1308, because the inflated
  // edges had been discouraging the partitioning that pays.
  //
  // Communication still reaches the pipeline objective, through `commFloor()`:
  // a throughput bound is a `max` term, which is the shape a fabric limit
  // actually has. Makespan mode keeps the edge delay, because there the
  // modelled quantity is a schedule length that a transfer genuinely extends
  // and no external measurement contradicts it.
  double scheduleEdgeDelay(const TaskGraphNode *producer,
                           const TaskGraphNode *consumer) const {
    if (use_pipeline)
      return 0.0;
    return commDelay(producer, consumer);
  }

  // Throughput floor from communication, in the same ceil(demand / capacity)
  // form as the resource floor.
  //
  // `commCost()` is the fabric-time the graph demands if every transfer ran one
  // after another. They do not: the fabric carries roughly
  // `num_links / mean_hops` of them at once, because a message in flight holds
  // about `mean_hops` links. Dividing by that concurrency is what turns a raw
  // demand into a floor.
  //
  // This replaced adding `comm_weight * commCost()` straight onto the interval.
  // That additive form is not a floor, it is a surcharge, and it asserts that
  // the fabric and the CGRAs never overlap -- so it charged ~5% of the
  // objective unconditionally, on top of a bound that already covers the same
  // wall-clock. Every other term of this model is a floor inside a max;
  // communication is now one too.
  int64_t commFloor() const {
    if (comm_weight <= 0.0)
      return 0;
    double demand = comm_weight * commCost();
    return (int64_t)std::llround(demand / std::max(1.0, comm_concurrency));
  }

  //===--------------------------------------------------------------------===//
  // Resource-constrained makespan
  //===--------------------------------------------------------------------===//
  //
  // Greedy list schedule of ONE input instance over a hard capacity of
  // kTotalCGRAs, longest-remaining-path first. A node occupies
  // `allocatedCgras()` cells for `estimatedLatency()` cycles; a node whose
  // tiling is not yet materialised expands into `tiling` sub-tasks, so the
  // search scores what materialisation will produce.
  struct ScheduleItem {
    const TaskGraphNode *node;
    int sub_index;
    int64_t duration;
    int area;
    int64_t start = -1;
    int64_t finish = -1;
  };

  int64_t makespan(SmallVectorImpl<ScheduleItem> *out = nullptr) const {
    SmallVector<ScheduleItem> items;
    DenseMap<const TaskGraphNode *, SmallVector<unsigned, 4>> node_items;
    for (auto &n : nodes) {
      int copies = std::max(1, n->tiling);
      for (int t = 0; t < copies; ++t) {
        node_items[n.get()].push_back(items.size());
        items.push_back({n.get(), t, n->estimatedLatency(),
                         std::max(1, n->allocatedCgras()), -1, -1});
      }
    }
    if (items.empty())
      return 0;

    SmallVector<SmallVector<unsigned, 4>> preds(items.size());
    SmallVector<SmallVector<std::pair<unsigned, int64_t>, 4>> succs(
        items.size());
    // A dropped intra-group chain is replaced by a fan-out/fan-in, not by
    // nothing: TaskTiler rewires the original consumers onto the LAST tile, so
    // simply deleting the chain would leave tiles 0..n-2 with no path to the
    // consumer and they could be ordered after it.
    DenseMap<int, SmallVector<const TaskGraphNode *, 4>> group_members;
    for (auto &n : nodes)
      if (n->tile_group >= 0 && n->tile_parallel)
        group_members[n->tile_group].push_back(n.get());
    auto expand = [&](const TaskGraphNode *n) {
      SmallVector<const TaskGraphNode *, 4> out;
      if (n->tile_group >= 0 && n->tile_parallel) {
        auto it = group_members.find(n->tile_group);
        if (it != group_members.end()) {
          out.assign(it->second.begin(), it->second.end());
          return out;
        }
      }
      out.push_back(n);
      return out;
    };
    for (auto &n : nodes) {
      for (auto *s : n->successors) {
        if (n->tile_group >= 0 && n->tile_group == s->tile_group &&
            n->tile_parallel)
          continue;
        int64_t delay = (int64_t)std::llround(scheduleEdgeDelay(n.get(), s));
        for (const TaskGraphNode *pn : expand(n.get()))
          for (const TaskGraphNode *sn : expand(s))
            for (unsigned pi : node_items[pn])
              for (unsigned si : node_items[sn])
                if (pi != si) {
                  preds[si].push_back(pi);
                  succs[pi].push_back({si, delay});
                }
      }
    }

    SmallVector<int64_t> prio(items.size(), 0);
    {
      SmallVector<unsigned> order, indeg(items.size(), 0), ready;
      for (unsigned i = 0; i < items.size(); ++i)
        indeg[i] = preds[i].size();
      for (unsigned i = 0; i < items.size(); ++i)
        if (!indeg[i])
          ready.push_back(i);
      while (!ready.empty()) {
        unsigned u = ready.pop_back_val();
        order.push_back(u);
        for (auto &[v, d] : succs[u])
          if (--indeg[v] == 0)
            ready.push_back(v);
      }
      for (auto it = order.rbegin(); it != order.rend(); ++it) {
        int64_t best = 0;
        for (auto &[v, d] : succs[*it])
          best = std::max(best, prio[v] + d);
        prio[*it] = best + items[*it].duration;
      }
    }

    SmallVector<int64_t> ready_at(items.size(), 0);
    SmallVector<unsigned> remaining(items.size(), 0);
    for (unsigned i = 0; i < items.size(); ++i)
      remaining[i] = preds[i].size();
    SmallVector<bool> started(items.size(), false);
    int64_t now = 0, makespan_v = 0;
    int free_area = kTotalCGRAs;
    SmallVector<unsigned> running;
    unsigned completed = 0;
    while (completed < items.size()) {
      SmallVector<unsigned> candidates;
      for (unsigned i = 0; i < items.size(); ++i)
        if (!started[i] && remaining[i] == 0 && ready_at[i] <= now)
          candidates.push_back(i);
      llvm::sort(candidates, [&](unsigned a, unsigned b) {
        if (prio[a] != prio[b])
          return prio[a] > prio[b];
        return a < b;
      });
      for (unsigned i : candidates) {
        int need = std::min(items[i].area, kTotalCGRAs);
        if (need > free_area && !(running.empty() && free_area == kTotalCGRAs))
          continue;
        free_area -= need;
        started[i] = true;
        items[i].start = now;
        items[i].finish = now + items[i].duration;
        running.push_back(i);
      }
      if (running.empty()) {
        int64_t next = INT64_MAX;
        for (unsigned i = 0; i < items.size(); ++i)
          if (!started[i] && remaining[i] == 0)
            next = std::min(next, ready_at[i]);
        if (next == INT64_MAX)
          break;
        now = next;
        continue;
      }
      int64_t next_finish = INT64_MAX;
      for (unsigned i : running)
        next_finish = std::min(next_finish, items[i].finish);
      now = next_finish;
      SmallVector<unsigned> still;
      for (unsigned i : running) {
        if (items[i].finish > now) {
          still.push_back(i);
          continue;
        }
        ++completed;
        free_area += std::min(items[i].area, kTotalCGRAs);
        makespan_v = std::max(makespan_v, items[i].finish);
        for (auto &[v, d] : succs[i]) {
          ready_at[v] = std::max(ready_at[v], items[i].finish + d);
          --remaining[v];
        }
      }
      running = still;
    }
    if (out)
      *out = std::move(items);
    return makespan_v;
  }

  // Component breakdown, for checking that the search and the rewritten IR
  // agree term by term rather than only in the total.
  void explainPipelineInterval(llvm::raw_ostream &os, StringRef tag) const {
    int64_t task_floor = 0, area_time = 0;
    for (auto &n : nodes) {
      int64_t lat = n->estimatedLatency();
      task_floor = std::max(task_floor, lat);
      area_time += lat * (int64_t)std::max(1, n->footprintCgras());
    }
    int64_t edges = 0;
    for (auto &n : nodes)
      edges += n->successors.size();
    int64_t sched_items = 0, sched_edges = 0;
    const int64_t cycle = pipelineCycle(&sched_items, &sched_edges);
    os << "[Comm] " << tag << ": nodes=" << nodes.size() << " edges=" << edges
       << " sched_items=" << sched_items << " sched_edges=" << sched_edges
       << " task_floor=" << task_floor << " resource_floor="
       << llvm::divideCeil(area_time, (int64_t)kTotalCGRAs)
       << " pipeline_cycle=" << cycle
       << " comm_demand=" << llvm::format("%.1f", commCost())
       << " comm_floor=" << commFloor() << " total=" << pipelineInterval()
       << "\n";
  }

  int64_t pipelineInterval() const {
    // The two idealised floors. `resource_floor` assumes the work is perfectly
    // divisible across the 16 cells -- and that assumption is exactly why an
    // earlier version of this model could not see what tiling buys: both DLP
    // moves keep lat*area constant (replicas: lat/k on k*area; tiling: t items
    // of lat/t on the same area), so the ratio never moves and the two knobs
    // are indistinguishable.
    int64_t task_floor = 0, area_time = 0;
    for (auto &n : nodes) {
      int64_t lat = n->estimatedLatency();
      task_floor = std::max(task_floor, lat);
      area_time += lat * (int64_t)std::max(1, n->footprintCgras());
    }
    int64_t floor_lb = std::max<int64_t>(
        task_floor, llvm::divideCeil(area_time, (int64_t)kTotalCGRAs));

    // The floors alone cannot separate the two DLP knobs -- they keep
    // `lat * area` constant -- so `pipelineCycle()` does the work: it models an
    // assignment, and an assignment is where granularity finally matters. A
    // replicated task is ONE rigid item of `cgra_count * replicas` cells; a
    // tiled task is `tiling` independent items of `cgra_count` cells that may
    // land on different cells. Fat rigid items leave slack; thin ones pack
    // flush.
    return std::max({floor_lb, pipelineCycle(), commFloor()});
  }

  // The steady-state interval, modelled the way the analyzer defines it.
  //
  // `--analyze-task-pipeline-interval` reports neither a makespan nor a packing
  // ratio. For each physical CGRA it closes a CYCLE -- that cell's last task
  // feeds the next iteration's first task -- and returns
  //
  //     max over cells of [ longest path(first task -> last task) + dur(last) ]
  //
  // over the graph of data dependences PLUS the cell's own context order. Two
  // things follow, and the model had each of them wrong at some point:
  //
  //  * the cell's own chain lies inside that path, so the cell's total
  //    occupancy is a lower bound. A packing bound alone is therefore valid but
  //    loose: it said 557k where the schedule measured 3.77M.
  //  * the GLOBAL critical path is NOT a bound. Give every task its own cell
  //  and
  //    each cycle is a single task, however deep the DAG. Charging the global
  //    longest path over-predicted gpt2_decode by 2.07x.
  //
  // What sets the cycle is which tasks end up SHARING a cell, and that is a
  // scheduling question. So this runs the same kind of schedule the
  // orchestrator does -- dependence-respecting, list-scheduled onto kTotalCGRAs
  // cells, with communication delay on the edges -- keeps the cell assignment,
  // and then evaluates the analyzer's formula on it.
  // `out_items` / `out_edges` report the schedule this actually ran on, which
  // is what has to match between a collapsed graph and the materialised one.
  // Durations and areas already provably match -- `task_floor` and
  // `resource_floor` are identical across `apply` -- so any remaining
  // difference in the cycle is in the precedence structure, and these two
  // counts are the cheapest way to see it.
  int64_t pipelineCycle(int64_t *out_items = nullptr,
                        int64_t *out_edges = nullptr) const {
    struct Item {
      const TaskGraphNode *node;
      int64_t duration;
      int area;
      int64_t start = 0;
      int64_t finish = 0;
      SmallVector<unsigned, 4> cells;
    };
    SmallVector<Item> items;
    DenseMap<const TaskGraphNode *, SmallVector<unsigned, 4>> node_items;
    // Nodes whose proposed cut is legal but ORDERED. Their items are chained
    // below; leaving them unordered prices such a cut at latency/N when the
    // tiles cannot overlap and it costs latency.
    DenseSet<const TaskGraphNode *> ordered_cut;
    for (auto &n : nodes) {
      const int64_t lat = n->estimatedLatency();
      const int area = std::min(std::max(1, n->allocatedCgras()), kTotalCGRAs);
      const int tiling = std::max(1, n->tiling);
      if (tiling > 1 && n->tile_group < 0 && n->op &&
          !TaskTiler::tilingIsParallel(n->op, tiling))
        ordered_cut.insert(n.get());
      for (int t = 0; t < tiling; ++t) {
        node_items[n.get()].push_back(items.size());
        items.push_back({n.get(), lat, area});
      }
    }
    if (items.empty())
      return 0;

    // Precedence between ITEMS. A parallel tile group does not chain
    // internally; every other producer/consumer pair does, and a consumer waits
    // for all of the producer's tiles (the orchestrator and the analyzer both
    // enforce that, and not doing so is what let a consumer overtake tile 0).
    SmallVector<SmallVector<std::pair<unsigned, int64_t>, 4>> successors_of(
        items.size());
    SmallVector<int> remaining(items.size(), 0);
    for (const TaskGraphNode *node : ordered_cut) {
      const SmallVector<unsigned, 4> &tiles = node_items[node];
      for (size_t tile = 1; tile < tiles.size(); ++tile) {
        successors_of[tiles[tile - 1]].push_back({tiles[tile], 0});
        ++remaining[tiles[tile]];
      }
    }
    for (auto &n : nodes) {
      for (const TaskGraphNode *successor : n->successors) {
        if (n->tile_group >= 0 && n->tile_group == successor->tile_group &&
            n->tile_parallel)
          continue;
        const int64_t delay =
            (int64_t)std::llround(scheduleEdgeDelay(n.get(), successor));
        auto producer_items_it = node_items.find(n.get());
        auto successor_items_it = node_items.find(successor);
        if (producer_items_it == node_items.end() || successor_items_it == node_items.end())
          continue;
        // All pairs, deliberately, even though `TaskTiler::tile` threads the
        // tiles and rewires consumers onto the LAST one, so the emitted graph
        // has a single entry and exit per cut. Wiring only those endpoints was
        // tried and measured worse: over 16 programs the model's mean error
        // against the analyzer went from 5.2% to 30.9%, exact agreement fell
        // from 13/16 to 3/16, and the measured interval regressed on five
        // programs (cnn_tiled +23.9%, slam_jtj_21 +21.7%, dense_gemm_20
        // +18.6%). A consumer does wait for the whole cut, and charging it
        // against every tile is what reproduces that.
        for (unsigned producer_item : producer_items_it->second)
          for (unsigned successor_item : successor_items_it->second)
            if (producer_item != successor_item) {
              successors_of[producer_item].push_back({successor_item, delay});
              ++remaining[successor_item];
            }
      }
    }

    if (out_items)
      *out_items = (int64_t)items.size();
    if (out_edges) {
      int64_t edge_count = 0;
      for (auto &succs : successors_of)
        edge_count += (int64_t)succs.size();
      *out_edges = edge_count;
    }

    // Longest remaining path first, the standard list-schedule priority.
    SmallVector<int64_t> prio(items.size(), 0);
    {
      SmallVector<unsigned> order;
      SmallVector<int> indeg(remaining.begin(), remaining.end());
      SmallVector<unsigned> ready;
      for (unsigned i = 0; i < items.size(); ++i)
        if (indeg[i] == 0)
          ready.push_back(i);
      while (!ready.empty()) {
        unsigned ready_item = ready.pop_back_val();
        order.push_back(ready_item);
        for (auto &[succ_item, delay] : successors_of[ready_item])
          if (--indeg[succ_item] == 0)
            ready.push_back(succ_item);
      }
      for (int i = (int)order.size() - 1; i >= 0; --i) {
        unsigned ready_item = order[i];
        int64_t best = 0;
        for (auto &[succ_item, delay] : successors_of[ready_item])
          best = std::max(best, delay + prio[succ_item]);
        prio[ready_item] = items[ready_item].duration + best;
      }
    }

    SmallVector<int64_t> ready_at(items.size(), 0);
    SmallVector<bool> started(items.size(), false);
    SmallVector<unsigned> free_cells;
    for (unsigned c = 0; c < (unsigned)kTotalCGRAs; ++c)
      free_cells.push_back(c);
    // Cells are handed out most-recently-freed first, so an item tends to land
    // where its predecessor just finished.
    //
    // The downstream placer's scorer looks like it wants the opposite: it
    // charges kGamma = 1000 for reusing an occupied CGRA against kAlpha = 10
    // per hop of SSA distance, so it takes a never-used cell while one exists.
    // Preferring fresh cells here was tried and is worse, because what this
    // function must predict is not where the placer puts the first 16 tasks but
    // what the analyzer then measures, and the interval is set by the cells
    // that do end up stacked. Measured on bicg: reuse-first predicts 9730
    // against a measured 9730, fresh-first predicts 9458 against a measured
    // 45045 -- and the search, trusting the optimistic number, allocated 46
    // tasks where 20 was better.
    SmallVector<unsigned> running;
    unsigned completed = 0;
    int64_t now = 0;
    while (completed < items.size()) {
      SmallVector<unsigned> candidates;
      for (unsigned i = 0; i < items.size(); ++i)
        if (!started[i] && remaining[i] == 0 && ready_at[i] <= now)
          candidates.push_back(i);
      llvm::sort(candidates, [&](unsigned a, unsigned b) {
        if (prio[a] != prio[b])
          return prio[a] > prio[b];
        return a < b;
      });
      for (unsigned i : candidates) {
        const int need = items[i].area;
        // An item wider than the whole grid can only run alone; letting it
        // through on an empty grid keeps the schedule from deadlocking.
        if ((int)free_cells.size() < need &&
            !(running.empty() && (int)free_cells.size() == kTotalCGRAs))
          continue;
        const int take = std::min(need, (int)free_cells.size());
        for (int cell_slot = 0; cell_slot < take; ++cell_slot) {
          items[i].cells.push_back(free_cells.back());
          free_cells.pop_back();
        }
        started[i] = true;
        items[i].start = now;
        items[i].finish = now + items[i].duration;
        running.push_back(i);
      }
      if (running.empty()) {
        int64_t next = INT64_MAX;
        for (unsigned i = 0; i < items.size(); ++i)
          if (!started[i] && remaining[i] == 0)
            next = std::min(next, ready_at[i]);
        if (next == INT64_MAX || next <= now)
          break; // nothing schedulable: a cycle, or nothing left to do
        now = next;
        continue;
      }
      int64_t next_finish = INT64_MAX;
      for (unsigned i : running)
        next_finish = std::min(next_finish, items[i].finish);
      now = next_finish;
      SmallVector<unsigned> still_running;
      for (unsigned i : running) {
        if (items[i].finish > now) {
          still_running.push_back(i);
          continue;
        }
        ++completed;
        for (unsigned c : items[i].cells)
          free_cells.push_back(c);
        for (auto &[succ_item, delay] : successors_of[i]) {
          ready_at[succ_item] = std::max(ready_at[succ_item], items[i].finish + delay);
          --remaining[succ_item];
        }
      }
      running = still_running;
    }

    // The analyzer's formula, on the cells this schedule produced.
    SmallVector<SmallVector<unsigned, 4>> cell_items(kTotalCGRAs);
    for (unsigned i = 0; i < items.size(); ++i)
      for (unsigned c : items[i].cells)
        cell_items[c].push_back(i);
    int64_t worst = 0;
    for (auto &n : nodes)
      worst = std::max(worst, n->estimatedLatency());
    for (int c = 0; c < kTotalCGRAs; ++c) {
      auto &on_cell = cell_items[c];
      if (on_cell.empty())
        continue;
      llvm::sort(on_cell, [&](unsigned a, unsigned b) {
        if (items[a].start != items[b].start)
          return items[a].start < items[b].start;
        return a < b;
      });
      int64_t occupancy = 0;
      for (unsigned i : on_cell)
        occupancy += items[i].duration;
      int64_t cycle = occupancy;
      const Item &first = items[on_cell.front()];
      const Item &last = items[on_cell.back()];
      if (first.node != last.node) {
        // `longestPathBetween` counts BOTH endpoints, so it is already the
        // analyzer's quantity: the analyzer's edges carry the source's
        // duration, which excludes the target, and it then adds the target's
        // duration once as the cycle's closing latency. Adding `last.duration`
        // here charged it twice. On the vendored `chain_reuse_cgra0` case that
        // reported 15 against the analyzer's 10, and the error scales with the
        // last task on the cell, so it biased the search rather than shifting
        // every configuration equally.
        const int64_t detour = longestPathBetween(first.node, last.node);
        cycle = std::max(cycle, detour);
      }
      worst = std::max(worst, cycle);
    }
    return worst;
  }

  // Longest dependence path from `from` to `to`, weighted by task latency and
  // the communication delay on each edge, or 0 when `to` is unreachable.
  // Parallel tile groups collapse to one node, exactly as criticalPath does, so
  // the value does not change when a cut is materialised.
  int64_t longestPathBetween(const TaskGraphNode *from,
                             const TaskGraphNode *to) const {
    auto key = [](const TaskGraphNode *n) -> const void * {
      if (n->tile_group >= 0 && n->tile_parallel)
        return (const void *)(intptr_t)(n->tile_group + 1);
      return (const void *)n;
    };
    const void *target = key(to);
    DenseMap<const void *, int64_t> weight;
    DenseMap<const void *, SmallVector<std::pair<const void *, int64_t>, 4>>
        succ;
    DenseSet<std::pair<const void *, const void *>> seen;
    for (auto &n : nodes) {
      const void *node_key = key(n.get());
      int64_t &node_weight = weight[node_key];
      node_weight = std::max(node_weight, n->estimatedLatency());
      for (const TaskGraphNode *succ_key_of : n->successors) {
        const void *succ_key = key(succ_key_of);
        if (succ_key == node_key)
          continue;
        if (seen.insert({node_key, succ_key}).second)
          succ[node_key].push_back(
              {succ_key,
               (int64_t)std::llround(scheduleEdgeDelay(n.get(), succ_key_of))});
      }
    }
    // Longest path to `target`; -1 marks "cannot reach", so an unreachable
    // branch never contributes.
    DenseMap<const void *, int64_t> memo;
    std::function<int64_t(const void *)> best = [&](const void *node_key) -> int64_t {
      if (node_key == target)
        return 0;
      auto it = memo.find(node_key);
      if (it != memo.end())
        return it->second;
      memo[node_key] = -1; // cycle guard
      int64_t longest_tail = -1;
      auto succ_it = succ.find(node_key);
      if (succ_it != succ.end())
        for (auto &[succ_key_of, delay] : succ_it->second) {
          int64_t tail_len = best(succ_key_of);
          if (tail_len < 0)
            continue;
          longest_tail = std::max(longest_tail, weight.lookup(succ_key_of) + delay + tail_len);
        }
      memo[node_key] = longest_tail;
      return longest_tail;
    };
    const void *src = key(from);
    if (src == target)
      return 0;
    int64_t tail = best(src);
    return tail < 0 ? 0 : weight.lookup(src) + tail;
  }

  // The single joint objective moved by all six knobs (fusion,
  // fission/replicas, tiling, cgra-shape, communication, spatial-temporal).
  //
  //   objective-mode=interval (default, the metric this pass shipped with):
  //     max task latency, plus the comm proxy, times the context waves needed.
  //   objective-mode=makespan:
  //     resource-constrained schedule length of ONE input instance.
  //   objective-mode=pipeline:
  //     steady-state pipeline interval -- the quantity the downstream analyzer
  //     actually reports, so this is the mode to optimise when that is the
  //     number being scored.
  int64_t objective() const {
    if (use_pipeline)
      return pipelineInterval();
    if (use_makespan)
      return makespan();
    int64_t interval = 0;
    for (auto &n : nodes)
      interval = std::max(interval, n->serialTiledLatency());
    double value = (double)interval + comm_weight * commCost();
    return (int64_t)std::llround(value * (double)wavesNeeded());
  }

  void build(func::FuncOp func, bool skip_mapper = false) {
    // 1. Creates TaskGraphNodes.
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskGraphNode>(task_id++, task);

      // If the task already has profiling attributes (e.g., from fusion),
      // skip expensive speculative lowering and use those directly.
      bool has_precomputed =
          task->hasAttr("compiled_ii") && task->hasAttr("profile_info");
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
      if (auto profile = task->getAttrOfType<DictionaryAttr>("profile_info"))
        if (auto dur = dyn_cast_or_null<IntegerAttr>(profile.get("duration")))
          node->steps = dur.getInt();
      if (auto attr = task->getAttrOfType<IntegerAttr>("compiled_ii")) {
        node->ii = attr.getInt();
      }
      if (auto attr = task->getAttrOfType<IntegerAttr>("cgra_count")) {
        node->cgra_count = attr.getInt();
      }
      if (auto attr = task->getAttrOfType<IntegerAttr>("replicas")) {
        node->replicas = std::max(1, (int)attr.getInt());
      }
      if (auto attr = task->getAttrOfType<IntegerAttr>("tiling")) {
        node->tiling = std::max(1, (int)attr.getInt());
      }
      // DLP capability comes from ClassifyTaskAndCounterPass: a task is
      // replicable/partitionable iff it has a counter dimension free of
      // loop-carried dependence.
      if (auto attr = task->getAttrOfType<BoolAttr>("dlp_replicable")) {
        node->dlp_replicable = attr.getValue();
      }
      // Restores the shape decision from the IR. Without this a node rebuilt
      // from attributes keeps the 1x1 default while carrying cgra_count > 1,
      // and the next write-back publishes that inconsistent pair — the
      // downstream placer then honours the shape and silently under-allocates.
      // Only when the shape search is on: restoring the shape changes which
      // tile array a rebuilt node is profiled on, which moves the balancer's
      // decisions, and the in-tree RESOPT expectations pin those. Off by
      // default the node keeps the historical 1x1 initial shape (profile_fn
      // sets the real one before every probe anyway).
      if (search_shape) {
        if (auto attr = task->getAttrOfType<StringAttr>("cgra_shape")) {
          if (auto parsed = parseRectShape(attr.getValue(), node->cgra_count))
            node->shape = *parsed;
          else
            node->shape = pickBestShape(node->cgra_count);
        } else {
          node->shape = pickBestShape(node->cgra_count);
        }
      }
      // Size of the partition space both DLP moves draw from: the product of
      // every legally-cuttable level of the loop NEST, not just the outer trip
      // count (an 8x8 nest offers 64 partitions, not 8).
      node->split_space = TaskTiler::partitionSpace(task);
      node->root_trip = TaskTiler::parallelPartitionSpace(task);
      // MEMORY legality, which `dlp_replicable` does not check: a reduction
      // that accumulates through a memref (histogram) has no loop-carried SSA
      // dependence and is therefore tagged replicable, but splitting it makes
      // the partitions race on the same addresses.
      if (node->dlp_replicable && !TaskTiler::rootDimIsPartitionable(task)) {
        node->dlp_replicable = false;
        node->partition_illegal = true;
      }
      // Tiles materialised by a previous outer iteration.
      if (auto attr = task->getAttrOfType<IntegerAttr>("tile_group"))
        node->tile_group = (int)attr.getInt();
      if (auto attr = task->getAttrOfType<BoolAttr>("tile_parallel"))
        node->tile_parallel = attr.getValue();
      if (auto attr = task->getAttrOfType<DenseI64ArrayAttr>("tile_range"))
        if (attr.size() >= 1)
          node->tile_dim = (int)attr[0]; // [counter_id, lb, ub, ...]

      op_to_node[task] = node.get();
      nodes.push_back(std::move(node));
    });

    // 2. Builds SSA edges (value dependencies between tasks).
    for (auto &consumer : nodes) {
      for (Value operand : consumer->op.getValueInputs()) {
        if (auto producer_op = operand.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get(), operand);
          }
        }
      }
      for (Value operand : consumer->op->getOperands()) {
        if (auto producer_op = operand.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get(), operand);
          }
        }
      }
    }

    // 3. Builds memory edges.
    for (auto &consumer : nodes) {
      // RAW: producer wrote a memref that this task reads.
      for (Value memref : consumer->op.getWillReads()) {
        if (auto producer_op = memref.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get());
          }
        }
      }
      // WAW/WAR: producer wrote or read a memref that this task writes.
      for (Value memref : consumer->op.getWillWrites()) {
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

  // Returns total CGRAs allocated: each replica occupies its own tile array of
  // `cgra_count` CGRAs, so the budget is charged cgra_count * replicas.
  int getTotalAllocatedCGRAs() const {
    int total = 0;
    for (auto &node : nodes) {
      total += node->footprintCgras();
    }
    return total;
  }

  // Profiles a single TaskflowTaskOp: clones the task, wraps the kernel in a
  // standalone func, and runs InsertDataMov + MapToAcceleratorPass to obtain
  // ii.  skip_mapper: use only ResMII/RecMII analytical estimates.
  // When skip_mapper=true, only ResMII/RecMII analytical estimates are used
  // (no MapToAcceleratorPass). This is safe for speculative balance checks
  // where the mapper may backtrack indefinitely on larger tile arrays.
  // What one profile produces. The result depends only on the task's body and
  // the tile array it is mapped to, never on replicas or tiling, which divide
  // the iteration space without touching the mapped loop body.
  struct ProfileResult {
    int64_t ii;
    int64_t steps;
    double avg_hop;
    double predicted_cost;
    // Shape-independent, but cached with the rest: a hit on a node that has
    // not been profiled before would otherwise leave them at zero, and the
    // shape ranking reads them.
    int64_t n_ops;
    int64_t rec_mii;
  };
  // (task identity, shape, mapper-or-not) -> profile.
  using ProfileCache =
      std::map<std::tuple<std::string, std::string, int>, ProfileResult>;
  // (task, rows, cols, non-rectangular fingerprint) -> profile.
  //
  // Profiling clones the task into a temporary module and re-runs the Neura
  // lowering on it, so it is by far the most expensive operation in the pass.
  // The balancer re-profiles every task tied at the bottleneck on every
  // candidate move, which without this cache means thousands of lowerings and a
  // search that does not finish: axpy_20 ran over 10 minutes and was still
  // going.
  //
  // Identity is the TILE GROUP when the task belongs to one, not the operation.
  // Tiles cut from a single loop share a body and a tile array and differ only
  // in their loop bounds, and the mapped DFG does not depend on the bounds: on
  // the 28-task transformer every member of every group comes out with an
  // identical (ii, steps, cgra_count). Keying on the operation made a t-way cut
  // cost t mappings for one distinct kernel, which is most of what made
  // partitioning look expensive to compile.
  //
  // `skip_mapper` is part of the key because the two modes answer different
  // questions. Without it an analytical result cached during the search is
  // handed to the final mapper verification, which then verifies nothing.
  // The shape enters the key as its IR string, not as a bounding box plus a
  // cell count. Two non-rectangular shapes can share both -- nothing in the
  // shape tables forbids a second four-cell shape in a 2x3 box -- and they
  // would then share a cache entry and one would silently be given the other's
  // II. `irAttr()` distinguishes them because it lists the occupied positions,
  // which is exactly what the mapper's tile array is built from.
  // std::map, not DenseMap: LLVM ships no DenseMapInfo for std::string. The
  // container is only ever looked up, never iterated, so its ordering does not
  // reach any decision.
  // Shared across every graph the pass builds, and keyed on the task NAME.
  //
  // `global-fusion` scores O(V^2) candidate pairs, and each score clones the
  // function, rebuilds a graph and balances it. With the cache owned per graph
  // and keyed on the task's `Operation *`, every one of those clones started
  // cold and re-profiled every task: on bicg, 19 tasks meant 171 pairs and 136
  // seconds where the same run without fusion ranking took 15.6. A clone's ops
  // live at different addresses, so a shared cache alone would still have
  // missed; the name is what survives cloning.
  //
  // Names identify a body within one run: fusion emits `A_B_utilfused` and
  // tiling emits `base_tileN`, so a name is never reused for different
  // contents, and a tiled node is keyed by its group anyway.
  std::shared_ptr<ProfileCache> profile_cache =
      std::make_shared<ProfileCache>();

  // The three body facts the ranking needs, read straight off the IR the main
  // pipeline already produced. No clone, no temporary module, no pass.
  //
  // The filters match the ones the cloned path uses, so the numbers agree:
  // `n_ops` uses `calculateResMii`'s filter (skip DataMov/CtrlMov/Reserve and
  // anything inside a FusedOp) and `rec_mii`/`steps` come from the same
  // `collectRecurrenceCycles` and ALAP-level helpers. Returns false when the
  // task has no kernel to read, so the caller falls back rather than inventing
  // numbers.
  bool readBodyFactsInPlace(TaskflowTaskOp task, int64_t &n_ops,
                            int64_t &rec_mii, int64_t &steps) {
    neura::KernelOp kernel;
    task.walk([&](neura::KernelOp k) { kernel = k; });
    if (!kernel)
      return false;
    Region &region = kernel.getBody();
    if (region.empty())
      return false;

    n_ops = 0;
    region.walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) ||
          isa<neura::CtrlMovOp, neura::DataMovOp, neura::ReserveOp>(op))
        return;
      if (op->getParentOp() && isa<neura::FusedOp>(op->getParentOp()))
        return;
      ++n_ops;
    });

    rec_mii = 1;
    auto cycles = neura::collectRecurrenceCycles(region);
    for (auto &cycle : cycles)
      rec_mii = std::max<int64_t>(rec_mii, cycle.length);

    steps = 1;
    std::set<Operation *> critical_ops;
    for (auto &cycle : cycles)
      for (Operation *op : cycle.operations)
        critical_ops.insert(op);
    auto sorted_ops = neura::getTopologicallySortedOps(region);
    if (!sorted_ops.empty())
      steps = std::max<int64_t>(
          1, (int64_t)neura::getOpsInAlapLevels(sorted_ops, critical_ops)
                 .size());
    return n_ops > 0;
  }

  void profileTask(TaskGraphNode *node, TaskflowTaskOp task,
                   bool skip_mapper = false) {
    // The function name is part of the identity because the pass object, and
    // therefore this cache, outlives one function: two functions in a module
    // both have a `Task_0`, and they are not the same body.
    std::string scope;
    if (auto parent = task->getParentOfType<func::FuncOp>())
      scope = parent.getName().str();
    const std::string identity =
        scope + "/" +
        (node->tile_group >= 0
             ? ("tile_group:" + std::to_string(node->tile_group))
             : ("task:" + task.getTaskName().str()));
    // An ANALYTICAL profile does not depend on the shape, so it is not keyed on
    // one.
    //
    // Everything `profileTaskUncached` learns from the clone -- `n_ops`,
    // `rec_mii`, `steps` -- is a property of the task BODY. The shape reaches
    // the answer only through the closed form below, which needs no clone at
    // all. Keying the analytical entry on the shape therefore paid the most
    // expensive operation in the pass for a number arithmetic already had:
    // gemver profiled Task_0 at 1x1, 1x2, 2x1, 2x2, 1x3, 3x1, 1x4, 4x1 and 2x3
    // -- nine clones and nine full Neura lowerings -- and every one of them
    // came back with the same n_ops and the same rec_mii. Across its three
    // tasks that was 21 lowerings spent to rank shapes, which is exactly the
    // enumerate-and-lower pattern the ranking is supposed to replace. The
    // ranking was closed-form in the joint pool and not in
    // `profileWithBestShape`, and the shape axis is where most candidates are.
    //
    // A VERIFIED profile still carries the shape: there the mapper or the
    // solver is handed the actual tile array, and that is the whole point of
    // running it.
    const auto key = std::make_tuple(
        identity, skip_mapper ? std::string("*") : node->shape.irAttr(),
        skip_mapper ? 1 : 0);
    auto cached = profile_cache->find(key);
    // The ranking compiles nothing at all.
    //
    // An analytical profile used to clone the kernel into a temporary module,
    // run InsertDataMov over it, and only then count -- per task, per shape.
    // None of the three numbers it was after needs any of that. `n_ops` is
    // counted with a filter that EXCLUDES exactly the ops InsertDataMov adds;
    // `rec_mii` and the ALAP depth read the dataflow IR the main pipeline has
    // already produced. The shape enters only as the tile count in the closed
    // form below.
    //
    // So the facts are read in place, once per task body, and every shape after
    // that is arithmetic. On gemver this replaced 30 clone-and-lower rounds
    // with 3 IR walks.
    if (skip_mapper && cached == profile_cache->end()) {
      int64_t body_ops = 0, body_rec = 1, body_steps = 1;
      if (readBodyFactsInPlace(task, body_ops, body_rec, body_steps)) {
        ProfileResult facts;
        facts.ii = 1;
        facts.steps = body_steps;
        facts.avg_hop = 0.0;
        facts.predicted_cost = 0.0;
        facts.n_ops = body_ops;
        facts.rec_mii = body_rec;
        cached = profile_cache->emplace(key, facts).first;
      }
    }
    if (cached != profile_cache->end()) {
      node->steps = cached->second.steps;
      node->n_ops = cached->second.n_ops;
      node->rec_mii = cached->second.rec_mii;
      if (!skip_mapper) {
        node->ii = cached->second.ii;
        node->avg_hop = cached->second.avg_hop;
        node->predicted_cost = cached->second.predicted_cost;
        return;
      }
      // Re-derive the two shape-dependent quantities. Same expressions the
      // uncached analytical path uses, so a hit and a miss agree.
      const int64_t tiles = (int64_t)node->shape.rows * per_cgra_rows *
                            (int64_t)node->shape.cols * per_cgra_cols;
      node->ii = node->n_ops > 0
                     ? std::max<int64_t>(
                           llvm::divideCeil(node->n_ops,
                                            std::max<int64_t>(1, tiles)),
                           node->rec_mii)
                     : cached->second.ii;
      node->avg_hop = hopForShape(node->shape);
      node->predicted_cost =
          predictedCost(node->ii, node->avg_hop, node->steps, hop_coef);
      return;
    }
    profileTaskUncached(node, task, skip_mapper);
    (*profile_cache)[key] = {node->ii,           node->steps,
                             node->avg_hop,       node->predicted_cost,
                             node->n_ops,         node->rec_mii};
  }

  void profileTaskUncached(TaskGraphNode *node, TaskflowTaskOp task,
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

    double prof_avg_hop = 0.0;
    // Shape-independent facts about the body; see `TaskGraphNode::n_ops`.
    int64_t prof_n_ops = 0, prof_rec_mii = 1;

    if (succeeded(runNeuraPipelineOnKernel(
            ctx, cloned_kernel, phase2_module, compiled_ii, cp_depth, x_tiles,
            y_tiles, valid_tiles, skip_mapper, &prof_avg_hop, &prof_n_ops,
            &prof_rec_mii))) {
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
    node->avg_hop = prof_avg_hop;
    if (prof_n_ops > 0) {
      node->n_ops = prof_n_ops;
      node->rec_mii = prof_rec_mii;
    }
    node->predicted_cost =
        predictedCost(node->ii, node->avg_hop, node->steps, hop_coef);

    llvm::errs() << "[profileTask] " << task.getTaskName()
                 << ": compiled_ii=" << node->ii << ", steps=" << node->steps
                 << "\n";

    // Erases the temporary module.
    tmp_mod.erase();
  }

  // Profiles the task at its current cgra_count, choosing the SHAPE by the
  // shape-aware predicted II (LB + hop) over every shape of that cgra_count,
  // instead of pickBestShape's geometric rule. Leaves the node profiled with
  // the winning shape. Falls back to the geometric pick when search_shape is
  // off, so default behaviour is bit-identical.
  // Emits the shape-selection ranking experiment for one task, in the real
  // pipeline: every candidate shape at every cgra_count is scored BOTH with the
  // cheap model (LB, and LB + hop) and with the real mapper, so top-k coverage
  // of the true optimum can be measured directly rather than replayed offline.
  //
  //   LB        = analytical floor, shape-blind at fixed area
  //   (ResMII=ops/tiles) LB + hop  = LB + max(0, coef * avg_hop * cp_depth - 1)
  //   <- our model true      = MapToAcceleratorPass II  (the oracle we are
  //   trying to rank to)
  void probeShapeCoverage(TaskGraphNode *node, TaskflowTaskOp task) {
    const int saved_cgra_count = node->cgra_count;
    const CgraShape saved_shape = node->shape;
    const int64_t saved_ii = node->ii, saved_steps = node->steps;
    for (int cgra_count = 1; cgra_count <= kMaxCgrasPerTask; ++cgra_count) {
      if (!canFitOnGrid(cgra_count))
        continue;
      SmallVector<CgraShape> candidates = getRectangularShapes(cgra_count);
      for (const auto &shape : getNonRectangularShapes(cgra_count))
        candidates.push_back(shape);
      if (candidates.size() < 2)
        continue; // Nothing to rank.
      node->cgra_count = cgra_count;
      for (const CgraShape &candidate : candidates) {
        node->shape = candidate;
        // Cheap pass: analytical only.
        profileTask(node, task, /*skip_mapper=*/true);
        const int64_t analytical_lb = node->ii;
        const double avg_hop = node->avg_hop;
        const double predicted = node->predicted_cost;
        const int64_t depth = node->steps;
        // Oracle pass: the real mapper on the same shape.
        profileTask(node, task, /*skip_mapper=*/false);
        llvm::errs() << "[ShapeProbe] task=" << task.getTaskName()
                     << " cgra_count=" << cgra_count
                     << " shape=" << candidate.describe(cgra_count)
                     << " lb=" << analytical_lb
                     << " hop=" << llvm::format("%.4f", avg_hop)
                     << " depth=" << depth
                     << " pred=" << llvm::format("%.4f", predicted)
                     << " true_ii=" << node->ii << " true_steps=" << node->steps
                     << "\n";
      }
    }
    node->cgra_count = saved_cgra_count;
    node->shape = saved_shape;
    node->ii = saved_ii;
    node->steps = saved_steps;
  }

  // Mean hop over the tile array a shape defines, memoised.
  //
  // This is the same quantity `profileTask` reports as `avg_hop`, computed
  // without compiling anything: the architecture is cloned to the shape's tile
  // dimensions and walked. It depends on the shape and on nothing else, so one
  // entry per shape serves every task.
  //
  // On a fabric whose tiles are uniform it cannot separate two shapes of equal
  // area: the mean distance over an R x C mesh is symmetric in R and C, so 4x8
  // and 8x4 score alike. `mem-weighted-hop` exists to break that tie by
  // weighting toward memory-capable tiles, and on `architecture_4x4.yaml` it
  // cannot, because `tile_defaults` gives every tile `mem` and `mem_indexed`
  // and no override takes it away. Measured: both settings yield the same two
  // distinct values across every shape this benchmark set reaches.
  mutable std::map<std::pair<int, int>, double> hop_by_shape;

  double hopForShape(const CgraShape &shape) const {
    const auto key = std::make_pair(shape.rows * per_cgra_rows,
                                    shape.cols * per_cgra_cols);
    auto cached = hop_by_shape.find(key);
    if (cached != hop_by_shape.end())
      return cached->second;
    std::unique_ptr<neura::Architecture> array =
        neura::getArchitecture().cloneWithNewDimensions(key.first, key.second);
    const double hops =
        array ? averageHop(*array, mem_weighted_hop)
              : averageHop(neura::getArchitecture(), mem_weighted_hop);
    hop_by_shape[key] = hops;
    return hops;
  }

  void profileWithBestShape(TaskGraphNode *node, TaskflowTaskOp task,
                            bool skip_mapper = false) {
    if (!search_shape) {
      node->shape = pickBestShape(node->cgra_count);
      profileTask(node, task, skip_mapper);
      return;
    }
    SmallVector<CgraShape> candidates = getRectangularShapes(node->cgra_count);
    for (const auto &shape : getNonRectangularShapes(node->cgra_count))
      candidates.push_back(shape);
    if (candidates.size() <= 1) {
      node->shape = candidates.empty() ? pickBestShape(node->cgra_count)
                                       : candidates.front();
      profileTask(node, task, skip_mapper);
      return;
    }

    // Rank cheaply, verify the top few. Profiling every candidate to find out
    // which is best is the pattern this work replaces on every other axis, and
    // it was the most expensive one left: on bicg it ran 1512 profiles.
    //
    // One profile gives the facts no shape changes -- the operation count and
    // the recurrence bound -- and every candidate's lower bound then follows in
    // closed form, ceil(n_ops / tiles(shape)) against rec_mii. The topology
    // term comes from the shape's own tile array, which is a graph walk rather
    // than a compilation.
    node->shape = pickBestShape(node->cgra_count);
    profileTask(node, task, skip_mapper);
    const int64_t body_ops = node->n_ops;
    const int64_t body_recurrence = node->rec_mii;
    const int64_t body_depth = node->steps;

    SmallVector<std::pair<double, unsigned>> ranked;
    for (auto [index, candidate] : llvm::enumerate(candidates)) {
      const int64_t tiles = (int64_t)candidate.rows * per_cgra_rows *
                            (int64_t)candidate.cols * per_cgra_cols;
      const int64_t lower_bound =
          body_ops > 0
              ? std::max<int64_t>(
                    llvm::divideCeil(body_ops, std::max<int64_t>(1, tiles)),
                    body_recurrence)
              : node->ii;
      ranked.push_back({predictedCost(lower_bound, hopForShape(candidate),
                                      body_depth, hop_coef),
                        (unsigned)index});
    }
    llvm::sort(ranked);

    // Verification is a real profile, and it obeys the caller's `skip_mapper`.
    // With the mapper skipped it cannot separate two shapes of the same tile
    // count on a uniform fabric -- their lower bound is equal and the mean hop
    // over an R x C mesh is symmetric in R and C -- so the ranking's own order
    // decides. That is a property of this architecture, whose every tile is
    // memory-capable, not of the search.
    const unsigned to_verify =
        verify_top_k <= 0 ? (unsigned)ranked.size()
                          : std::min<unsigned>(verify_top_k, ranked.size());

    CgraShape best_shape = candidates[ranked.front().second];
    double best_pred = std::numeric_limits<double>::infinity();
    int64_t best_ii = INT64_MAX, best_steps = INT64_MAX;
    double best_hop = 0.0;
    for (unsigned position = 0; position < to_verify; ++position) {
      const CgraShape &candidate = candidates[ranked[position].second];
      node->shape = candidate;
      profileTask(node, task, skip_mapper);
      // Selection is on what the profile MEASURED, not on the score that
      // ordered the shortlist. Picking by `predicted_cost` here threw the
      // verification away: the whole point of mapping k shapes is to learn
      // their real II, and the winner was then chosen by the same closed form
      // that had already ranked them -- so the mapper's answer changed nothing
      // and the shortlist could not be wrong in a way the search would notice.
      // Ties on II break on depth, then on the ranking score.
      const bool better =
          node->ii < best_ii ||
          (node->ii == best_ii &&
           (node->steps < best_steps ||
            (node->steps == best_steps &&
             node->predicted_cost < best_pred - 1e-9)));
      if (better) {
        best_pred = node->predicted_cost;
        best_shape = candidate;
        best_ii = node->ii;
        best_steps = node->steps;
        best_hop = node->avg_hop;
      }
    }
    if (candidates.size() > 1) {
      CgraShape geometric = pickBestShape(node->cgra_count);
      bool differs = geometric.rows != best_shape.rows ||
                     geometric.cols != best_shape.cols;
      llvm::errs() << "  [ShapeSearch] " << task.getTaskName()
                   << " cgra_count=" << node->cgra_count << " -> "
                   << best_shape.describe(node->cgra_count)
                   << " (predicted=" << llvm::format("%.4f", best_pred)
                   << ", avg_hop=" << llvm::format("%.3f", best_hop)
                   << ", geometric pick was "
                   << geometric.describe(node->cgra_count)
                   << (differs ? " [DIFFERS]" : " [same]") << ")\n";
    }
    // Re-profiles at the winner so the node carries its numbers.
    node->shape = best_shape;
    profileTask(node, task, skip_mapper);
    node->ii = best_ii;
    node->steps = best_steps;
  }

private:
  llvm::DenseSet<std::pair<TaskGraphNode *, TaskGraphNode *>> edge_set;

public:
  // Elements crossing each edge, when the carrier memref has a static shape.
  llvm::DenseMap<std::pair<TaskGraphNode *, TaskGraphNode *>, double>
      edge_volume;

private:
  void addEdge(TaskGraphNode *from, TaskGraphNode *to, Value carried = {}) {
    auto key = std::make_pair(from, to);
    if (edge_set.insert(key).second) {
      from->successors.push_back(to);
      to->predecessors.push_back(from);
    }
    // Data volume actually crossing this edge. `trip_count` -- what the proxy
    // used before -- is the producer's ITERATION count, which has nothing to do
    // with how many elements it hands over: a 524288-iteration task may write a
    // single scalar. Use the element count of the memref that carries the
    // dependence when its shape is static.
    if (carried) {
      if (auto mt = dyn_cast<MemRefType>(carried.getType())) {
        if (mt.hasStaticShape()) {
          int64_t n = 1;
          for (int64_t d : mt.getShape())
            n *= std::max<int64_t>(1, d);
          double &v = edge_volume[key];
          v = std::max(v, (double)n);
        }
      }
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
      const std::string &valid_tiles = "", bool skip_mapper = false,
      double *out_avg_hop = nullptr, int64_t *out_n_ops = nullptr,
      int64_t *out_rec_mii = nullptr) {
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
        if (use_full_cost_model) {
          // Full parametric model: II = max of all resource bounds, computed by
          // computeAnalyticalII. Do NOT fold in the crude res_mii/rec_mii here.
          // The crude rec_mii is max(cycle.length) — the raw hop count around the
          // recurrence, INCLUDING the non-materialized reserve/ctrl_mov/data_mov
          // hops — so it can exceed the parametric (latency-weighted, materialized-
          // only) RecMII inside bd.final_ii. Folding it back in would let that
          // coarse hop-count override the parametric prediction on recurrence-
          // bound kernels and perturb cross-kernel ranking — exactly the metric
          // this model replaces. bd.final_ii already includes IssueMII, which
          // subsumes the crude res_mii.
          neura::AnalyticalIIBreakdown bd =
              neura::computeAnalyticalII(region, architecture);
          llvm::errs() << "[cost-model-analytical] task profiling:\n";
          bd.print(llvm::errs());
          // `final_ii` is a sound floor and nothing more. Measured against the
          // exact CP-SAT mapper it under-predicts wherever routing, not
          // throughput, is what binds: `plain_gemm` on 4x4 schedules at II=1,
          // fails to ROUTE there, and the true optimum is 2. Reporting the
          // floor as if it were the II is what made an interval computed from
          // it look 3x better than the same program measured any other way.
          // `predicted_ii` adds the distance-charged routing term on top of the
          // same floor; it is opt-in so the floor stays available unchanged to
          // every caller that needs soundness rather than accuracy.
          compiled_ii = std::max(
              compiled_ii, use_predicted_ii ? bd.predicted_ii : bd.final_ii);
        } else {
          compiled_ii = std::max({compiled_ii, res_mii, rec_mii});
        }
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

        if (out_avg_hop)
          *out_avg_hop = averageHop(architecture, mem_weighted_hop);
        if (out_rec_mii)
          *out_rec_mii = rec_mii;
        if (out_n_ops) {
          // Counted with exactly the filter neura::calculateResMii uses, so
          // that ceilDiv(n_ops, tiles) reproduces its ResMII.
          int64_t counted = 0;
          region.walk([&](Operation *op) {
            if (isa<func::FuncOp>(op) ||
                isa<neura::CtrlMovOp, neura::DataMovOp, neura::ReserveOp>(op))
              return;
            if (isa<neura::FusedOp>(op->getParentOp()))
              return;
            ++counted;
          });
          *out_n_ops = counted;
        }
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

    // Configurable, because at 150 this guard decides the experiment rather
    // than protecting it. Measured on bicg: the guard fires 85 times, on tasks
    // of 1050 and 2099 ops, so every request for a real lowering is silently
    // served from the analytical estimate. On this benchmark set that leaves
    // "rank cheaply, lower the top k" with nothing to lower, and no shortlist
    // depth can be shown to matter while it holds. Default unchanged.
    const int kMapperOpLimit = mapper_op_limit;
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

    // The exact mapper is the verifier this design calls for. The heuristic
    // backtracker it replaces is not a measurement: it OVER-predicts (it puts
    // `plain_gemm` on 4x4 at II=3 where the true optimum is 2), and on a task
    // of ~400 ops a single `map()` call does not return -- which is what made
    // "lower the top k and keep the best" unusable on anything but small
    // kernels. CP-SAT proves each II infeasible or produces a routed witness,
    // and it takes a time budget, so a hard case degrades into a stated bound
    // instead of a hang.
    if (verifier_is_cpsat && all_data_movs_ok) {
      int solved_ii = 0;
      bool proven = false;
      // `compiled_ii` here is the analytical floor the block above just
      // computed, and the solver starts its ladder there instead of at 1.
      //
      // Every II below a sound floor is infeasible by arithmetic, so probing
      // one spends the whole per-II budget re-deriving what the cost model
      // already proved -- and spends it the expensive way, because an
      // infeasible II usually TIMES OUT rather than coming back UNSAT. On
      // gesummv, whose true II is 3, a single solver call cost 48.4s: 20s
      // failing to settle II=1, 20s failing to settle II=2, and the rest
      // actually solving II=3.
      //
      // It also tightens the verdict. With the ladder starting at 1 a witness
      // found at the floor is tagged an upper bound, because `--fallback`
      // cannot distinguish "II-1 was proved infeasible" from "II-1 timed out".
      // Starting at a proven floor, a witness AT the floor is optimal.
      if (solveWithCpsat(dst_module, ctx, x_tiles, y_tiles, valid_tiles,
                         std::max(1, compiled_ii), solved_ii, proven)) {
        compiled_ii = solved_ii;
        llvm::errs() << "[profileTask] cpsat returned II=" << solved_ii
                     << (proven ? " (proven optimal)" : " (upper bound)")
                     << "\n";
        return success();
      }
      // No verdict inside the budget. Keeping the analytical estimate is the
      // same fallback the mapper path uses, but it is announced: a silent
      // fallback is how a run reports "verified" for a task nothing verified.
      llvm::errs() << "[profileTask] WARNING: cpsat produced no verdict; "
                   << "keeping analytical compiled_ii=" << compiled_ii << "\n";
      return success();
    }

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

  // Runs the exact CP-SAT modulo mapper on one lowered kernel and returns the
  // II it certifies.
  //
  // The solver is a separate process (`test/cost-model/exact_mapper_cpsat.py`,
  // OR-Tools) because that is where it already lives and where PR #340's
  // `*.exact-mapping.json` witnesses come from; keeping one implementation
  // means the II verified here and the mapping `--import-mapping` replays are
  // produced by the same code. The exchange format is the DFG the existing
  // `--dump-dfg-json` emits, so nothing about the DFG or the architecture is
  // re-derived on this path.
  //
  // `--fallback` is the project's ii+1-on-timeout convention: an II the solver
  // cannot settle within its budget is skipped rather than aborting the task.
  // The result is then a routable II found within budget -- an upper bound --
  // and `out_proven` says so, because a table that cannot tell a proven optimum
  // from a budget-limited one will eventually report the second as the first.
  bool solveWithCpsat(ModuleOp dst_module, MLIRContext *ctx, int x_tiles,
                      int y_tiles, StringRef valid_tiles, int min_ii,
                      int &out_ii, bool &out_proven) {
    llvm::SmallString<128> dfg_path;
    if (llvm::sys::fs::createTemporaryFile("neura-dfg", "json", dfg_path)) {
      llvm::errs() << "[cpsat] cannot create a temporary file for the DFG\n";
      return false;
    }
    auto remove_dfg = llvm::make_scope_exit(
        [&] { llvm::sys::fs::remove(dfg_path.str()); });

    {
      PassManager dump_pm(ctx);
      dump_pm.enableVerifier(false);
      neura::DumpDfgJsonOptions dump_options;
      dump_options.x_tiles = x_tiles;
      dump_options.y_tiles = y_tiles;
      dump_options.valid_tiles = valid_tiles.str();
      dump_options.output_file = dfg_path.str().str();
      dump_pm.addPass(neura::createDumpDfgJsonPass(dump_options));
      if (failed(dump_pm.run(dst_module))) {
        llvm::errs() << "[cpsat] dump-dfg-json failed\n";
        return false;
      }
    }

    llvm::SmallString<128> out_path;
    if (llvm::sys::fs::createTemporaryFile("neura-cpsat", "txt", out_path)) {
      llvm::errs() << "[cpsat] cannot create a temporary file for the output\n";
      return false;
    }
    auto remove_out = llvm::make_scope_exit(
        [&] { llvm::sys::fs::remove(out_path.str()); });

    std::string seconds = std::to_string(std::max(1, cpsat_seconds));
    // The ladder starts at the analytical lower bound, not at 1.
    std::string min_ii_str = std::to_string(std::max(1, min_ii));
    SmallVector<StringRef> argv = {cpsat_python, cpsat_script, dfg_path.str(),
                                   "--seconds",  seconds,      "--fallback",
                                   "--min-ii",   min_ii_str,   "-v"};
    argv.push_back(cpsat_minimize_routing ? "--minimize-routing"
                                          : "--no-minimize-routing");
    // The solver prints its verdict on stdout and its per-II trace on stderr,
    // and the trace is what distinguishes a proven optimum from a skipped II,
    // so both are captured into the same file.
    std::optional<StringRef> redirects[] = {std::nullopt, StringRef(out_path),
                                            StringRef(out_path)};
    std::string error_message;
    // Wall-clock ceiling: the per-II budget times the II range it may walk,
    // plus interpreter start-up. Without it a solver that ignores its own
    // budget would reintroduce exactly the hang this path exists to remove.
    const unsigned wall_seconds =
        (unsigned)(std::max(1, cpsat_seconds) * 2 * 24 + 60);
    int rc = llvm::sys::ExecuteAndWait(cpsat_python, argv, /*Env=*/std::nullopt,
                                       redirects, wall_seconds,
                                       /*MemoryLimit=*/0, &error_message);
    if (rc != 0) {
      llvm::errs() << "[cpsat] solver exited " << rc
                   << (error_message.empty() ? "" : ": ") << error_message
                   << "\n";
      return false;
    }

    auto buffer = llvm::MemoryBuffer::getFile(out_path.str());
    if (!buffer) {
      llvm::errs() << "[cpsat] cannot read the solver output\n";
      return false;
    }
    StringRef text = (*buffer)->getBuffer();

    // `TRUE_MIN_II = N` or `MAPPED_II = N`. The tag does not settle optimality:
    // `--fallback` prints MAPPED_II unconditionally. Two things in the trace
    // do. A skipped II ("schedule timeout") decided nothing about that II. A
    // "routing=FAIL" rejected only the one placement the solver's first stage
    // returned -- it is a two-stage mapper, place-then-route, so a routing
    // failure is a statement about that witness and not about the II. Both must
    // be absent for the answer to be the proven minimum; otherwise it is a
    // routable II found within budget, i.e. an upper bound.
    for (StringRef tag : {StringRef("TRUE_MIN_II = "),
                          StringRef("MAPPED_II = ")}) {
      size_t at = text.find(tag);
      if (at == StringRef::npos)
        continue;
      StringRef rest = text.substr(at + tag.size());
      int value = 0;
      if (rest.take_while(llvm::isDigit).getAsInteger(10, value) || value <= 0)
        continue;
      out_ii = value;
      // With the ladder starting at the analytical floor, a witness AT that
      // floor is optimal without the solver having to prove anything below it:
      // those IIs are infeasible by the cost model's own arithmetic. Above the
      // floor the old rule stands, because then the solver did climb past IIs
      // it may only have failed to settle.
      out_proven = value <= std::max(1, min_ii) ||
                   (!text.contains("schedule timeout") &&
                    !text.contains("routing=FAIL"));
      return true;
    }
    // `... > N` is the solver's other outcome: no II up to the architecture's
    // `ctrl_mem_items` ceiling admits a mapping, i.e. this body does not fit
    // this shape at all. It is a real answer and must not be reported as "the
    // solver said nothing" -- that reads as a tooling failure when it is a
    // statement about the candidate. The caller keeps the analytical estimate,
    // which is already clamped to the same ceiling.
    if (text.contains("TRUE_MIN_II > ") || text.contains("MAPPED_II > ")) {
      llvm::errs() << "[cpsat] not mappable at this shape within the II "
                   << "ceiling; the body needs more tiles or more fission\n";
      return false;
    }
    return false;
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
  // Iterations of the single root taskflow.counter, i.e. the size of the only
  // dimension the tiling/replication moves may split. Returns 1 when the task
  // has no unique constant-bound root (nothing to partition).
  static int64_t rootTripCountOf(TaskflowTaskOp task) {
    if (!task.getBody().hasOneBlock())
      return 1;
    TaskflowCounterOp root;
    unsigned n_roots = 0;
    for (Operation &op : task.getBody().front()) {
      auto counter = dyn_cast<TaskflowCounterOp>(&op);
      if (!counter || counter.getParentIndex())
        continue;
      ++n_roots;
      root = counter;
    }
    if (n_roots != 1 || !root)
      return 1;
    auto cst = [](Value v) -> std::optional<int64_t> {
      if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
        return c.value();
      return std::nullopt;
    };
    auto lb = cst(root.getLowerBound()), ub = cst(root.getUpperBound()),
         st = cst(root.getStep());
    if (!lb || !ub || !st || *st <= 0 || *ub <= *lb)
      return 1;
    return llvm::divideCeil(*ub - *lb, *st);
  }

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
  // Measures the CURRENT configuration through the real downstream pipeline
  // (orchestrate + the interval analysis). Injected rather than called
  // directly, because it lives on the pass and this is a separate class; the
  // balancer must not grow a dependency on the pass to reach it.
  using RealIntervalFn = std::function<int64_t()>;
  RealIntervalFn real_interval_fn;
  // Profiles a task the EXPENSIVE way: the real mapper, or the exact solver
  // when one is configured. `profile_fn` obeys `estimation-mode`, so under
  // `cost-model-analytical` it never maps anything -- which is right for the
  // search, and wrong for the shortlist.
  //
  // Without this the design did not do what it claims. The pool was ranked in
  // closed form (correct, and the point), but the top k were then "verified"
  // with the same analytical II the ranking already used, so the verification
  // could only re-order candidates the estimate had already ordered. The
  // solver ran once, after the search, on whatever the search had settled on.
  // "Rank cheaply, lower the top k for real" needs the second half.
  ProfileFn verify_profile_fn;
  // Publishes the current configuration onto the IR so `real_interval_fn` sees
  // it. Same reason for the indirection.
  using PublishFn = std::function<void(TaskDependencyGraph &)>;
  PublishFn publish_fn;

  // Enables the two data-level-parallelism moves on top of the classic
  // add-a-CGRA move. Both are gated on the task's `dlp_replicable` tag.
  bool enable_replicas = false;
  bool enable_tiling = false;
  int max_tiling = 8;
  // When set, the allocation may exceed the grid; the extra area is paid for
  // with context waves inside the objective instead of being forbidden.
  bool allow_temporal = false;
  // Shortlist depth for the per-move ranking; see the pass option of the same
  // name, which governs every shortlist in this pass.
  int verify_top_k = 4;

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
  // NOTE on the greedy's quality. Hill-climbing one bottleneck at a time is a
  // weak search over this space, and offering it three moves makes it weaker,
  // not stronger: on cnn, with tiling available it takes a cheap DLP step at
  // cgra_count=1 and settles at (c=1,k=2,t=8)=422 using 2 of 16 cells, while
  // without tiling it is forced to grow the array first (II 21 -> 6) and
  // reaches 358. A two-phase variant (saturate LargerTileArray, then
  // parallelise) was tried and merely moves the trap: it over-commits area to
  // cgra_count and lands on 434 / gemm 158. The exact allocator finds 358 / 131
  // and never tiles when tiling does not pay, so the fix belongs in the search,
  // not in the move set -- which is exactly the "rank cheaply, then verify
  // exactly" split this cost model is for.

  // Which quantity a candidate move has to improve.
  //
  //   false (shipped)  the BOTTLENECK TASK's own latency. This is AMOEBA's
  //                    Alg. 1: walk the tasks by criticality and grow each one
  //                    until it stops getting faster.
  //   true  (joint)    the whole graph's objective.
  //
  // They are not the same rule, and the difference is the point of this work.
  // Once the top tasks saturate, `findBottleneck` hands back a task that is not
  // at the maximum at all -- multi-nested reaches Task_4 at latency 74 while
  // the interval is 196 -- and the shipped rule spends a CGRA on it because ITS
  // latency drops, even though the interval cannot move. The joint rule
  // declines.
  //
  // Default false so the shipped path is bit-identical; the pass options that
  // opt into the joint search turn it on.
  bool joint_criterion = false;
  // Rounds of the joint search. Each round re-centres the per-axis option lists
  // on the winner, which is what stops "options built with the other axes held
  // at their current values" from being a one-shot approximation.
  int joint_rounds = 0;
  // Depth of the joint shortlist. Separate from `verify_top_k`, which caps the
  // balancer's per-step MOVE list: reusing that one silently made the joint
  // search verify the whole pool, because its default is 0 meaning "all".
  int joint_top_k = 4;


  bool balance(TaskDependencyGraph &graph, const ProfileFn &profile_fn) {
    // Warm-start the joint climb from the shipped allocation.
    //
    // Both rules are hill-climbs, and they climb different surfaces, so the
    // joint one can settle in a worse basin than the per-task one even though
    // its objective is the one being measured: on the 28-task transformer it
    // reached a point its own model scored 10.1M where the shipped rule reached
    // 6.3M. Running the cheap per-task rule first costs one pass over the tasks
    // and makes the joint arm a refinement of the baseline rather than an
    // independent search that has to rediscover it.
    if (joint_criterion) {
      // Kept only if it helps. The warm start optimises the per-task rule,
      // which is a different surface, and the outer loop calls balance() again
      // on an already-allocated graph -- there it can push cgra_count up in
      // ways the joint objective does not want, and the climb cannot undo a
      // move because reverting is not in the move set. On bicg that walked an
      // allocation the model scored 9730 to one it scored 13950.
      SmallVector<std::tuple<int, int, int>> before;
      SmallVector<std::pair<int64_t, int64_t>> before_profile;
      for (auto &n : graph.nodes) {
        before.emplace_back(n->cgra_count, n->replicas, n->tiling);
        before_profile.emplace_back(n->ii, n->steps);
      }
      const int64_t before_obj = graph.objective();
      joint_criterion = false;
      balanceImpl(graph, profile_fn);
      joint_criterion = true;
      if (graph.objective() > before_obj) {
        for (auto [i, n] : llvm::enumerate(graph.nodes)) {
          std::tie(n->cgra_count, n->replicas, n->tiling) = before[i];
          n->ii = before_profile[i].first;
          n->steps = before_profile[i].second;
        }
      }
    }
    // The joint search replaces the one-axis climb rather than following it:
    // running both would let the greedy rule commit a move the joint pool was
    // meant to weigh against its alternatives.
    if (joint_rounds > 0)
      return jointBalanceImpl(graph, profile_fn);
    return balanceImpl(graph, profile_fn);
  }

  //===--------------------------------------------------------------------===//
  // Joint search: one score over the whole configuration, global top-k.
  //===--------------------------------------------------------------------===//
  //
  // `balanceImpl` below is a hill-climb. It picks a bottleneck and moves ONE
  // axis -- `cgra_count`, or `replicas`, or `tiling` -- keeping whichever move
  // scored best. Shape is chosen separately inside `profileWithBestShape`, and
  // fusion separately again in `speculate`. Three shortlists on three surfaces,
  // advanced one at a time.
  //
  // That cannot reach a configuration whose axes must move TOGETHER. If
  // widening the array is a loss at the current tiling, and doubling the tiling
  // is a loss at the current width, a one-axis climb rejects both and never
  // tries the pair -- even when the pair is the best point in the space.
  //
  // So the candidate here is a whole configuration, the pool is the cross
  // product of the per-axis options, and every member is scored by the SAME
  // closed form the objective already defines:
  //
  //   score = max(task_floor, resource_floor, pipelineCycle(), commFloor())
  //
  // Ranking is free -- no candidate is compiled to be scored. Only the top k
  // are lowered for real, and the real result picks the winner. Rounds repeat
  // with the winner as the new centre, which is what makes the per-axis option
  // lists (built with the other axes held at their current values) progressively
  // less of an approximation.
  //
  // What "for real" covers, precisely: the top-k are re-measured through
  // `realPipelineIntervalOf`, which clones the module and runs the actual
  // `orchestrate-tasks-on-accelerators` and `analyze-task-pipeline-interval`.
  // So placement, temporal context assignment and the reported interval are the
  // real ones. Tiling is the exception -- materialising it erases task ops, so
  // it cannot be applied and rolled back per candidate, and on that one axis
  // the verification still reads the attribute rather than the rewritten loop.
  struct JointConfig {
    SmallVector<int> cgra_count, replicas, tiling;
    SmallVector<CgraShape> shape;
    SmallVector<int64_t> ii, steps;
  };

  JointConfig captureConfig(const TaskDependencyGraph &graph) const {
    JointConfig config;
    for (auto &n : graph.nodes) {
      config.cgra_count.push_back(n->cgra_count);
      config.replicas.push_back(n->replicas);
      config.tiling.push_back(n->tiling);
      config.shape.push_back(n->shape);
      config.ii.push_back(n->ii);
      config.steps.push_back(n->steps);
    }
    return config;
  }

  void restoreConfig(TaskDependencyGraph &graph, const JointConfig &config) {
    for (auto [i, n] : llvm::enumerate(graph.nodes)) {
      n->cgra_count = config.cgra_count[i];
      n->replicas = config.replicas[i];
      n->tiling = config.tiling[i];
      n->shape = config.shape[i];
      n->ii = config.ii[i];
      n->steps = config.steps[i];
    }
  }

  // The options this node may take on each axis, including its current value so
  // "leave it alone" is always in the product.
  struct AxisOptions {
    SmallVector<int> cgra_count, replicas, tiling;
  };

  AxisOptions axisOptionsFor(TaskGraphNode *node, int budget_left) const {
    AxisOptions options;
    options.cgra_count.push_back(node->cgra_count);
    for (int next : {node->cgra_count + 1, node->cgra_count * 2}) {
      if (next != node->cgra_count && canFitOnGrid(next) &&
          next <= node->trip_count &&
          next * std::max(1, node->replicas) * std::max(1, node->tiling) <=
              node->cgra_count + budget_left)
        options.cgra_count.push_back(next);
    }
    options.replicas.push_back(node->replicas);
    if (node->dlp_replicable &&
        TaskGraphNode::replicasFitGrid(node->cgra_count, node->replicas + 1) &&
        (int64_t)(node->replicas + 1) * std::max(1, node->tiling) <=
            node->root_trip &&
        TaskTiler::canReplicate(node->op, node->replicas + 1))
      options.replicas.push_back(node->replicas + 1);
    options.tiling.push_back(node->tiling);
    // `canTile` asks whether the cut is LEGAL; `tilingIsParallel` asks whether
    // the pieces may overlap in time. The pool needs both, and asking only the
    // first is what let it price an ordered cut at latency/N.
    //
    // On bicg no dimension indexes every store, so an 8-way cut is legal and
    // strictly sequential. The pool offered it, the closed form scored it at
    // trip/8, and the shortlist measured 8195 -> 4099 -> 2051 -> 1027 -> 515
    // through the pre-materialisation model. Materialising it produced eight
    // serialised tasks and an interval of 8216 -- worse than the 8195 it
    // started from, and 16x what the search believed. An ordered cut costs the
    // same latency on more area, so it is never the right answer and the pool
    // should not carry it.
    if (node->tile_group < 0 && node->tiling * 2 <= max_tiling &&
        (int64_t)std::max(1, node->replicas) * (node->tiling * 2) <=
            node->split_space &&
        TaskTiler::canTile(node->op, node->tiling * 2) &&
        TaskTiler::tilingIsParallel(node->op, node->tiling * 2))
      options.tiling.push_back(node->tiling * 2);
    return options;
  }

  bool jointBalanceImpl(TaskDependencyGraph &graph,
                        const ProfileFn &profile_fn) {
    bool changed = false;
    const int grid_budget =
        allow_temporal ? kTotalCGRAs * kMaxTemporalWaves : kTotalCGRAs;

    if (!real_interval_fn || !publish_fn) {
      // A `[Joint]` prefix here reads as "the joint search ran", which is the
      // opposite of what happened. This is the speculative path, where a
      // downstream run per candidate would cost more than the lookahead it
      // informs, so it deliberately uses the cheap climb.
      llvm::errs() << "[Balance] speculative call: one-axis climb (the joint "
                   << "search needs the real-measurement callback)\n";
      return balanceImpl(graph, profile_fn);
    }
    // Falls back to the search's own profiler when no verifier is wired, so a
    // configuration without one behaves exactly as before rather than silently
    // losing the shortlist step.
    const ProfileFn &verify =
        verify_profile_fn ? verify_profile_fn : profile_fn;
    for (int round = 0; round < joint_rounds; ++round) {
      DenseSet<TaskGraphNode *> none;
      TaskGraphNode *bottleneck = findBottleneck(graph, none);
      if (!bottleneck)
        break;
      // The centre is measured with the SAME profiler as the candidates.
      //
      // Verifying the candidates while leaving the centre on its analytical II
      // would decide the round on which yardstick was applied, not on which
      // configuration is better: the analytical II is a floor, so an
      // unverified centre reads as faster than it is and no candidate can ever
      // beat it. Verify first, then snapshot, so `restoreConfig` puts the
      // verified value back and `centre_measured` below is comparable.
      verify(bottleneck, bottleneck->op);
      const JointConfig centre = captureConfig(graph);
      const int64_t centre_score = graph.objective();
      const int budget_left = grid_budget - graph.getTotalAllocatedCGRAs() +
                              bottleneck->footprintCgras();
      AxisOptions axes = axisOptionsFor(bottleneck, budget_left);

      // The pool. Every member differs from the centre on any subset of the
      // axes, which is the whole point -- a one-axis climb only ever sees the
      // members that differ on exactly one.
      struct Scored {
        int cgra_count, replicas, tiling;
        CgraShape shape;
        int64_t score;
      };
      // NOTHING in this loop compiles anything. That is the claim the design
      // rests on: the pool is ranked in closed form, and a lowering is spent
      // only on the k that survive.
      //
      // One profile of the centre supplies the two facts no candidate changes
      // -- the body's op count and its recurrence bound -- and every member's
      // II then follows from the tile count of its own shape:
      //
      //   ii(shape) = max( ceil(n_ops / tiles(shape)), rec_mii )
      //
      // An earlier version of this loop called `profile_fn` whenever the tile
      // array moved, which ran InsertDataMov and the mapper per candidate. That
      // is the cost the ranking exists to avoid; scoring the pool that way
      // makes the shortlist pointless.
      const int64_t body_ops = bottleneck->n_ops;
      const int64_t body_recurrence = bottleneck->rec_mii;
      const int64_t body_steps = bottleneck->steps;
      SmallVector<Scored> pool;
      for (int cgras : axes.cgra_count) {
        SmallVector<CgraShape> shapes = getRectangularShapes(cgras);
        for (const auto &s : getNonRectangularShapes(cgras))
          shapes.push_back(s);
        if (shapes.empty())
          shapes.push_back(pickBestShape(cgras));
        SmallVector<CgraShape> unique;
        for (const CgraShape &shape : shapes) {
          bool seen = false;
          for (const CgraShape &kept : unique)
            seen |= kept.irAttr() == shape.irAttr();
          if (!seen)
            unique.push_back(shape);
        }
        for (const CgraShape &shape : unique) {
          const int64_t tiles = (int64_t)shape.rows * graph.per_cgra_rows *
                                (int64_t)shape.cols * graph.per_cgra_cols;
          const int64_t closed_form_ii =
              body_ops > 0
                  ? std::max<int64_t>(
                        llvm::divideCeil(body_ops, std::max<int64_t>(1, tiles)),
                        body_recurrence)
                  : centre.ii[bottleneck->id];
          for (int replicas : axes.replicas) {
            for (int tiling : axes.tiling) {
              if (cgras * std::max(1, replicas) * std::max(1, tiling) >
                  bottleneck->cgra_count + budget_left)
                continue;
              // The cross product pairs a cgra_count with a replica count the
              // axis enumerator cleared against a DIFFERENT cgra_count, so the
              // spatial check belongs here as well as there.
              if (!TaskGraphNode::replicasFitGrid(cgras, replicas))
                continue;
              bottleneck->cgra_count = cgras;
              bottleneck->shape = shape;
              bottleneck->replicas = replicas;
              bottleneck->tiling = tiling;
              bottleneck->ii = closed_form_ii;
              bottleneck->steps = body_steps;
              pool.push_back({cgras, replicas, tiling, shape,
                              graph.objective()});
              restoreConfig(graph, centre);
            }
          }
        }
      }
      if (pool.empty())
        break;

      llvm::sort(pool, [](const Scored &a, const Scored &b) {
        return a.score < b.score;
      });

      // Drop members that ARE another member. `getRectangularShapes` and
      // `getNonRectangularShapes` both offer a 1x2 for two CGRAs, and the
      // per-cgra_count dedup above compares only shapes, so the same
      // (cgra_count, shape, replicas, tiling) reaches the pool more than once.
      //
      // It cost real verifications. On gemver round 0 the shortlist was
      //
      //   cand 0 cgras=2 shape=2x1 replicas=2 tiling=2 scored=9229 measured=4100
      //   cand 1 cgras=1 shape=1x1 replicas=2 tiling=2 scored=9229 measured=4100
      //   cand 2 cgras=2 shape=1x2 replicas=2 tiling=2 scored=9229 measured=4100
      //   cand 3 cgras=2 shape=1x2 replicas=2 tiling=2 scored=9229 measured=4100
      //
      // -- four lowerings for three configurations, because cand 2 and cand 3
      // are the same one. k is the budget the whole design rests on, so
      // spending a quarter of it re-measuring a config already measured is not
      // a cosmetic defect: it silently makes the shortlist shallower than the
      // flag says it is.
      //
      // Only exact repeats go. Ties on score are left alone -- two different
      // configurations that score the same are exactly what the verification
      // step exists to separate, and dropping one of them because the ranking
      // could not tell them apart would put the greedy choice back.
      SmallVector<Scored> distinct;
      unsigned dropped = 0;
      for (const Scored &candidate : pool) {
        bool repeat = false;
        for (const Scored &kept : distinct)
          repeat |= kept.cgra_count == candidate.cgra_count &&
                    kept.replicas == candidate.replicas &&
                    kept.tiling == candidate.tiling &&
                    kept.shape.irAttr() == candidate.shape.irAttr();
        if (repeat)
          ++dropped;
        else
          distinct.push_back(candidate);
      }
      if (dropped > 0) {
        llvm::errs() << "[Joint]   pool: dropped " << dropped
                     << " duplicate configurations, " << distinct.size()
                     << " distinct\n";
        pool = std::move(distinct);
      }

      const unsigned keep = std::min<unsigned>(
          pool.size(), joint_top_k > 0 ? joint_top_k : pool.size());

      llvm::errs() << "[Joint] round " << round << ": task="
                   << bottleneck->op.getTaskName() << " pool=" << pool.size()
                   << " centre=" << centre_score << " best_scored="
                   << pool.front().score << " verifying top " << keep << "\n";

      // Cash the shortlist in: lower each of the k and let the measurement,
      // not the score that produced the ranking, choose. Selecting on the score
      // here would make the verification unable to change any outcome.
      int64_t best_measured = INT64_MAX;
      const Scored *winner = nullptr;
      for (unsigned i = 0; i < keep; ++i) {
        const Scored &candidate = pool[i];
        bottleneck->cgra_count = candidate.cgra_count;
        bottleneck->shape = candidate.shape;
        bottleneck->replicas = candidate.replicas;
        bottleneck->tiling = candidate.tiling;
        verify(bottleneck, bottleneck->op);
        publish_fn(graph);
        const int64_t measured = real_interval_fn();
        llvm::errs() << "[Joint]   cand " << i << " cgras="
                     << candidate.cgra_count << " shape="
                     << candidate.shape.describe(candidate.cgra_count)
                     << " replicas=" << candidate.replicas << " tiling="
                     << candidate.tiling << " scored=" << candidate.score
                     << " measured=" << measured << "\n";
        if (measured > 0 && measured < best_measured) {
          best_measured = measured;
          winner = &candidate;
        }
        restoreConfig(graph, centre);
      }

      if (!winner) {
        llvm::errs() << "[Joint]   no candidate measured; stopping\n";
        break;
      }
      // Adopt only on a real improvement, so a round that finds nothing leaves
      // the allocation exactly as it was.
      restoreConfig(graph, centre);
      publish_fn(graph);
      const int64_t centre_measured = real_interval_fn();
      if (centre_measured > 0 && best_measured >= centre_measured) {
        llvm::errs() << "[Joint]   centre measured " << centre_measured
                     << " <= best candidate " << best_measured
                     << "; converged\n";
        break;
      }
      bottleneck->cgra_count = winner->cgra_count;
      bottleneck->shape = winner->shape;
      bottleneck->replicas = winner->replicas;
      bottleneck->tiling = winner->tiling;
      verify(bottleneck, bottleneck->op);
      changed = true;
      llvm::errs() << "[Joint]   adopted, measured " << best_measured
                   << " (was " << centre_measured << ")\n";
    }
    return changed;
  }

  bool balanceImpl(TaskDependencyGraph &graph, const ProfileFn &profile_fn) {
    bool changed = false;
    // Tracks nodes for which adding one more CGRA did not reduce latency.
    // These are skipped in subsequent iterations.
    llvm::DenseSet<TaskGraphNode *> saturated_nodes;

    for (int iter = 0; iter < kMaxBalanceIterations; ++iter) {
      int total_cgras = graph.getTotalAllocatedCGRAs();
      if (!allow_temporal && total_cgras >= kTotalCGRAs) {
        break;
      }
      if (allow_temporal && total_cgras >= kTotalCGRAs * kMaxTemporalWaves) {
        break;
      }

      // Recomputes critical path each iteration (may shift after rebalance).
      TaskGraphNode *bottleneck = findBottleneck(graph, saturated_nodes);
      if (!bottleneck) {
        break;
      }

      // Every task tied with the bottleneck has to move together. The objective
      // is a MAX over tasks (and, in pipeline mode, over grid cells), so
      // improving one member of a tied set changes nothing and the move gets
      // rejected -- the balancer then declares the task saturated and stalls.
      // That is fatal for exactly the case loop partitioning creates: N tiles
      // cut from one loop are identical by construction, so every one of them
      // sits at the maximum and no single-task step can ever lower it.
      SmallVector<TaskGraphNode *, 8> peers;
      if (joint_criterion) {
        const int64_t top = bottleneck->estimatedLatency();
        for (auto &n : graph.nodes)
          if (!saturated_nodes.count(n.get()) &&
              n->cgra_count < kMaxCgrasPerTask && n->estimatedLatency() == top)
            peers.push_back(n.get());
      }
      if (peers.empty())
        peers.push_back(bottleneck);

      // The quantity a move has to improve; see `joint_criterion`.
      auto score = [&]() -> int64_t {
        return joint_criterion ? graph.objective()
                               : bottleneck->estimatedLatency();
      };

      // Saves state for potential rollback.
      const int old_cgra_count = bottleneck->cgra_count;
      const int old_replicas = bottleneck->replicas;
      const int old_tiling = bottleneck->tiling;
      const int64_t old_latency = score();
      const int grid_budget =
          allow_temporal ? kTotalCGRAs * kMaxTemporalWaves : kTotalCGRAs;
      const int budget_left =
          grid_budget - total_cgras + bottleneck->footprintCgras();

      // Builds the candidate moves for this bottleneck (AMOEBA Alg. 1 lines
      // 6-7, plus loop partitioning):
      //   LargerTileArray : cgra_count + 1     -- lowers II (ResMII =
      //   ops/tiles) MoreReplicas    : replicas   + 1     -- divides the
      //   iteration space Tiling          : tiling     * 2     -- divides the
      //   iteration space
      //                                           but scales the per-iter work
      // The last two need a partitionable counter dimension (`dlp_replicable`)
      // and cannot divide the iteration space below one iteration.
      enum MoveKind { kLargerTileArray, kMoreReplicas, kTiling };
      struct Move {
        MoveKind kind;
        int cgra_count, replicas, tiling;
        const char *name;
      };
      SmallVector<Move, 3> candidates;
      if (canFitOnGrid(old_cgra_count + 1) &&
          (old_cgra_count + 1) * old_replicas * std::max(1, old_tiling) <=
              budget_left) {
        candidates.push_back({kLargerTileArray, old_cgra_count + 1,
                              old_replicas, old_tiling, "LargerTileArray"});
      }
      // Same rule as tiling: only propose a replica count whose partition table
      // the compiler can actually emit.
      if (enable_replicas && bottleneck->dlp_replicable &&
          TaskGraphNode::replicasFitGrid(old_cgra_count, old_replicas + 1) &&
          old_cgra_count * (old_replicas + 1) * std::max(1, old_tiling) <=
              budget_left &&
          bottleneck->effectiveTripCount() > 1 &&
          (int64_t)(old_replicas + 1) * old_tiling <= bottleneck->root_trip &&
          TaskTiler::canReplicate(bottleneck->op, old_replicas + 1)) {
        candidates.push_back({kMoreReplicas, old_cgra_count, old_replicas + 1,
                              old_tiling, "MoreReplicas"});
      }
      // The tiling factor is only proposed when the cut can actually be
      // MATERIALISED: the task needs a single constant-bound root counter with
      // at least `factor` iterations and a chainable yield. Searching a factor
      // the IR rewrite would later refuse is how a knob ends up decided but not
      // applied.
      // No dlp_replicable gate on tiling: splitting a loop is sound for any
      // loop, and TaskTiler::canTile already checks that the cut is realisable.
      // Whether the pieces may overlap is recorded separately as tile_parallel.
      if (enable_tiling && bottleneck->tile_group < 0 &&
          old_tiling * 2 <= max_tiling &&
          old_cgra_count * std::max(1, old_replicas) * (old_tiling * 2) <=
              budget_left &&
          (int64_t)old_replicas * (old_tiling * 2) <= bottleneck->split_space &&
          TaskTiler::canTile(bottleneck->op, old_tiling * 2)) {
        candidates.push_back(
            {kTiling, old_cgra_count, old_replicas, old_tiling * 2, "Tiling"});
      }
      if (candidates.empty()) {
        saturated_nodes.insert(bottleneck);
        continue;
      }

      // Speculatively evaluates every candidate and keeps the best. Only the
      // LargerTileArray move changes the tile array, so only it needs a
      // re-profile; replicas/tiling are analytic on top of the same profile.
      int64_t best_latency = old_latency;
      const Move *best_move = nullptr;
      // Snapshot the peers so a candidate can be applied to all of them.
      struct PeerState {
        TaskGraphNode *node;
        int cgra_count, replicas, tiling;
        int64_t ii, steps;
        CgraShape shape;
      };
      SmallVector<PeerState, 8> peer_state;
      for (TaskGraphNode *peer : peers)
        peer_state.push_back({peer, peer->cgra_count, peer->replicas,
                              peer->tiling, peer->ii, peer->steps,
                              peer->shape});
      auto restorePeers = [&]() {
        for (PeerState &saved : peer_state) {
          saved.node->cgra_count = saved.cgra_count;
          saved.node->replicas = saved.replicas;
          saved.node->tiling = saved.tiling;
          saved.node->ii = saved.ii;
          saved.node->steps = saved.steps;
          saved.node->shape = saved.shape;
        }
      };
      // Applies move `m` (expressed as a delta on the bottleneck) to every peer
      // whose own limits allow it. Returns the CGRAs the set would then hold.
      auto applyToPeers = [&](const Move &move, bool group) {
        for (PeerState &saved : peer_state) {
          TaskGraphNode *peer = saved.node;
          // Single-task step unless the group step is being tried.
          if (!group && peer != bottleneck)
            continue;
          switch (move.kind) {
          case kLargerTileArray:
            // `findBottleneck` never returns a task already holding as many
            // CGRAs as it has iterations, because the extra tiles cannot be
            // filled. The bottleneck therefore satisfies this by construction,
            // but a peer tied with it need not, and without the guard the group
            // step spends budget on tiles that task cannot use.
            if (!canFitOnGrid(saved.cgra_count + 1) ||
                saved.cgra_count + 1 > peer->trip_count)
              continue;
            peer->cgra_count = saved.cgra_count + 1;
            profile_fn(peer, peer->op);
            break;
          case kMoreReplicas:
            if (!peer->dlp_replicable ||
                (int64_t)(saved.replicas + 1) * saved.tiling >
                    peer->root_trip ||
                !TaskTiler::canReplicate(peer->op, saved.replicas + 1))
              continue;
            peer->replicas = saved.replicas + 1;
            break;
          case kTiling:
            if (peer->tile_group >= 0 || saved.tiling * 2 > max_tiling ||
                (int64_t)saved.replicas * (saved.tiling * 2) >
                    peer->split_space ||
                !TaskTiler::canTile(peer->op, saved.tiling * 2))
              continue;
            peer->tiling = saved.tiling * 2;
            break;
          }
        }
        return graph.getTotalAllocatedCGRAs();
      };

      // Each move is tried twice: on the bottleneck alone, and on every task
      // tied with it.
      //
      // The group step exists because the objective is a max, so when N tasks
      // sit at the maximum -- which is exactly what an N-way loop cut produces,
      // since the pieces are identical by construction -- no single-task step
      // can lower it and the balancer declares the task saturated. But applying
      // the step to everything tied also overshoots: it buys area for tasks
      // that did not need it, and on bicg, ffn and slam_jtj that lands worse
      // than the one-at-a-time walk. Trying both and keeping the better one
      // dominates either rule alone.
      bool best_group = false;

      // Rank the legal moves cheaply, then score the top few for real.
      //
      // `score()` is `graph.objective()`, which in pipeline mode list-schedules
      // the whole program onto the grid and walks each cell's longest path.
      // The balancer called it once per legal move per iteration -- 477 times
      // on bicg -- and the ranking below replaces most of those with two O(V)
      // sums. The candidate set here is small (three moves, each on the
      // bottleneck alone or on its whole tied set), so this saves a fraction
      // rather than an order of magnitude; it is the same discipline the
      // fusion, allocation and shape shortlists use, applied where it also
      // happens to be the largest remaining cost.
      auto cheapMoveScore = [&]() -> int64_t {
        int64_t task_floor = 0, area_time = 0;
        for (auto &node : graph.nodes) {
          const int64_t latency = node->estimatedLatency();
          task_floor = std::max(task_floor, latency);
          area_time += latency * node->footprintCgras();
        }
        return std::max<int64_t>(
            task_floor, llvm::divideCeil(area_time, (int64_t)kTotalCGRAs));
      };

      struct RankedMove {
        int64_t cheap_score;
        const Move *move;
        bool group;
      };
      SmallVector<RankedMove, 6> ranked_moves;
      for (const Move &move : candidates) {
        for (int group = 0; group < (peers.size() > 1 ? 2 : 1); ++group) {
          restorePeers();
          int demand = applyToPeers(move, group == 1);
          if (demand <= grid_budget)
            ranked_moves.push_back({cheapMoveScore(), &move, group == 1});
          restorePeers();
        }
      }
      // Stable, so equal scores keep the enumeration order. The acceptance
      // test below is strict, which means ties go to whichever move is seen
      // first, and an unstable sort silently re-picks among them: on plain_gemm
      // that alone moved the result from 262147 to 393218, with every move
      // still verified. A shortlist may drop candidates; it must not reorder
      // the ones it keeps.
      llvm::stable_sort(ranked_moves,
                        [](const RankedMove &lhs, const RankedMove &rhs) {
                          return lhs.cheap_score < rhs.cheap_score;
                        });
      const size_t moves_to_verify =
          verify_top_k <= 0
              ? ranked_moves.size()
              : std::min<size_t>(verify_top_k, ranked_moves.size());

      for (size_t position = 0; position < moves_to_verify; ++position) {
        const RankedMove &candidate = ranked_moves[position];
        restorePeers();
        applyToPeers(*candidate.move, candidate.group);
        int64_t cand_latency = score();
        llvm::errs() << "  Balance: trying Task " << bottleneck->id << " ("
                     << bottleneck->op.getTaskName().str() << ") "
                     << candidate.move->name
                     << (candidate.group ? " [group]" : " [single]")
                     << " -> cgra_count=" << candidate.move->cgra_count
                     << ", replicas=" << candidate.move->replicas
                     << ", tiling=" << candidate.move->tiling
                     << ", ii=" << bottleneck->effectiveII()
                     << ", trip=" << bottleneck->effectiveTripCount()
                     << ", lat=" << old_latency << "->" << cand_latency << "\n";
        if (cand_latency < best_latency) {
          best_latency = cand_latency;
          best_move = candidate.move;
          best_group = candidate.group;
        }
        restorePeers();
      }

      if (best_move) {
        // Accepted: commit the best-improving move across the whole tied set.
        changed = true;
        applyToPeers(*best_move, best_group);
        llvm::errs() << "  Balance: ACCEPTED Task " << bottleneck->id << " ("
                     << bottleneck->op.getTaskName().str() << ") "
                     << best_move->name
                     << " cgra_count=" << bottleneck->cgra_count
                     << ", replicas=" << bottleneck->replicas
                     << ", tiling=" << bottleneck->tiling
                     << ", lat=" << old_latency << "->" << best_latency
                     << ", total_cgras=" << graph.getTotalAllocatedCGRAs()
                     << "\n";
      } else {
        // Rejected: no move improved latency — mark saturated.
        llvm::errs() << "  Balance: REJECTED Task " << bottleneck->id
                     << " (no move beats lat=" << old_latency
                     << "). Saturating.\n";
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
            const ProfileFn &profile_fn) {
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

  // Public: returns true if tasks a and b can be legally fused (independent,
  // single-block bodies, dominance-safe). Used by the global fusion search to
  // enumerate candidate pairs.
  bool canFuse(TaskGraphNode *a, TaskGraphNode *b, TaskDependencyGraph &graph) {
    if (!graph.areIndependent(a, b))
      return false;
    if (!a->op.getBody().hasOneBlock() || !b->op.getBody().hasOneBlock())
      return false;
    return canSafelyFuse(a, b, graph);
  }

  // Public: fuse two SPECIFIC nodes (bypasses the trip-diff selection). Used by
  // the global fusion search to commit / speculate a chosen pair.
  bool fuseNodes(func::FuncOp func, TaskGraphNode *a, TaskGraphNode *b,
                 TaskDependencyGraph &graph, const ProfileFn &profile_fn) {
    return performFusion(func, a, b, graph, profile_fn);
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

    // IR-level dominance soundness. performFusion inserts the fused task right
    // after the LATEST-defined operand of either task, then replaces every use
    // of both tasks' results with the fused task's results. That rewrite is
    // legal only if every EXISTING use of either task's results is positioned
    // strictly after that insertion point. The TaskDependencyGraph only tracks
    // dependencies whose memref is directly a task result; memref dependencies
    // threaded through memref.alloc + memref.copy (or bufferization.to_tensor)
    // are INVISIBLE to it, so areIndependent can wrongly report an early
    // producer (e.g. a bias-init task whose result is copied downstream) as
    // independent of a late task. Fusing those places the fused op after an
    // early consumer of the producer's result -> "operand does not dominate
    // this use". Detect and reject such pairs here using the real SSA uses,
    // which remain visible even when the graph edge is missing.
    Operation *latest_def = task_a; // task_a is the earlier op (ensured above).
    auto updateLatest = [&](ValueRange operands) {
      for (Value operand : operands)
        if (Operation *def_op = operand.getDefiningOp())
          if (def_op->getBlock() == task_a->getBlock() &&
              latest_def->isBeforeInBlock(def_op))
            latest_def = def_op;
    };
    auto task_a_op = cast<TaskflowTaskOp>(task_a);
    auto task_b_op = cast<TaskflowTaskOp>(task_b);
    updateLatest(task_a_op.getWillReads());
    updateLatest(task_a_op.getWillWrites());
    updateLatest(task_a_op.getValueInputs());
    updateLatest(task_b_op.getWillReads());
    updateLatest(task_b_op.getWillWrites());
    updateLatest(task_b_op.getValueInputs());

    Block *task_block = task_a->getBlock();
    auto allUsesAfterInsertion = [&](Operation *task) -> bool {
      for (Value result : task->getResults()) {
        for (OpOperand &use : result.getUses()) {
          // Walk up to the ancestor op that lives in the tasks' block.
          Operation *owner = use.getOwner();
          while (owner && owner->getBlock() != task_block)
            owner = owner->getParentOp();
          if (!owner || owner == task_a || owner == task_b)
            continue;
          // The use must be strictly after the fused op's insertion point.
          if (!latest_def->isBeforeInBlock(owner))
            return false;
        }
      }
      return true;
    };
    if (!allUsesAfterInsertion(task_a) || !allUsesAfterInsertion(task_b))
      return false;

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
                     const ProfileFn &profile_fn) {
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
    updateLatest(task_a.getWillReads());
    updateLatest(task_a.getWillWrites());
    updateLatest(task_a.getValueInputs());
    updateLatest(task_b.getWillReads());
    updateLatest(task_b.getWillWrites());
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

    addUnique(merged_read_memrefs, task_a.getWillReads());
    addUnique(merged_read_memrefs, task_b.getWillReads());
    addUnique(merged_write_memrefs, task_a.getWillWrites());
    addUnique(merged_write_memrefs, task_b.getWillWrites());
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
      unsigned read_count = orig_task.getWillReads().size();
      unsigned write_count = orig_task.getWillWrites().size();

      for (unsigned i = 0; i < read_count; ++i) {
        Value orig_memref = orig_task.getWillReads()[i];
        auto it = llvm::find(merged_read_memrefs, orig_memref);
        assert(it != merged_read_memrefs.end());
        unsigned fused_idx = std::distance(merged_read_memrefs.begin(), it);
        mapping.map(src_entry.getArgument(src_idx + i),
                    fused_region.front().getArgument(fused_idx));
      }
      src_idx += read_count;

      for (unsigned i = 0; i < write_count; ++i) {
        Value orig_memref = orig_task.getWillWrites()[i];
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
      OpBuilder yield_builder = OpBuilder::atBlockEnd(entry_block);
      yield_builder.create<TaskflowYieldOp>(fused_task.getLoc(), yield_reads,
                                            yield_writes, yield_values);
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
      {
        OpBuilder b(fused_task);
        MLIRContext *ctx = fused_task->getContext();
        SmallVector<NamedAttribute, 1> profile_attrs;
        profile_attrs.push_back(
            NamedAttribute(StringAttr::get(ctx, "duration"),
                           b.getI32IntegerAttr(fused_node.steps)));
        fused_task->setAttr("profile_info",
                            DictionaryAttr::get(ctx, profile_attrs));
        fused_task->setAttr("compiled_ii", b.getI64IntegerAttr(fused_node.ii));
      }
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
    for (unsigned i = 0; i < orig_task.getDoneReads().size(); ++i) {
      Value orig_result = orig_task.getDoneReads()[i];
      Value orig_read = orig_task.getWillReads()[i];
      unsigned fused_idx = findOperandIndex(merged_read_memrefs, orig_read);
      orig_result.replaceAllUsesWith(fused_task.getDoneReads()[fused_idx]);
    }
    // Writes outputs: maps by matching the original write memref to its
    // position in the merged write memrefs list.
    for (unsigned i = 0; i < orig_task.getDoneWrites().size(); ++i) {
      Value orig_result = orig_task.getDoneWrites()[i];
      Value orig_write = orig_task.getWillWrites()[i];
      unsigned fused_idx = findOperandIndex(merged_write_memrefs, orig_write);
      orig_result.replaceAllUsesWith(fused_task.getDoneWrites()[fused_idx]);
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
  //   "cost-model-analytical": full parametric cost model (max of ResMII,
  //       RecMII, MemMII, RouteMII, RegMII, IssueMII); no mapper.
  //   "analytical": legacy ResMII/RecMII-only estimate; no mapper.
  Option<std::string> estimationMode{
      *this, "estimation-mode",
      llvm::cl::desc(
          "Profiling estimation mode: 'compiled' (default) runs the full "
          "Neura lowering + mapping pipeline; 'cost-model-analytical' uses the "
          "full parametric cost model (all resource bounds); 'analytical' uses "
          "only ResMII/RecMII. The latter two never invoke the mapper."),
      llvm::cl::init("compiled")};

  // Whether `cost-model-analytical` serves its shape-aware prediction or its
  // sound floor. The floor is a lower bound by construction, so it can only
  // under-predict; the exact CP-SAT mapper puts `plain_gemm` on 4x4 at II=2
  // where the floor says 1 (II=1 schedules but does not route). Opt-in, so a
  // run that needs a bound rather than an estimate still gets one.
  // Which mapper cashes the shortlist in. `heuristic` is the shipped
  // backtracker and stays the default so nothing in tree changes; `cpsat` runs
  // the exact modulo mapper from PR #340, which is the one that certifies an
  // II rather than reporting whatever its search happened to reach.
  Option<std::string> verifier{
      *this, "verifier",
      llvm::cl::desc("Mapper used to verify the shortlist: 'heuristic' "
                     "(default, the shipped backtracker) or 'cpsat' (the exact "
                     "modulo mapper; needs cpsat-python and cpsat-script)."),
      llvm::cl::init("heuristic")};
  Option<std::string> cpsatPython{
      *this, "cpsat-python",
      llvm::cl::desc("Interpreter that has OR-Tools installed, for "
                     "verifier=cpsat."),
      llvm::cl::init("python3")};
  Option<std::string> cpsatScript{
      *this, "cpsat-script",
      llvm::cl::desc("Path to test/cost-model/exact_mapper_cpsat.py."),
      llvm::cl::init("")};
  // `mr` in the solver: bias stage 1 toward a placement stage 2 can route.
  //
  // Without it the two stages disagree, and the disagreement is visible: on the
  // graph-isomorphic shapes 8x2 and 2x8 -- same tile count, same link count,
  // same mean hop, homogeneous FUs -- the solver returns 3 and 2, reproducibly
  // and with no timeouts, purely because row-major tile ids lead stage 1 to
  // different placements. With the bias on, both return 3.
  //
  // It buys consistency, not tightness: that same 2x8 goes from 2 to 3, because
  // stage 1 now spends its budget optimising a routing proxy instead of
  // searching for a routable witness. Both numbers are upper bounds either way
  // -- an II rejected by a routing failure was never proven infeasible. Turn it
  // off to trade that consistency back for speed.
  // Measurement-only: after the decisions are materialised, re-read every
  // emitted task's II with the exact solver. Changes nothing the search did,
  // and exists so two arms that SEARCH differently can still be READ with one
  // instrument. Set it on the baseline to compare like with like.
  Option<std::string> rescoreWith{
      *this, "rescore-with",
      llvm::cl::desc("Re-measure the materialised tasks with 'cpsat' after all "
                     "decisions are made (measurement only, no decision "
                     "changes). Empty (default) leaves the arm's own estimate."),
      llvm::cl::init("")};
  Option<bool> cpsatMinimizeRouting{
      *this, "cpsat-minimize-routing",
      llvm::cl::desc("Bias the solver's placement stage toward routable "
                     "placements (default true). False is faster and lets two "
                     "isomorphic shapes disagree."),
      llvm::cl::init(true)};
  Option<int> cpsatSeconds{
      *this, "cpsat-seconds",
      llvm::cl::desc("CP-SAT budget per II attempt (default 30). An II the "
                     "solver cannot settle in this time is skipped and the "
                     "search continues at II+1."),
      llvm::cl::init(30)};

  Option<bool> usePredictedII{
      *this, "use-predicted-ii",
      llvm::cl::desc(
          "With estimation-mode=cost-model-analytical, use the cost model's "
          "shape-aware predicted_ii (floor + distance-charged routing term) "
          "instead of the sound lower bound final_ii (default: false)."),
      llvm::cl::init(false)};

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

  // Cost-aware fusion guard. When true, UtilizationFusion is DEMAND-DRIVEN:
  // a fusion is attempted only when the CGRA budget is actually under pressure
  // (total allocated CGRAs >= the grid capacity), i.e. when co-locating tasks
  // is required to free tiles for the pipeline bottleneck. The default (false)
  // preserves the legacy behavior of fusing unconditionally whenever any legal
  // independent pair exists — which over-fuses and can raise the pipeline
  // interval when there is spare budget. See findBestFusionCandidate.
  Option<bool> costGuidedFusion{
      *this, "cost-guided-fusion",
      llvm::cl::desc(
          "Only fuse tasks when the CGRA budget is under pressure "
          "(demand-driven), instead of unconditionally (default: false)."),
      llvm::cl::init(false)};

  // Ablation: disable UtilizationFusion entirely (balance-only). Used to
  // isolate the contribution of fusion vs. the pipeline-balance replica
  // allocation.
  Option<bool> disableFusion{
      *this, "disable-fusion",
      llvm::cl::desc(
          "Never fuse tasks (balance-only ablation; default false)."),
      llvm::cl::init(false)};

  // Global (best-first) fusion: instead of committing the first cost-guided
  // fusion (greedy), score EVERY candidate pair by the balanced interval it
  // would yield (analytical lookahead) and commit the globally-best pair, only
  // if it beats not fusing. Still Pareto-safe; replaces the greedy trip-diff
  // ordering with a global ranking of fusion candidates each step.
  Option<bool> globalFusion{
      *this, "global-fusion",
      llvm::cl::desc("Best-first fusion: rank all candidate pairs with the "
                     "cost model, verify the top `fusion-candidates` by "
                     "actually fusing and balancing, and commit the best only "
                     "if it beats not fusing (default false)."),
      llvm::cl::init(false)};

  // One shortlist depth for every place this pass ranks cheaply and verifies
  // for real: fusion pairs and complete allocations. Both enumerate a space too
  // large to evaluate exactly -- O(V^2) pairs, and one allocation per (rule,
  // latency target) -- and both have a closed-form score that orders it.
  Option<int> verifyTopK{
      *this, "verify-top-k",
      llvm::cl::desc("How many of the cheaply-ranked candidates to verify with "
                     "the real objective, for both the fusion-pair and the "
                     "allocation shortlists. 0 verifies every candidate "
                     "(default 4)."),
      llvm::cl::init(4)};

  // Shortlist depth for the two axes that are NOT the fusion/allocation
  // shortlist, kept separate so that `verify-top-k` cannot reach the evaluator.
  //
  // It used to. `configureGraph` and `configureBalancer` both read
  // `verify-top-k`, and both are called from `speculate`, so raising k made
  // fusion's own yardstick longer at the same time as its candidate list.
  // Measured on gpt2_prefill_s128: on one identical graph with the same 15
  // legal pairs, the no-fusion baseline was 354,189,345 at k=1 and 266,797,098
  // at k=8. Top-8 contains top-1, but it was being compared against a 25%
  // stronger baseline, so k=8 rejected the pair k=1 accepted and the two runs
  // diverged from the first iteration. A shortlist can only be sound if the
  // thing that scores it holds still.
  // This is the shortlist the whole approach rests on. The speculative probes
  // call `profileWithBestShape` analytically, where the depth is nearly free,
  // but the converged allocation calls it with the real mapper -- so `k` is
  // literally how many shapes per task get lowered, against the enumeration
  // over every legal shape that the incremental search it replaces would run.
  // Turn the staged one-axis climb into a joint search: build the cross
  // product of the per-axis options, score every member with the SAME
  // objective, and lower only the top `verify-top-k`. 0 rounds keeps the
  // shipped hill-climb, so this is opt-in and the default path is unchanged.
  Option<int> jointRounds{
      *this, "joint-rounds",
      llvm::cl::desc("Rounds of joint candidate search (0 = off, the shipped "
                     "one-axis hill-climb). Each round scores the cross "
                     "product of the axis options and lowers the top k."),
      llvm::cl::init(0)};

  Option<int> shapeTopK{
      *this, "shape-top-k",
      llvm::cl::desc("How many of the ranked shapes to lower for real when "
                     "picking a task's shape. 0 lowers every shape (default "
                     "4)."),
      llvm::cl::init(4)};

  // Score a shortlisted fusion candidate on the graph its decisions produce, by
  // materialising them on the speculative clone before measuring.
  //
  // Off by default, on measurement. It does what it claims -- on
  // transformer_static_affine the objective's error against the analyzer falls
  // from 23.3% to 0.0% -- and the measured interval still got 43.3% worse,
  // because fusion is one of four axes and the other three keep scoring the
  // collapsed graph. Fusion then optimises a different quantity from the
  // balancer that runs after it, and the search walks somewhere the objective
  // correctly reports as worse. Converting one axis at a time makes the
  // objective inconsistent with itself; the fix belongs at the end of the
  // search, where one evaluator can judge whole configurations.
  Option<bool> verifyOnMaterialised{
      *this, "verify-on-materialised",
      llvm::cl::desc("Materialise a shortlisted fusion candidate's tiling and "
                     "replication before scoring it (default false -- see the "
                     "comment; this is inconsistent with the other axes)."),
      llvm::cl::init(false)};

  // What the downstream orchestration will be run with. The allocator cannot
  // read the orchestrator's own command line, so verifying against the real
  // pipeline requires being told which pipeline that is; a mismatch here would
  // verify against a program nobody runs.
  Option<std::string> verifyPipelineOptions{
      *this, "verify-pipeline-options",
      llvm::cl::desc("Options passed to orchestrate-tasks-on-accelerators when "
                     "a candidate is verified against the real pipeline."),
      llvm::cl::init("scheduling-mode=spatial-temporal")};

  // Diagnostic: after the decisions are applied, run the real pipeline on a
  // clone and report it next to the model. Validates that the in-pass pipeline
  // reproduces what the command line produces before anything is scored by it.
  Option<bool> reportRealInterval{
      *this, "report-real-interval",
      llvm::cl::desc("Run the downstream pipeline on a clone after applying "
                     "decisions and report the analyzer's interval (default "
                     "false)."),
      llvm::cl::init(false)};

  // The guard that decides whether a "real lowering" is real. At the shipped
  // 150 it fires on every task of every large program here, so a request for
  // the mapper is served analytically and the shortlist has nothing to cash in.
  // Raise it to measure what the mapper would have said.
  Option<int> mapperOpLimit{
      *this, "mapper-op-limit",
      llvm::cl::desc("Kernels larger than this many ops are profiled "
                     "analytically instead of through the mapper (default "
                     "150, the shipped value)."),
      llvm::cl::init(150)};

  Option<int> balanceTopK{
      *this, "balance-top-k",
      llvm::cl::desc("How many of the ranked balancer moves to evaluate per "
                     "step. 0 evaluates every move (default 0). The candidate "
                     "set is at most six, and an A/B over 11 programs found "
                     "ranking changed neither the result nor the runtime, so "
                     "exhaustive is the honest default."),
      llvm::cl::init(0)};

  // Force a SPECIFIC fusion partition, e.g. "0,1;2,3,4;5" fuses original tasks
  // {0,1}, {2,3,4}, leaves 5 a singleton. Used by the offline exhaustive search
  // (enumerate all partitions -> evaluate each -> global optimum). Overrides
  // the other fusion modes when non-empty.
  Option<std::string> forcePartition{
      *this, "force-partition",
      llvm::cl::desc("Fuse exactly the given groups of original task indices, "
                     "e.g. \"0,1;2,3\" (default: empty = off)."),
      llvm::cl::init("")};

  // Cost-model fission: replace Neura's greedy add-CGRA-to-bottleneck balance
  // (AMOEBA's MoreReplicas hill-climb) with the EXACT min-max replica
  // allocation over the analytical latency(task, cgra_count) curve (minimize
  // the maximum task latency s.t. sum of cgra_count <= grid). Compares optimal
  // vs greedy fission. Shape selection driven by the shape-aware PREDICTED II
  // (LB + hop term) instead of pickBestShape's purely geometric rule. The sound
  // LB is unaffected; the prediction is used only to rank candidate shapes.
  Option<bool> searchShape{
      *this, "search-shape",
      llvm::cl::desc("Choose the CGRA shape by predicted II (LB + hop) over "
                     "all shapes of the same cgra_count, instead of the "
                     "geometric pickBestShape (default false)."),
      llvm::cl::init(false)};

  Option<double> commWeight{
      *this, "comm-weight",
      llvm::cl::desc("Weight of the communication proxy in the joint "
                     "objective (default 0 = compute only)."),
      llvm::cl::init(0.0)};

  Option<double> commMsgCost{
      *this, "comm-msg-cost",
      llvm::cl::desc("Cycles charged per inter-task message. Partitioning "
                     "multiplies the message count (N*M between an N-way and "
                     "an M-way tile group), so this is what makes tiling pay "
                     "for the traffic it creates. Negative (the default) "
                     "derives it per edge from the two endpoints' footprints; "
                     "a non-negative value pins every message to that "
                     "constant."),
      llvm::cl::init(-1.0)};

  Option<bool> allowTemporal{
      *this, "allow-temporal",
      llvm::cl::desc("Let the allocation exceed the grid and pay for the "
                     "surplus with context waves inside the objective "
                     "(default false)."),
      llvm::cl::init(false)};

  Option<bool> memWeightedHop{
      *this, "mem-weighted-hop",
      llvm::cl::desc("Weight the topology term by distance to the nearest "
                     "memory FU instead of the all-pairs mean; unlike the mean "
                     "it is orientation-sensitive (8x4 vs 4x8) (default "
                     "false)."),
      llvm::cl::init(false)};

  Option<double> hopCoef{
      *this, "hop-coef",
      llvm::cl::desc("Coefficient of the topology term in the predicted II: "
                     "predicted = LB + max(0, coef*avg_hop*cp_depth - 1) "
                     "(default 0.05)."),
      llvm::cl::init(0.05)};

  // Data-level-parallelism decision dimensions (AMOEBA Alg. 1 line 7 and loop
  // partitioning). Both are gated per task on `dlp_replicable`. Default off so
  // existing results stay reproducible; turn on to widen the decision space.
  Option<bool> enableReplicas{
      *this, "enable-replicas",
      llvm::cl::desc("Allow the balancer to allocate multiple data-parallel "
                     "replicas of a dlp_replicable task (default false)."),
      llvm::cl::init(false)};

  Option<bool> enableTiling{
      *this, "enable-tiling",
      llvm::cl::desc("Allow the balancer to loop-partition (tile) a "
                     "dlp_replicable task, trading per-iteration work for a "
                     "shorter iteration space (default false)."),
      llvm::cl::init(false)};

  Option<int> maxTiling{
      *this, "max-tiling",
      llvm::cl::desc("Upper bound on the loop-partitioning factor "
                     "(default 8)."),
      llvm::cl::init(8)};

  // Diagnostic: for every task, score EVERY candidate shape with the cheap
  // model and with the real mapper, so top-k coverage of the true optimum is
  // measured in the real pipeline. Expensive (one mapper run per shape).
  Option<bool> shapeCoverageProbe{
      *this, "shape-coverage-probe",
      llvm::cl::desc("Emit [ShapeProbe] lines scoring every candidate shape "
                     "with LB, LB+hop and the real mapper (default false)."),
      llvm::cl::init(false)};

  // Tile the program UP FRONT, before any allocation, instead of searching the
  // tiling factor inside the balance loop. This is the order the hand-tiled
  // experiments used: cut the loops into many small tasks first, then let the
  // allocator size and replicate those. It sidesteps the greedy trap entirely,
  // because by the time the balancer runs, the tasks are already small and the
  // only moves left are the ones that lower II and add replicas.
  Option<int> preTile{
      *this, "pre-tile",
      llvm::cl::desc("Cut every partitionable task into this many tiles BEFORE "
                     "allocation (1 = off, the default)."),
      llvm::cl::init(1)};

  Option<bool> optimalFission{
      *this, "optimal-fission",
      llvm::cl::desc("Sweep the per-task min-max frontier -- every distinct "
                     "latency target, two per-task rules -- and keep the "
                     "allocation the joint objective scores best, instead of "
                     "greedy bottleneck balancing. Exact for the two floors; "
                     "not exact once the objective contains the pipeline "
                     "cycle, which depends on how the whole allocation packs "
                     "(default false)."),
      llvm::cl::init(false)};

  // The two halves of "what does the best allocation in this space actually
  // score?". The pass already enumerates every legal (cgra_count, replicas,
  // tiling) per task while profiling; `dump-config-space` writes that
  // enumeration out, so an external solver can optimise over exactly the space
  // the pass searches -- no re-derivation, no second model of legality --
  // and `import-allocation` plays the answer back through the same
  // materialisation and the same downstream passes. Without the pair, an
  // "optimum" would be a number from a different model than the one being
  // compared against.
  Option<std::string> dumpConfigSpace{
      *this, "dump-config-space",
      llvm::cl::desc("Write the enumerated per-task (cgra_count, replicas, "
                     "tiling) space plus latency, area and the task DAG to "
                     "this JSON path, then continue normally."),
      llvm::cl::init("")};

  Option<std::string> dumpAllocation{
      *this, "dump-allocation",
      llvm::cl::desc("Write the allocation this run settled on to this JSON "
                     "path, in the format import-allocation reads. Lets one "
                     "search seed another."),
      llvm::cl::init("")};

  Option<std::string> importAllocation{
      *this, "import-allocation",
      llvm::cl::desc("Read a per-task {cgra_count, replicas, tiling} JSON and "
                     "use it instead of searching. Combine with "
                     "apply-decisions to materialise it."),
      llvm::cl::init("")};

  // Scoring metric. `interval` is the metric this pass shipped with (max task
  // latency, assumes every task is resident at once). `makespan` is the
  // resource-constrained schedule length; it is the only mode that can see what
  // loop partitioning buys, because it is the only one that charges area over
  // time. Default keeps existing results reproducible.
  Option<std::string> objectiveMode{
      *this, "objective-mode",
      llvm::cl::desc("Objective scored by every knob: 'interval' (default, max "
                     "task latency), 'makespan' (resource-constrained schedule "
                     "length of one instance), or 'pipeline' (steady-state "
                     "pipeline interval, the metric the downstream analyzer "
                     "reports)."),
      llvm::cl::init("interval")};

  // Materialise the searched decisions into the IR: cut loop bounds for tiling
  // and emit the replica partition tables. Without this the two DLP knobs are
  // recorded as attributes but nothing downstream can act on them.
  Option<bool> applyDecisions{
      *this, "apply-decisions",
      llvm::cl::desc("Materialise the searched tiling/replica decisions into "
                     "the IR (split tasks, emit partition tables) instead of "
                     "only annotating them (default false)."),
      llvm::cl::init(false)};

  // Single place where the pass options reach the cost model, so the
  // speculative clones used for fusion scoring cannot drift from the real
  // graph — a knob that is off in the speculation makes fusion look free.
  // Runs the real downstream pipeline on a module and returns the interval the
  // analysis reports, or -1 if it could not be produced.
  //
  // This is the number the paper quotes, so a candidate scored with it is
  // scored against the ground truth rather than against a model of it. The two
  // passes cost 0.00-0.07s per program against runs of 4.5-37s -- 0.2% -- which
  // is why the modelled second layer is worth replacing at the points where a
  // shortlist has already narrowed the field.
  //
  // Clones the whole enclosing module and returns the interval the real
  // downstream pipeline reports for `name`, or -1.
  //
  // The whole module, not just the function: a func that reads a weight through
  // `memref.get_global` refers to a symbol declared at module scope, and
  // cloning the function alone leaves that behind. The clone then fails to
  // verify with "'__constant_64x256xf32' does not reference a valid global
  // memref" -- which is exactly the five programs that carry constants, and
  // exactly the five this returned -1 for.
  int64_t realPipelineIntervalOf(func::FuncOp func) {
    auto parent = func->getParentOfType<ModuleOp>();
    if (!parent)
      return -1;
    OwningOpRef<ModuleOp> clone(parent.clone());
    StringRef name = func.getName();
    int64_t interval = -1;
    clone->walk([&](func::FuncOp candidate) {
      if (candidate.getName() == name)
        interval = 0; // found; the pipeline decides the value below
    });
    if (interval != 0)
      return -1;
    return realPipelineInterval(*clone, name);
  }

  // `mod` must be a standalone clone: a pass manager may not run on anything
  // nested inside the operation the current pass is processing.
  int64_t realPipelineInterval(ModuleOp mod, StringRef only = StringRef()) {
    PassManager pm(mod.getContext());
    // Commas stand in for the spaces that separate options inside `{}`: this
    // string arrives as the value of one of THIS pass's options, and a space
    // there would be read as the end of it.
    std::string downstream = verifyPipelineOptions.getValue();
    std::replace(downstream.begin(), downstream.end(), ',', ' ');
    // No `builtin.module(...)` wrapper: the manager is already anchored there,
    // so wrapping again asks it to find a module nested inside this one. There
    // is none, nothing runs, and `run` reports success -- which is how this
    // first returned -1 with no diagnostic at all.
    std::string pipeline = "func.func(orchestrate-tasks-on-accelerators{" +
                           downstream + "},analyze-task-pipeline-interval)";
    if (failed(parsePassPipeline(pipeline, pm, llvm::errs()))) {
      llvm::errs() << "[RealPipeline] could not parse: " << pipeline << "\n";
      return -1;
    }
    if (failed(pm.run(mod))) {
      llvm::errs() << "[RealPipeline] pipeline failed on the clone\n";
      return -1;
    }

    int64_t best = -1;
    int seen_funcs = 0, seen_info = 0;
    mod.walk([&](func::FuncOp f) {
      if (!only.empty() && f.getName() != only)
        return;
      ++seen_funcs;
      auto info = f->getAttrOfType<DictionaryAttr>("task_pipeline_interval_info");
      if (!info)
        return;
      ++seen_info;
      // The exact value is published separately whenever it does not fit the
      // i32 the in-tree expectations pin; prefer it when it is there.
      if (auto exact = info.getAs<IntegerAttr>("pipeline_interval_cycles"))
        best = std::max(best, exact.getInt());
      else if (auto narrow = info.getAs<IntegerAttr>("pipeline_interval"))
        best = std::max(best, narrow.getInt());
    });
    if (best < 0)
      llvm::errs() << "[RealPipeline] ran, but no interval: " << seen_funcs
                   << " funcs, " << seen_info << " carried the attribute\n";
    return best;
  }

  struct MaterialiseCounts {
    int tiled = 0, replicated = 0, refused = 0;
  };

  // Rewrites the IR so the search's two DLP decisions are real: tiling cuts the
  // loop into separate tasks, replication emits the partition table. Up to here
  // `replicas` and `tiling` are only annotations, and a decision that never
  // reaches the IR is not a decision.
  //
  // Shared with speculation, which is the point. Scoring a candidate on the
  // collapsed graph -- one node carrying `tiling = t` -- and emitting the
  // expanded one is how the objective came to be 30-43% optimistic on every
  // program that tiles: the same graph reports 274,346,824 collapsed and
  // 387,789,643 once its 8 groups are cut, against an analyzer reading of
  // 388,379,469. The expanded number is right to 0.15%, so the fix is to score
  // what gets emitted rather than to keep adjusting the collapsed model.
  MaterialiseCounts materialiseDecisions(TaskDependencyGraph &graph,
                                         int &group_id, bool quiet) {
    MaterialiseCounts counts;
    // Snapshot first: tile() erases task ops, invalidating graph nodes.
    struct Decision {
      TaskflowTaskOp op;
      int tiling, replicas, cgra_count;
      int64_t ii, steps;
    };
    SmallVector<Decision> decisions;
    for (auto &node : graph.nodes)
      decisions.push_back({node->op, node->tiling, node->replicas,
                           node->cgra_count, node->ii, node->steps});
    for (Decision &d : decisions) {
      // Tiling FIRST: it cuts the root range, and the replica partition table
      // must be computed over each tile's own (cut) range, not over the
      // original one — otherwise every tile ships the same table covering the
      // whole loop and the two moves overlap.
      SmallVector<TaskflowTaskOp> targets;
      if (d.tiling > 1) {
        auto tiles = TaskTiler::tile(d.op, d.tiling, group_id, d.ii, d.steps,
                                     d.cgra_count, d.replicas);
        if (tiles.empty()) {
          ++counts.refused;
          if (!quiet)
            llvm::errs() << "  [Apply] REFUSED tiling=" << d.tiling << " on "
                         << d.op.getTaskName() << "\n";
          OpBuilder refuse_builder(d.op);
          d.op->setAttr("tiling", refuse_builder.getI32IntegerAttr(1));
          targets.push_back(d.op);
        } else {
          if (!quiet)
            llvm::errs() << "  [Apply] tiling=" << d.tiling << " -> "
                         << tiles.size() << " tasks, group " << group_id
                         << "\n";
          ++counts.tiled;
          ++group_id;
          targets.assign(tiles.begin(), tiles.end());
        }
      } else {
        targets.push_back(d.op);
      }

      if (d.replicas > 1) {
        for (TaskflowTaskOp t : targets) {
          if (TaskTiler::emitPartitionConfig(t, d.replicas)) {
            ++counts.replicated;
          } else {
            ++counts.refused;
            if (!quiet)
              llvm::errs() << "  [Apply] REFUSED replicas=" << d.replicas
                           << " on " << t.getTaskName()
                           << " (no partitionable root counter)\n";
            OpBuilder refuse_builder(t);
            t->setAttr("replicas", refuse_builder.getI32IntegerAttr(1));
          }
        }
      }
    }
    return counts;
  }

  // `top_k_override` is how a speculative clone pins its own evaluator depth.
  // Passing 0 from `speculate`/`speculatePair` is what keeps the yardstick the
  // same length for every candidate and for every value of `verify-top-k`.
  void configureGraph(TaskDependencyGraph &g, int top_k_override = -1) {
    // Every graph in this pass -- the real one, the speculative clones fusion
    // scoring builds, the final rebuild -- profiles the same task bodies on the
    // same tile arrays, so they share one cache. Without this the O(V^2) fusion
    // sweep re-profiles the whole program per candidate pair.
    g.profile_cache = shared_profile_cache;
    g.use_full_cost_model =
        (estimationMode.getValue() == "cost-model-analytical");
    g.use_predicted_ii = usePredictedII.getValue();
    g.verifier_is_cpsat = (verifier.getValue() == "cpsat");
    g.cpsat_python = cpsatPython.getValue();
    g.cpsat_script = cpsatScript.getValue();
    g.cpsat_seconds = cpsatSeconds.getValue();
    g.cpsat_minimize_routing = cpsatMinimizeRouting.getValue();
    g.search_shape = searchShape.getValue();
    g.hop_coef = hopCoef.getValue();
    g.mem_weighted_hop = memWeightedHop.getValue();
    g.verify_top_k =
        top_k_override >= 0 ? top_k_override : shapeTopK.getValue();
    g.mapper_op_limit = mapperOpLimit.getValue();
    g.comm_weight = commWeight.getValue();
    g.allow_temporal = allowTemporal.getValue();
    g.use_makespan = (objectiveMode.getValue() == "makespan");
    g.use_pipeline = (objectiveMode.getValue() == "pipeline");
    // Negative means "derive it from the architecture" (the default): a
    // message costs at least mean_hops * link_latency. A hand-set value stays
    // authoritative so the term can still be swept.
    const neura::Architecture &arch = neura::getArchitecture();
    double concurrency = 1.0, link_latency = 1.0;
    fabricMessageLatency(arch, &concurrency, &link_latency);
    g.comm_concurrency = concurrency;
    g.comm_link_latency = link_latency;
    g.per_cgra_rows = arch.getPerCgraRows();
    g.per_cgra_cols = arch.getPerCgraColumns();
    // Negative keeps the derived per-edge distance; a hand-set value pins every
    // message to the same constant so the term can be swept.
    g.comm_msg_cost = commMsgCost.getValue();
    g.comm_msg_cost_derived = commMsgCost.getValue() < 0.0;
  }

  // True when the caller opted into the DLP / scheduling machinery. The new
  // attributes are published only then: the in-tree RESOPT expectations pin the
  // exact attribute dictionary, so emitting extras unconditionally would change
  // the shipped output.
  bool emitsNewAttrs() const {
    // `optimalFission` belongs here as well as in `jointSearchEnabled`. On its
    // own it turns on the exact allocator but published no `est_latency`, and
    // the analyzer then falls back to `profile_info.duration` -- the DFG depth,
    // a different unit -- so that arm alone was measured on a different scale
    // from every other arm.
    return enableReplicas.getValue() || enableTiling.getValue() ||
           applyDecisions.getValue() || optimalFission.getValue() ||
           objectiveMode.getValue() != "interval";
  }

  // True when the caller opted into searching the joint space. Gates both the
  // acceptance rule and the shortlist; the shipped invocation must see neither.
  // One profile cache for the whole pass. See `TaskDependencyGraph`.
  std::shared_ptr<TaskDependencyGraph::ProfileCache> shared_profile_cache =
      std::make_shared<TaskDependencyGraph::ProfileCache>();

  // Cells one task may hold. `allow-temporal` lets the allocation exceed the
  // grid and pay for the surplus in context waves, and both allocators size the
  // SUM against that budget, so a per-task cap must use it too.
  int perTaskCellBudget() const {
    return allowTemporal.getValue() ? kTotalCGRAs * kMaxTemporalWaves
                                    : kTotalCGRAs;
  }

  bool jointSearchEnabled() const {
    return enableReplicas.getValue() || enableTiling.getValue() ||
           optimalFission.getValue() || objectiveMode.getValue() != "interval";
  }

  // Publishes the current configuration onto the IR so a downstream run reads
  // it. Only the attributes the orchestrator and the interval analysis consume;
  // no rewriting, so the caller can restore and re-publish freely.
  // The function this run is processing, so an injected callback can reach
  // the real downstream pipeline without threading it through every layer.
  func::FuncOp current_func;

  void writeDecisionAttributes(TaskDependencyGraph &graph) {
    for (auto &node : graph.nodes) {
      OpBuilder builder(node->op);
      node->op->setAttr("cgra_count",
                        builder.getI32IntegerAttr(node->cgra_count));
      node->op->setAttr("replicas", builder.getI32IntegerAttr(node->replicas));
      node->op->setAttr("trip_count",
                        builder.getI32IntegerAttr(node->trip_count));
      node->op->setAttr("est_latency",
                        builder.getI64IntegerAttr(node->estimatedLatency()));
      node->op->setAttr("cgra_shape",
                        builder.getStringAttr(node->shape.irAttr()));
    }
  }

  void configureBalancer(PipelineBalancer &b, int top_k_override = -1) {
    b.enable_replicas = enableReplicas.getValue();
    b.enable_tiling = enableTiling.getValue();
    b.max_tiling = std::max(1, maxTiling.getValue());
    b.joint_rounds = jointRounds.getValue();
    b.joint_top_k = verifyTopK.getValue();
    b.publish_fn = [this](TaskDependencyGraph &g) {
      writeDecisionAttributes(g);
    };
    // Only the outer, committing balance gets the real measurement. The two
    // speculative call sites pass `top_k_override=0`, and letting a lookahead
    // clone run the whole downstream pipeline per candidate would cost far more
    // than the decision it is informing.
    if (top_k_override < 0) {
      b.real_interval_fn = [this]() -> int64_t {
        return current_func ? realPipelineIntervalOf(current_func) : -1;
      };
    }
    b.verify_top_k =
        top_k_override >= 0 ? top_k_override : balanceTopK.getValue();
    b.allow_temporal = allowTemporal.getValue();
    // Gated on actually SEARCHING the joint space, not on `emitsNewAttrs()`.
    //
    // The two are different questions and tying them together made the baseline
    // unmeasurable. `est_latency` is published under emitsNewAttrs(), and the
    // downstream analyzer prefers it over `profile_info.duration`; so a
    // "shipped allocator" arm that did not publish it was scored in different
    // units from ours -- bicg came out 840 against 9730 purely because one arm
    // was measured in per-stage `duration` (84/task) and the other in real
    // cycles (973/task). Now `apply-decisions=true` alone gives the shipped
    // AMOEBA acceptance rule WITH comparable units.
    b.joint_criterion = jointSearchEnabled();
  }

  // Serialises the decision space and the DAG it is scored on.
  //
  // Everything a solver needs to reproduce `objective-mode=pipeline` without
  // re-implementing the pass: per task its legal configs with (latency, area),
  // the successor edges, and the constants the floors use. Hand-rolled JSON
  // because MLIR has no JSON writer and the schema is four fields deep.
  void writeConfigSpace(TaskDependencyGraph &graph,
                        const std::vector<std::vector<TaskConfig>> &configs,
                        StringRef path) {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      llvm::errs() << "[ConfigSpace] cannot write " << path << ": "
                   << ec.message() << "\n";
      return;
    }
    llvm::DenseMap<const TaskGraphNode *, int> index;
    for (int i = 0; i < (int)graph.nodes.size(); ++i)
      index[graph.nodes[i].get()] = i;

    os << "{\n  \"total_cgras\": " << kTotalCGRAs
       << ",\n  \"max_waves\": " << kMaxTemporalWaves
       << ",\n  \"allow_temporal\": "
       << (allowTemporal.getValue() ? "true" : "false")
       << ",\n  \"tasks\": [\n";
    for (int i = 0; i < (int)graph.nodes.size(); ++i) {
      TaskGraphNode *nd = graph.nodes[i].get();
      os << "    {\"index\": " << i << ", \"name\": \"" << nd->op.getTaskName()
         << "\", \"trip_count\": " << nd->trip_count << ", \"succ\": [";
      bool first = true;
      for (const TaskGraphNode *s : nd->successors) {
        auto it = index.find(s);
        if (it == index.end())
          continue;
        os << (first ? "" : ", ") << it->second;
        first = false;
      }
      os << "], \"configs\": [";
      for (size_t j = 0; j < configs[i].size(); ++j) {
        const TaskConfig &cf = configs[i][j];
        os << (j ? ", " : "") << "{\"c\": " << cf.cgra_count << ", \"k\": " << cf.replicas
           << ", \"t\": " << cf.tiling << ", \"area\": " << cf.cells
           << ", \"lat\": " << cf.latency << "}";
      }
      os << "]}" << (i + 1 < (int)graph.nodes.size() ? "," : "") << "\n";
    }
    os << "  ]\n}\n";
    llvm::errs() << "[ConfigSpace] wrote " << graph.nodes.size() << " tasks to "
                 << path << "\n";
  }

  // The inverse of applyImportedAllocation: what this run decided.
  void writeAllocation(TaskDependencyGraph &graph, StringRef path) {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      llvm::errs() << "[DumpAlloc] cannot write " << path << ": "
                   << ec.message() << "\n";
      return;
    }
    os << "{\n  \"alloc\": {\n";
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
      TaskGraphNode *nd = graph.nodes[i].get();
      os << "    \"" << nd->op.getTaskName() << "\": {\"c\": " << nd->cgra_count
         << ", \"k\": " << std::max(1, nd->replicas)
         << ", \"t\": " << std::max(1, nd->tiling) << "}"
         << (i + 1 < graph.nodes.size() ? "," : "") << "\n";
    }
    os << "  }\n}\n";
  }

  // Reads back a `{"alloc": {"Task_3": {"c": 2, "k": 1, "t": 4}, ...}}`
  // allocation and installs it. Tasks the file does not mention keep what they
  // have, so a partial allocation is a legal input.
  bool applyImportedAllocation(TaskDependencyGraph &graph, StringRef path) {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
      llvm::errs() << "[ImportAlloc] cannot read " << path << "\n";
      return false;
    }
    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
      llvm::errs() << "[ImportAlloc] bad JSON in " << path << "\n";
      llvm::consumeError(parsed.takeError());
      return false;
    }
    llvm::json::Object *root = parsed->getAsObject();
    llvm::json::Object *alloc = root ? root->getObject("alloc") : nullptr;
    if (!alloc) {
      llvm::errs() << "[ImportAlloc] no \"alloc\" object in " << path << "\n";
      return false;
    }
    bool changed = false;
    int matched = 0, unmatched = 0;
    for (auto &node : graph.nodes) {
      llvm::json::Object *entry = alloc->getObject(node->op.getTaskName());
      if (!entry) {
        // Silently keeping the searched value here would make the run a hybrid
        // of the imported allocation and this run's own search, reported as if
        // it were the imported one. Names diverge when the two runs fuse
        // differently, which they can: fusion is scored on the allocation, and
        // the imported allocation is not the one the dumping run searched.
        ++unmatched;
        llvm::errs() << "  [ImportAlloc] NO ENTRY for "
                     << node->op.getTaskName() << ", keeping searched value\n";
        continue;
      }
      ++matched;
      const int cgra_count =
          (int)entry->getInteger("c").value_or(node->cgra_count);
      const int replicas =
          (int)entry->getInteger("k").value_or(node->replicas);
      const int tiling = (int)entry->getInteger("t").value_or(node->tiling);
      if (node->cgra_count != cgra_count || node->replicas != replicas ||
          node->tiling != tiling)
        changed = true;
      node->cgra_count = cgra_count;
      node->replicas = replicas;
      node->tiling = tiling;
      // Re-profile so the shape and II match the imported tile array; without
      // this the objective would be scored on the previous array's II.
      graph.profileWithBestShape(node.get(), node->op, /*skip_mapper=*/true);
      llvm::errs() << "  [ImportAlloc] " << node->op.getTaskName()
                   << " -> cgra_count=" << cgra_count
                   << ", replicas=" << replicas << ", tiling=" << tiling
                   << ", est_latency=" << node->estimatedLatency() << "\n";
    }
    llvm::errs() << "[ImportAlloc] matched=" << matched
                 << " unmatched=" << unmatched << " of " << graph.nodes.size()
                 << " tasks\n";
    return changed;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    current_func = func;

    // cost-model-analytical: full parametric model (all bounds), no mapper.
    // analytical: legacy ResMII/RecMII-only, no mapper.
    // compiled (default): full lowering + mapper oracle.
    bool use_full_cost_model =
        (estimationMode.getValue() == "cost-model-analytical");
    bool use_analytical =
        (estimationMode.getValue() == "analytical") || use_full_cost_model;

    // Both gating predicates test `objectiveMode != "interval"`, so an
    // unrecognised value turns the joint acceptance rule and the new attribute
    // set on while leaving the objective at the shipped `interval` -- a typo
    // silently produces a fourth configuration that was never intended.
    const StringRef objective_mode = objectiveMode.getValue();
    if (objective_mode != "interval" && objective_mode != "makespan" &&
        objective_mode != "pipeline") {
      func.emitError() << "unknown objective-mode '" << objective_mode
                       << "'; expected interval, makespan or pipeline";
      return signalPassFailure();
    }
    const StringRef estimation_mode = estimationMode.getValue();
    if (estimation_mode != "compiled" && estimation_mode != "analytical" &&
        estimation_mode != "cost-model-analytical") {
      func.emitError() << "unknown estimation-mode '" << estimation_mode
                       << "'; expected compiled, analytical or "
                          "cost-model-analytical";
      return signalPassFailure();
    }

    if (commMsgCost.getValue() >= 0.0 && commWeight.getValue() <= 0.0)
      func.emitWarning()
          << "comm-msg-cost is set but comm-weight is 0, so the communication "
             "term is switched off and the message cost has no effect";
    // The four fusion options are a precedence chain, not a set: force-partition
    // beats disable-fusion, which beats global-fusion, which beats
    // cost-guided-fusion. Combining them silently kept only the first.
    if (!forcePartition.getValue().empty() && disableFusion.getValue())
      func.emitWarning() << "force-partition overrides disable-fusion; tasks "
                            "in the given groups will still be fused";
    if (globalFusion.getValue() && costGuidedFusion.getValue())
      func.emitWarning() << "global-fusion overrides cost-guided-fusion";

    llvm::errs() << "=== ResourceAwareTaskOptimization on " << func.getName()
                 << " (estimation-mode=" << estimationMode.getValue()
                 << ") ===\n";
    if (commWeight.getValue() > 0.0) {
      TaskDependencyGraph probe;
      configureGraph(probe);
      llvm::errs() << "[Comm] msg_cost=";
      if (probe.comm_msg_cost_derived)
        llvm::errs() << "per-edge hops * "
                     << llvm::format("%.2f", probe.comm_link_latency)
                     << " cycles/link (1-CGRA pair: "
                     << llvm::format("%.2f", probe.blockPairHops(1, 1) *
                                                 probe.comm_link_latency)
                     << ")";
      else
        llvm::errs() << llvm::format("%.3f", probe.comm_msg_cost)
                     << " (flat, set on the command line)";
      llvm::errs() << ", BW=" << TaskDependencyGraph::kCommBW
                   << ", fabric concurrency="
                   << llvm::format("%.1f", probe.comm_concurrency)
                   << " transfers\n";
    }

    // Pre-tiling runs once, on the raw task list, before the first profile.
    if (preTile.getValue() > 1) {
      SmallVector<TaskflowTaskOp> tasks;
      func.walk([&](TaskflowTaskOp t) { tasks.push_back(t); });
      int group = 0, cut = 0;
      for (TaskflowTaskOp t : tasks) {
        int64_t space = TaskTiler::partitionSpace(t);
        int factor = (int)std::min<int64_t>(preTile.getValue(), space);
        // Fall back to the largest realisable factor at or below the request.
        while (factor > 1 && !TaskTiler::canTile(t, factor))
          --factor;
        if (factor <= 1)
          continue;
        auto tiles = TaskTiler::tile(t, factor, group, /*ii=*/0, /*steps=*/0,
                                     /*cgra_count=*/1, /*replicas=*/1);
        if (tiles.empty())
          continue;
        llvm::errs() << "[PreTile] " << tiles.size() << " tiles (requested "
                     << preTile.getValue() << ", space " << space
                     << ") for group " << group << "\n";
        ++group;
        cut += tiles.size();
      }
      llvm::errs() << "[PreTile] " << group << " groups, " << cut
                   << " tasks after pre-tiling\n";
    }

    constexpr int kMaxOuterIterations = 10;

    // Best allocation seen across the whole search, keyed by task name.
    //
    // The outer loop rebuilds the graph from the IR each round, so a hill-climb
    // that is monotone WITHIN one round is not monotone across rounds: the
    // rebuilt graph re-derives latencies and partition spaces, and the next
    // round can climb away from a better point. On bicg the search ended at an
    // allocation its own objective scored 13950 while it had already visited
    // one it scored 9730. Remembering the best and restoring it before
    // materialising costs one map and removes the failure mode.
    llvm::StringMap<std::tuple<int, int, int>> best_seen;
    int64_t best_seen_obj = INT64_MAX;

    for (int outer = 0; outer < kMaxOuterIterations; ++outer) {
      // Rebuilds graph from current IR state.
      TaskDependencyGraph graph;
      configureGraph(graph);
      graph.build(func, use_analytical);

      if (graph.nodes.empty()) {
        return;
      }

      int num_tasks = graph.nodes.size();

      // NOTE: the previous hard assert(num_tasks <= kTotalCGRAs) is removed so
      // that programs with MORE tasks than the 4x4 grid (AMOEBA scale, 9-28
      // tasks) can run: fusion + spatial-temporal context reuse are meant to
      // fit them onto the 16 cores. Grid stays 4x4 (AMOEBA architecture); we do
      // NOT increase kTotalCGRAs.

      llvm::errs() << "[ResourceAware] Iteration " << outer << ": " << num_tasks
                   << " tasks\n";

      if (shapeCoverageProbe.getValue() && outer == 0) {
        for (auto &node : graph.nodes)
          graph.probeShapeCoverage(node.get(), node->op);
      }
      for (auto &node : graph.nodes) {
        llvm::errs()
            << "  Task " << node->id << " (" << node->op.getTaskName()
            << "): trip_count=" << node->trip_count
            << ", cgra_count=" << node->cgra_count
            << ", partitions=" << node->root_trip
            << (node->partition_illegal
                    ? " [dlp REVOKED: no dimension indexes every store]"
                    : (node->dlp_replicable ? " [dlp]" : ""))
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
      // Cost-guided fusion is PARETO-SAFE via one-step speculative lookahead:
      // clone the function and compute the converged-this-step interval both
      // WITHOUT fusing (balance only) and WITH the next fusion (fuse +
      // balance); commit the fusion only if it STRICTLY lowers the balanced
      // interval. This never does worse than balance-only (over-fusion can't
      // sneak in) and still captures beneficial fusion (freeing CGRAs for the
      // bottleneck) — no separate budget guard needed. Legacy behavior fuses
      // unconditionally. Fusion is scored on the SAME joint objective the
      // balancer optimises, not on bare max-latency: fusion trades CGRA area
      // (which the makespan and the wave count charge for) and the
      // `dlp_replicable` tag (which the two DLP moves are gated on) for a
      // shorter task list. Scoring it on a metric blind to both makes fusion
      // look free.
      auto graphInterval = [](TaskDependencyGraph &g) -> int64_t {
        return g.objective();
      };
      // Scores a speculative clone the way the emitted program will read: cut
      // the tiles, emit the partition tables, rebuild, and measure that. The
      // clone is discarded either way, so materialising it costs one rebuild
      // and buys the difference between a collapsed graph and a real one.
      auto scoreClone = [&](TaskDependencyGraph &g,
                            func::FuncOp clone_func) -> int64_t {
        if (!verifyOnMaterialised.getValue())
          return graphInterval(g);
        int speculative_group_id = 0;
        materialiseDecisions(g, speculative_group_id, /*quiet=*/true);
        TaskDependencyGraph emitted;
        configureGraph(emitted, /*top_k_override=*/0);
        emitted.build(clone_func, use_analytical);
        return graphInterval(emitted);
      };
      // Speculatively runs (optional) one fusion + one balance pass on a clone
      // of the current function and returns the resulting pipeline interval.
      auto speculate = [&](bool do_fuse) -> int64_t {
        auto tmp_mod = ModuleOp::create(func.getLoc());
        OpBuilder tmp_builder(tmp_mod.getBodyRegion());
        auto tmp_func = cast<func::FuncOp>(tmp_builder.clone(*func.getOperation()));
        TaskDependencyGraph speculative_graph;
        configureGraph(speculative_graph, /*top_k_override=*/0);
        speculative_graph.build(tmp_func, use_analytical);
        if (do_fuse) {
          auto fusion_profile_fn = [&speculative_graph, use_analytical](TaskGraphNode *node, TaskflowTaskOp task) {
            speculative_graph.profileTask(node, task, /*skip_mapper=*/use_analytical);
          };
          UtilizationFuser fuser;
          if (fuser.fuse(tmp_func, speculative_graph, fusion_profile_fn)) {
            speculative_graph = TaskDependencyGraph();
            configureGraph(speculative_graph, /*top_k_override=*/0);
            speculative_graph.build(tmp_func, use_analytical);
          } else {
            tmp_mod.erase();
            return -1; // no fusion possible
          }
        }
        // `profileWithBestShape`, not `profileTask`: the latter profiles at
        // whatever `shape` the node already carries, which `build` leaves at
        // 1x1. A speculative LargerTileArray move would then set
        // cgra_count = c+1, re-profile on a 1x1 array, hit the same cache
        // entry, and see no improvement -- so every such move was rejected and
        // fusion was scored against a balancer that could not balance. The
        // real path (`balance_profile_fn`) has always used the shape-deriving
        // one.
        auto speculative_profile_fn = [&speculative_graph](TaskGraphNode *node,
                                           TaskflowTaskOp task) {
          speculative_graph.profileWithBestShape(node, task, /*skip_mapper=*/true);
        };
        // The fusion decision must be SCORED in the same decision space the
        // real balancer will use, otherwise fusion looks free when it actually
        // costs the `dlp_replicable` tag that replicas/tiling are gated on.
        PipelineBalancer balancer;
        configureBalancer(balancer, /*top_k_override=*/0);
        balancer.balance(speculative_graph, speculative_profile_fn);
        int64_t interval = scoreClone(speculative_graph, tmp_func);
        tmp_mod.erase();
        return interval;
      };
      // Speculatively fuses the SPECIFIC pair (i,j) of the current graph on a
      // clone, balances, and returns the resulting interval (-1 if illegal).
      // Node order is deterministic across graph.build, so the index pair
      // identifies the same tasks in the clone as in the real graph.
      auto speculatePair = [&](int first_task_idx,
                               int second_task_idx) -> int64_t {
        auto tmp_mod = ModuleOp::create(func.getLoc());
        OpBuilder tmp_builder(tmp_mod.getBodyRegion());
        auto tmp_func =
            cast<func::FuncOp>(tmp_builder.clone(*func.getOperation()));
        TaskDependencyGraph speculative_graph;
        configureGraph(speculative_graph, /*top_k_override=*/0);
        speculative_graph.build(tmp_func, use_analytical);
        const int node_count = (int)speculative_graph.nodes.size();
        if (first_task_idx >= node_count || second_task_idx >= node_count) {
          tmp_mod.erase();
          return -1;
        }
        TaskGraphNode *first_node =
            speculative_graph.nodes[first_task_idx].get();
        TaskGraphNode *second_node =
            speculative_graph.nodes[second_task_idx].get();
        UtilizationFuser fuser;
        if (!fuser.canFuse(first_node, second_node, speculative_graph)) {
          tmp_mod.erase();
          return -1;
        }
        auto fusion_profile_fn = [&speculative_graph, use_analytical](
                                     TaskGraphNode *node, TaskflowTaskOp task) {
          speculative_graph.profileTask(node, task,
                                        /*skip_mapper=*/use_analytical);
        };
        fuser.fuseNodes(tmp_func, first_node, second_node, speculative_graph,
                        fusion_profile_fn);
        speculative_graph = TaskDependencyGraph();
        configureGraph(speculative_graph, /*top_k_override=*/0);
        speculative_graph.build(tmp_func, use_analytical);
        // `profileWithBestShape` for the same reason as in `speculate` above.
        auto speculative_profile_fn = [&speculative_graph](
                                          TaskGraphNode *node,
                                          TaskflowTaskOp task) {
          speculative_graph.profileWithBestShape(node, task,
                                                 /*skip_mapper=*/true);
        };
        PipelineBalancer balancer;
        configureBalancer(balancer, /*top_k_override=*/0);
        balancer.balance(speculative_graph, speculative_profile_fn);
        int64_t interval = scoreClone(speculative_graph, tmp_func);
        tmp_mod.erase();
        return interval;
      };
      // --force-partition parsing: original task index -> target group id.
      llvm::DenseMap<int, int> forceGroup;
      if (!forcePartition.getValue().empty()) {
        llvm::SmallVector<llvm::StringRef> groups;
        llvm::StringRef(forcePartition.getValue()).split(groups, ';');
        for (int group_idx = 0; group_idx < (int)groups.size();
             ++group_idx) {
          llvm::SmallVector<llvm::StringRef> ids;
          groups[group_idx].split(ids, ',');
          for (auto id : ids) {
            int task_index;
            if (!id.trim().getAsInteger(10, task_index))
              forceGroup[task_index] = group_idx;
          }
        }
      }
      // Extracts original task indices from a (possibly fused) task name like
      // "Task_3_Task_5_utilfused" -> {3,5}.
      auto originalTaskIndices = [](llvm::StringRef name) {
        llvm::SmallVector<int> task_indices;
        size_t scan_pos = 0;
        while ((scan_pos = name.find("Task_", scan_pos)) !=
               llvm::StringRef::npos) {
          scan_pos += 5;
          size_t digits_end = scan_pos;
          while (digits_end < name.size() && name[digits_end] >= '0' &&
                 name[digits_end] <= '9')
            ++digits_end;
          int task_index;
          if (digits_end > scan_pos &&
              !name.substr(scan_pos, digits_end - scan_pos)
                   .getAsInteger(10, task_index))
            task_indices.push_back(task_index);
          scan_pos = digits_end;
        }
        return task_indices;
      };
      bool fuse_changed = false;
      if (!forcePartition.getValue().empty()) {
        // Fuse one within-group pair per iteration until each target group is a
        // single task (offline exhaustive-partition evaluation).
        TaskGraphNode *first_node = nullptr, *second_node = nullptr;
        const int node_count = (int)graph.nodes.size();
        for (int first = 0; first < node_count && !first_node; ++first)
          for (int second = first + 1; second < node_count; ++second) {
            auto first_indices = originalTaskIndices(
                graph.nodes[first]->op.getTaskName());
            auto second_indices = originalTaskIndices(
                graph.nodes[second]->op.getTaskName());
            if (first_indices.empty() || second_indices.empty())
              continue;
            const int group_id = forceGroup.count(first_indices[0])
                                     ? forceGroup[first_indices[0]]
                                     : -1;
            bool same_group = (group_id >= 0);
            for (int task_index : first_indices)
              same_group = same_group && forceGroup.count(task_index) &&
                           forceGroup[task_index] == group_id;
            for (int task_index : second_indices)
              same_group = same_group && forceGroup.count(task_index) &&
                           forceGroup[task_index] == group_id;
            if (same_group && fuser.canFuse(graph.nodes[first].get(),
                                            graph.nodes[second].get(), graph)) {
              first_node = graph.nodes[first].get();
              second_node = graph.nodes[second].get();
              break;
            }
          }
        if (first_node)
          fuse_changed = fuser.fuseNodes(func, first_node, second_node, graph,
                                         profile_fn);
      } else if (disableFusion.getValue()) {
        // balance-only ablation: never fuse.
      } else if (globalFusion.getValue()) {
        // Rank every candidate pair cheaply, then verify only the top few.
        //
        // The selection is still global -- no pair is excluded by a local rule
        // like Neura's trip-diff -- but the expensive step is not. Scoring a
        // pair by actually fusing it costs a module clone, a graph rebuild and
        // a balance, and doing that for all O(V^2) pairs, every outer
        // iteration, is what made this pass slow: on `bicg` (19 tasks, 171
        // pairs) the run took 136s against 15.6s with fusion ranking off, and
        // the ranking is the whole of that difference.
        //
        // That is the same failure this project set out to fix on the shape
        // axis -- mapping every candidate to compare it does not scale -- so it
        // gets the same treatment: rank with the cost model, verify the top-k.
        const int64_t interval_without_fusion = speculate(/*do_fuse=*/false);
        int64_t best_interval = interval_without_fusion;
        int best_first = -1, best_second = -1;
        const int node_count = (int)graph.nodes.size();

        // Cheap score for fusing (a, b), with no clone: the two floors of the
        // objective, evaluated on the graph that fusion would produce. Fusing
        // puts both bodies on one tile array, so their ResMII adds and their
        // trip counts merge to the larger -- the mechanism that makes
        // unconditional fusion lose. The pipeline cycle is left out; this value
        // only orders the shortlist, and the survivors are scored for real.
        auto rankFusion = [&](int first, int second) -> int64_t {
          const TaskGraphNode *lhs = graph.nodes[first].get();
          const TaskGraphNode *rhs = graph.nodes[second].get();
          const int64_t fused_ii = lhs->ii + rhs->ii;
          const int64_t fused_steps = std::max(lhs->steps, rhs->steps);
          const int64_t fused_trip =
              std::max(lhs->trip_count, rhs->trip_count);
          const int64_t fused_latency =
              fused_ii * std::max<int64_t>(0, fused_trip - 1) + fused_steps;
          const int fused_cells =
              std::max(lhs->allocatedCgras(), rhs->allocatedCgras());

          int64_t task_floor = fused_latency;
          int64_t area_time = fused_latency * fused_cells;
          for (auto &other : graph.nodes) {
            if (other.get() == lhs || other.get() == rhs)
              continue;
            const int64_t latency = other->estimatedLatency();
            task_floor = std::max(task_floor, latency);
            area_time += latency * other->allocatedCgras();
          }
          return std::max<int64_t>(
              task_floor, llvm::divideCeil(area_time, (int64_t)kTotalCGRAs));
        };

        SmallVector<std::pair<int64_t, std::pair<int, int>>> ranked;
        for (int first = 0; first < node_count; ++first)
          for (int second = first + 1; second < node_count; ++second)
            if (fuser.canFuse(graph.nodes[first].get(),
                              graph.nodes[second].get(), graph))
              ranked.push_back(
                  {rankFusion(first, second), {first, second}});
        llvm::sort(ranked);

        const int verify_limit = verifyTopK.getValue();
        const int to_verify =
            verify_limit <= 0
                ? (int)ranked.size()
                : std::min<int>(verify_limit, (int)ranked.size());
        llvm::errs() << "[ResourceAware] global-fusion: " << ranked.size()
                     << " legal pairs, verifying top " << to_verify << "\n";
        for (int candidate = 0; candidate < to_verify; ++candidate) {
          const auto [first, second] = ranked[candidate].second;
          const int64_t candidate_interval = speculatePair(first, second);
          if (candidate_interval >= 0 && candidate_interval < best_interval) {
            best_interval = candidate_interval;
            best_first = first;
            best_second = second;
          }
        }
        if (best_first >= 0) {
          llvm::errs() << "[ResourceAware] global-fusion: commit best pair ("
                       << best_first << "," << best_second << ") interval "
                       << interval_without_fusion << "->" << best_interval
                       << "\n";
          fuse_changed = fuser.fuseNodes(func, graph.nodes[best_first].get(),
                                         graph.nodes[best_second].get(), graph,
                                         profile_fn);
        } else {
          llvm::errs() << "[ResourceAware] global-fusion: no improving pair "
                          "(nofuse="
                       << interval_without_fusion << ")\n";
        }
      } else if (!costGuidedFusion.getValue()) {
        fuse_changed =
            fuser.fuse(func, graph, profile_fn); // legacy unconditional
      } else {
        const int64_t interval_with_fusion = speculate(/*do_fuse=*/true);
        if (interval_with_fusion < 0) {
          // no legal fusion available.
        } else {
          const int64_t interval_without_fusion = speculate(/*do_fuse=*/false);
          if (interval_with_fusion < interval_without_fusion) {
            fuse_changed =
                fuser.fuse(func, graph, profile_fn); // commit -- it helps
          } else {
            llvm::errs() << "[ResourceAware] cost-guided: skip fusion (nofuse="
                         << interval_without_fusion
                         << " <= fuse=" << interval_with_fusion << ")\n";
          }
        }
      }

      llvm::errs() << "[ResourceAware] After fusion: total_cgras="
                   << graph.getTotalAllocatedCGRAs() << "\n";

      // Rebuilds graph after fusion (tasks may have been erased/created).
      if (fuse_changed) {
        graph = TaskDependencyGraph();
        configureGraph(graph);
        graph.build(func, use_analytical);
      }

      // Phase 2: Latency-Aware Pipeline Balance.
      //
      // Obeys `balance-skip-mapper`, which is what the flag has always claimed
      // to do -- "set to false for accurate compiled_ii during balance at the
      // cost of compile time". It could not: the value was hardcoded here, so
      // the climb settled every shape analytically no matter what was passed,
      // and the flag reached only the post-convergence verification below.
      //
      // The knob decides which baseline the compile time is measured against,
      // so a knob that does not turn decides it silently. With it stuck at
      // true the shipped climb lowers once or twice per kernel -- fewer times
      // than the shortlist it is supposed to be more expensive than -- because
      // it is not searching, not because it is searching cheaply. With it
      // false the climb does what the greedy pattern actually costs: a
      // complete lowering per probe, per shape it tries.
      //
      // Same expression as the verification below, so one flag means one thing
      // in both places.
      const bool balance_probes_skip_mapper =
          use_analytical && balanceSkipMapper.getValue();
      auto balance_profile_fn = [&graph, balance_probes_skip_mapper](
                                    TaskGraphNode *node, TaskflowTaskOp task) {
        graph.profileWithBestShape(node, task, balance_probes_skip_mapper);
      };
      // What the shortlist cashes in: one COMPLETE lowering of a candidate.
      // The mapper runs, or the exact solver when one is configured, and the
      // II that comes back is a measurement rather than the floor the ranking
      // already used.
      //
      // `profileTask`, not `profileWithBestShape`, because the joint pool
      // carries its own shape axis and already set `node->shape` to the
      // candidate's. `profileWithBestShape` starts by overwriting that with
      // `pickBestShape(cgra_count)` and runs its own shape shortlist, so
      // routing the candidates through it discarded the axis the pool had just
      // enumerated -- every candidate at a given cgra_count was profiled at the
      // same shape, which is why they kept coming back with identical
      // measurements.
      auto balance_verify_fn = [&graph](TaskGraphNode *node,
                                        TaskflowTaskOp task) {
        graph.profileTask(node, task, /*skip_mapper=*/false);
      };
      // Cost-model (optimal) fission: exact min-max allocation over the
      // analytical latency curve, vs Neura's greedy bottleneck balance.
      //
      // The configuration of a task is the triple (c, k, T):
      // Every legal (cgra_count, replicas, tiling) for every task, with the
      // latency and cell count each one costs. This is THE decision space: the
      // exact allocator optimises over it, `dump-config-space` writes it out so
      // an external solver can optimise over the same one, and nothing else
      // re-derives what is legal.
      //
      // Only `cgra_count` changes the mapped tile array, so only it needs a
      // re-profile; replicas and tiling are analytic on top of that profile.
      // Each task is therefore profiled kMaxCgrasPerTask times, not once per
      // configuration.
      auto enumerateConfigSpace =
          [&](TaskDependencyGraph &graph)
          -> std::vector<std::vector<TaskConfig>> {
        const int task_count = (int)graph.nodes.size();
        const int replica_limit = enableReplicas.getValue() ? kTotalCGRAs : 1;
        const int tiling_limit =
            enableTiling.getValue() ? std::max(1, maxTiling.getValue()) : 1;
        std::vector<std::vector<TaskConfig>> configs(task_count);
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          TaskGraphNode *node = graph.nodes[task_idx].get();
          const int saved_cgra_count = node->cgra_count;
          const int saved_replicas = node->replicas;
          const int saved_tiling = node->tiling;
          const CgraShape saved_shape = node->shape;
          const int64_t saved_ii = node->ii, saved_steps = node->steps;
          for (int cgra_count = 1; cgra_count <= kMaxCgrasPerTask;
               ++cgra_count) {
            if (!canFitOnGrid(cgra_count))
              continue;
            node->cgra_count = cgra_count;
            graph.profileWithBestShape(node, node->op, /*skip_mapper=*/true);
            const int max_replicas = node->dlp_replicable ? replica_limit : 1;
            const int max_tiling = tiling_limit; // tiling needs no DLP tag
            for (int replicas = 1; replicas <= max_replicas; ++replicas) {
              for (int tiling = 1; tiling <= max_tiling; tiling *= 2) {
                // A tiled task becomes `tiling` separate tasks, EACH holding
                // cgra_count * replicas cells. Charging only cgra_count *
                // replicas (as this allocator used to) hides a factor of
                // `tiling` and lets the "exact" allocation ask for 128 cells on
                // a 16-cell grid.
                const int64_t cells =
                    (int64_t)cgra_count * replicas * tiling;
                // The cap is the same budget the allocators enforce. Capping a
                // single task at the spatial grid while the allocators size the
                // SUM against the temporal budget made the enumerated space a
                // strict subset of the greedy's, so the exact allocator and any
                // solver reading `--dump-config-space` optimised over a smaller
                // problem than the one they were compared against.
                if (cells > perTaskCellBudget())
                  break;
                // Replicas are concurrent, so they answer to the spatial grid
                // and not to the temporal budget `perTaskCellBudget()` allows.
                if (!TaskGraphNode::replicasFitGrid(cgra_count, replicas))
                  break;
                // Both DLP moves draw from the same partition space.
                if ((int64_t)replicas * tiling > node->split_space)
                  break;
                // Tiling must be realisable as a cut of the loop nest.
                if (tiling > 1 && !TaskTiler::canTile(node->op, tiling))
                  continue;
                node->replicas = replicas;
                node->tiling = tiling;
                configs[task_idx].push_back({cgra_count, replicas, tiling,
                                             (int)cells,
                                             node->estimatedLatency()});
              }
              if ((int64_t)cgra_count * (replicas + 1) > kTotalCGRAs)
                break;
            }
          }
          node->cgra_count = saved_cgra_count;
          node->replicas = saved_replicas;
          node->tiling = saved_tiling;
          node->shape = saved_shape;
          node->ii = saved_ii;
          node->steps = saved_steps;
        }
        return configs;
      };

      auto optimalFissionAllocate = [&](TaskDependencyGraph &graph) -> bool {
        int task_count = (int)graph.nodes.size();
        if (task_count == 0)
          return false;
        using Config = TaskConfig;
        std::vector<std::vector<Config>> configs = enumerateConfigSpace(graph);
        for (int i = 0; i < task_count; ++i)
          if (configs[i].empty())
            return false;

        // Minimum feasible latency target: for each task take the cheapest config
        // meeting latency_target; feasible iff the total CGRA cost fits the grid.
        std::vector<int64_t> targets;
        for (auto &row : configs)
          for (auto &config : row)
            targets.push_back(config.latency);
        llvm::sort(targets);
        targets.erase(llvm::unique(targets), targets.end());

        // Sweeps the latency target. For each target take the cheapest config
        // per task that meets it; that is the exact min-max allocation under a
        // simultaneous-residency area budget.
        //
        // The old version stopped at the FIRST feasible target, which is the
        // min-max optimum but NOT the optimum of the objective the rest of the
        // pass is minimising -- that is why its prediction did not survive
        // materialisation. Every feasible target is now scored with
        // graph.objective() and the best is kept, so "exact" means exact for the
        // objective actually in use.
        std::vector<Config> best_picked;
        int64_t best_objective = INT64_MAX;
        const int grid_budget = allowTemporal.getValue()
                                    ? kTotalCGRAs * kMaxTemporalWaves
                                    : kTotalCGRAs;
        // Saves the live state so scoring can mutate the graph freely.
        std::vector<std::tuple<int, int, int>> saved_state;
        for (auto &node : graph.nodes)
          saved_state.emplace_back(node->cgra_count, node->replicas, node->tiling);

        // Installs an allocation and re-profiles every task on it.
        //
        // `graph.objective()` derives each task's latency from `node->ii`, and
        // `ii` is a property of the tile array, not of the allocation record.
        // `enumerateConfigSpace` restores the incumbent's `ii` on exit, so
        // scoring a candidate without re-profiling scores its cgra_count
        // against the incumbent's II: every move that grows the array is priced
        // as if it bought nothing, and every move that shrinks it is priced as
        // if it cost nothing. On gemv, whose II falls from 26 to 8 between
        // cgra_count 1 and 3, that returned cgra_count=1 for all 18 tasks.
        // The profile cache absorbs the cost -- `enumerateConfigSpace` has
        // already profiled every one of these configurations.
        auto installAndProfile = [&](llvm::ArrayRef<Config> allocation) {
          for (int i = 0; i < task_count; ++i) {
            TaskGraphNode *node = graph.nodes[i].get();
            node->cgra_count = allocation[i].cgra_count;
            node->replicas = allocation[i].replicas;
            node->tiling = allocation[i].tiling;
            graph.profileWithBestShape(node, node->op, /*skip_mapper=*/true);
          }
        };

        // Every (rule, target) pair produces one complete allocation, and
        // there are many: 84 on bicg, 104 on cnn_tiled. Scoring one for real
        // means re-profiling every task and running the pipeline-cycle list
        // schedule, so scoring them all is the same cost blow-up the fusion
        // ranking had -- and it gets the same treatment. Each candidate is
        // ranked by a closed-form score that reads only the enumerated
        // (latency, cells) records, then the top `verify-top-k` are installed,
        // profiled and scored by the objective that actually decides.
        //
        // The ranking score is the two floors. It is exactly the part of the
        // objective that is separable, so it costs nothing beyond arithmetic
        // already done during enumeration; the part it omits, the pipeline
        // cycle, is what the verification adds back.
        auto cheapAllocationScore =
            [&](llvm::ArrayRef<Config> allocation) -> int64_t {
          int64_t task_floor = 0, area_time = 0;
          for (const Config &config : allocation) {
            task_floor = std::max(task_floor, config.latency);
            area_time += config.latency * (int64_t)config.cells;
          }
          return std::max<int64_t>(
              task_floor, llvm::divideCeil(area_time, (int64_t)kTotalCGRAs));
        };
        SmallVector<std::pair<int64_t, std::vector<Config>>> candidates;

        // Two per-task rules, both scored with the real objective.
        //
        // Minimising `lat * area` is exact for `max(task_floor,
        // resource_floor)`, because that is a separable sum. It is NOT exact
        // once the objective also contains the pipeline cycle, which depends on
        // how the whole allocation packs. Rather than pick a rule and hope,
        // both are enumerated at every latency target and the objective
        // decides. On cnn_tiled the area rule alone returned an allocation the
        // objective scored 154550 while the hill-climb reached 60342.
        for (int rule = 0; rule < 2; ++rule)
          for (int64_t latency_target : targets) {
            int64_t total_cells = 0;
            std::vector<Config> picked;
            bool ok = true;
            for (int i = 0; i < task_count && ok; ++i) {
              const Config *pick = nullptr;
              for (const Config &config : configs[i]) {
                if (config.latency > latency_target)
                  continue;
                // Which config is "best at this latency target" depends on the
                // objective, and getting this wrong is what made the exact
                // allocator lose to the greedy one:
                //
                //  pipeline: the objective is max(max_t lat, ceil(sum_t
                //    lat*area / 16)). Given the target latency_target, the task floor is
                //    already satisfied by lat <= latency_target, so the exact choice per
                //    task is the one minimising its own lat*area contribution
                //    to the resource floor -- a separable sum, so termwise
                //    minimisation is exact. Minimising raw AREA instead (the
                //    old rule) picks cnn's (k=1,t=8) at 806 over the affordable
                //    (k=2,t=8) at 422.
                //
                //  interval / makespan: min-max under an area budget, where the
                //    cheapest config meeting the target is the right pick.
                bool better;
                if (rule == 0) {
                  const int64_t cand_area_time =
                      config.latency * (int64_t)config.cells;
                  const int64_t best_area_time =
                      pick ? pick->latency * (int64_t)pick->cells : 0;
                  better = !pick || cand_area_time < best_area_time ||
                           (cand_area_time == best_area_time &&
                            config.latency < pick->latency);
                } else {
                  better = !pick || config.cells < pick->cells ||
                           (config.cells == pick->cells && config.latency < pick->latency);
                }
                if (better)
                  pick = &config;
              }
              if (!pick)
                ok = false;
              else {
                picked.push_back(*pick);
                total_cells += pick->cells;
              }
            }
            if (!ok || total_cells > grid_budget)
              continue;
            candidates.push_back({cheapAllocationScore(picked),
                                  std::move(picked)});
          }
        llvm::sort(candidates, [](const auto &lhs, const auto &rhs) {
          return lhs.first < rhs.first;
        });
        const int verify_limit = verifyTopK.getValue();
        const int to_verify =
            verify_limit <= 0
                ? (int)candidates.size()
                : std::min<int>(verify_limit, (int)candidates.size());
        llvm::errs() << "  [OptimalFission] " << candidates.size()
                     << " feasible allocations, verifying top " << to_verify
                     << "\n";
        for (int candidate = 0; candidate < to_verify; ++candidate) {
          installAndProfile(candidates[candidate].second);
          const int64_t objective_value = graph.objective();
          if (objective_value < best_objective) {
            best_objective = objective_value;
            best_picked = candidates[candidate].second;
          }
        }

        // The incumbent is a candidate too. The sweep only visits one
        // allocation per (rule, target), so nothing guarantees it dominates the
        // allocation the graph arrived with. It is re-profiled like any other
        // candidate: `graph.objective()` reads each node's `ii`, so scoring the
        // incumbent on the II left behind by the last candidate compares two
        // different allocations.
        SmallVector<Config> incumbent;
        for (int i = 0; i < task_count; ++i) {
          Config config;
          std::tie(config.cgra_count, config.replicas, config.tiling) =
              saved_state[i];
          incumbent.push_back(config);
        }
        installAndProfile(incumbent);
        if (best_picked.empty() || graph.objective() <= best_objective)
          return false;

        bool changed = false;
        for (int i = 0; i < task_count; ++i) {
          TaskGraphNode *node = graph.nodes[i].get();
          const Config &config = best_picked[i];
          if (node->cgra_count != config.cgra_count || node->replicas != config.replicas ||
              node->tiling != config.tiling)
            changed = true;
          node->cgra_count = config.cgra_count;
          node->replicas = config.replicas;
          node->tiling = config.tiling;
          graph.profileWithBestShape(node, node->op, /*skip_mapper=*/true);
          llvm::errs() << "  [OptimalFission] " << node->op.getTaskName()
                       << " -> cgra_count=" << config.cgra_count << ", replicas=" << config.replicas
                       << ", tiling=" << config.tiling << ", cores=" << config.cells
                       << ", est_latency=" << node->estimatedLatency() << "\n";
        }
        return changed;
      };
      // Writes the decision space (and the DAG it is scored on) so an external
      // solver can be held to exactly this problem. Emitted before the
      // allocation runs, because profiling mutates the nodes, and only on the
      // first outer iteration: later iterations see a graph fusion has already
      // rewritten, so writing every time left the file describing a different
      // problem from the one the comment promises, at the cost of a full
      // re-enumeration each round.
      if (!dumpConfigSpace.getValue().empty() && outer == 0)
        writeConfigSpace(graph, enumerateConfigSpace(graph),
                         dumpConfigSpace.getValue());

      PipelineBalancer balancer;
      configureBalancer(balancer);
      // Only the committing balance gets it. The speculative call sites have no
      // `real_interval_fn` and fall back to the one-axis climb, where a
      // complete lowering per candidate would cost more than the lookahead it
      // informs.
      balancer.verify_profile_fn = balance_verify_fn;
      bool balance_changed = false;
      if (!importAllocation.getValue().empty()) {
        balance_changed =
            applyImportedAllocation(graph, importAllocation.getValue());
      } else if (!jointSearchEnabled()) {
        // Shipped path: the per-task rule alone, exactly as before.
        balance_changed = balancer.balance(graph, balance_profile_fn);
      } else {
        // Score a shortlist of complete allocations, keep the best.
        //
        // This one is not ranked-then-verified, unlike the fusion-pair,
        // allocation, shape and per-move shortlists. Those enumerate candidates
        // that are cheap to describe and expensive to evaluate, which is what
        // makes a cheap ranking worth having. Here each candidate IS the output
        // of a whole search, so generating it is the expensive part and there
        // is nothing to rank before paying for it; scoring the four that come
        // out costs four objective evaluations.
        //
        // Each strategy is a hill-climb on its own surface and each has a
        // failure mode the others do not: the per-task rule cannot see the
        // interval, the joint climb can settle in a worse basin than the
        // per-task rule reaches, and the exact step is exact only for the two
        // floors. Patching the order does not fix this. What fixes it is
        // running them all and letting ONE objective choose, which costs a
        // handful of extra profiles because they share the profile cache.
        //
        // On bicg the per-task rule reaches an allocation scoring 9730 while
        // the joint climb reaches 13950; before this, the pass shipped the
        // 13950 one because it was the last strategy to run.
        auto snapshot = [&]() {
          SmallVector<std::tuple<int, int, int>> a;
          for (auto &n : graph.nodes)
            a.emplace_back(n->cgra_count, std::max(1, n->replicas),
                           std::max(1, n->tiling));
          return a;
        };
        // One yardstick for the whole portfolio, and it is the same one the
        // joint search uses inside a strategy.
        //
        // It was two. A strategy chose among ITS candidates on the verified
        // interval -- lower the shortlist, measure it through orchestration and
        // the analysis -- and then the portfolio chose among STRATEGIES on the
        // analytical objective of an analytically re-profiled graph. The same
        // allocation therefore had two scores, and on bicg they disagreed by
        // 8x: the joint climb adopted tiling 2, 4 and 8, measuring 8195, 4099,
        // 2051, 1027, and the portfolio scored that same allocation at 8195 and
        // discarded it for the strategy that had not moved at all. The emitted
        // IR carried tiling=1. A search whose selection rule changes between
        // levels is not selecting on one objective, which is the property this
        // design exists to have.
        const bool measured_portfolio = jointSearchEnabled() && current_func &&
                                        (bool)balancer.verify_profile_fn;
        auto restore = [&](const SmallVector<std::tuple<int, int, int>> &a) {
          for (auto [i, n] : llvm::enumerate(graph.nodes)) {
            std::tie(n->cgra_count, n->replicas, n->tiling) = a[i];
            // Verified when the portfolio is measured, so the II the interval
            // is derived from is the solver's, exactly as inside a strategy.
            // The profile cache makes this a handful of extra solves: the
            // strategies revisit the same (task, shape) pairs.
            graph.profileWithBestShape(n.get(), n->op,
                                       /*skip_mapper=*/!measured_portfolio);
          }
        };
        // The score a candidate allocation is judged on.
        auto scoreOf = [&]() -> int64_t {
          if (!measured_portfolio)
            return graph.objective();
          writeDecisionAttributes(graph);
          const int64_t measured = realPipelineIntervalOf(current_func);
          return measured > 0 ? measured : graph.objective();
        };

        const auto initial = snapshot();
        SmallVector<std::tuple<int, int, int>> best = initial;
        int64_t best_obj = scoreOf();

        auto consider =
            [&](const SmallVector<std::tuple<int, int, int>> &cand) {
              restore(cand);
              const int64_t obj = scoreOf();
              if (obj < best_obj) {
                best_obj = obj;
                best = cand;
              }
            };

        // 1. The shipped strategy: per-task acceptance AND only the tile-array
        //    move. Leaving the DLP moves on here does not reproduce it -- under
        //    a per-task criterion, dividing a task's own iteration space always
        //    looks cheaper than growing its array, so the walk never grows
        //    `cgra_count` at all and the shipped allocation is never among the
        //    candidates. That is why bicg kept shipping 13950 with a 9730
        //    allocation available.
        const bool saved_repl = balancer.enable_replicas;
        const bool saved_tile = balancer.enable_tiling;
        balancer.joint_criterion = false;
        balancer.enable_replicas = false;
        balancer.enable_tiling = false;
        balancer.balance(graph, balance_profile_fn);
        balancer.enable_replicas = saved_repl;
        balancer.enable_tiling = saved_tile;
        consider(snapshot());

        // 2. The joint climb, from scratch.
        restore(initial);
        balancer.joint_criterion = true;
        balancer.balance(graph, balance_profile_fn);
        consider(snapshot());

        // 3. The exact allocation over the per-task config space, then a joint
        //    climb from it.
        if (optimalFission.getValue()) {
          restore(initial);
          if (optimalFissionAllocate(graph)) {
            consider(snapshot());
            balancer.balance(graph, balance_profile_fn);
            consider(snapshot());
          }
        }

        restore(best);
        balance_changed = (best != initial);
      }

      // Record this round's allocation if it is the best the search has seen,
      // and otherwise fall back to the best one before anything is written.
      {
        const int64_t obj = graph.objective();
        if (obj < best_seen_obj) {
          best_seen_obj = obj;
          best_seen.clear();
          for (auto &node : graph.nodes)
            best_seen[node->op.getTaskName()] = {node->cgra_count,
                                                 std::max(1, node->replicas),
                                                 std::max(1, node->tiling)};
        } else if (!best_seen.empty() && obj > best_seen_obj) {
          for (auto &node : graph.nodes) {
            auto it = best_seen.find(node->op.getTaskName());
            if (it == best_seen.end())
              continue;
            std::tie(node->cgra_count, node->replicas, node->tiling) =
                it->second;
            graph.profileWithBestShape(node.get(), node->op,
                                       /*skip_mapper=*/true);
          }
          balance_changed = true;
        }
      }

      if (!dumpAllocation.getValue().empty())
        writeAllocation(graph, dumpAllocation.getValue());

      // Writes back attributes so the next iteration sees them.
      if (balance_changed || fuse_changed) {
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          node->op->setAttr("cgra_count",
                            b.getI32IntegerAttr(node->cgra_count));
          if (emitsNewAttrs()) {
            node->op->setAttr("replicas", b.getI32IntegerAttr(node->replicas));
            node->op->setAttr("tiling", b.getI32IntegerAttr(node->tiling));
          }
          if (node->ii != kUnprofiled) {
            node->op->setAttr("compiled_ii", b.getI32IntegerAttr(node->ii));
          }
          if (node->steps != kUnprofiled) {
            SmallVector<NamedAttribute, 1> profile_attrs;
            profile_attrs.push_back(
                NamedAttribute(StringAttr::get(b.getContext(), "duration"),
                               b.getI32IntegerAttr(node->steps)));
            node->op->setAttr(
                "profile_info",
                DictionaryAttr::get(b.getContext(), profile_attrs));
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

      // Finalisation runs when the search converges, and also on the last
      // outer iteration if it never does.
      //
      // Materialisation used to sit behind convergence alone. A search that
      // still had a move to make on iteration 10 therefore left the loop with
      // `replicas` and `tiling` written as attributes but never applied, while
      // `est_latency` already reflected the division they promise. Everything
      // downstream then measured a program in which each task does half its
      // work and the other half exists nowhere: fft_staged reported an interval
      // of 192 against a true 12652.
      const bool converged = !balance_changed && !fuse_changed;
      const bool last_iteration = (outer == kMaxOuterIterations - 1);
      if (converged || last_iteration) {
        if (!converged)
          llvm::errs() << "[ResourceAware] Outer loop hit its iteration limit; "
                       << "materialising the current allocation.\n";
        // Converged — optionally re-profile with the real mapper for accuracy.
        // When balance-skip-mapper=false, runs the mapper once per task to get
        // an authoritative compiled_ii/steps for the allocation that was
        // chosen.
        //
        // `use_analytical` used to force this off, which conflated two
        // different questions: whether the SEARCH consults the mapper, and
        // whether the RESULT is verified by it. Conflating them makes the
        // configuration this cost model exists for -- search analytically, map
        // once at the end -- impossible to express, and therefore impossible to
        // time against a baseline that does map. An explicit
        // `balance-skip-mapper=false` now wins; the default is unchanged,
        // because balanceSkipMapper defaults to true.
        bool balance_skip_mapper =
            use_analytical && balanceSkipMapper.getValue();
        if (!balance_skip_mapper) {
          llvm::errs() << "[ResourceAware] Running final mapper verification "
                       << "for converged allocation...\n";
          for (auto &node : graph.nodes) {
            // This is the one place a real lowering is worth spending, so it is
            // the one place the shortlist is cashed in: rank every legal shape
            // by the closed form, then run the actual mapper on the top k and
            // keep whichever it measures fastest.
            //
            // It used to map exactly one shape -- whatever the analytical
            // search had settled on -- which made `search-shape` a decision
            // taken entirely on the model. Ranking is still free and still
            // covers every candidate; only k of them are lowered.
            if (!searchShape.getValue()) {
              node->shape = pickBestShape(node->cgra_count);
              graph.profileTask(node.get(), node->op, /*skip_mapper=*/false);
              continue;
            }
            graph.profileWithBestShape(node.get(), node->op,
                                       /*skip_mapper=*/false);
          }
        }

        // Writes ALL attributes (cgra_count, ii, steps) to IR for every task.
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          // Do NOT overwrite a searched shape here: this assignment used to
          // reset cgra_shape to the geometric pick unconditionally, which made
          // search-shape decide nothing no matter what it found. The area check
          // keeps shape and cgra_count consistent for the downstream placer.
          if (!searchShape.getValue() ||
              node->shape.area() < node->cgra_count ||
              (node->shape.is_rectangular &&
               node->shape.area() != node->cgra_count))
            node->shape = pickBestShape(node->cgra_count);
          node->op->setAttr("cgra_count",
                            b.getI32IntegerAttr(node->cgra_count));
          if (emitsNewAttrs()) {
            node->op->setAttr("replicas", b.getI32IntegerAttr(node->replicas));
            node->op->setAttr("tiling", b.getI32IntegerAttr(node->tiling));
          }
          node->op->setAttr("compiled_ii", b.getI32IntegerAttr(node->ii));
          {
            SmallVector<NamedAttribute, 1> profile_attrs;
            profile_attrs.push_back(
                NamedAttribute(StringAttr::get(b.getContext(), "duration"),
                               b.getI32IntegerAttr(node->steps)));
            node->op->setAttr(
                "profile_info",
                DictionaryAttr::get(b.getContext(), profile_attrs));
          }
          node->op->setAttr("trip_count",
                            b.getI32IntegerAttr(node->trip_count));
          // The downstream orchestrator occupies a CGRA for `profile_info.
          // duration`, which is the DFG depth — independent of how many
          // iterations the task runs. Publish the execution latency so the
          // spatial-temporal scheduler can charge the real residency.
          if (emitsNewAttrs())
            node->op->setAttr("est_latency",
                              b.getI64IntegerAttr(node->estimatedLatency()));
          // Writes cgra_shape attribute: simple "NxM" bounding-box string.
          // The detailed occupancy diagram is printed in the summary below.
          std::string shape_str = node->shape.irAttr();
          node->op->setAttr("cgra_shape", b.getStringAttr(shape_str));
        }

        // Materialises the two DLP decisions. Up to here `replicas` and
        // `tiling` are only annotations; a decision that never reaches the IR
        // is not a decision. Tiling rewrites the loop bounds into separate
        // tasks; replication emits the partition table the runtime needs.
        if (applyDecisions.getValue()) {
          int64_t predicted = graph.objective();
          if (commWeight.getValue() > 0.0)
            graph.explainPipelineInterval(llvm::errs(), "pre-apply");
          int group_id = 0, n_tiled = 0, n_replicated = 0, n_refused = 0;
          // Snapshot first: tile() erases task ops, invalidating graph nodes.
          struct Decision {
            TaskflowTaskOp op;
            int tiling, replicas, cgra_count;
            int64_t ii, steps;
          };
          MaterialiseCounts counts =
              materialiseDecisions(graph, group_id, /*quiet=*/false);
          n_tiled = counts.tiled;
          n_replicated = counts.replicated;
          n_refused = counts.refused;
          // Re-measures on the REWRITTEN IR rather than trusting the search's
          // own estimate, so predicted and materialised can be compared.
          TaskDependencyGraph applied;
          configureGraph(applied);
          applied.build(func, use_analytical);
          // The verification above ran on the graph BEFORE these decisions were
          // materialised, so it profiled a task that tiling was about to cut
          // into `tiling` pieces. That is the wrong body to certify: bicg's
          // tasks are 404 ops before the cut, which needs ResMII 26 against a
          // `ctrl_mem_items` ceiling of 20, so the solver correctly answers
          // "not mappable at this shape" for a task that is never emitted at
          // that size. Re-verifying here puts the certified II on the bodies
          // the IR actually carries.
          //
          // It runs only for the exact solver. The heuristic mapper is not
          // worth a second pass over every emitted task, and this path exists
          // because the certified number is the one that gets published.
          // `rescore-with=cpsat` is a MEASUREMENT, not a decision. Every choice
          // above has already been made and materialised; this only re-reads
          // the result with the exact solver.
          //
          // It exists because the arms are otherwise measured with different
          // instruments. The shipped allocator estimates II with the heuristic
          // mapper, which runs well above what the solver certifies, so a raw
          // baseline-vs-ours interval ratio contains the baseline's estimator
          // error as well as its decisions. Giving the baseline the solver as
          // its SEARCH estimator would be worse -- that is a compiler nobody
          // ships. Letting each arm search exactly as it does and then reading
          // both outcomes with one instrument is what makes the two numbers
          // comparable.
          const bool rescore =
              applied.verifier_is_cpsat || rescoreWith.getValue() == "cpsat";
          if (rescore) {
            const bool saved = applied.verifier_is_cpsat;
            applied.verifier_is_cpsat = true;
            llvm::errs() << "[ResourceAware] Re-scoring " <<
                applied.nodes.size() << " materialised tasks with the exact "
                "solver (measurement only; no decision changes)...\n";
            for (auto &node : applied.nodes)
              applied.profileTask(node.get(), node->op, /*skip_mapper=*/false);
            applied.verifier_is_cpsat = saved;
          }
          // Refreshes the derived attributes on the rewritten task list: the
          // tiles inherited the pre-cut trip_count / est_latency from their
          // parent when they were cloned.
          for (auto &node : applied.nodes) {
            OpBuilder b(node->op);
            node->op->setAttr("trip_count",
                              b.getI32IntegerAttr(node->trip_count));
            node->op->setAttr("est_latency",
                              b.getI64IntegerAttr(node->estimatedLatency()));
          }
          if (commWeight.getValue() > 0.0)
            applied.explainPipelineInterval(llvm::errs(), "post-apply");
          llvm::errs() << "[Apply] tiled=" << n_tiled
                       << " groups, replicated=" << n_replicated
                       << " tasks, refused=" << n_refused << "; tasks "
                       << graph.nodes.size() << "->" << applied.nodes.size()
                       << ", objective predicted=" << predicted
                       << " materialised=" << applied.objective() << "\n";
          if (reportRealInterval.getValue())
            llvm::errs() << "[Apply] real pipeline interval="
                         << realPipelineIntervalOf(func) << "\n";
        }
        break;
      }
    }

    // Performs final validation and tile occupation summary with visual 4x4
    // grid.
    {
      TaskDependencyGraph final_graph;
      // Configured like every other graph in this pass: the makespan and the
      // schedule_start/schedule_finish attributes reported below are computed
      // from it, and an unconfigured graph reports them with comm_weight = 0
      // even when the run priced communication.
      configureGraph(final_graph);
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
        // Reads back the shape that was actually committed to the IR, so the
        // occupancy summary reflects the search result rather than re-deriving
        // the geometric pick.
        CgraShape shape = pickBestShape(node->cgra_count);
        if (auto sa = node->op->getAttrOfType<StringAttr>("cgra_shape")) {
          StringRef s = sa.getValue();
          auto x = s.find('x');
          if (x != StringRef::npos) {
            int r = 0, c = 0;
            if (!s.take_front(x).getAsInteger(10, r) &&
                !s.drop_front(x + 1)
                     .take_while(llvm::isDigit)
                     .getAsInteger(10, c) &&
                r > 0 && c > 0 && s.find('[') == StringRef::npos)
              shape = {r, c, true, {}};
          }
        }
        int tile_rows = shape.rows * neura::getArchitecture().getPerCgraRows();
        int tile_cols =
            shape.cols * neura::getArchitecture().getPerCgraColumns();

        // Per-task grid (shape.rows x shape.cols bbox, filled up to
        // cgra_count).
        llvm::errs() << "\n  [" << task_idx << "] " << node->op.getTaskName()
                     << "  cgra_count=" << node->cgra_count
                     << "  shape=" << shape.describe(node->cgra_count)
                     << "  tile_array=" << tile_rows << "x" << tile_cols
                     << "  ii=" << node->effectiveII()
                     << "  steps=" << node->steps
                     << "  trip_count=" << node->trip_count
                     << "  replicas=" << node->replicas
                     << "  tiling=" << node->tiling
                     << "  eff_trip=" << node->effectiveTripCount()
                     << "  est_latency=" << node->estimatedLatency() << "\n";

        // Draws a per-task bounding-box grid (shape.rows x shape.cols).
        int remaining = node->allocatedCgras();
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

      // Emits the resource-constrained schedule. This is what the temporal knob
      // actually decides, so it is written to the IR (schedule_start /
      // schedule_finish per task) and printed, instead of being summarised as a
      // context-wave multiplier nobody can check.
      {
        SmallVector<TaskDependencyGraph::ScheduleItem> items;
        int64_t ms = final_graph.makespan(&items);
        DenseMap<const TaskGraphNode *, int64_t> first_start, last_finish;
        for (auto &it : items) {
          auto fs = first_start.find(it.node);
          if (fs == first_start.end() || it.start < fs->second)
            first_start[it.node] = it.start;
          auto lf = last_finish.find(it.node);
          if (lf == last_finish.end() || it.finish > lf->second)
            last_finish[it.node] = it.finish;
        }
        if (emitsNewAttrs()) {
          for (auto &node : final_graph.nodes) {
            OpBuilder b(node->op);
            node->op->setAttr("schedule_start",
                              b.getI64IntegerAttr(first_start[node.get()]));
            node->op->setAttr("schedule_finish",
                              b.getI64IntegerAttr(last_finish[node.get()]));
          }
        }
        int64_t interval = 0, work = 0;
        for (auto &node : final_graph.nodes) {
          interval = std::max(interval, node->estimatedLatency());
          work += node->estimatedLatency() *
                  (int64_t)std::max(1, node->footprintCgras());
        }
        double util = ms > 0 ? (double)work / ((double)ms * kTotalCGRAs) : 0.0;
        llvm::errs() << "[ResourceAware] Schedule: makespan=" << ms
                     << ", interval=" << interval
                     << ", grid_utilisation=" << llvm::format("%.3f", util)
                     << "\n";
      }
      // A program with more tasks than grid cells does not fit spatially; the
      // orchestrate pass time-multiplexes the surplus via temporal context
      // waves. So >kTotalCGRAs is a legitimate multi-wave case, not an error.
      if (final_total > kTotalCGRAs)
        llvm::errs() << "[ResourceAware] Note: " << final_total << " CGRAs > "
                     << kTotalCGRAs
                     << " grid cells; temporal reuse (context "
                        "waves) will time-multiplex the surplus.\n";
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

//===----------------------------------------------------------------------===//
// Shared fusion entry point (see TaskFusionUtil.h)
//===----------------------------------------------------------------------===//
//
// Reuses the file-local UtilizationFuser / TaskDependencyGraph -- the exact
// tested merge path used by the resource-aware pass, dominance-safety guard
// included -- so external-decision replay (e.g. --import-joint-mapping) never
// re-implements the DFG merge. Defined here (not in a separate TU) precisely so
// it can name the anonymous-namespace fusion classes above.
mlir::FailureOr<mlir::taskflow::TaskflowTaskOp>
mlir::taskflow::fuseTaskGroup(func::FuncOp func,
                              llvm::ArrayRef<std::string> task_names,
                              bool analytical, std::string &err) {
  if (task_names.size() < 2) {
    err = "fuseTaskGroup requires at least 2 tasks in the group";
    return failure();
  }

  // Locates a TaskGraphNode by task name in a freshly built graph.
  auto findNode = [](TaskDependencyGraph &g,
                     llvm::StringRef name) -> TaskGraphNode * {
    for (auto &n : g.nodes)
      if (n->op.getTaskName() == name)
        return n.get();
    return nullptr;
  };

  // Pairwise chain: current = t0; for i in 1..N-1: current = fuse(current, ti).
  // performFusion names the result "<earlier>_<later>_utilfused" (earlier =
  // whichever op comes first in the block); we recompute that name so the next
  // iteration can find the just-created fused task after the graph rebuild.
  std::string current_name = task_names[0];
  for (size_t i = 1; i < task_names.size(); ++i) {
    llvm::StringRef next_name = task_names[i];

    // Rebuild the dependency graph from the CURRENT IR state each step (the
    // previous fusion erased two tasks and created one).
    TaskDependencyGraph graph;
    graph.build(func, /*skip_mapper=*/analytical);

    TaskGraphNode *na = findNode(graph, current_name);
    TaskGraphNode *nb = findNode(graph, next_name);
    if (!na) {
      err = "fuseTaskGroup: task '" + current_name + "' not found in function";
      return failure();
    }
    if (!nb) {
      err =
          "fuseTaskGroup: task '" + next_name.str() + "' not found in function";
      return failure();
    }
    if (na == nb) {
      err = "fuseTaskGroup: task '" + current_name +
            "' appears more than once in the group";
      return failure();
    }

    UtilizationFuser fuser;
    // Same legality gate as resource-aware: independence + single-block bodies
    // + the SSA dominance-safety guard. Rejecting here (instead of forcing the
    // merge) is what prevents "operand does not dominate this use".
    if (!fuser.canFuse(na, nb, graph)) {
      err = "fuseTaskGroup: cannot fuse '" + current_name + "' + '" +
            next_name.str() +
            "' (tasks are dependent, have multi-block bodies, or the merge "
            "would be dominance-unsafe)";
      return failure();
    }

    // Recompute the name performFusion will assign to the merged task.
    Operation *opa = na->op.getOperation();
    Operation *opb = nb->op.getOperation();
    std::string earlier = na->op.getTaskName().str();
    std::string later = nb->op.getTaskName().str();
    if (!opa->isBeforeInBlock(opb))
      std::swap(earlier, later);
    std::string fused_name = earlier + "_" + later + "_utilfused";

    auto profile_fn = [&graph, analytical](TaskGraphNode *n, TaskflowTaskOp t) {
      graph.profileTask(n, t, /*skip_mapper=*/analytical);
    };
    if (!fuser.fuseNodes(func, na, nb, graph, profile_fn)) {
      err = "fuseTaskGroup: fusion failed for '" + current_name + "' + '" +
            next_name.str() + "'";
      return failure();
    }
    current_name = fused_name;
  }

  // Return the final fused task (located by its computed name).
  TaskflowTaskOp result;
  func.walk([&](TaskflowTaskOp t) {
    if (t.getTaskName() == current_name)
      result = t;
  });
  if (!result) {
    err = "fuseTaskGroup: fused task '" + current_name +
          "' not found after fusion";
    return failure();
  }
  return result;
}
