//===- MapTaskOnCgraPass.cpp - Task to CGRA Mapping Pass ----------------===//
//
// This pass maps Taskflow tasks onto a 2D CGRA grid array:
// 1. Places tasks with SSA dependencies on adjacent CGRAs.
// 2. Assigns memrefs to SRAMs (each MemRef is assigned to exactly one SRAM,
//    determined by proximity to the task that first accesses it).
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// CGRA Grid Position
//===----------------------------------------------------------------------===//
/// Represents a position on the 2D CGRA grid.
struct CGRAPosition {
  int row;
  int col;

  bool operator==(const CGRAPosition &other) const {
    return row == other.row && col == other.col;
  }

  bool operator!=(const CGRAPosition &other) const { return !(*this == other); }

  /// Computes Manhattan distance to another position.
  int manhattanDistance(const CGRAPosition &other) const {
    return std::abs(row - other.row) + std::abs(col - other.col);
  }

  /// Checks if adjacent (Manhattan distance = 1).
  bool isAdjacent(const CGRAPosition &other) const {
    return manhattanDistance(other) == 1;
  }
};

//===----------------------------------------------------------------------===//
// Task Placement Info
//===----------------------------------------------------------------------===//
/// Stores placement info for a task: can span multiple combined CGRAs.
struct TaskPlacement {
  SmallVector<CGRAPosition> cgra_positions; // CGRAs assigned to this task.

  /// Returns the primary (first) position.
  CGRAPosition primary() const {
    return cgra_positions.empty() ? CGRAPosition{-1, -1} : cgra_positions[0];
  }

  /// Returns the number of CGRAs assigned.
  size_t cgraCount() const { return cgra_positions.size(); }

  /// Checks if any CGRA in this task is adjacent to any in other task.
  bool hasAdjacentCGRA(const TaskPlacement &other) const {
    for (const auto &pos : cgra_positions) {
      for (const auto &other_pos : other.cgra_positions) {
        if (pos.isAdjacent(other_pos)) {
          return true;
        }
      }
    }
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Task-Memory Graph
//===----------------------------------------------------------------------===//

struct MemoryNode;

/// Represents a Task node in the graph.
struct TaskNode {
  size_t id;
  TaskflowTaskOp op;
  int dependency_depth = 0; // Longest path to any sink in the dependency graph.

  // Edges based on original memory access.
  SmallVector<MemoryNode *> read_memrefs;  // Original read memrefs.
  SmallVector<MemoryNode *> write_memrefs; // Original write memrefs.
  SmallVector<TaskNode *> ssa_users;
  SmallVector<TaskNode *> ssa_operands;

  // Placement result
  SmallVector<CGRAPosition> placement;

  TaskNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}
};

/// Represents a Memory node (MemRef) in the graph.
struct MemoryNode {
  Value memref;

  // Edges.
  SmallVector<TaskNode *> readers;
  SmallVector<TaskNode *> writers;

  // Mapping result.
  std::optional<CGRAPosition> assigned_sram_pos;

  MemoryNode(Value memref) : memref(memref) {}
};

/// The Task-Memory Dependency Graph.
class TaskMemoryGraph {
public:
  SmallVector<std::unique_ptr<TaskNode>> task_nodes;
  SmallVector<std::unique_ptr<MemoryNode>> memory_nodes;
  DenseMap<Value, MemoryNode *> memref_to_node;
  DenseMap<Operation *, TaskNode *> op_to_node;

  void build(func::FuncOp func) {
    // 1. Creates TaskNodes.
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskNode>(task_id++, task);
      op_to_node[task] = node.get();
      task_nodes.push_back(std::move(node));
    });

    // 2. Creates MemoryNodes using ORIGINAL memrefs (canonical identity).
    // Uses original_read_memrefs/original_write_memrefs to ensure aliased
    // memories share the same MemoryNode.
    for (auto &t_node : task_nodes) {
      // Uses original_read_memrefs for canonical memory identity.
      for (Value orig_memref : t_node->op.getOriginalReadMemrefs()) {
        MemoryNode *m_node = getOrCreateMemoryNode(orig_memref);
        t_node->read_memrefs.push_back(m_node);
        m_node->readers.push_back(t_node.get());
      }

      // Uses original_write_memrefs for canonical memory identity.
      for (Value orig_memref : t_node->op.getOriginalWriteMemrefs()) {
        MemoryNode *m_node = getOrCreateMemoryNode(orig_memref);
        t_node->write_memrefs.push_back(m_node);
        m_node->writers.push_back(t_node.get());
      }
    }

    // 3. Builds SSA Edges (Inter-Task Value Dependencies).
    // Identifies if a task uses a value produced by another task.
    for (auto &consumer_node : task_nodes) {
      // Iterates all operands for now to be safe.
      for (Value operand : consumer_node->op.getValueInputs()) {
        if (auto producer_op = operand.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer_node = op_to_node[producer_op]) {
            producer_node->ssa_users.push_back(consumer_node.get());
            consumer_node->ssa_operands.push_back(producer_node);
          }
        }
      }
    }
  }

private:
  MemoryNode *getOrCreateMemoryNode(Value memref) {
    if (memref_to_node.count(memref)) {
      return memref_to_node[memref];
    }

    auto node = std::make_unique<MemoryNode>(memref);
    MemoryNode *ptr = node.get();
    memref_to_node[memref] = ptr;
    memory_nodes.push_back(std::move(node));
    return ptr;
  }
};

//===----------------------------------------------------------------------===//
// Task Mapper
//===----------------------------------------------------------------------===//
/// Maps a task-memory graph onto a 2D CGRA grid.

class TaskMapper {
public:
  TaskMapper(int grid_rows, int grid_cols)
      : grid_rows_(grid_rows), grid_cols_(grid_cols) {
    occupied_.resize(grid_rows_);
    for (auto &row : occupied_) {
      row.resize(grid_cols_, false);
    }
  }

  /// Places all tasks and performs memory mapping.
  void place(func::FuncOp func) {
    SmallVector<TaskflowTaskOp> tasks;
    func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

    if (tasks.empty()) {
      llvm::errs() << "No tasks to place.\n";
      return;
    }

    // Builds Task-Memory Graph.
    TaskMemoryGraph graph;
    graph.build(func);

    if (graph.task_nodes.empty()) {
      llvm::errs() << "No tasks to place.\n";
      return;
    }

    // Computes Dependency Depth for each task.
    // Dependency depth = longest path from this node to any sink in the
    // dependency graph (considering both SSA and memory edges). Tasks with
    // higher depth are more "critical" and are placed first to ensure their
    // dependent chains have good locality.
    computeDependencyDepth(graph);

    // Sorts tasks by dependency depth (Critical Path First).
    SmallVector<TaskNode *> sorted_tasks;
    for (auto &node : graph.task_nodes)
      sorted_tasks.push_back(node.get());

    std::stable_sort(sorted_tasks.begin(), sorted_tasks.end(),
                     [](TaskNode *a, TaskNode *b) {
                       return a->dependency_depth > b->dependency_depth;
                     });

    // Critical-path-first placement:
    // 1. Computes dependency depth for each task (longest path to sink).
    // 2. Sorts tasks by dependency depth (higher = more critical).
    // 3. Places tasks in sorted order with heuristic scoring.
    // Iterative Refinement Loop (Coordinate Descent).
    // Alternates between Task Placement (Phase 1) and SRAM Assignment (Phase
    // 2).
    constexpr int kMaxIterations = 10;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
      // Phase 1: Place Tasks (assuming fixed SRAMs).
      if (iter > 0) {
        resetTaskPlacements(graph);
      }

      for (TaskNode *task_node : sorted_tasks) {
        int cgra_count = 1;
        if (auto attr =
                task_node->op->getAttrOfType<IntegerAttr>("cgra_count")) {
          cgra_count = attr.getInt();
        }

        // Finds best placement using SRAM positions from previous iter (or
        // -1/default).
        TaskPlacement placement =
            findBestPlacement(task_node, cgra_count, graph);

        // Commits Placement.
        task_node->placement.push_back(placement.primary());
        // Handles mapping one task on multi-CGRAs.
        // TODO: Introduce explicit multi-CGRA binding logic.
        for (size_t i = 1; i < placement.cgra_positions.size(); ++i) {
          task_node->placement.push_back(placement.cgra_positions[i]);
        }

        // Marks occupied.
        for (const auto &pos : placement.cgra_positions) {
          if (pos.row >= 0 && pos.row < grid_rows_ && pos.col >= 0 &&
              pos.col < grid_cols_) {
            occupied_[pos.row][pos.col] = true;
          }
        }
      }

      // Phase 2: Assign SRAMs (assuming fixed tasks).
      bool sram_moved = assignAllSRAMs(graph);

      // Convergence Check.
      // If SRAMs didn't move, it means task placement based on them likely
      // won't change either.
      if (iter > 0 && !sram_moved) {
        break;
      }
    }

    // Annotates result.
    OpBuilder builder(func.getContext());
    for (auto &task_node : graph.task_nodes) {
      if (task_node->placement.empty()) {
        continue;
      }

      SmallVector<NamedAttribute, 4> mapping_attrs;

      // 1. CGRA positions.
      SmallVector<Attribute> pos_attrs;
      for (const auto &pos : task_node->placement) {
        SmallVector<NamedAttribute, 2> coord_attrs;
        coord_attrs.push_back(
            NamedAttribute(StringAttr::get(func.getContext(), "row"),
                           builder.getI32IntegerAttr(pos.row)));
        coord_attrs.push_back(
            NamedAttribute(StringAttr::get(func.getContext(), "col"),
                           builder.getI32IntegerAttr(pos.col)));
        pos_attrs.push_back(
            DictionaryAttr::get(func.getContext(), coord_attrs));
      }
      mapping_attrs.push_back(
          NamedAttribute(StringAttr::get(func.getContext(), "cgra_positions"),
                         builder.getArrayAttr(pos_attrs)));

      // 2. Reads SRAM Locations.
      SmallVector<Attribute> read_sram_attrs;
      for (MemoryNode *mem : task_node->read_memrefs) {
        if (mem->assigned_sram_pos) {
          SmallVector<NamedAttribute, 2> sram_coord;
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "row"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "col"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));
          read_sram_attrs.push_back(
              DictionaryAttr::get(func.getContext(), sram_coord));
        }
      }
      mapping_attrs.push_back(NamedAttribute(
          StringAttr::get(func.getContext(), "read_sram_locations"),
          builder.getArrayAttr(read_sram_attrs)));

      // 3. Writes SRAM Locations.
      SmallVector<Attribute> write_sram_attrs;
      for (MemoryNode *mem : task_node->write_memrefs) {
        if (mem->assigned_sram_pos) {
          SmallVector<NamedAttribute, 2> sram_coord;
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "row"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "col"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));

          write_sram_attrs.push_back(
              DictionaryAttr::get(func.getContext(), sram_coord));
        }
      }
      mapping_attrs.push_back(NamedAttribute(
          StringAttr::get(func.getContext(), "write_sram_locations"),
          builder.getArrayAttr(write_sram_attrs)));

      // Sets Attribute.
      task_node->op->setAttr(
          "task_mapping_info",
          DictionaryAttr::get(func.getContext(), mapping_attrs));
    }
  }

private:
  /// Clears task placement and occupied grid.
  void resetTaskPlacements(TaskMemoryGraph &graph) {
    for (auto &task : graph.task_nodes) {
      task->placement.clear();
    }
    // Clears grid.
    for (int r = 0; r < grid_rows_; ++r) {
      std::fill(occupied_[r].begin(), occupied_[r].end(), false);
    }
  }

  /// Assigns all memory nodes to SRAMs based on centroid of accessing tasks.
  /// Returns true if any SRAM assignment changed.
  bool assignAllSRAMs(TaskMemoryGraph &graph) {
    bool changed = false;
    for (auto &mem_node : graph.memory_nodes) {
      // Computes centroid of all tasks that access this memory.
      int total_row = 0, total_col = 0, count = 0;
      for (TaskNode *reader : mem_node->readers) {
        if (!reader->placement.empty()) {
          total_row += reader->placement[0].row;
          total_col += reader->placement[0].col;
          count++;
        }
      }
      for (TaskNode *writer : mem_node->writers) {
        if (!writer->placement.empty()) {
          total_row += writer->placement[0].row;
          total_col += writer->placement[0].col;
          count++;
        }
      }

      std::optional<CGRAPosition> new_sram_pos;
      if (count > 0) {
        // Rounds to the nearest integer.
        int avg_row = (total_row + count / 2) / count;
        int avg_col = (total_col + count / 2) / count;
        new_sram_pos = CGRAPosition{avg_row, avg_col};
      }

      if (mem_node->assigned_sram_pos != new_sram_pos) {
        mem_node->assigned_sram_pos = new_sram_pos;
        changed = true;
      }
    }
    return changed;
  }

  /// Finds best placement for a task.
  /// TODO: Currently defaults to single-CGRA placement. Multi-CGRA binding
  /// logic (cgra_count > 1) is experimental/placeholder and should ideally be
  /// handled by an upstream resource binding pass.
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph) {
    int best_score = INT_MIN;
    TaskPlacement best_placement;

    // Baseline: For cgra_count=1, finds single best position.
    for (int r = 0; r < grid_rows_; ++r) {
      for (int c = 0; c < grid_cols_; ++c) {
        if (occupied_[r][c]) {
          continue;
        }

        TaskPlacement candidate;
        candidate.cgra_positions.push_back({r, c});

        int score = computeScore(task_node, candidate, graph);
        if (score > best_score) {
          best_score = score;
          best_placement = candidate;
        }
      }
    }

    // Error handling: No available position found (grid over-subscribed).
    if (best_placement.cgra_positions.empty()) {
      assert(false &&
             "No available CGRA position found (grid over-subscribed).");
    }

    return best_placement;
  }

  /// Computes placement score based on Task-Memory Graph.
  /// TODO: Introduce explicit 'direct_wires' attributes in the IR for
  /// downstream hardware generators to configure fast bypass paths between
  /// adjacent PEs with dependencies.
  ///
  /// Score = α·SSA_Dist + β·Mem_Dist.
  ///
  /// SSA_Dist: Minimize distance to placed SSA predecessors (ssa_operands).
  /// Mem_Dist: Minimize distance to assigned SRAMs for read/write memrefs.
  int computeScore(TaskNode *task_node, const TaskPlacement &placement,
                   TaskMemoryGraph &graph) {
    // Weight constants (tunable).
    constexpr int kAlpha = 10; // SSA proximity weight.
    constexpr int kBeta = 50;  // Memory proximity weight (high priority).

    int ssa_score = 0;
    int mem_score = 0;

    CGRAPosition current_pos = placement.primary();

    // 1. SSA proximity (predecessors & successors).
    for (TaskNode *producer : task_node->ssa_operands) {
      if (!producer->placement.empty()) {
        int dist = current_pos.manhattanDistance(producer->placement[0]);
        // Uses negative distance to penalize far-away placements.
        ssa_score -= dist;
      }
    }
    for (TaskNode *consumer : task_node->ssa_users) {
      if (!consumer->placement.empty()) {
        int dist = current_pos.manhattanDistance(consumer->placement[0]);
        ssa_score -= dist;
      }
    }

    // 2. Memory proximity.
    // For read memrefs.
    for (MemoryNode *mem : task_node->read_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = current_pos.manhattanDistance(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }
    // For write memrefs.
    // If we write to a memory that is already assigned (e.g. read by previous
    // task), we want to be close to it too.
    for (MemoryNode *mem : task_node->write_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = current_pos.manhattanDistance(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }

    return kAlpha * ssa_score + kBeta * mem_score;
  }

  /// Computes dependency depth for all tasks in the graph.
  ///
  /// Dependency depth = longest path from this node to any sink node in the
  /// dependency graph (via SSA or memory edges).
  ///
  /// Tasks with higher dependency depth have longer chains of dependent tasks
  /// after them. By placing these tasks first:
  /// 1. They get priority access to good grid positions.
  /// 2. Their dependent tasks can then be positioned adjacent to them,
  ///    minimizing inter-task communication distance.
  void computeDependencyDepth(TaskMemoryGraph &graph) {
    DenseMap<TaskNode *, int> depth_cache;
    for (auto &node : graph.task_nodes) {
      node->dependency_depth = calculateDepth(node.get(), depth_cache);
    }
  }

  /// Recursively calculates dependency depth for a single task.
  int calculateDepth(TaskNode *node, DenseMap<TaskNode *, int> &depth_cache) {
    if (depth_cache.count(node)) {
      return depth_cache[node];
    }

    int max_child_depth = 0;
    // SSA dependencies.
    for (TaskNode *child : node->ssa_users) {
      max_child_depth =
          std::max(max_child_depth, calculateDepth(child, depth_cache) + 1);
    }

    // Memory dependencies (Producer -> Mem -> Consumer).
    for (MemoryNode *mem : node->write_memrefs) {
      for (TaskNode *reader : mem->readers) {
        if (reader != node) {
          max_child_depth = std::max(max_child_depth,
                                     calculateDepth(reader, depth_cache) + 1);
        }
      }
    }

    return depth_cache[node] = max_child_depth;
  }

  int grid_rows_;
  int grid_cols_;
  std::vector<std::vector<bool>> occupied_;
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
struct MapTaskOnCgraPass
    : public PassWrapper<MapTaskOnCgraPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapTaskOnCgraPass)

  MapTaskOnCgraPass() = default;

  StringRef getArgument() const override { return "map-task-on-cgra"; }

  StringRef getDescription() const override {
    return "Maps Taskflow tasks onto a 2D CGRA grid with adjacency "
           "optimization and memory mapping.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    constexpr int kDefaultGridRows = 3;
    constexpr int kDefaultGridCols = 3;
    TaskMapper mapper(kDefaultGridRows, kDefaultGridCols);
    mapper.place(func);
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createMapTaskOnCgraPass() {
  return std::make_unique<MapTaskOnCgraPass>();
}

} // namespace taskflow
} // namespace mlir
