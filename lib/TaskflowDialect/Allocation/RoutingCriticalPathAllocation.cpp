//===- RoutingCriticalPathAllocation.cpp - Routing-critical-path-first ----===//
//
// Implements the RoutingCriticalPathAllocation strategy.  The core algorithm
// lives in the TaskMapper class (internal to this file); runAllocation()
// instantiates it and delegates to TaskMapper::place().
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/Allocation/RoutingCriticalPathAllocation.h"
#include "TaskflowDialect/Allocation/allocation_utils.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <optional>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// CGRA Grid Position
//===----------------------------------------------------------------------===//
/// Represents a position on the 2D CGRA grid.
struct CgraPosition {
  int row;
  int col;

  bool operator==(const CgraPosition &other) const {
    return row == other.row && col == other.col;
  }

  bool operator!=(const CgraPosition &other) const { return !(*this == other); }

  int manhattanDistance(const CgraPosition &other) const {
    return std::abs(row - other.row) + std::abs(col - other.col);
  }

  /// Returns true if the two positions are directly adjacent (Manhattan
  /// distance == 1), i.e. share an edge on the grid.
  bool isAdjacent(const CgraPosition &other) const {
    return manhattanDistance(other) == 1;
  }
};

//===----------------------------------------------------------------------===//
// Task Placement Info
//===----------------------------------------------------------------------===//
/// Stores the placement result for a task: the set of CGRAs assigned to it.
/// A task can span one or more contiguous CGRAs (rectangular or non-rect).
struct TaskPlacement {
  SmallVector<CgraPosition> cgra_positions; // CGRAs assigned to this task.

  /// Returns the primary (first) CGRA position.
  CgraPosition primary() const {
    return cgra_positions.empty() ? CgraPosition{-1, -1} : cgra_positions[0];
  }

  /// Returns the number of CGRAs assigned to this task.
  size_t cgraCount() const { return cgra_positions.size(); }

  /// Returns true if any CGRA in this task is grid-adjacent to any CGRA
  /// in `other`, indicating that direct data forwarding between tasks is
  /// possible without going through the network.
  bool hasTaskAdjacentCgra(const TaskPlacement &other) const {
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

/// Represents a Task node in the dependency graph.
struct TaskNode {
  size_t id;
  TaskflowTaskOp op;
  int dependency_depth = 0; // Longest path to any sink in the dependency graph.

  // Edges based on original (pre-streaming-fusion) memory accesses.
  SmallVector<MemoryNode *> read_memrefs;  // MemoryNodes this task reads.
  SmallVector<MemoryNode *> write_memrefs; // MemoryNodes this task writes.
  // SSA value edges between tasks.
  SmallVector<TaskNode *> ssa_users; // Tasks that consume this task's output.
  SmallVector<TaskNode *>
      ssa_operands; // Tasks whose output this task consumes.

  // Placement result.
  SmallVector<CgraPosition> placement;

  TaskNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}
};

/// Represents a MemRef node in the dependency graph.
struct MemoryNode {
  Value memref;

  // Access edges.
  SmallVector<TaskNode *> readers; // Tasks that read this memref.
  SmallVector<TaskNode *> writers; // Tasks that write this memref.

  // SRAM assignment result — populated by TaskMapper::assignAllSrams().
  std::optional<CgraPosition> assigned_sram_pos;

  MemoryNode(Value memref) : memref(memref) {}
};

class TaskMemoryGraph {
public:
  SmallVector<std::unique_ptr<TaskNode>> task_nodes;
  SmallVector<std::unique_ptr<MemoryNode>> memory_nodes;
  DenseMap<Value, MemoryNode *> memref_to_node;
  DenseMap<Operation *, TaskNode *> op_to_node;

  void build(func::FuncOp func) {
    // Phase 1: Creates a TaskNode for every TaskflowTaskOp in the function.
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskNode>(task_id++, task);
      op_to_node[task] = node.get();
      task_nodes.push_back(std::move(node));
    });

    // Phase 2: Creates MemoryNodes using ORIGINAL memrefs (canonical identity).
    // Uses original_read_memrefs / original_write_memrefs so that aliased
    // memories (created by streaming-fusion) share the same MemoryNode.
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

    // Phase 3: Build SSA edges (inter-task value dependencies).
    // A consumer task directly uses a value produced by a producer task.
    for (auto &consumer_node : task_nodes) {
      // Iterates all operands to be safe (not only getValueInputs()).
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
/// Maps a task-memory graph onto a 2D multi-CGRA grid using
/// routing-critical-path-first ordering.
///
/// Uses a two-phase fixed-point iteration:
///   Phase 1: Place tasks on the grid (scoring by SSA + memory proximity),
///            processing tasks in routing-critical-path-first order.
///   Phase 2: Assign each MemRef to the nearest SRAM given task positions.
/// Iterates until SRAM assignments converge.
class TaskMapper {
public:
  TaskMapper(int grid_rows, int grid_cols)
      : grid_rows_(grid_rows), grid_cols_(grid_cols) {
    occupied_.resize(grid_rows_);
    for (auto &row : occupied_) {
      row.resize(grid_cols_, false);
    }
  }

  /// Places all tasks and performs iterative SRAM assignment for `func`.
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

    // Computes dependency depth for each task.
    // Dependency depth = longest path from this node to any sink node in the
    // dependency graph (via SSA or memory edges).  Tasks with higher depth
    // have longer dependent chains after them; placing them first gives their
    // successors the best chance of landing on adjacent grid cells.
    computeDependencyDepth(graph);

    // Sorts tasks by dependency depth (routing-critical-path-first).
    SmallVector<TaskNode *> sorted_tasks;
    for (auto &node : graph.task_nodes)
      sorted_tasks.push_back(node.get());

    std::stable_sort(sorted_tasks.begin(), sorted_tasks.end(),
                     [](TaskNode *a, TaskNode *b) {
                       return a->dependency_depth > b->dependency_depth;
                     });

    // Fixed-point iteration: task placement scoring depends on SRAM
    // positions (memory proximity), and SRAM assignment depends on task
    // positions (centroid of accessing tasks).  Each iteration re-places
    // all tasks using the latest SRAM assignments, then re-assigns SRAMs.
    // On iteration 0, all SRAM positions are unset (no initial random
    // distribution), so the first task placement is driven purely by SSA
    // proximity; SRAM influence is introduced from iteration 1 onwards.
    // Converges when SRAM assignments stabilise (no change between iters).
    constexpr int kMaxIterations = 10;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
      if (iter > 0) {
        resetTaskPlacements(graph);
      }

      // Phase 1: Place tasks (scoring uses current SRAM assignments).
      for (TaskNode *task_node : sorted_tasks) {
        int cgra_count = 1;
        if (auto attr =
                task_node->op->getAttrOfType<IntegerAttr>("cgra_count")) {
          cgra_count = attr.getInt();
        }

        TaskPlacement placement =
            findBestPlacement(task_node, cgra_count, graph);

        assert(!placement.cgra_positions.empty() &&
               "findBestPlacement must succeed: cgra_count should be "
               "validated by the upstream resource-aware optimization pass "
               "or manually assigned resource binding attributes");

        // Commits placement and marks occupied grid cells.
        for (const auto &pos : placement.cgra_positions) {
          task_node->placement.push_back(pos);
        }

        for (const auto &pos : placement.cgra_positions) {
          if (pos.row >= 0 && pos.row < grid_rows_ && pos.col >= 0 &&
              pos.col < grid_cols_) {
            occupied_[pos.row][pos.col] = true;
          }
        }
      }

      // Phase 2: Assign SRAMs (assuming fixed task positions).
      // If nothing moved, task scores won't change -> convergence reached.
      bool sram_moved = assignAllSrams(graph);

      if (iter > 0 && !sram_moved) {
        break;
      }
    }

    // Annotates result: writes task_allocation_info attribute to each task op.
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

      // 2. Read SRAM locations.
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

      // 3. Write SRAM locations.
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

      // Sets task_allocation_info attribute on the task op.
      task_node->op->setAttr(
          "task_allocation_info",
          DictionaryAttr::get(func.getContext(), mapping_attrs));

      // Removes upstream resource-binding attributes that have now been
      // consumed by allocation. Downstream passes should read the
      // task_allocation_info attribute instead.
      task_node->op->removeAttr("cgra_count");
      task_node->op->removeAttr("cgra_shape");
    }
  }

private:
  /// Clears all task placements and resets the occupied-cell grid.
  void resetTaskPlacements(TaskMemoryGraph &graph) {
    for (auto &task : graph.task_nodes) {
      task->placement.clear();
    }
    // Clears grid.
    for (int r = 0; r < grid_rows_; ++r) {
      std::fill(occupied_[r].begin(), occupied_[r].end(), false);
    }
  }

  /// Assigns each MemoryNode to the SRAM at the centroid of all CGRAs that
  /// access it (readers + writers).  Returns true if any assignment changed,
  /// which is used as the convergence criterion for the outer iteration loop.
  bool assignAllSrams(TaskMemoryGraph &graph) {
    bool changed = false;
    for (auto &mem_node : graph.memory_nodes) {
      int total_row = 0, total_col = 0, count = 0;
      // Computes centroid of all tasks that read this memory.
      for (TaskNode *reader : mem_node->readers) {
        for (const CgraPosition &pos : reader->placement) {
          total_row += pos.row;
          total_col += pos.col;
          count++;
        }
      }
      // Computes centroid of all tasks that write this memory.
      for (TaskNode *writer : mem_node->writers) {
        for (const CgraPosition &pos : writer->placement) {
          total_row += pos.row;
          total_col += pos.col;
          count++;
        }
      }

      std::optional<CgraPosition> new_sram_pos;
      if (count > 0) {
        // Rounds to the nearest integer (round-half-up).
        int avg_row = (total_row + count / 2) / count;
        int avg_col = (total_col + count / 2) / count;
        new_sram_pos = CgraPosition{avg_row, avg_col};
      }

      if (mem_node->assigned_sram_pos != new_sram_pos) {
        mem_node->assigned_sram_pos = new_sram_pos;
        changed = true;
      }
    }
    return changed;
  }

  // Finds the best placement for `task_node` on the 2D multi-CGRA grid.
  //
  // The `cgra_shape` attribute set by the upstream
  // ResourceAwareTaskOptimization pass determines which base shape to use. This
  // function enumerates all rotations of that shape and picks the
  // highest-scoring free grid position across all rotations.
  //
  // If no `cgra_shape` attribute is present (e.g. cgra_count == 1 or the task
  // was not processed by the resource-binding pass), falls back to enumerating
  // all rectangular factorizations of cgra_count, then a polyomino DFS.
  //
  // Returns an empty TaskPlacement only if no valid position exists on the grid
  // (should not happen if cgra_count was validated upstream).
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph) {
    // Reads the pre-determined base shape from the upstream pass.
    // getAllPlacementShapes() enumerates all valid rectangular factorizations.
    SmallVector<CgraShape> shapes_to_try;
    if (auto attr = task_node->op->getAttrOfType<StringAttr>("cgra_shape")) {
      StringRef cgra_shape_str = attr.getValue();
      if (!cgra_shape_str.empty()) {
        CgraShape base = parseCgraShapeToBase(cgra_shape_str, cgra_count);
        shapes_to_try = rotationsOf(base);
      }
    }

    // Fallback: no cgra_shape attribute, enumerates all valid shapes.
    if (shapes_to_try.empty()) {
      shapes_to_try = getAllPlacementShapes(cgra_count);
    }

    int best_score = INT_MIN;
    TaskPlacement best_placement;

    for (const CgraShape &shape : shapes_to_try) {
      SmallVector<std::pair<int, int>> shape_offsets;
      if (shape.is_rectangular) {
        for (int r = 0; r < shape.rows; ++r)
          for (int c = 0; c < shape.cols; ++c)
            shape_offsets.push_back({c, r});
      } else {
        shape_offsets = SmallVector<std::pair<int, int>>(
            shape.cgra_positions.begin(), shape.cgra_positions.end());
      }

      for (int origin_row = 0; origin_row < grid_rows_; ++origin_row) {
        for (int origin_col = 0; origin_col < grid_cols_; ++origin_col) {
          // Checks that every cell of the rectangle is within bounds and free.
          bool valid = true;
          TaskPlacement candidate;
          for (auto &[col_off, row_off] : shape_offsets) {
            int abs_row = origin_row + row_off;
            int abs_col = origin_col + col_off;
            if (abs_row < 0 || abs_row >= grid_rows_ || abs_col < 0 ||
                abs_col >= grid_cols_ || occupied_[abs_row][abs_col]) {
              valid = false;
              break;
            }
            candidate.cgra_positions.push_back({abs_row, abs_col});
          }
          if (!valid) {
            continue;
          }
          // Scores the candidate by proximity to dependent tasks and SRAMs.
          int score = computeScore(task_node, candidate, graph);
          if (score > best_score) {
            best_score = score;
            best_placement = candidate;
          }
        }
      }
    }
    return best_placement;
  }

  // Parses a `cgra_shape` IR attribute string into a base CgraShape.
  // Accepts "NxM" (rectangular) and "NxM[(c0,r0)(c1,r1)...]" (non-rectangular).
  CgraShape parseCgraShapeToBase(StringRef cgra_shape, int cgra_count) {
    size_t bracket_pos = cgra_shape.find('[');
    auto [rows_str, rest] = cgra_shape.split('x');
    int rows = 1, cols = 1;
    rows_str.getAsInteger(10, rows);

    if (bracket_pos == StringRef::npos) {
      // Rectangular: "NxM"
      rest.getAsInteger(10, cols);
      return CgraShape{rows, cols, /*is_rectangular=*/true, {}};
    }

    // Non-rectangular: "NxM[(c0,r0)(c1,r1)...]"
    StringRef cols_str = rest.take_until([](char c) { return c == '['; });
    cols_str.getAsInteger(10, cols);

    SmallVector<std::pair<int, int>> positions;
    StringRef positions_str = cgra_shape.substr(bracket_pos);
    size_t pos = 0;
    while (pos < positions_str.size()) {
      size_t open = positions_str.find('(', pos);
      if (open == StringRef::npos)
        break;
      size_t close = positions_str.find(')', open);
      if (close == StringRef::npos)
        break;
      StringRef pair_str = positions_str.slice(open + 1, close);
      auto [col_str, row_str] = pair_str.split(',');
      int col_off = 0, row_off = 0;
      col_str.getAsInteger(10, col_off);
      row_str.getAsInteger(10, row_off);
      positions.push_back({col_off, row_off});
      pos = close + 1;
    }
    return CgraShape{rows, cols, /*is_rectangular=*/false,
                     std::move(positions)};
  }

  // Generates all unique rotations of `base` as CgraShapes.
  // Rectangular shapes produce both orientations (rows×cols and cols×rows).
  // Non-rectangular shapes produce up to four 90° rotations, deduplicated.
  SmallVector<CgraShape> rotationsOf(const CgraShape &base) {
    SmallVector<CgraShape> result;

    if (base.is_rectangular) {
      result.push_back(base);
      if (base.rows != base.cols)
        result.push_back(CgraShape{base.cols, base.rows, true, {}});
      return result;
    }

    // Non-rectangular: generate 4 × 90° CW rotations, deduplicate by hash.
    // Rotation formula in (col, row) space: (col, row) -> (row, -col).
    llvm::DenseSet<int64_t> seen_hashes;
    auto current_positions = SmallVector<std::pair<int, int>>(
        base.cgra_positions.begin(), base.cgra_positions.end());

    for (int rotation_count = 0; rotation_count < 4; ++rotation_count) {
      // Normalises to non-negative offsets starting from (0, 0).
      int min_col = INT_MAX, min_row = INT_MAX;
      for (auto &[col, row] : current_positions) {
        min_col = std::min(min_col, col);
        min_row = std::min(min_row, row);
      }

      SmallVector<std::pair<int, int>> normalised_positions;
      for (auto &[col, row] : current_positions)
        normalised_positions.push_back({col - min_col, row - min_row});

      // Hashes to deduplicate.
      auto sorted_positions = normalised_positions;
      llvm::sort(sorted_positions,
                 [](const std::pair<int, int> &a,
                    const std::pair<int, int> &b) { return a < b; });

      int64_t position_hash = 0;
      for (auto &[col, row] : sorted_positions)
        position_hash = position_hash * 131 + col * 17 + row;

      if (seen_hashes.insert(position_hash).second) {
        int max_col = 0, max_row = 0;
        for (auto &[col, row] : normalised_positions) {
          max_col = std::max(max_col, col);
          max_row = std::max(max_row, row);
        }
        result.push_back(
            CgraShape{max_row + 1, max_col + 1, false, normalised_positions});
      }

      // Applies 90° CW rotation: (col, row) -> (row, -col).
      SmallVector<std::pair<int, int>> rotated_positions;
      for (auto &[col, row] : current_positions)
        rotated_positions.push_back({row, -col});
      current_positions = rotated_positions;
    }
    return result;
  }

  /// Computes the placement score for `task_node` at `placement`.
  ///
  /// Score = α·SSA_Dist + β·Mem_Dist.
  ///   SSA_Dist : sum of distances to already-placed SSA predecessors and
  ///              successors (negative; penalises far-away neighbours).
  ///   Mem_Dist : sum of distances to assigned SRAMs for read/write memrefs
  ///              (negative; memory proximity is weighted more heavily).
  ///
  /// Higher score is better; 0 means all neighbours are co-located.
  int computeScore(TaskNode *task_node, const TaskPlacement &placement,
                   TaskMemoryGraph &graph) {
    // Weight constants (tunable).
    constexpr int kAlpha = 10; // SSA proximity weight.
    constexpr int kBeta = 50;  // Memory proximity weight (high priority).

    int ssa_score = 0;
    int mem_score = 0;

    auto minDistToPlacement =
        [&](const SmallVector<CgraPosition> &other) -> int {
      int min_dist = INT_MAX;
      for (const auto &pos : placement.cgra_positions) {
        for (const auto &opos : other) {
          min_dist = std::min(min_dist, pos.manhattanDistance(opos));
        }
      }
      return min_dist;
    };

    auto minDistToTarget = [&](const CgraPosition &target) -> int {
      int min_dist = INT_MAX;
      for (const auto &pos : placement.cgra_positions) {
        min_dist = std::min(min_dist, pos.manhattanDistance(target));
      }
      return min_dist;
    };

    // 1. SSA proximity — penalise distance to producers and consumers.
    for (TaskNode *producer : task_node->ssa_operands) {
      if (!producer->placement.empty()) {
        // Uses negative distance: closer = higher score.
        int dist = minDistToPlacement(producer->placement);
        ssa_score -= dist;
      }
    }
    for (TaskNode *consumer : task_node->ssa_users) {
      if (!consumer->placement.empty()) {
        int dist = minDistToPlacement(consumer->placement);
        ssa_score -= dist;
      }
    }

    // 2. Memory proximity — penalise distance to assigned SRAMs.
    // For read memrefs (data sources).
    for (MemoryNode *mem : task_node->read_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = minDistToTarget(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }
    // For write memrefs: if the SRAM is already assigned (e.g. read by a
    // previous task), we want to be close to it too.
    for (MemoryNode *mem : task_node->write_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = minDistToTarget(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }

    return kAlpha * ssa_score + kBeta * mem_score;
  }

  /// Computes dependency depth for every task in the graph.
  ///
  /// Routing Critical Path: the longest chain of dependent tasks in the
  /// task graph, where each edge represents either an SSA value dependency or
  /// a memory dependency (RAW, WAR, WAW) between tasks.  This is the
  /// allocation analogue of the critical path in scheduling: the chain of
  /// tasks that constrains the minimum total inter-CGRA communication
  /// distance.
  ///
  /// How we identify it: for every task node we compute the "dependency
  /// depth": the longest path (in edges) from that node to any sink in the
  /// graph, traversing SSA edges (producer -> consumer) and all memory
  /// dependency edges.  The task with the greatest depth lies at the head
  /// of the routing critical path.
  ///
  /// Why it matters: tasks on the routing critical path
  /// have the most downstream dependents.  Placing them first
  /// (routing-critical-path-first ordering) ensures that:
  ///   1. They receive priority access to good grid positions.
  ///   2. Their dependent tasks can later be placed adjacent, minimising
  ///      inter-task communication distance on the critical path.
  void computeDependencyDepth(TaskMemoryGraph &graph) {
    DenseMap<TaskNode *, int> depth_cache;
    DenseSet<TaskNode *> visiting; // Tracks nodes on the current DFS path.
    for (auto &node : graph.task_nodes) {
      node->dependency_depth =
          calculateDepth(node.get(), depth_cache, visiting);
    }
  }

  /// Recursively calculates dependency depth for a single task (memoised).
  ///
  /// Traverses both SSA edges and all three kinds of memory dependencies:
  ///   - RAW (Read-After-Write): a writer's reader depends on the write.
  ///   - WAR (Write-After-Read): a reader's subsequent writer depends on
  ///     the read completing first.
  ///   - WAW (Write-After-Write): two writers to the same memref must be
  ///     ordered.
  ///
  /// This pass must remain modular and not assume any upstream pass (e.g.
  /// streaming-fusion) has already transformed certain dependency patterns;
  /// therefore all memory dependency types are considered.
  ///
  /// Memory dependency edges can form cycles (e.g. RAW A→B and WAR B→A for
  /// the same memref).  The `visiting` set detects back-edges and breaks
  /// cycles by treating them as depth 0.
  int calculateDepth(TaskNode *node, DenseMap<TaskNode *, int> &depth_cache,
                     DenseSet<TaskNode *> &visiting) {
    if (depth_cache.count(node)) {
      return depth_cache[node];
    }
    // Cycle detection: if this node is already on the current DFS path,
    // return 0 to break the cycle.
    if (!visiting.insert(node).second) {
      return 0;
    }

    int max_child_depth = 0;
    // SSA dependencies: tasks that consume this task's output values.
    for (TaskNode *child : node->ssa_users) {
      max_child_depth = std::max(
          max_child_depth, calculateDepth(child, depth_cache, visiting) + 1);
    }

    // Memory dependencies — all three types:
    //
    // RAW (Read-After-Write): this task writes a memref, downstream tasks
    // read the same memref and depend on this write.
    for (MemoryNode *mem : node->write_memrefs) {
      for (TaskNode *reader : mem->readers) {
        if (reader != node) {
          max_child_depth =
              std::max(max_child_depth,
                       calculateDepth(reader, depth_cache, visiting) + 1);
        }
      }
    }

    // WAR (Write-After-Read): this task reads a memref, downstream tasks
    // that write the same memref must wait for the read to complete.
    for (MemoryNode *mem : node->read_memrefs) {
      for (TaskNode *writer : mem->writers) {
        if (writer != node) {
          max_child_depth =
              std::max(max_child_depth,
                       calculateDepth(writer, depth_cache, visiting) + 1);
        }
      }
    }

    // WAW (Write-After-Write): this task writes a memref, other tasks that
    // also write to the same memref have an ordering dependency.
    // Example:
    //   %out0 = task0 write_memref(%mem0) origin_write_memref(%mem0)
    //   %out1 = task1 write_memref(%out0) origin_write_memref(%mem0)
    // task1 can only execute after task0 because both write to %mem0.
    for (MemoryNode *mem : node->write_memrefs) {
      for (TaskNode *other_writer : mem->writers) {
        if (other_writer != node) {
          max_child_depth =
              std::max(max_child_depth,
                       calculateDepth(other_writer, depth_cache, visiting) + 1);
        }
      }
    }

    visiting.erase(node);
    return depth_cache[node] = max_child_depth;
  }

  int grid_rows_;
  int grid_cols_;
  std::vector<std::vector<bool>> occupied_;
};

} // namespace

//===----------------------------------------------------------------------===//
// RoutingCriticalPathAllocation::runAllocation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace taskflow {

bool RoutingCriticalPathAllocation::runAllocation(func::FuncOp func) {
  TaskMapper mapper(grid_rows_, grid_cols_);
  mapper.place(func);
  return true;
}

} // namespace taskflow
} // namespace mlir
