// Shared CGRA orchestration utilities.

#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using llvm::ArrayRef;
using llvm::SmallVector;

namespace mlir {
namespace taskflow {

// CgraShape member implementations

std::string CgraShape::describe(int cgra_count) const {
  std::string s = std::to_string(rows) + "x" + std::to_string(cols);
  if (!is_rectangular) {
    s += "(non-rect, " + std::to_string(cgra_count) + " CGRAs:";
    for (auto &[c, r] : cgra_positions)
      s += " (" + std::to_string(c) + "," + std::to_string(r) + ")";
    s += ")";
  }
  return s;
}

std::string CgraShape::irAttr() const {
  std::string s = std::to_string(rows) + "x" + std::to_string(cols);
  if (!is_rectangular && !cgra_positions.empty()) {
    s += "[";
    for (auto &[c, r] : cgra_positions)
      s += "(" + std::to_string(c) + "," + std::to_string(r) + ")";
    s += "]";
  }
  return s;
}

// Internal helpers

namespace {

// Returns the set of non-rectangular shapes for `cgra_count` CGRAs.
// Currently defined for cgra_count == 3 (L-shape) and cgra_count == 4
// (L-shape and T-shape variants).
SmallVector<CgraShape> getNonRectangularShapes(int cgra_count) {
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

} // namespace

// getAllPlacementShapes

SmallVector<CgraShape> getAllPlacementShapes(int cgra_count) {
  SmallVector<CgraShape> shapes;

  // 1. Rectangular shapes with both orientations, deduplicated.
  {
    llvm::DenseSet<int64_t> seen_keys; // encodes (rows<<16)|cols
    for (int row_dim = 1; row_dim <= kCgraGridRows; ++row_dim) {
      for (int col_dim = 1; col_dim <= kCgraGridCols; ++col_dim) {
        if (row_dim * col_dim == cgra_count) {
          int64_t key = ((int64_t)row_dim << 16) | col_dim;
          if (seen_keys.insert(key).second) {
            shapes.push_back({row_dim, col_dim, true, {}});
            // Adds the rotated orientation if different (e.g. 1×4 -> 4×1).
            if (row_dim != col_dim) {
              int64_t rotated_key = ((int64_t)col_dim << 16) | row_dim;
              if (seen_keys.insert(rotated_key).second) {
                shapes.push_back({col_dim, row_dim, true, {}});
              }
            }
          }
        }
      }
    }
    // Sorts rectangles: prefer more square-like (smaller |rows-cols|), then
    // smaller bounding-box area as tiebreaker.
    llvm::sort(shapes, [](const CgraShape &lhs, const CgraShape &rhs) {
      int squareness_lhs = std::abs(lhs.rows - lhs.cols);
      int squareness_rhs = std::abs(rhs.rows - rhs.cols);
      if (squareness_lhs != squareness_rhs)
        return squareness_lhs < squareness_rhs;
      return lhs.area() < rhs.area();
    });
  }

  // 2. Non-rectangular shapes with all four 90° rotations.
  auto base_non_rect = getNonRectangularShapes(cgra_count);
  for (const auto &base : base_non_rect) {
    // Generates 4 rotations of the cgra_positions list.
    // Rotation by 90° CW: (col, row) -> (row, -col).
    // Each rotation is normalised so that offsets start from (0, 0).
    SmallVector<SmallVector<std::pair<int, int>>, 4> rotation_variants;
    rotation_variants.push_back(
        SmallVector<std::pair<int, int>>(base.cgra_positions));

    auto prev_positions = base.cgra_positions;
    for (int rotation_idx = 0; rotation_idx < 3; ++rotation_idx) {
      SmallVector<std::pair<int, int>> rotated_positions;
      for (auto &[col_off, row_off] : prev_positions)
        rotated_positions.push_back(
            {row_off, -col_off}); // 90° CW in (col, row) space

      // Normalises to non-negative offsets starting from (0, 0).
      int min_col = INT_MAX, min_row = INT_MAX;
      for (auto &[col_off, row_off] : rotated_positions) {
        min_col = std::min(min_col, col_off);
        min_row = std::min(min_row, row_off);
      }
      for (auto &[col_off, row_off] : rotated_positions) {
        col_off -= min_col;
        row_off -= min_row;
      }
      rotation_variants.push_back(rotated_positions);
      prev_positions = rotated_positions;
    }

    // Deduplicates rotations that produce the same position set.
    // Hash parameters: multiplier 131 and positional weight 17 are chosen to
    // give low collision rates for small integer coordinate sets.
    llvm::DenseSet<int64_t> seen_hashes;
    for (auto &positions : rotation_variants) {
      auto sorted_positions = positions;
      llvm::sort(sorted_positions,
                 [](const std::pair<int, int> &lhs,
                    const std::pair<int, int> &rhs) { return lhs < rhs; });
      int64_t hash = 0;
      for (auto &[col_off, row_off] : sorted_positions)
        hash = hash * 131 + col_off * 17 + row_off;
      if (!seen_hashes.insert(hash).second) {
        continue;
      }
      // Computes bounding box for this rotation.
      int max_col = 0, max_row = 0;
      for (auto &[col_off, row_off] : positions) {
        max_col = std::max(max_col, col_off);
        max_row = std::max(max_row, row_off);
      }
      shapes.push_back({max_row + 1, max_col + 1, false, std::move(positions)});
    }
  }

  return shapes;
}

// canAllTasksFitOnGrid

bool canAllTasksFitOnGrid(ArrayRef<int> task_cgra_counts) {
  constexpr int kTotalCGRAs = kCgraGridRows * kCgraGridCols;

  // Quick capacity check: total CGRAs must not exceed grid size.
  int total_cgras = 0;
  for (int count : task_cgra_counts)
    total_cgras += count;
  if (total_cgras > kTotalCGRAs) {
    return false;
  }

  // Simulates placement on a grid.
  bool occupied[kCgraGridRows][kCgraGridCols] = {};

  // Sorts tasks by descending cgra_count for better packing (largest-first
  // decreasing, a standard bin-packing heuristic).  Each task may have a
  // different cgra_count because the balance phase only increments one
  // bottleneck at a time; this array reflects the heterogeneous orchestration
  // across all tasks in the current trial configuration.
  SmallVector<int> sorted_counts(task_cgra_counts.begin(),
                                 task_cgra_counts.end());
  llvm::sort(sorted_counts, [](int lhs, int rhs) { return lhs > rhs; });

  for (int cgra_count : sorted_counts) {
    SmallVector<CgraShape> candidates = getAllPlacementShapes(cgra_count);
    bool placed = false;

    for (const auto &shape : candidates) {
      if (placed)
        break;

      if (shape.is_rectangular) {
        // Rectangular: tries every origin where the rows×cols bbox fits.
        for (int origin_row = 0;
             origin_row <= kCgraGridRows - shape.rows && !placed;
             ++origin_row) {
          for (int origin_col = 0;
               origin_col <= kCgraGridCols - shape.cols && !placed;
               ++origin_col) {
            bool fits = true;
            for (int delta_row = 0; delta_row < shape.rows && fits; ++delta_row)
              for (int delta_col = 0; delta_col < shape.cols && fits;
                   ++delta_col)
                if (occupied[origin_row + delta_row][origin_col + delta_col])
                  fits = false;
            if (fits) {
              for (int delta_row = 0; delta_row < shape.rows; ++delta_row)
                for (int delta_col = 0; delta_col < shape.cols; ++delta_col)
                  occupied[origin_row + delta_row][origin_col + delta_col] =
                      true;
              placed = true;
            }
          }
        }
      } else {
        // Non-rectangular: cgra_positions stores (col, row) offsets.
        for (int origin_row = 0; origin_row < kCgraGridRows && !placed;
             ++origin_row) {
          for (int origin_col = 0; origin_col < kCgraGridCols && !placed;
               ++origin_col) {
            bool fits = true;
            for (auto &[col_off, row_off] : shape.cgra_positions) {
              int abs_row = origin_row + row_off;
              int abs_col = origin_col + col_off;
              if (abs_row < 0 || abs_row >= kCgraGridRows || abs_col < 0 ||
                  abs_col >= kCgraGridCols || occupied[abs_row][abs_col]) {
                fits = false;
                break;
              }
            }
            if (fits) {
              for (auto &[col_off, row_off] : shape.cgra_positions)
                occupied[origin_row + row_off][origin_col + col_off] = true;
              placed = true;
            }
          }
        }
      }
    }

    if (!placed) {
      return false;
    }
  }
  return true;
}

// Task scheduling utilities

// CGRA Grid Position (spatial + temporal)
// Represents a spatial-temporal orchestration for a task on the 2D CGRA grid.
//
// A task assigned to (row, col) occupies that CGRA for the half-open interval
// [start_time, start_time + duration).  Two tasks may share the same CGRA as
// long as their intervals do not overlap, enabling time-multiplexed reuse.
//
// start_time and duration are internal scheduling quantities; they are not
// written to the IR directly.  Instead the output attribute uses context_id —
// the 0-based index of this task in the sorted list of tasks assigned to the
// same (row, col) CGRA tile (sorted by start_time ascending).  context_id
// maps directly to the hardware context-memory index.
struct CgraPosition {
  int row;
  int col;
  int start_time = 0; // Internal scheduling; not emitted to IR.
  int duration = 1;   // Read from profile_info; not emitted to IR.
  int context_id = 0; // Emitted to IR as task_orchestration_info.

  bool operator==(const CgraPosition &other) const {
    return row == other.row && col == other.col;
  }

  bool operator!=(const CgraPosition &other) const { return !(*this == other); }

  int manhattanDistance(const CgraPosition &other) const {
    return std::abs(row - other.row) + std::abs(col - other.col);
  }

  // Returns true if the two positions are directly adjacent (Manhattan
  // distance == 1), i.e. share an edge on the grid.
  bool isAdjacent(const CgraPosition &other) const {
    return manhattanDistance(other) == 1;
  }
};

// Task Placement Info
// Stores the placement result for a task: the set of CGRAs assigned to it.
// A task can span one or more contiguous CGRAs (rectangular or non-rect).
struct TaskPlacement {
  SmallVector<CgraPosition> cgra_positions; // CGRAs assigned to this task.

  // Returns the primary (first) CGRA position.
  CgraPosition primary() const {
    return cgra_positions.empty() ? CgraPosition{-1, -1, 0, 0}
                                  : cgra_positions[0];
  }

  // Returns the number of CGRAs assigned to this task.
  size_t cgraCount() const { return cgra_positions.size(); }

  // Returns true if any CGRA in this task is grid-adjacent to any CGRA
  // in `other`, indicating that direct data forwarding between tasks is
  // possible without going through the network.
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

// Task-Memory Graph

struct MemoryNode;

// Represents a Task node in the dependency graph.
struct TaskNode {
  size_t id;
  TaskflowTaskOp op;

  // Edges based on original (pre-streaming-fusion) memory accesses.
  SmallVector<MemoryNode *> read_memrefs;  // MemoryNodes this task reads.
  SmallVector<MemoryNode *> write_memrefs; // MemoryNodes this task writes.
  // Explicit taskflow dependency edges between tasks, including value inputs
  // and dependency read/write token inputs.
  SmallVector<TaskNode *> ssa_users;    // Tasks that depend on this task.
  SmallVector<TaskNode *> ssa_operands; // Tasks this task depends on.

  // Placement result.
  SmallVector<CgraPosition> placement;

  TaskNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}

  // Returns the task's execution duration in time slots.
  //
  // Prefers `est_latency` = II*(trip_count-1) + steps, written by
  // ResourceAwareTaskOptimizationPass. A task holds its CGRA for its whole
  // execution, not just its pipeline depth, so `profile_info.duration` (the DFG
  // depth) understates residency by the entire iteration count — and, being
  // independent of trip_count, it makes every decision that changes the
  // iteration space (loop partitioning, replication) invisible to this
  // scheduler. Falls back to the depth, then to 1, when latency is absent.
  int getDuration() const {
    if (auto lat = op->getAttrOfType<IntegerAttr>("est_latency")) {
      int64_t v = lat.getInt();
      if (v > 0)
        return static_cast<int>(std::min<int64_t>(v, INT32_MAX));
    }
    if (auto profile = op->getAttrOfType<DictionaryAttr>("profile_info")) {
      if (auto dur = dyn_cast_or_null<IntegerAttr>(profile.get("duration"))) {
        return std::max(1, static_cast<int>(dur.getInt()));
      }
    }
    return 1;
  }

  // Replicas the grid could actually accommodate (<= getReplicas()).
  int replicas_placed = 1;

  // Number of data-parallel replicas of this task. Each replica is a separate
  // tile array running the SAME configuration on a disjoint data partition, so
  // they must all be resident at the same time.
  int getReplicas() const {
    if (auto attr = op->getAttrOfType<IntegerAttr>("replicas"))
      return std::max(1, static_cast<int>(attr.getInt()));
    return 1;
  }
};

// Represents a MemRef node in the dependency graph.
struct MemoryNode {
  Value memref;

  // Access edges.
  SmallVector<TaskNode *> readers; // Tasks that read this memref.
  SmallVector<TaskNode *> writers; // Tasks that write this memref.

  // SRAM assignment result, populated by TaskScheduler::assignAllSrams().
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

    // Phase 3: Build explicit task dependency edges. Keep scalar/value SSA
    // dependencies visible through value_inputs, and also scan all operands to
    // catch taskflow dependency_read_in/dependency_write_in token edges.
    for (auto &consumer_node : task_nodes) {
      for (Value value_input : consumer_node->op.getValueInputs()) {
        addProducerDependency(value_input, consumer_node.get());
      }

      for (Value operand : consumer_node->op->getOperands()) {
        addProducerDependency(operand, consumer_node.get());
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

  void addProducerDependency(Value operand, TaskNode *consumer) {
    if (auto producer_op = operand.getDefiningOp<TaskflowTaskOp>()) {
      if (auto *producer = op_to_node[producer_op]) {
        addDependencyEdge(producer, consumer);
      }
    }
  }

  void addDependencyEdge(TaskNode *producer, TaskNode *consumer) {
    if (producer == consumer) {
      return;
    }
    if (!llvm::is_contained(producer->ssa_users, consumer)) {
      producer->ssa_users.push_back(consumer);
    }
    if (!llvm::is_contained(consumer->ssa_operands, producer)) {
      consumer->ssa_operands.push_back(producer);
    }
  }
};

// TaskPipelineIntervalAnalyzer

TaskPipelineIntervalAnalyzer::TaskPipelineIntervalAnalyzer(
    ArrayRef<TaskScheduleResult> schedule_result)
    : schedule_result_(schedule_result) {}

TaskPipelineIntervalResult TaskPipelineIntervalAnalyzer::analyze() {
  TaskPipelineIntervalResult result;
  if (schedule_result_.empty()) {
    return result;
  }

  buildTaskIndex();
  task_graph_.resize(schedule_result_.size());
  buildDataDependenceEdges();
  buildCgraExecutionOrderEdgesAndPipelineCycles();
  return computeLongestPipelineCycle();
}

int TaskPipelineIntervalAnalyzer::getTaskDuration(int task_idx) const {
  return std::max(1, schedule_result_[task_idx].duration);
}

int64_t TaskPipelineIntervalAnalyzer::encodeCgraLocation(
    const TaskScheduleResult::CgraOccupancy &occupancy) const {
  return (static_cast<int64_t>(occupancy.row) << 32) |
         static_cast<uint32_t>(occupancy.col);
}

void TaskPipelineIntervalAnalyzer::addExecutionOrderEdge(int task,
                                                         int next_task) {
  if (task < 0 || next_task < 0 || task == next_task) {
    return;
  }
  task_graph_[task].push_back({next_task, getTaskDuration(task)});
}

void TaskPipelineIntervalAnalyzer::buildTaskIndex() {
  for (auto [idx, task_result] : llvm::enumerate(schedule_result_)) {
    TaskflowTaskOp task = task_result.task;
    task_to_index_[task.getOperation()] = static_cast<int>(idx);
  }
}

void TaskPipelineIntervalAnalyzer::buildDataDependenceEdges() {
  for (auto [task_idx, task_result] : llvm::enumerate(schedule_result_)) {
    for (TaskflowTaskOp pred : task_result.predecessor_tasks) {
      auto pred_it = task_to_index_.find(pred.getOperation());
      if (pred_it == task_to_index_.end()) {
        continue;
      }
      addExecutionOrderEdge(pred_it->second, static_cast<int>(task_idx));
    }
  }
}

void TaskPipelineIntervalAnalyzer::
    buildCgraExecutionOrderEdgesAndPipelineCycles() {
  DenseMap<int64_t, SmallVector<int>> cgra_location_to_tasks;
  for (auto [idx, task_result] : llvm::enumerate(schedule_result_)) {
    for (const TaskScheduleResult::CgraOccupancy &occupancy :
         task_result.cgra_occupancies) {
      cgra_location_to_tasks[encodeCgraLocation(occupancy)].push_back(
          static_cast<int>(idx));
    }
  }

  for (auto &entry : cgra_location_to_tasks) {
    SmallVector<int> &tasks = entry.second;
    llvm::sort(tasks, [&](int lhs, int rhs) {
      const TaskScheduleResult &lhs_result = schedule_result_[lhs];
      const TaskScheduleResult &rhs_result = schedule_result_[rhs];
      if (lhs_result.start_time != rhs_result.start_time) {
        return lhs_result.start_time < rhs_result.start_time;
      }
      return lhs < rhs;
    });

    for (size_t i = 1; i < tasks.size(); ++i) {
      addExecutionOrderEdge(tasks[i - 1], tasks[i]);
    }

    int first = tasks.front();
    int last = tasks.back();
    cgra_pipeline_cycles_.push_back({last, first, getTaskDuration(last)});
  }
}

TaskPipelineIntervalAnalyzer::LongestExecutionPath
TaskPipelineIntervalAnalyzer::findLongestPathToTarget(
    int current, int target, DenseSet<int> &visiting,
    DenseMap<int, LongestExecutionPath> &memo) const {
  if (current == target) {
    LongestExecutionPath result;
    result.found = true;
    result.path.push_back(current);
    return result;
  }

  if (visiting.contains(current)) {
    return LongestExecutionPath();
  }

  // Memoised on `current` for a fixed `target`. The plain recursion re-walks
  // every path through every diamond, which is exponential in the graph: once
  // the resource pass partitions a program into ~100 tasks with fan-out, this
  // analysis stops terminating (axpy_20 ran past 400s where placement itself
  // took 0.07s). `computeStartTimes` has already rejected any cycle by the time
  // this runs, so on a DAG the memo is exact, not an approximation.
  auto hit = memo.find(current);
  if (hit != memo.end()) {
    return hit->second;
  }

  visiting.insert(current);
  LongestExecutionPath best;
  for (const ExecutionOrderEdge &edge : task_graph_[current]) {
    LongestExecutionPath suffix =
        findLongestPathToTarget(edge.next_task, target, visiting, memo);
    if (!suffix.found) {
      continue;
    }

    int total_latency = edge.latency + suffix.total_latency;
    if (!best.found || total_latency > best.total_latency) {
      best.found = true;
      best.total_latency = total_latency;
      best.path.clear();
      best.path.push_back(current);
      best.path.append(suffix.path.begin(), suffix.path.end());
    }
  }
  visiting.erase(current);
  memo[current] = best;
  return best;
}

TaskPipelineIntervalResult
TaskPipelineIntervalAnalyzer::computeLongestPipelineCycle() const {
  TaskPipelineIntervalResult result;
  for (const CgraPipelineCycle &pipeline_cycle : cgra_pipeline_cycles_) {
    DenseSet<int> visiting;
    // One memo per target: the value cached is "longest path from `current` to
    // THIS cycle's last task", so it cannot be shared across cycles.
    DenseMap<int, LongestExecutionPath> memo;
    LongestExecutionPath path = findLongestPathToTarget(
        pipeline_cycle.first_task, pipeline_cycle.last_task, visiting, memo);
    if (!path.found) {
      continue;
    }

    int interval = path.total_latency + pipeline_cycle.latency;
    if (interval <= result.pipeline_interval) {
      continue;
    }

    result.pipeline_interval = interval;
    result.critical_path.clear();

    int bottleneck_idx = pipeline_cycle.last_task;
    int bottleneck_duration = getTaskDuration(bottleneck_idx);
    for (int idx : path.path) {
      const TaskScheduleResult &task_result = schedule_result_[idx];
      result.critical_path.push_back(task_result.task);
      int duration = getTaskDuration(idx);
      if (duration > bottleneck_duration) {
        bottleneck_idx = idx;
        bottleneck_duration = duration;
      }
    }
    result.bottleneck_task = schedule_result_[bottleneck_idx].task;
  }
  return result;
}

// TaskScheduler
// Orchestrates a task-memory graph onto a 2D multi-CGRA grid using the
// priority provided by the caller.
//
// Uses a two-phase fixed-point iteration:
//   Phase 1: Place tasks on the grid (scoring by SSA + memory proximity),
//            processing tasks in priority order.
//   Phase 2: Assign each MemRef to the nearest SRAM given task positions.
// Iterates until SRAM assignments converge.
//
// In SpatialTemporal mode, ASAP scheduling is applied via
// computeEarliestStartTime() so that each task starts as soon as all explicit
// taskflow dependencies have completed.
TaskScheduler::TaskScheduler(int grid_rows, int grid_cols, SchedulingMode mode)
    : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode) {
  cgra_occupancy_.resize(grid_rows_);
  for (auto &row : cgra_occupancy_) {
    row.resize(grid_cols_);
  }
}

// Schedules all tasks and performs iterative SRAM assignment for `func`.
bool TaskScheduler::schedule(func::FuncOp func,
                             const TaskPriorityMap &priority) {
  schedule_result_.clear();

  SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

  if (tasks.empty()) {
    llvm::errs() << "No tasks to place.\n";
    return true;
  }

  // Builds Task-Memory Graph.
  TaskMemoryGraph graph;
  graph.build(func);

  if (graph.task_nodes.empty()) {
    llvm::errs() << "No tasks to place.\n";
    return true;
  }

  // Sorts tasks by orchestration-provided priority. The scheduler does not
  // infer a critical path; orchestration algorithms provide that policy.
  SmallVector<TaskNode *> sorted_tasks;
  for (auto &node : graph.task_nodes) {
    sorted_tasks.push_back(node.get());
  }
  auto getPriority = [&](TaskNode *node) {
    auto it = priority.find(node->op.getOperation());
    return it == priority.end() ? 0 : it->second;
  };
  std::stable_sort(sorted_tasks.begin(), sorted_tasks.end(),
                   [&](TaskNode *a, TaskNode *b) {
                     int a_priority = getPriority(a);
                     int b_priority = getPriority(b);
                     if (a_priority != b_priority) {
                       return a_priority > b_priority;
                     }
                     return a->id < b->id;
                   });

  // Fixed-point iteration: placement scoring depends on SRAM positions, and
  // SRAM assignment depends on task positions.  Converges when SRAMs are
  // stable.  On iteration 0 SRAMs are unset, so placement is driven purely
  // by SSA proximity.
  constexpr int kMaxIterations = 10;

  // Stores the total number of tasks so findBestPlacement can compute a
  // sufficient time horizon even on very small grids where the multi-CGRA
  // grid area (grid_rows_ * grid_cols_) is much smaller than task_count
  // (e.g. 5 tasks on a 1x1 grid).
  total_task_count_ = static_cast<int>(sorted_tasks.size());

  for (int iter = 0; iter < kMaxIterations; ++iter) {
    if (iter > 0) {
      resetTaskPlacements(graph);
    }

    // Phase 1: Place tasks.
    for (TaskNode *task_node : sorted_tasks) {
      int cgra_count = 1;
      if (auto attr = task_node->op->getAttrOfType<IntegerAttr>("cgra_count")) {
        cgra_count = attr.getInt();
      }

      TaskPlacement placement = findBestPlacement(task_node, cgra_count, graph);

      assert(!placement.cgra_positions.empty() &&
             "findBestPlacement must succeed: cgra_count should be "
             "validated by the upstream resource-aware optimization pass "
             "or manually assigned resource binding attributes");

      for (const auto &pos : placement.cgra_positions) {
        task_node->placement.push_back(pos);
      }

      for (const auto &pos : placement.cgra_positions) {
        if (posInBounds(pos)) {
          markOccupied(pos.row, pos.col, pos.start_time, pos.duration);
        }
      }

      // Data-parallel replicas: each is another tile array of the same shape
      // running concurrently, so they are pinned to the first replica's slot.
      // A replica that does not fit is dropped and the achieved count is
      // recorded, because the grid — not the cost model — has the final say.
      int replicas = task_node->getReplicas();
      int placed = 1;
      if (replicas > 1) {
        int t0 = placement.cgra_positions.front().start_time;
        for (int r = 1; r < replicas; ++r) {
          TaskPlacement extra = findBestPlacement(task_node, cgra_count, graph,
                                                  /*force_start=*/t0);
          if (extra.cgra_positions.empty())
            break;
          for (const auto &pos : extra.cgra_positions) {
            task_node->placement.push_back(pos);
            if (posInBounds(pos))
              markOccupied(pos.row, pos.col, pos.start_time, pos.duration);
          }
          ++placed;
        }
      }
      task_node->replicas_placed = placed;
    }

    // Phase 2: Assign SRAMs.
    bool sram_moved = assignAllSrams(graph);
    if (iter > 0 && !sram_moved) {
      break;
    }
  }

  // Compute context_id for each task at each assigned CGRA cell.
  // For every physical CGRA (row, col), sort all tasks assigned to it by
  // their internal start_time, then assign context_id = 0, 1, 2, ...
  // This maps directly to the hardware context-memory index.
  using TaskInterval = std::pair<int, TaskNode *>; // (start_time, node)
  std::vector<std::vector<SmallVector<TaskInterval, 4>>> cell_tasks(
      grid_rows_, std::vector<SmallVector<TaskInterval, 4>>(grid_cols_));

  for (auto &task_node : graph.task_nodes) {
    for (CgraPosition &pos : task_node->placement) {
      if (posInBounds(pos)) {
        cell_tasks[pos.row][pos.col].push_back(
            {pos.start_time, task_node.get()});
      }
    }
  }

  for (int r = 0; r < grid_rows_; ++r) {
    for (int c = 0; c < grid_cols_; ++c) {
      auto &tasks_at_cell = cell_tasks[r][c];
      std::stable_sort(tasks_at_cell.begin(), tasks_at_cell.end(),
                       [](const TaskInterval &a, const TaskInterval &b) {
                         return a.first < b.first;
                       });
      for (int ctx = 0; ctx < static_cast<int>(tasks_at_cell.size()); ++ctx) {
        TaskNode *tn = tasks_at_cell[ctx].second;
        for (CgraPosition &pos : tn->placement) {
          if (pos.row == r && pos.col == c) {
            pos.context_id = ctx;
          }
        }
      }
    }
  }

  recordScheduleResult(graph);

  // Write output attributes.
  OpBuilder builder(func.getContext());
  for (auto &task_node : graph.task_nodes) {
    if (task_node->placement.empty()) {
      continue;
    }

    SmallVector<NamedAttribute, 4> mapping_attrs;

    // 1. CGRA positions.
    // Keys are in alphabetical order as required by DictionaryAttr:
    // col < context_id < row.
    SmallVector<Attribute> pos_attrs;
    for (const auto &pos : task_node->placement) {
      SmallVector<NamedAttribute, 3> coord_attrs;
      coord_attrs.push_back(
          NamedAttribute(StringAttr::get(func.getContext(), "col"),
                         builder.getI32IntegerAttr(pos.col)));
      coord_attrs.push_back(
          NamedAttribute(StringAttr::get(func.getContext(), "context_id"),
                         builder.getI32IntegerAttr(pos.context_id)));
      coord_attrs.push_back(
          NamedAttribute(StringAttr::get(func.getContext(), "row"),
                         builder.getI32IntegerAttr(pos.row)));
      pos_attrs.push_back(DictionaryAttr::get(func.getContext(), coord_attrs));
    }
    mapping_attrs.push_back(
        NamedAttribute(StringAttr::get(func.getContext(), "cgra_positions"),
                       builder.getArrayAttr(pos_attrs)));

    // 2. Reads SRAM locations.
    SmallVector<Attribute> read_sram_attrs;
    for (MemoryNode *mem : task_node->read_memrefs) {
      if (mem->assigned_sram_pos) {
        SmallVector<NamedAttribute, 2> sram_coord;
        sram_coord.push_back(NamedAttribute(
            StringAttr::get(func.getContext(), "col"),
            builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));
        sram_coord.push_back(NamedAttribute(
            StringAttr::get(func.getContext(), "row"),
            builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
        read_sram_attrs.push_back(
            DictionaryAttr::get(func.getContext(), sram_coord));
      }
    }
    mapping_attrs.push_back(NamedAttribute(
        StringAttr::get(func.getContext(), "read_sram_locations"),
        builder.getArrayAttr(read_sram_attrs)));

    // 3. Writes SRAM locations.
    SmallVector<Attribute> write_sram_attrs;
    for (MemoryNode *mem : task_node->write_memrefs) {
      if (mem->assigned_sram_pos) {
        SmallVector<NamedAttribute, 2> sram_coord;
        sram_coord.push_back(NamedAttribute(
            StringAttr::get(func.getContext(), "col"),
            builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));
        sram_coord.push_back(NamedAttribute(
            StringAttr::get(func.getContext(), "row"),
            builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
        write_sram_attrs.push_back(
            DictionaryAttr::get(func.getContext(), sram_coord));
      }
    }
    mapping_attrs.push_back(NamedAttribute(
        StringAttr::get(func.getContext(), "write_sram_locations"),
        builder.getArrayAttr(write_sram_attrs)));

    task_node->op->setAttr(
        "task_orchestration_info",
        DictionaryAttr::get(func.getContext(), mapping_attrs));

    // Write profile_info = {duration: N} if not already present so that
    // downstream passes can read the task duration without re-computing it.
    if (!task_node->op->hasAttr("profile_info")) {
      SmallVector<NamedAttribute, 1> profile_attrs;
      profile_attrs.push_back(
          NamedAttribute(StringAttr::get(func.getContext(), "duration"),
                         builder.getI32IntegerAttr(task_node->getDuration())));
      task_node->op->setAttr(
          "profile_info",
          DictionaryAttr::get(func.getContext(), profile_attrs));
    }

    // Removes upstream resource-binding attributes that have been consumed.
    task_node->op->removeAttr("cgra_count");
    task_node->op->removeAttr("cgra_shape");
  }

  // Reports the schedule this pass actually produced. start_time and duration
  // stay out of the IR (the attribute contract is the placement), but without
  // them there is no way to read back what the spatial-temporal scheduler did,
  // so they are printed: makespan, peak concurrent CGRAs, context depth, and
  // the per-task interval.
  {
    int makespan = 0, peak_contexts = 0;
    int64_t busy_area = 0;
    llvm::errs() << "\n=== Orchestrated Schedule ("
                 << (mode_ == SchedulingMode::Spatial ? "spatial"
                                                      : "spatial-temporal")
                 << ", " << grid_rows_ << "x" << grid_cols_ << " CGRAs) ===\n";
    for (auto &task_node : graph.task_nodes) {
      if (task_node->placement.empty())
        continue;
      int start = INT32_MAX, finish = 0, ctx = 0;
      for (const CgraPosition &pos : task_node->placement) {
        start = std::min(start, pos.start_time);
        finish = std::max(finish, pos.start_time + pos.duration);
        ctx = std::max(ctx, pos.context_id);
      }
      makespan = std::max(makespan, finish);
      peak_contexts = std::max(peak_contexts, ctx + 1);
      busy_area += (int64_t)(finish - start) * task_node->placement.size();
      llvm::errs() << "  " << task_node->op.getTaskName() << ": start=" << start
                   << " finish=" << finish
                   << " cgras=" << task_node->placement.size()
                   << " replicas=" << task_node->replicas_placed << "/"
                   << task_node->getReplicas() << " max_context=" << ctx
                   << "\n";
    }
    int64_t grid_area = (int64_t)grid_rows_ * grid_cols_;
    double util = (makespan > 0 && grid_area > 0)
                      ? (double)busy_area / ((double)makespan * grid_area)
                      : 0.0;
    llvm::errs() << "[Orchestrate] makespan=" << makespan
                 << " tasks=" << graph.task_nodes.size()
                 << " max_contexts_per_cgra=" << peak_contexts
                 << " grid_utilisation=" << llvm::format("%.3f", util) << "\n";
  }
  return true;
}

void TaskScheduler::recordScheduleResult(const TaskMemoryGraph &graph) {
  schedule_result_.clear();
  for (const auto &task_node : graph.task_nodes) {
    if (task_node->placement.empty()) {
      continue;
    }

    TaskScheduleResult task_result;
    task_result.task = task_node->op;
    task_result.start_time = task_node->placement.front().start_time;
    task_result.duration = task_node->placement.front().duration;
    task_result.end_time = task_result.start_time + task_result.duration;

    for (const CgraPosition &pos : task_node->placement) {
      task_result.start_time = std::min(task_result.start_time, pos.start_time);
      task_result.end_time =
          std::max(task_result.end_time, pos.start_time + pos.duration);
      task_result.cgra_occupancies.push_back(
          {pos.row, pos.col, pos.start_time, pos.duration, pos.context_id});
    }
    task_result.duration = task_result.end_time - task_result.start_time;

    for (TaskNode *pred : task_node->ssa_operands) {
      task_result.predecessor_tasks.push_back(pred->op);
    }
    for (TaskNode *succ : task_node->ssa_users) {
      task_result.successor_tasks.push_back(succ->op);
    }

    schedule_result_.push_back(std::move(task_result));
  }
}

bool TaskScheduler::posInBounds(const CgraPosition &pos) const {
  return pos.row >= 0 && pos.row < this->grid_rows_ && pos.col >= 0 &&
         pos.col < this->grid_cols_;
}

// Returns true if CGRA (row, col) is occupied during
// [start_time, start_time + duration).
//
// Spatial mode: occupied once any task is assigned (permanently taken).
// SpatialTemporal mode: occupied if any existing interval overlaps.
bool TaskScheduler::isOccupied(int row, int col, int start_time,
                               int duration) const {
  if (mode_ == SchedulingMode::Spatial) {
    return !cgra_occupancy_[row][col].empty();
  }
  for (auto [occupied_start, occupied_end] : cgra_occupancy_[row][col]) {
    if (start_time < occupied_end && start_time + duration > occupied_start) {
      return true;
    }
  }
  return false;
}

void TaskScheduler::markOccupied(int row, int col, int start_time,
                                 int duration) {
  cgra_occupancy_[row][col].push_back({start_time, start_time + duration});
}

void TaskScheduler::resetTaskPlacements(TaskMemoryGraph &graph) {
  for (auto &task : graph.task_nodes) {
    task->placement.clear();
  }
  for (auto &row : this->cgra_occupancy_) {
    for (auto &col_intervals : row) {
      col_intervals.clear();
    }
  }
}

// Computes the earliest feasible start time for `task_node` such that all
// explicit taskflow dependencies have completed.
int TaskScheduler::computeEarliestStartTime(const TaskNode *task_node) const {
  int min_time = 0;

  // Tiles cut from the same loop by the resource-aware pass carry a shared
  // tile_group. They are threaded through the memref dependence-state SSA
  // because that version chain is linear, but the dimension they partition is
  // dependence-free (that is the precondition the cut was made under), so the
  // chain is an ordering artefact. Honouring it here would serialise every tile
  // and cancel the partitioning outright.
  auto tileParallel = [](const TaskNode *n) -> bool {
    auto attr = n->op->template getAttrOfType<BoolAttr>("tile_parallel");
    return !attr || attr.getValue();
  };
  auto tileGroupOf = [](const TaskNode *n) -> int {
    if (auto attr = n->op->template getAttrOfType<IntegerAttr>("tile_group"))
      return static_cast<int>(attr.getInt());
    return -1;
  };
  const int my_group = tileGroupOf(task_node);

  auto updateFromPlacement = [&](const TaskNode *other) {
    if (other == task_node || other->placement.empty())
      return;
    // Only a parallel-safe group may ignore its own chain; an ordered group
    // (a reduction cut into pieces) must keep it.
    if (my_group >= 0 && tileGroupOf(other) == my_group &&
        tileParallel(task_node))
      return;
    const CgraPosition &pos = other->placement[0];
    min_time = std::max(min_time, pos.start_time + pos.duration);
    // TaskTiler rewires consumers onto the LAST tile of a group, so depending
    // on that one tile is not the same as depending on the group. Wait for
    // every sibling of the producer too, or tiles 0..n-2 -- which have no other
    // path here once their chain is dropped -- could still be running.
    int g = tileGroupOf(other);
    if (g >= 0 && g != my_group) {
      // The siblings are reachable by walking the SSA chain backwards: tile i
      // consumes tile i-1, so the last tile transitively names them all.
      const TaskNode *cur = other;
      for (int guard = 0; guard < 4096 && cur; ++guard) {
        const TaskNode *prev = nullptr;
        for (const TaskNode *p : cur->ssa_operands)
          if (tileGroupOf(p) == g) {
            prev = p;
            break;
          }
        if (!prev || prev->placement.empty())
          break;
        const CgraPosition &sp = prev->placement[0];
        min_time = std::max(min_time, sp.start_time + sp.duration);
        cur = prev;
      }
    }
  };

  for (const TaskNode *pred : task_node->ssa_operands) {
    updateFromPlacement(pred);
  }
  // A tile inherits its group's external predecessors: without this a tile
  // whose only SSA operand is its sibling would float to time 0.
  if (my_group >= 0) {
    for (const TaskNode *pred : task_node->ssa_operands) {
      if (tileGroupOf(pred) != my_group)
        continue;
      for (const TaskNode *grand : pred->ssa_operands)
        updateFromPlacement(grand);
    }
  }
  return min_time;
}

// Assigns each MemoryNode to the SRAM at the centroid of all accessing
// CGRAs.  Returns true if any assignment changed (convergence criterion).
bool TaskScheduler::assignAllSrams(TaskMemoryGraph &graph) {
  bool changed = false;
  for (auto &mem_node : graph.memory_nodes) {
    int total_row = 0, total_col = 0, count = 0;
    for (TaskNode *reader : mem_node->readers) {
      for (const CgraPosition &pos : reader->placement) {
        total_row += pos.row;
        total_col += pos.col;
        count++;
      }
    }
    for (TaskNode *writer : mem_node->writers) {
      for (const CgraPosition &pos : writer->placement) {
        total_row += pos.row;
        total_col += pos.col;
        count++;
      }
    }

    std::optional<CgraPosition> new_sram_pos;
    if (count > 0) {
      int avg_row = (total_row + count / 2) / count;
      int avg_col = (total_col + count / 2) / count;
      new_sram_pos = CgraPosition{avg_row, avg_col, 0, 0};
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
// In SpatialTemporal mode an outer time loop applies ASAP scheduling:
// the earliest feasible start time is computed from dependency constraints,
// then incremented by task_duration until a valid grid position is found.
TaskPlacement TaskScheduler::findBestPlacement(TaskNode *task_node,
                                               int cgra_count,
                                               TaskMemoryGraph &graph,
                                               int force_start) {
  SmallVector<CgraShape> shapes_to_try;
  if (auto attr = task_node->op->getAttrOfType<StringAttr>("cgra_shape")) {
    StringRef cgra_shape_str = attr.getValue();
    if (!cgra_shape_str.empty()) {
      CgraShape base = parseCgraShapeToBase(cgra_shape_str, cgra_count);
      shapes_to_try = rotationsOf(base);
    }
  }
  if (shapes_to_try.empty()) {
    shapes_to_try = getAllPlacementShapes(cgra_count);
  }

  int task_duration = task_node->getDuration();

  int t_start = (mode_ == SchedulingMode::SpatialTemporal)
                    ? computeEarliestStartTime(task_node)
                    : 0;
  if (force_start >= 0) {
    // Replica placement: all replicas of one task run concurrently, so this
    // one is pinned to the slot the first replica took.
    TaskPlacement pinned;
    int64_t pinned_score = INT64_MIN;
    for (const CgraShape &shape : shapes_to_try) {
      SmallVector<std::pair<int, int>> offsets;
      if (shape.is_rectangular) {
        for (int r = 0; r < shape.rows; ++r)
          for (int c = 0; c < shape.cols; ++c)
            offsets.push_back({c, r});
      } else {
        offsets = SmallVector<std::pair<int, int>>(shape.cgra_positions.begin(),
                                                   shape.cgra_positions.end());
      }
      for (int origin_row = 0; origin_row < grid_rows_; ++origin_row) {
        for (int origin_col = 0; origin_col < grid_cols_; ++origin_col) {
          bool valid = true;
          TaskPlacement candidate;
          for (auto &[col_off, row_off] : offsets) {
            int abs_row = origin_row + row_off;
            int abs_col = origin_col + col_off;
            if (abs_row < 0 || abs_row >= grid_rows_ || abs_col < 0 ||
                abs_col >= grid_cols_ ||
                isOccupied(abs_row, abs_col, force_start, task_duration)) {
              valid = false;
              break;
            }
            candidate.cgra_positions.push_back(
                {abs_row, abs_col, force_start, task_duration, 0});
          }
          if (!valid)
            continue;
          int64_t score = computeScore(task_node, candidate, graph);
          if (score > pinned_score) {
            pinned_score = score;
            pinned = candidate;
          }
        }
      }
    }
    return pinned;
  }
  // Candidate start times.
  //
  // The earliest time a task can start is `t_start` or the instant some cell
  // frees up, and nothing in between: within a gap between two occupancy
  // events the feasible set does not change, so any later time in that gap is
  // dominated. Enumerating the occupancy END times therefore covers every
  // placement a finer search could find, exactly, with at most one candidate
  // per already-placed interval.
  //
  // The previous form swept `t_start, t_start + d, t_start + 2d, ...` up to
  // `max(grid_area, task_count) * d`, i.e. it measured the horizon in units of
  // THIS task's duration. That holds only when tasks have similar durations.
  // On a GPT-2 block they do not: a 768x768x768 projection occupies a cell for
  // ~10^8 cycles while a residual add runs for ~10^5, so the short task ran out
  // of horizon long before the long one released the grid, findBestPlacement
  // returned empty, and the scheduler asserted.
  SmallVector<int> candidate_times;
  candidate_times.push_back(t_start);
  if (mode_ == SchedulingMode::SpatialTemporal) {
    for (const auto &row : cgra_occupancy_)
      for (const auto &cell : row)
        for (auto [occupied_start, occupied_end] : cell)
          if (occupied_end > t_start)
            candidate_times.push_back(occupied_end);
    llvm::sort(candidate_times);
    candidate_times.erase(
        std::unique(candidate_times.begin(), candidate_times.end()),
        candidate_times.end());
  }

  for (int t : candidate_times) {
    int64_t best_score = INT64_MIN;
    TaskPlacement best_at_t;

    for (const CgraShape &shape : shapes_to_try) {
      SmallVector<std::pair<int, int>> shape_offsets;
      if (shape.is_rectangular) {
        for (int r = 0; r < shape.rows; ++r) {
          for (int c = 0; c < shape.cols; ++c) {
            shape_offsets.push_back({c, r});
          }
        }
      } else {
        shape_offsets = SmallVector<std::pair<int, int>>(
            shape.cgra_positions.begin(), shape.cgra_positions.end());
      }

      for (int origin_row = 0; origin_row < grid_rows_; ++origin_row) {
        for (int origin_col = 0; origin_col < grid_cols_; ++origin_col) {
          bool valid = true;
          TaskPlacement candidate;
          for (auto &[col_off, row_off] : shape_offsets) {
            int abs_row = origin_row + row_off;
            int abs_col = origin_col + col_off;
            if (abs_row < 0 || abs_row >= grid_rows_ || abs_col < 0 ||
                abs_col >= grid_cols_ ||
                isOccupied(abs_row, abs_col, t, task_duration)) {
              valid = false;
              break;
            }
            candidate.cgra_positions.push_back(
                {abs_row, abs_col, t, task_duration, 0});
          }
          if (!valid) {
            continue;
          }
          int64_t score = computeScore(task_node, candidate, graph);
          if (score > best_score) {
            best_score = score;
            best_at_t = candidate;
          }
        }
      }
    }

    if (!best_at_t.cgra_positions.empty()) {
      return best_at_t;
    }

    if (mode_ == SchedulingMode::Spatial) {
      break;
    }
  }

  return TaskPlacement{};
}

CgraShape TaskScheduler::parseCgraShapeToBase(StringRef cgra_shape,
                                              int cgra_count) {
  size_t bracket_pos = cgra_shape.find('[');
  auto [rows_str, rest] = cgra_shape.split('x');
  int rows = 1, cols = 1;
  rows_str.getAsInteger(10, rows);

  if (bracket_pos == StringRef::npos) {
    rest.getAsInteger(10, cols);
    return CgraShape{rows, cols, /*is_rectangular=*/true, {}};
  }

  StringRef cols_str = rest.take_until([](char c) { return c == '['; });
  cols_str.getAsInteger(10, cols);

  SmallVector<std::pair<int, int>> positions;
  StringRef positions_str = cgra_shape.substr(bracket_pos);
  size_t pos = 0;
  while (pos < positions_str.size()) {
    size_t open = positions_str.find('(', pos);
    if (open == StringRef::npos) {
      break;
    }
    size_t close = positions_str.find(')', open);
    if (close == StringRef::npos) {
      break;
    }
    StringRef pair_str = positions_str.slice(open + 1, close);
    auto [col_str, row_str] = pair_str.split(',');
    int col_off = 0, row_off = 0;
    col_str.getAsInteger(10, col_off);
    row_str.getAsInteger(10, row_off);
    positions.push_back({col_off, row_off});
    pos = close + 1;
  }
  return CgraShape{rows, cols, /*is_rectangular=*/false, std::move(positions)};
}

SmallVector<CgraShape> TaskScheduler::rotationsOf(const CgraShape &base) {
  SmallVector<CgraShape> result;

  if (base.is_rectangular) {
    result.push_back(base);
    if (base.rows != base.cols) {
      result.push_back(CgraShape{base.cols, base.rows, true, {}});
    }
    return result;
  }

  llvm::DenseSet<int64_t> seen_hashes;
  auto current_positions = SmallVector<std::pair<int, int>>(
      base.cgra_positions.begin(), base.cgra_positions.end());

  for (int rotation_count = 0; rotation_count < 4; ++rotation_count) {
    int min_col = INT_MAX, min_row = INT_MAX;
    for (auto &[col, row] : current_positions) {
      min_col = std::min(min_col, col);
      min_row = std::min(min_row, row);
    }

    SmallVector<std::pair<int, int>> normalised_positions;
    for (auto &[col, row] : current_positions) {
      normalised_positions.push_back({col - min_col, row - min_row});
    }

    auto sorted_positions = normalised_positions;
    llvm::sort(sorted_positions,
               [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                 return a < b;
               });

    int64_t position_hash = 0;
    for (auto &[col, row] : sorted_positions) {
      position_hash = position_hash * 131 + col * 17 + row;
    }

    if (seen_hashes.insert(position_hash).second) {
      int max_col = 0, max_row = 0;
      for (auto &[col, row] : normalised_positions) {
        max_col = std::max(max_col, col);
        max_row = std::max(max_row, row);
      }
      result.push_back(
          CgraShape{max_row + 1, max_col + 1, false, normalised_positions});
    }

    SmallVector<std::pair<int, int>> rotated_positions;
    for (auto &[col, row] : current_positions) {
      rotated_positions.push_back({row, -col});
    }
    current_positions = rotated_positions;
  }
  return result;
}

// Computes the placement score for `task_node` at `placement`.
//
// Score = α·SSA_Dist + β·Mem_Dist - γ·Context_Reuse.
//   SSA_Dist : sum of distances to already-placed SSA predecessors and
//              successors (negative; penalises far-away neighbours).
//   Mem_Dist : sum of distances to assigned SRAMs for read/write memrefs
//              (negative; memory proximity is weighted more heavily).
//   Context_Reuse : penalty for reusing a CGRA that already has another
//                   task in a different context.
//
// Higher score is better; 0 means all neighbours are co-located.
int64_t TaskScheduler::computeScore(TaskNode *task_node,
                                    const TaskPlacement &placement,
                                    TaskMemoryGraph &graph) {
  // Weight constants (tunable).
  constexpr int64_t kAlpha = 10;   // SSA proximity weight.
  constexpr int64_t kBeta = 50;    // Memory proximity weight (high priority).
  constexpr int64_t kGamma = 1000; // Context switch cost is higher than NoC.

  // OURS -- comm-aware placement (research toggle NEURA_COMM_AWARE=1): weight
  // each memory-proximity penalty by the transferred DATA VOLUME (memref
  // element count), so the placement minimizes sum(volume * manhattan_distance)
  // = communication. Baseline (env unset) weights every memref uniformly (the
  // original behavior).
  static const bool comm_aware = (std::getenv("NEURA_COMM_AWARE") != nullptr);
  auto volOf = [](MemoryNode *mem) -> int64_t {
    if (auto st = llvm::dyn_cast<ShapedType>(mem->memref.getType()))
      if (st.hasStaticShape())
        return std::max<int64_t>(1, st.getNumElements());
    return 1;
  };

  int64_t ssa_score = 0, mem_score = 0, context_reuse_penalty = 0;

  auto minDistToPlacement = [&](const SmallVector<CgraPosition> &other) -> int {
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
      ssa_score -= minDistToPlacement(producer->placement);
    }
  }
  for (TaskNode *consumer : task_node->ssa_users) {
    if (!consumer->placement.empty()) {
      ssa_score -= minDistToPlacement(consumer->placement);
    }
  }

  // 2. Memory proximity — penalise distance to assigned SRAMs.
  // OURS: comm-aware weights the penalty by the memref volume (data size);
  // baseline uses 1. For read memrefs (data sources).
  for (MemoryNode *mem : task_node->read_memrefs) {
    if (mem->assigned_sram_pos) {
      int64_t w = comm_aware ? volOf(mem) : 1;
      mem_score -= w * minDistToTarget(*mem->assigned_sram_pos);
    }
  }
  // For write memrefs: if the SRAM is already assigned (e.g. read by a
  // previous task), we want to be close to it too.
  for (MemoryNode *mem : task_node->write_memrefs) {
    if (mem->assigned_sram_pos) {
      int64_t w = comm_aware ? volOf(mem) : 1;
      mem_score -= w * minDistToTarget(*mem->assigned_sram_pos);
    }
  }

  if (mode_ == SchedulingMode::SpatialTemporal) {
    for (const CgraPosition &pos : placement.cgra_positions) {
      if (!cgra_occupancy_[pos.row][pos.col].empty()) {
        ++context_reuse_penalty;
      }
    }
  }

  return kAlpha * ssa_score + kBeta * mem_score -
         kGamma * context_reuse_penalty;
}

} // namespace taskflow
} // namespace mlir
