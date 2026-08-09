// Shared CGRA orchestration utilities.

#ifndef TASKFLOW_ORCHESTRATION_UTILS_H
#define TASKFLOW_ORCHESTRATION_UTILS_H

#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace mlir {
namespace taskflow {

// Grid constants.

constexpr int kCgraGridRows = 4;
constexpr int kCgraGridCols = 4;

// CgraShape.

// Represents a CGRA orchestration shape on the grid.
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
  llvm::SmallVector<std::pair<int, int>> cgra_positions;

  // Returns the bounding-box area (rows * cols).  For rectangular shapes this
  // equals cgra_count; for non-rectangular shapes it is larger than cgra_count
  // (some cells in the bbox are unoccupied).  Used only for shape sorting
  // (prefer smaller bounding boxes), not for counting occupied CGRAs.
  int area() const { return rows * cols; }

  // Returns a human-readable description for log messages only (not IR).
  std::string describe(int cgra_count) const;

  // Returns the shape string written into the IR cgra_shape attribute.
  // For rectangular shapes: "NxM" (e.g. "2x2").
  // For non-rectangular shapes: "NxM[(c0,r0)(c1,r1)...]" listing only the
  // occupied CGRA positions so that downstream passes can reconstruct the
  // exact valid tile set for multi-CGRA mapping.
  std::string irAttr() const;
};

// Shape enumeration utilities.

// Generates all placement-candidate shapes for `cgra_count` CGRAs, including
// rotations. Rectangular shapes include both orientations (rows×cols and
// cols×rows, deduplicated for squares). Non-rectangular shapes include all
// four 90° rotations.
//
// Ordering (tried first to last):
//   1. Rectangular shapes, sorted by squareness (e.g. 2×2 before 1×4),
//      with smaller bounding-box area as tiebreaker.
//   2. Non-rectangular shapes (L, T, etc.) in all unique rotations.
llvm::SmallVector<CgraShape> getAllPlacementShapes(int cgra_count);

// Global placement feasibility.

// Simulates greedy placement of all tasks' shapes on the kCgraGridRows x
// kCgraGridCols grid to verify that they physically fit without overlap.
//
// For each task, all valid shapes (including rotations) are tried. Rectangular
// shapes prefer square-like orientations (e.g. 2x2 over 1x4). Non-rectangular
// shapes are tried in all four 90 degree rotations.
//
// `task_cgra_counts` contains the cgra_count for every task in the graph
// (including the speculatively modified one).
//
// Returns true if all tasks can be placed without overlap.
bool canAllTasksFitOnGrid(llvm::ArrayRef<int> task_cgra_counts);

// Task scheduling utilities.

// Controls whether a time-slot dimension is added to every CGRA assignment.
enum class SchedulingMode {
  // Each CGRA is assigned to at most one task. Asserts if tasks exceed the
  // grid size.
  Spatial,
  // Adds a time-slot dimension. Tasks that would over-subscribe the grid are
  // scheduled at a later time slot, enabling temporal reuse of CGRAs.
  SpatialTemporal,
};

// One scheduled CGRA cell assignment for a task.
struct CgraPosition;

// Complete CGRA placement chosen for one task.
struct TaskPlacement;

// Internal task node used by the scheduler's dependency graph.
struct TaskNode;

// Internal task/memory dependency graph built from one func.func.
class TaskMemoryGraph;

// Caller-provided task priority; higher values are scheduled earlier.
using TaskPriorityMap = llvm::DenseMap<Operation *, int>;

// Concrete schedule result for one task after TaskScheduler placement.
// Times and durations are 64-bit because they are cycle counts of whole
// kernels, not op counts. A GPT-2 prefill block already measures 1.85e9 cycles,
// 86% of INT32_MAX, and start times accumulate along a path, so 32-bit
// arithmetic here overflows on the programs this compiler is meant for.
struct TaskScheduleResult {
  // One CGRA cell occupied by this task.
  struct CgraOccupancy {
    int row = 0;
    int col = 0;
    int64_t start_time = 0;
    int64_t duration = 1;
    int context_id = 0;
  };

  TaskflowTaskOp task;
  int64_t start_time = 0;
  int64_t duration = 1;
  int64_t end_time = 1;
  llvm::SmallVector<CgraOccupancy> cgra_occupancies;
  llvm::SmallVector<TaskflowTaskOp> predecessor_tasks;
  llvm::SmallVector<TaskflowTaskOp> successor_tasks;
};

// Pipeline interval analysis result for a concrete task schedule.
struct TaskPipelineIntervalResult {
  int64_t pipeline_interval = 0;
  TaskflowTaskOp bottleneck_task;
  llvm::SmallVector<TaskflowTaskOp> critical_path;
};

// Builds a task-level schedule analysis graph and derives the steady-state
// pipeline interval for the concrete schedule produced by TaskScheduler.
//
// The graph nodes are the scheduled tasks for one input instance. The graph
// contains two kinds of execution-order edges:
//
//   1. Data-dependence edges.
//      If T1 consumes a value/token produced by T0, add T0 -> T1 with latency
//      latency(T0). This says T1 cannot execute until T0 has completed.
//
//   2. CGRA execution-order edges.
//      If a physical CGRA runs T0 before T2 for the same input instance, add
//      T0 -> T2 with latency latency(T0). This says T2 cannot use that CGRA
//      until T0 releases its context slot.
//
// The graph gives ordering within one input instance. To compute steady-state
// throughput, each CGRA also defines a pipeline cycle: the last task using
// that CGRA for this input instance must finish before the first task using
// that same CGRA for the next input instance can start.
//
//      first_task(this input) -> ... -> last_task(this input)
//      last_task(this input)  -> first_task(next input)
//
// The interval required by that CGRA is the latency of the longest path from
// first_task to last_task in the analysis graph, plus latency(last_task) for
// the transition to the next input instance. The overall pipeline interval is
// the maximum interval over all CGRA pipeline cycles.
//
// Example:
//   Data dependence: T0 -> T1 -> T2
//   Placement:       T0 on CGRA0, T1 on CGRA1, T2 on CGRA0
//
// The analysis graph contains data-dependence edges T0 -> T1 and T1 -> T2.
// Since T0 and T2 reuse CGRA0 in order, it also contains a CGRA
// execution-order edge T0 -> T2. CGRA0 then defines the pipeline cycle
// T0(this input) -> T1 -> T2 -> T0(next input), requiring
// latency(T0)+latency(T1)+latency(T2).
class TaskPipelineIntervalAnalyzer {
public:
  explicit TaskPipelineIntervalAnalyzer(
      llvm::ArrayRef<TaskScheduleResult> schedule_result);

  TaskPipelineIntervalResult analyze();

private:
  // Edge in the analysis graph that says `task` must execute before
  // `next_task`. It is created either by a data dependence or by sequential
  // reuse of the same CGRA context.
  // An edge between two TASK INDICES into `schedule_result_`, not between task
  // ops. `latency` is the source task's duration.
  struct ExecutionOrderEdge {
    int next_task_idx = -1;
    int64_t latency = 0;
  };

  // Pipeline cycle induced by one physical CGRA. `last_task` is the final task
  // using that CGRA for this input instance, and `first_task` is the first task
  // using that same CGRA for the next input instance.
  // The cycle one physical CGRA closes: its last task feeds the next input's
  // first task. Both fields are indices into `schedule_result_`.
  struct CgraPipelineCycle {
    int last_task_idx = -1;
    int first_task_idx = -1;
    int64_t latency = 0;
  };

  // Longest execution-order path found between two task nodes.
  // Longest path to a target, as TASK INDICES.
  struct LongestExecutionPath {
    bool found = false;
    int64_t total_latency = 0;
    llvm::SmallVector<int> path;
  };

  int64_t getTaskDuration(int task_idx) const;
  int64_t
  encodeCgraLocation(const TaskScheduleResult::CgraOccupancy &occupancy) const;
  void addExecutionOrderEdge(int task_idx, int next_task_idx);
  void buildTaskIndex();
  void buildDataDependenceEdges();
  void buildCgraExecutionOrderEdgesAndPipelineCycles();
  LongestExecutionPath findLongestPathToTarget(
      int current_task_idx, int target_task_idx, llvm::DenseSet<int> &visiting,
      llvm::DenseMap<int, LongestExecutionPath> &memo) const;
  TaskPipelineIntervalResult computeLongestPipelineCycle() const;

  llvm::ArrayRef<TaskScheduleResult> schedule_result_;
  llvm::DenseMap<Operation *, int> task_to_index_;
  llvm::SmallVector<llvm::SmallVector<ExecutionOrderEdge>> task_graph_;
  llvm::SmallVector<CgraPipelineCycle> cgra_pipeline_cycles_;
};

// Reusable one-shot scheduler/placer for Taskflow task graphs.
//
// Builds the task-memory graph, schedules tasks using the provided priority,
// places them on the CGRA grid, assigns SRAM locations, and emits
// task_orchestration_info/profile_info metadata.
class TaskScheduler {
public:
  // `comm_aware` weights each memory-proximity penalty by the transferred
  // data volume instead of counting every memref equally, so the placement
  // minimises sum(volume * distance) rather than sum(distance). Off by default:
  // it changes the placement the in-tree expectations pin.
  TaskScheduler(int grid_rows = kCgraGridRows, int grid_cols = kCgraGridCols,
                SchedulingMode mode = SchedulingMode::SpatialTemporal,
                bool comm_aware = false);

  // Schedules and places all Taskflow tasks in `func` using the caller-provided
  // task priority map.
  bool schedule(func::FuncOp func, const TaskPriorityMap &priority);

  // Returns the concrete task schedule produced by schedule().
  llvm::ArrayRef<TaskScheduleResult> getScheduleResult() const {
    return schedule_result_;
  }

private:
  // Records concrete schedule facts from the internal task placement.
  void recordScheduleResult(const TaskMemoryGraph &graph);

  // Returns true if a CGRA grid coordinate is inside the configured grid.
  bool posInBounds(const CgraPosition &pos) const;

  // Returns true if a CGRA cell is already occupied during the requested
  // time interval.
  bool isOccupied(int row, int col, int64_t start_time,
                  int64_t duration) const;

  // Marks a CGRA cell as occupied for the half-open interval
  // [start_time, start_time + duration).
  void markOccupied(int row, int col, int64_t start_time, int64_t duration);

  // Clears all task placements and CGRA occupancy state before another
  // fixed-point placement iteration.
  void resetTaskPlacements(TaskMemoryGraph &graph);

  // Computes the earliest start time allowed by already-placed predecessor
  // tasks.
  int64_t computeEarliestStartTime(const TaskNode *task_node) const;

  // Assigns every memory node to the SRAM location closest to its accessing
  // tasks and returns whether any assignment changed.
  bool assignAllSrams(TaskMemoryGraph &graph);

  // Searches legal grid positions and returns the best-scoring placement for
  // one task under the current scheduling mode.
  //
  // `force_start` pins the placement to one time slot instead of searching
  // forward from the earliest feasible one. It is used to place the replicas of
  // a data-parallel task, which must all be resident simultaneously; -1 means
  // the normal ASAP search.
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph,
                                  int64_t force_start = -1);

  // Parses a cgra_shape attribute string into its base placement shape.
  CgraShape parseCgraShapeToBase(StringRef cgra_shape, int cgra_count);

  // Generates all unique rotations of a placement shape.
  llvm::SmallVector<CgraShape> rotationsOf(const CgraShape &base);

  // Scores a candidate placement using proximity to dependent tasks, assigned
  // SRAMs, and context reuse cost.
  int64_t computeScore(TaskNode *task_node, const TaskPlacement &placement,
                       TaskMemoryGraph &graph);

  int grid_rows_;
  int grid_cols_;
  SchedulingMode mode_;
  bool comm_aware_ = false;
  int total_task_count_ = 0;
  llvm::SmallVector<TaskScheduleResult> schedule_result_;
  // Half-open [start, end) busy intervals per cell, in cycles.
  std::vector<std::vector<llvm::SmallVector<std::pair<int64_t, int64_t>, 4>>>
      cgra_occupancy_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_ORCHESTRATION_UTILS_H
