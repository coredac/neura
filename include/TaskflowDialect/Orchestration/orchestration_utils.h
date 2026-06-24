// Shared CGRA orchestration utilities.

#ifndef TASKFLOW_ORCHESTRATION_UTILS_H
#define TASKFLOW_ORCHESTRATION_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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

// Reusable one-shot scheduler/placer for Taskflow task graphs.
//
// Builds the task-memory graph, schedules tasks using the provided priority,
// places them on the CGRA grid, assigns SRAM locations, and emits
// task_orchestration_info/profile_info metadata.
class TaskScheduler {
public:
  TaskScheduler(int grid_rows = kCgraGridRows, int grid_cols = kCgraGridCols,
                SchedulingMode mode = SchedulingMode::SpatialTemporal);

  // Schedules and places all Taskflow tasks in `func` using the caller-provided
  // task priority map.
  bool schedule(func::FuncOp func, const TaskPriorityMap &priority);

private:
  // Returns true if a CGRA grid coordinate is inside the configured grid.
  bool posInBounds(const CgraPosition &pos) const;

  // Returns true if a CGRA cell is already occupied during the requested
  // time interval.
  bool isOccupied(int row, int col, int start_time, int duration) const;

  // Marks a CGRA cell as occupied for the half-open interval
  // [start_time, start_time + duration).
  void markOccupied(int row, int col, int start_time, int duration);

  // Clears all task placements and CGRA occupancy state before another
  // fixed-point placement iteration.
  void resetTaskPlacements(TaskMemoryGraph &graph);

  // Computes the earliest start time allowed by already-placed predecessor
  // tasks.
  int computeEarliestStartTime(const TaskNode *task_node) const;

  // Assigns every memory node to the SRAM location closest to its accessing
  // tasks and returns whether any assignment changed.
  bool assignAllSrams(TaskMemoryGraph &graph);

  // Searches legal grid positions and returns the best-scoring placement for
  // one task under the current scheduling mode.
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph);

  // Parses a cgra_shape attribute string into its base placement shape.
  CgraShape parseCgraShapeToBase(StringRef cgra_shape, int cgra_count);

  // Generates all unique rotations of a placement shape.
  llvm::SmallVector<CgraShape> rotationsOf(const CgraShape &base);

  // Scores a candidate placement using proximity to dependent tasks, assigned
  // SRAMs, and context reuse cost.
  int computeScore(TaskNode *task_node, const TaskPlacement &placement,
                   TaskMemoryGraph &graph);

  int grid_rows_;
  int grid_cols_;
  SchedulingMode mode_;
  int total_task_count_ = 0;
  std::vector<std::vector<llvm::SmallVector<std::pair<int, int>, 4>>>
      cgra_occupancy_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_ORCHESTRATION_UTILS_H
