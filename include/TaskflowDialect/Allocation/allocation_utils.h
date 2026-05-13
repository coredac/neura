//===- allocation_utils.h - Shared CGRA allocation utilities --------------===//
//
// Shared utility types and functions used by AllocateCgraToTaskPass and
// ResourceAwareTaskOptimizationPass for 2D multi-CGRA grid placement
// feasibility checks and task-to-CGRA mapping.
//
//===----------------------------------------------------------------------===//

#ifndef TASKFLOW_ALLOCATION_UTILS_H
#define TASKFLOW_ALLOCATION_UTILS_H

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace taskflow {

//===----------------------------------------------------------------------===//
// Grid constants
//===----------------------------------------------------------------------===//

constexpr int kCgraGridRows = 4;
constexpr int kCgraGridCols = 4;

//===----------------------------------------------------------------------===//
// CgraShape
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

//===----------------------------------------------------------------------===//
// Shape Enumeration Utilities
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Global Placement Feasibility
//===----------------------------------------------------------------------===//

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

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_CGRA_PLACEMENT_UTILS_H
