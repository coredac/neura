//===- allocation_utils.cpp - Shared CGRA allocation utilities ------------===//
//
// Implements shared utility functions for 2D multi-CGRA grid placement used
// by AllocateCgraToTaskPass and ResourceAwareTaskOptimizationPass.
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/Allocation/allocation_utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;
using llvm::ArrayRef;
using llvm::SmallVector;

//===----------------------------------------------------------------------===//
// CgraShape member implementations
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// getAllPlacementShapes
//===----------------------------------------------------------------------===//

SmallVector<CgraShape> mlir::taskflow::getAllPlacementShapes(int cgra_count) {
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

//===----------------------------------------------------------------------===//
// canAllTasksFitOnGrid
//===----------------------------------------------------------------------===//

bool mlir::taskflow::canAllTasksFitOnGrid(ArrayRef<int> task_cgra_counts) {
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
  // bottleneck at a time; this array reflects the heterogeneous allocation
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
