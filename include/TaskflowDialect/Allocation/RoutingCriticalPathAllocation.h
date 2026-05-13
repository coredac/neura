//===- RoutingCriticalPathAllocation.h - Routing-critical-path-first ------===//
//
// Concrete Allocation strategy that places Taskflow tasks onto a 2D
// multi-CGRA grid using a routing-critical-path-first ordering.
//
// Tasks with the longest downstream dependency chains are placed first so
// that their successors can land on adjacent CGRAs, minimizing inter-task
// communication distance along the critical path.
//
//===----------------------------------------------------------------------===//

#ifndef TASKFLOW_ROUTING_CRITICAL_PATH_ALLOCATION_H
#define TASKFLOW_ROUTING_CRITICAL_PATH_ALLOCATION_H

#include "TaskflowDialect/Allocation/Allocation.h"
#include "TaskflowDialect/Allocation/allocation_utils.h"

namespace mlir {
namespace taskflow {

/// Concrete allocation strategy: routing-critical-path-first.
///
/// Implements the two-phase fixed-point algorithm:
///   Phase 1: Places tasks in routing-critical-path-first order, scoring each
///            candidate grid position by proximity to SSA predecessors /
///            successors and assigned SRAMs.
///   Phase 2: Assigns each MemRef to the SRAM nearest to the centroid of all
///            CGRAs that access it.
/// Iterates until SRAM assignments converge.
class RoutingCriticalPathAllocation : public Allocation {
public:
  RoutingCriticalPathAllocation(int grid_rows = kCgraGridRows,
                                int grid_cols = kCgraGridCols)
      : grid_rows_(grid_rows), grid_cols_(grid_cols) {}

  /// Places all taskflow.task ops in `func` onto the grid, annotating each
  /// with a `task_allocation_info` attribute.  Returns true on success.
  bool runAllocation(mlir::func::FuncOp func) override;

  std::string getName() const override { return "routing-critical-path-first"; }

private:
  int grid_rows_;
  int grid_cols_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_ROUTING_CRITICAL_PATH_ALLOCATION_H
