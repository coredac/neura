//===- RoutingCriticalPathOrchestration.h -===//
//
// Concrete Orchestration strategy that places Taskflow tasks onto a 2D
// multi-CGRA grid using a routing-critical-path-first ordering.
//
// Tasks with the longest downstream dependency chains are placed first so
// that their successors can land on adjacent CGRAs, minimizing inter-task
// communication distance along the critical path.
//
//===----------------------------------------------------------------------===//

#ifndef TASKFLOW_ROUTING_CRITICAL_PATH_ORCHESTRATION_H
#define TASKFLOW_ROUTING_CRITICAL_PATH_ORCHESTRATION_H

#include "TaskflowDialect/Orchestration/Orchestration.h"
#include "TaskflowDialect/Orchestration/orchestration_utils.h"

namespace mlir {
namespace taskflow {

/// Controls whether a time-slot dimension is added to every CGRA assignment.
enum class OrchestrationMode {
  /// Each CGRA is assigned to at most one task.  Asserts if tasks exceed the
  /// grid size.
  Spatial,
  /// Adds a time-slot dimension.  Tasks that would over-subscribe the grid
  /// are scheduled at a later time slot, enabling temporal reuse of CGRAs.
  SpatialTemporal,
};

/// Concrete orchestration strategy: routing-critical-path-first.
///
/// Implements the two-phase fixed-point algorithm:
///   Phase 1: Places tasks in routing-critical-path-first order, scoring each
///            candidate grid position by proximity to SSA predecessors /
///            successors and assigned SRAMs.
///   Phase 2: Assigns each MemRef to the SRAM nearest to the centroid of all
///            CGRAs that access it.
/// Iterates until SRAM assignments converge.
///
/// In SpatialTemporal mode each task also receives a start_time (ASAP
/// scheduling) and duration (from profiling or analytical estimation).
/// The output attribute on each taskflow.task op is `task_orchestration_info`.
class RoutingCriticalPathOrchestration : public Orchestration {
public:
  RoutingCriticalPathOrchestration(
      int grid_rows = kCgraGridRows, int grid_cols = kCgraGridCols,
      OrchestrationMode mode = OrchestrationMode::SpatialTemporal)
      : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode) {}

  /// Places all taskflow.task ops in `func` onto the grid, annotating each
  /// with a `task_orchestration_info` attribute.  Returns true on success.
  bool runOrchestration(mlir::func::FuncOp func) override;

  std::string getName() const override { return "routing-critical-path-first"; }

private:
  int grid_rows_;
  int grid_cols_;
  OrchestrationMode mode_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_ROUTING_CRITICAL_PATH_ORCHESTRATION_H
