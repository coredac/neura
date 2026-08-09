// Routing-critical-path CGRA orchestration.

#ifndef TASKFLOW_ROUTING_CRITICAL_PATH_ORCHESTRATION_H
#define TASKFLOW_ROUTING_CRITICAL_PATH_ORCHESTRATION_H

#include "TaskflowDialect/Orchestration/Orchestration.h"
#include "TaskflowDialect/Orchestration/orchestration_utils.h"

namespace mlir {
namespace taskflow {

// Concrete orchestration strategy: routing-critical-path-first.
//
// Implements the two-phase fixed-point algorithm:
//   Phase 1: Places tasks in routing-critical-path-first order, scoring each
//            candidate grid position by proximity to SSA predecessors /
//            successors and assigned SRAMs.
//   Phase 2: Assigns each MemRef to the SRAM nearest to the centroid of all
//            CGRAs that access it.
// Iterates until SRAM assignments converge.
//
// In SpatialTemporal mode each task also receives a start_time (ASAP
// scheduling) and duration (from profiling or analytical estimation).
// The output attribute on each taskflow.task op is `task_orchestration_info`.
class RoutingCriticalPathOrchestration : public Orchestration {
public:
  RoutingCriticalPathOrchestration(
      int grid_rows = kCgraGridRows, int grid_cols = kCgraGridCols,
      SchedulingMode mode = SchedulingMode::SpatialTemporal,
      bool comm_aware = false)
      : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode),
        comm_aware_(comm_aware) {}

  // Places all taskflow.task ops in `func` onto the grid, annotating each
  // with a `task_orchestration_info` attribute.  Returns true on success.
  bool runTaskOrchestration(mlir::func::FuncOp func) override;

  std::string getName() const override { return "routing-critical-path-first"; }

private:
  // Task successor graph: producer task operation -> direct consumer task
  // operations.
  using TaskSuccessorMap =
      llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>>;

  // Adds one producer -> consumer edge to the task successor graph if it is
  // valid and not already present.
  void addDependencyEdge(TaskSuccessorMap &successors, Operation *producer,
                         Operation *consumer) const;

  // Computes the longest downstream dependency-chain length starting from
  // `task`. This depth is used as the routing-critical-path priority.
  int computeDependencyDepth(Operation *task, TaskSuccessorMap &successors,
                             llvm::DenseMap<Operation *, int> &depth_cache,
                             llvm::DenseSet<Operation *> &visiting) const;

  // Builds the task dependency graph from Taskflow SSA/token operands and
  // returns the priority map consumed by TaskScheduler.
  TaskPriorityMap computeRoutingCriticalPathPriority(func::FuncOp func) const;

  int grid_rows_;
  int grid_cols_;
  bool comm_aware_ = false;
  SchedulingMode mode_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_ROUTING_CRITICAL_PATH_ORCHESTRATION_H
