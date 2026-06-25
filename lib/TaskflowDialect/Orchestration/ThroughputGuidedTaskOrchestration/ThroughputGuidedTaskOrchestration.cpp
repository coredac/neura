// Implements the throughput-guided task orchestration strategy.

#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ThroughputGuidedTaskOrchestration.h"
#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ResourceAssignmentState.h"
#include "TaskflowDialect/Orchestration/orchestration_utils.h"

namespace mlir {
namespace taskflow {

bool ThroughputGuidedTaskOrchestration::runTaskOrchestration(
    func::FuncOp func) {
  ResourceAssignmentState assignment_state =
      ResourceAssignmentState::buildInitialResourceAssignment(func);
  TaskPriorityMap priority = assignment_state.buildInitialTaskPriority();
  TaskScheduler scheduler(grid_rows_, grid_cols_, mode_);
  return scheduler.schedule(func, priority);
}

} // namespace taskflow
} // namespace mlir
