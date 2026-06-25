// Implements the throughput-guided task orchestration strategy.

#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ThroughputGuidedTaskOrchestration.h"
#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"

namespace mlir {
namespace taskflow {

TaskPriorityMap ThroughputGuidedTaskOrchestration::buildInitialPriority(
    func::FuncOp func) const {
  TaskPriorityMap priority;
  func.walk([&](TaskflowTaskOp task) { priority[task.getOperation()] = 0; });
  return priority;
}

bool ThroughputGuidedTaskOrchestration::runTaskOrchestration(
    func::FuncOp func) {
  TaskPriorityMap priority = buildInitialPriority(func);
  TaskScheduler scheduler(grid_rows_, grid_cols_, mode_);
  return scheduler.schedule(func, priority);
}

} // namespace taskflow
} // namespace mlir
