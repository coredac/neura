// Implements the routing-critical-path orchestration strategy as a thin
// wrapper around the reusable TaskScheduler backend.

#include "TaskflowDialect/Orchestration/RoutingCriticalPathOrchestration/RoutingCriticalPathOrchestration.h"
#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace taskflow {

void RoutingCriticalPathOrchestration::addDependencyEdge(
    TaskSuccessorMap &successors, Operation *producer,
    Operation *consumer) const {
  if (!producer || !consumer || producer == consumer) {
    return;
  }
  SmallVector<Operation *> &producer_successors = successors[producer];
  if (!llvm::is_contained(producer_successors, consumer)) {
    producer_successors.push_back(consumer);
  }
}

int RoutingCriticalPathOrchestration::computeDependencyDepth(
    Operation *task, TaskSuccessorMap &successors,
    DenseMap<Operation *, int> &depth_cache,
    DenseSet<Operation *> &visiting) const {
  if (auto it = depth_cache.find(task); it != depth_cache.end()) {
    return it->second;
  }
  if (!visiting.insert(task).second) {
    return 0;
  }

  int max_child_depth = 0;
  for (Operation *successor : successors[task]) {
    max_child_depth = std::max(
        max_child_depth,
        computeDependencyDepth(successor, successors, depth_cache, visiting) +
            1);
  }

  visiting.erase(task);
  depth_cache[task] = max_child_depth;
  return max_child_depth;
}

TaskPriorityMap
RoutingCriticalPathOrchestration::computeRoutingCriticalPathPriority(
    func::FuncOp func) const {
  SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

  TaskSuccessorMap successors;
  for (TaskflowTaskOp task : tasks) {
    (void)successors[task.getOperation()];
  }

  for (TaskflowTaskOp consumer : tasks) {
    Operation *consumer_op = consumer.getOperation();
    auto addProducerFromValue = [&](Value value) {
      if (auto producer = value.getDefiningOp<TaskflowTaskOp>()) {
        addDependencyEdge(successors, producer.getOperation(), consumer_op);
      }
    };

    for (Value value_input : consumer.getValueInputs()) {
      addProducerFromValue(value_input);
    }
    for (Value operand : consumer->getOperands()) {
      addProducerFromValue(operand);
    }
  }

  TaskPriorityMap priority;
  DenseMap<Operation *, int> depth_cache;
  DenseSet<Operation *> visiting;
  for (TaskflowTaskOp task : tasks) {
    Operation *task_op = task.getOperation();
    priority[task_op] =
        computeDependencyDepth(task_op, successors, depth_cache, visiting);
  }
  return priority;
}

bool RoutingCriticalPathOrchestration::runTaskOrchestration(func::FuncOp func) {
  TaskPriorityMap priority = computeRoutingCriticalPathPriority(func);
  TaskScheduler scheduler(grid_rows_, grid_cols_, mode_);
  return scheduler.schedule(func, priority);
}

} // namespace taskflow
} // namespace mlir
