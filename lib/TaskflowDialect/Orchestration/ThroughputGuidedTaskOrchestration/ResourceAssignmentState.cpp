// Implements throughput-guided resource assignment state.

#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ResourceAssignmentState.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/IR/Attributes.h"

#include <algorithm>

namespace mlir {
namespace taskflow {

static bool getBoolAttr(Operation *op, StringRef attr_name) {
  if (auto attr = op->getAttrOfType<BoolAttr>(attr_name)) {
    return attr.getValue();
  }
  return false;
}

static int getProfiledDuration(Operation *op) {
  auto profile_info = op->getAttrOfType<DictionaryAttr>("profile_info");
  if (!profile_info) {
    return 1;
  }

  auto duration = dyn_cast_or_null<IntegerAttr>(profile_info.get("duration"));
  if (!duration) {
    return 1;
  }

  return std::max(1, static_cast<int>(duration.getInt()));
}

static TaskResourceAssignment buildTaskResourceAssignment(TaskflowTaskOp task) {
  Operation *op = task.getOperation();
  TaskResourceAssignment assignment;
  assignment.task = op;
  assignment.estimated_latency = getProfiledDuration(op);
  assignment.dlp_replicable = getBoolAttr(op, "dlp_replicable");
  return assignment;
}

ResourceAssignmentState
ResourceAssignmentState::buildInitialResourceAssignment(func::FuncOp func) {
  ResourceAssignmentState state;
  func.walk([&](TaskflowTaskOp task) {
    state.resource_assignments_.push_back(buildTaskResourceAssignment(task));
  });
  return state;
}

TaskPriorityMap ResourceAssignmentState::buildInitialTaskPriority() const {
  TaskPriorityMap priority;
  for (const TaskResourceAssignment &assignment : resource_assignments_) {
    priority[assignment.task] = 0;
  }
  return priority;
}

} // namespace taskflow
} // namespace mlir
