// Throughput-guided resource assignment state.

#ifndef TASKFLOW_THROUGHPUT_GUIDED_RESOURCE_ASSIGNMENT_STATE_H
#define TASKFLOW_THROUGHPUT_GUIDED_RESOURCE_ASSIGNMENT_STATE_H

#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir {
namespace taskflow {

// Static resource choice currently assigned to one task.
//
// The throughput-guided search mutates these fields speculatively before
// re-running TaskScheduler. The initial state is intentionally conservative:
// one CGRA and one active replica per task.
struct TaskResourceAssignment {
  Operation *task = nullptr;

  // Number of adjacent CGRAs composed into one larger execution unit for this
  // task. A value of 1 means the task uses one standalone CGRA.
  int composed_cgra_count = 1;

  // Shape of the composed CGRA group. "1x1" means no CGRA stitching.
  std::string composed_cgra_shape = "1x1";

  int active_replicas = 1;
  int estimated_latency = 1;
  bool dlp_replicable = false;
};

// Per-function static orchestration state used by throughput-guided search.
class ResourceAssignmentState {
public:
  // Creates a baseline resource assignment for every task in `func`.
  //
  // The baseline is intentionally conservative: one standalone CGRA and one
  // active replica per task.
  explicit ResourceAssignmentState(func::FuncOp func);

  // Returns all per-task resource assignments in deterministic IR walk order.
  llvm::ArrayRef<TaskResourceAssignment> getResourceAssignments() const {
    return resource_assignments_;
  }

  // Builds the neutral task-priority map used for the baseline schedule.
  TaskPriorityMap buildInitialTaskPriority() const;

private:
  llvm::SmallVector<TaskResourceAssignment> resource_assignments_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_THROUGHPUT_GUIDED_RESOURCE_ASSIGNMENT_STATE_H
