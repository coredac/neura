// TaskFusionUtil.h - Shared task-fusion entry point.
//
// Exposes ONE reusable operation: fuseTaskGroup, which merges a named group of
// taskflow tasks into a single task by reusing the tested UtilizationFuser IR
// merge (performFusion) from ResourceAwareTaskOptimizationPass -- the same code
// path, dominance-safety guard included, that resource-aware fusion uses. This
// lets other passes (e.g. --import-joint-mapping) replay an externally-computed
// fusion decision WITHOUT re-implementing the DFG merge (which is exactly where
// the "operand does not dominate this use" class of bug lived).
//
// The definition lives in ResourceAwareTaskOptimizationPass.cpp so it can call
// the file-local UtilizationFuser / TaskDependencyGraph directly; both the
// resource-aware pass and the import-joint-mapping pass are compiled into the
// MLIRTaskflowOptimization library, so the symbol resolves in-archive.

#ifndef TASKFLOW_TASK_FUSION_UTIL_H
#define TASKFLOW_TASK_FUSION_UTIL_H

#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace mlir {
namespace taskflow {

// Fuses a group of >= 2 taskflow tasks (given by name) into ONE task.
//
// Fusion is performed pairwise, left-to-right: the result of fusing the first
// two tasks is fused with the third, and so on -- exactly how resource-aware
// chains its "utilfused" tasks. Each pairwise step goes through the same
// UtilizationFuser::canFuse / performFusion used by resource-aware, so the
// dominance-safety guard is enforced: an illegal group (dependent tasks,
// multi-block bodies, or a dominance-unsafe merge) is REJECTED with a clear
// message rather than producing invalid IR.
//
//   func         : function containing the tasks (mutated in place on success).
//   task_names   : the group's member task names; size must be >= 2.
//   analytical   : if true, fused tasks are profiled with ResMII/RecMII only
//                  (no mapper) -- fast and mapper-free; recommended for replay.
//   err          : populated with a human-readable reason on failure.
//
// Returns the final fused TaskflowTaskOp on success, or failure() (with `err`
// set and no invalid IR emitted) if the group cannot be fused. On failure the
// IR may be partially fused (earlier legal pairs in the chain), but it is never
// left dominance-invalid, so the caller should signal pass failure.
FailureOr<TaskflowTaskOp> fuseTaskGroup(func::FuncOp func,
                                        llvm::ArrayRef<std::string> task_names,
                                        bool analytical, std::string &err);

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_TASK_FUSION_UTIL_H
