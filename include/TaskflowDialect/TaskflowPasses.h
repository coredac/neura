// TaskflowPasses.h - Header file for Taskflow passes

#ifndef TASKFLOW_PASSES_H
#define TASKFLOW_PASSES_H

#include "TaskflowDialect/Allocation/allocation_utils.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>
namespace mlir {
namespace taskflow {

void registerTaskflowConversionPassPipeline();
void registerTosaToAffineConversionPassPipeline();
void registerLinalgToAffineConversionPassPipeline();

// Passes defined in TaskflowPasses.td
#define GEN_PASS_DECL
#include "TaskflowDialect/TaskflowPasses.h.inc"
std::unique_ptr<mlir::Pass> createConstructHyperblockFromTaskPass();
std::unique_ptr<mlir::Pass> createClassifyCountersPass();
std::unique_ptr<mlir::Pass> createAllocateCgraToTaskPass();
std::unique_ptr<mlir::Pass> createMapTaskOnCgraPass();
std::unique_ptr<mlir::Pass> createFuseTaskPass();

//=========================================================//
// Optimization Passes
//=========================================================//
std::unique_ptr<mlir::Pass> createAffineLoopTreeSerializationPass();
std::unique_ptr<mlir::Pass> createAffineLoopPerfectionPass();
std::unique_ptr<mlir::Pass> createMemoryAccessStreamingFusionPass();
std::unique_ptr<mlir::Pass> createResourceAwareTaskOptimizationPass();

#define GEN_PASS_REGISTRATION
#include "TaskflowDialect/TaskflowPasses.h.inc"
} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_PASSES_H