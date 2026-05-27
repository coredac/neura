//===- OrchestrateTaskOnCgraPass.cpp - Task to CGRA Orchestration Pass ---===//
//
// Implements the orchestrate-task-on-cgra pass, which maps Taskflow tasks
// onto a 2D multi-CGRA grid array:
// 1. Places tasks with SSA dependencies (producer-consumer pairs) on
//    adjacent CGRAs to enable direct data forwarding.
// 2. Adds spatial-temporal scheduling (context_id) per task.
// 3. Assigns memrefs to SRAMs (each MemRef is assigned to exactly one SRAM,
//    determined by proximity to the task that first accesses it).
//
// Implementation: RoutingCriticalPathAllocation in
// lib/TaskflowDialect/Allocation/RoutingCriticalPathAllocation.cpp.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/Architecture/Architecture.h"
#include "TaskflowDialect/Allocation/RoutingCriticalPathAllocation.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

struct OrchestrateTaskOnCgraPass
    : public PassWrapper<OrchestrateTaskOnCgraPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrchestrateTaskOnCgraPass)

  OrchestrateTaskOnCgraPass() = default;
  OrchestrateTaskOnCgraPass(const OrchestrateTaskOnCgraPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "orchestrate-task-on-cgra"; }
  StringRef getDescription() const override {
    return "Orchestrates Taskflow tasks onto a 2D multi-CGRA grid (spatial or "
           "spatial-temporal)";
  }

  Option<std::string> orchestrationMode{
      *this, "orchestration-mode",
      llvm::cl::desc("Task orchestration mode: 'spatial' (one task per CGRA, "
                     "asserts if tasks exceed the grid size) or "
                     "'spatial-temporal' (default, time-multiplexes CGRAs so "
                     "task count is not bounded by grid size)."),
      llvm::cl::init("spatial-temporal")};

  void runOnOperation() override {
    OrchestrationMode mode = (orchestrationMode.getValue() == "spatial")
                                 ? OrchestrationMode::Spatial
                                 : OrchestrationMode::SpatialTemporal;
    const neura::Architecture &architecture = neura::getArchitecture();
    RoutingCriticalPathAllocation strategy(architecture.getMultiCgraRows(),
                                           architecture.getMultiCgraColumns(),
                                           mode);
    strategy.runAllocation(getOperation());
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createOrchestrateTaskOnCgraPass() {
  return std::make_unique<OrchestrateTaskOnCgraPass>();
}

} // namespace taskflow
} // namespace mlir
