// Orchestrate Taskflow tasks onto a multi-CGRA grid.

#include "NeuraDialect/Architecture/Architecture.h"
#include "TaskflowDialect/Orchestration/RoutingCriticalPathOrchestration/RoutingCriticalPathOrchestration.h"
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

  Option<std::string> schedulingMode{
      *this, "scheduling-mode",
      llvm::cl::desc("Task scheduling mode: 'spatial' (one task per CGRA, "
                     "asserts if tasks exceed the grid size) or "
                     "'spatial-temporal' (default, time-multiplexes CGRAs so "
                     "task count is not bounded by grid size)."),
      llvm::cl::init("spatial-temporal")};

  void runOnOperation() override {
    SchedulingMode mode = (schedulingMode.getValue() == "spatial")
                              ? SchedulingMode::Spatial
                              : SchedulingMode::SpatialTemporal;
    const neura::Architecture &architecture = neura::getArchitecture();
    RoutingCriticalPathOrchestration strategy(
        architecture.getMultiCgraRows(), architecture.getMultiCgraColumns(),
        mode);
    strategy.runTaskOrchestration(getOperation());
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
