// Orchestrate Taskflow tasks onto a multi-CGRA grid.

#include "NeuraDialect/Architecture/Architecture.h"
#include "TaskflowDialect/Orchestration/RoutingCriticalPathOrchestration/RoutingCriticalPathOrchestration.h"
#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ThroughputGuidedTaskOrchestration.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

std::unique_ptr<Orchestration>
createOrchestrationStrategy(StringRef strategy_name, int grid_rows,
                            int grid_cols, SchedulingMode mode) {
  return llvm::StringSwitch<std::unique_ptr<Orchestration>>(strategy_name)
      .Case("routing-critical-path",
            std::make_unique<RoutingCriticalPathOrchestration>(grid_rows,
                                                               grid_cols, mode))
      .Case("throughput-guided",
            std::make_unique<ThroughputGuidedTaskOrchestration>(
                grid_rows, grid_cols, mode))
      .Default(nullptr);
}

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

  Option<std::string> orchestrationStrategy{
      *this, "orchestration-strategy",
      llvm::cl::desc("Task orchestration strategy: 'routing-critical-path' "
                     "(default) or 'throughput-guided'."),
      llvm::cl::init("routing-critical-path")};

  void runOnOperation() override {
    SchedulingMode mode = (schedulingMode.getValue() == "spatial")
                              ? SchedulingMode::Spatial
                              : SchedulingMode::SpatialTemporal;
    const neura::Architecture &architecture = neura::getArchitecture();
    std::unique_ptr<Orchestration> strategy = createOrchestrationStrategy(
        orchestrationStrategy.getValue(), architecture.getMultiCgraRows(),
        architecture.getMultiCgraColumns(), mode);
    if (!strategy) {
      getOperation()->emitError() << "unknown task orchestration strategy: "
                                  << orchestrationStrategy.getValue();
      signalPassFailure();
      return;
    }

    strategy->runTaskOrchestration(getOperation());
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
