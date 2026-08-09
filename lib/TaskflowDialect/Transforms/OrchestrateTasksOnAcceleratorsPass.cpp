// Orchestrate Taskflow tasks onto a multi-CGRA grid.

#include "NeuraDialect/Architecture/Architecture.h"
#include "TaskflowDialect/Orchestration/RoutingCriticalPathOrchestration/RoutingCriticalPathOrchestration.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

struct OrchestrateTasksOnAcceleratorsPass
    : public PassWrapper<OrchestrateTasksOnAcceleratorsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OrchestrateTasksOnAcceleratorsPass)

  OrchestrateTasksOnAcceleratorsPass() = default;
  OrchestrateTasksOnAcceleratorsPass(
      const OrchestrateTasksOnAcceleratorsPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "orchestrate-tasks-on-accelerators";
  }
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

  Option<bool> commAware{
      *this, "comm-aware",
      llvm::cl::desc("Weight each memory-proximity penalty by the transferred "
                     "data volume, so placement minimises "
                     "sum(volume * distance) instead of sum(distance) "
                     "(default: false)."),
      llvm::cl::init(false)};

  void runOnOperation() override {
    const StringRef scheduling_mode = schedulingMode.getValue();
    if (scheduling_mode != "spatial" && scheduling_mode != "spatial-temporal") {
      getOperation().emitError()
          << "unknown scheduling-mode '" << scheduling_mode
          << "'; expected spatial or spatial-temporal";
      return signalPassFailure();
    }
    SchedulingMode mode = (scheduling_mode == "spatial")
                              ? SchedulingMode::Spatial
                              : SchedulingMode::SpatialTemporal;
    const neura::Architecture &architecture = neura::getArchitecture();
    RoutingCriticalPathOrchestration strategy(
        architecture.getMultiCgraRows(), architecture.getMultiCgraColumns(),
        mode, commAware.getValue());
    strategy.runTaskOrchestration(getOperation());
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createOrchestrateTasksOnAcceleratorsPass() {
  return std::make_unique<OrchestrateTasksOnAcceleratorsPass>();
}

} // namespace taskflow
} // namespace mlir
