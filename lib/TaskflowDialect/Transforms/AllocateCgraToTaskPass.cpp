//===- AllocateCgraToTaskPass.cpp - Task to CGRA Mapping Pass -===//
//
// This pass maps Taskflow tasks onto a 2D multi-CGRA grid array:
// 1. Places tasks with SSA dependencies (producer-consumer pairs) on
//    adjacent CGRAs to enable direct data forwarding.
// 2. Assigns memrefs to SRAMs (each MemRef is assigned to exactly one SRAM,
//    determined by proximity to the task that first accesses it).
//
// Implementation: RoutingCriticalPathAllocation in
// lib/TaskflowDialect/Allocation/RoutingCriticalPathAllocation.cpp.
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/Allocation/RoutingCriticalPathAllocation.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

struct AllocateCgraToTaskPass
    : public PassWrapper<AllocateCgraToTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AllocateCgraToTaskPass)

  StringRef getArgument() const override { return "allocate-cgra-to-task"; }
  StringRef getDescription() const override {
    return "Maps Taskflow tasks onto a 2D multi-CGRA grid with adjacency "
           "optimization and memory mapping.";
  }

  void runOnOperation() override {
    RoutingCriticalPathAllocation strategy(kCgraGridRows, kCgraGridCols);
    strategy.runAllocation(getOperation());
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createAllocateCgraToTaskPass() {
  return std::make_unique<AllocateCgraToTaskPass>();
}

} // namespace taskflow
} // namespace mlir
