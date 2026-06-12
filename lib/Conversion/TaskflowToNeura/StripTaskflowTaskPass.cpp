//===-- StripTaskflowTaskPass.cpp - Strip taskflow.task wrappers ---------===//
//
// Strips taskflow.task wrappers and promotes their inner operations
// (neura.kernel, arith constants, etc.) to the parent function level,
// producing pure Neura dialect output.
//
//===----------------------------------------------------------------------===//

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

// Strips taskflow.task wrappers by inlining their body into the parent block.
// For each taskflow.task op:
//   1. Find the neura.kernel inside (and any non-terminator non-counter ops).
//   2. Clone them before the task op.
//   3. Replace task results:
//      - Task result 0..N-2 are "read pass-through" results (same as inputs).
//      - Task result N-1 is the "write result" → map to kernel result.
//   4. Erase the task op.
struct StripTaskflowTaskPattern : public OpRewritePattern<TaskflowTaskOp> {
  using OpRewritePattern<TaskflowTaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskflowTaskOp task_op,
                                PatternRewriter &rewriter) const override {
    Block &task_block = task_op.getBody().front();

    // Find the neura.kernel and collect non-kernel ops.
    neura::KernelOp kernel_op = nullptr;
    SmallVector<Operation *> ops_to_clone;
    for (Operation &op : task_block) {
      if (auto k = dyn_cast<neura::KernelOp>(op)) {
        kernel_op = k;
      } else if (!isa<TaskflowYieldOp>(op) && !isa<TaskflowCounterOp>(op)) {
        ops_to_clone.push_back(&op);
      }
    }

    if (!kernel_op) {
      return failure();
    }

    // Clone kernel and non-kernel ops before the task op.
    rewriter.setInsertionPoint(task_op);
    IRMapping mapping;

    // Map task block arguments to task operands.
    for (unsigned i = 0; i < task_op.getODSOperands(0).size(); ++i) {
      mapping.map(task_block.getArgument(i), task_op.getODSOperands(0)[i]);
    }
    // Also handle subsequent operand groups (writes, values).
    // The task has 3 operand groups: reads, writes, values.
    // Block args are in order: reads, writes, values.
    unsigned num_reads = task_op.getODSOperands(0).size();
    unsigned num_writes = task_op.getODSOperands(1).size();
    // unsigned num_values = task_op.getODSOperands(2).size();

    for (unsigned i = 0; i < num_writes; ++i) {
      mapping.map(task_block.getArgument(num_reads + i),
                  task_op.getODSOperands(1)[i]);
    }
    for (unsigned i = 0; i < task_op.getODSOperands(2).size(); ++i) {
      mapping.map(task_block.getArgument(num_reads + num_writes + i),
                  task_op.getODSOperands(2)[i]);
    }

    // Clone non-kernel ops first.
    for (Operation *op : ops_to_clone) {
      rewriter.clone(*op, mapping);
    }

    // Clone the kernel.
    auto *cloned_kernel_op = rewriter.clone(*kernel_op.getOperation(), mapping);
    auto new_kernel = cast<neura::KernelOp>(cloned_kernel_op);

    // Build replacement values for task results.
    // Task results are: [reads..., writes...]
    // - Reads are pass-through (same as read inputs).
    // - Writes come from kernel results, or from write inputs if kernel
    //   has no results (side-effect-only kernel).
    SmallVector<Value> replacements;
    // Pass-through read results.
    for (unsigned i = 0; i < num_reads; ++i) {
      replacements.push_back(task_op.getODSOperands(0)[i]);
    }
    // Write results: prefer kernel results, fall back to mapped write inputs.
    if (new_kernel.getNumResults() > 0) {
      for (Value result : new_kernel.getResults()) {
        replacements.push_back(result);
      }
    } else {
      // Kernel produces no results (side-effect-only).
      // Write results are the same memrefs as write inputs (modified in-place).
      for (unsigned i = 0; i < num_writes; ++i) {
        replacements.push_back(
            mapping.lookupOrDefault(task_block.getArgument(num_reads + i)));
      }
    }

    rewriter.replaceOp(task_op, replacements);

    return success();
  }
};

struct StripTaskflowTaskPass
    : public PassWrapper<StripTaskflowTaskPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripTaskflowTaskPass)

  StringRef getArgument() const override { return "strip-taskflow-task"; }
  StringRef getDescription() const override {
    return "Strip taskflow.task wrappers, promoting inner ops to pure Neura "
           "dialect";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::taskflow::TaskflowDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<StripTaskflowTaskPattern>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createStripTaskflowTaskPass() {
  return std::make_unique<StripTaskflowTaskPass>();
}
