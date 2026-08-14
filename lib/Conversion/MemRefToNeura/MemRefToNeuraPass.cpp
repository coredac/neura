#include "Common/AcceleratorAttrs.h"
#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::neura;

namespace {

struct MemRefLoadLowering : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load_op,
                                PatternRewriter &rewriter) const override {
    // Creates a Neura LoadIndexedOp from the MemRef LoadOp.
    Type result_type = load_op.getType();
    Value memref = load_op.getMemRef();
    ValueRange indices = load_op.getIndices();
    rewriter.replaceOpWithNewOp<neura::LoadIndexedOp>(load_op, result_type,
                                                      memref, indices);
    return success();
  }
};

struct MemRefStoreLowering : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp store_op,
                                PatternRewriter &rewriter) const override {
    // Creates a Neura StoreIndexedOp from the MemRef StoreOp.
    Value value = store_op.getValueToStore();
    Value memref = store_op.getMemRef();
    ValueRange indices = store_op.getIndices();
    rewriter.replaceOpWithNewOp<neura::StoreIndexedOp>(store_op, value, memref,
                                                       indices);
    return success();
  }
};

struct MemRefAllocaToNeuraAlloca : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp alloca_op,
                                PatternRewriter &rewriter) const override {
    // Gets the result type.
    Type result_type = alloca_op.getType();

    // Checks if we have dynamic dimensions.
    if (!alloca_op.getDynamicSizes().empty()) {
      // For dynamic dimensions, we need to create the alloca with the size
      // arguments.
      rewriter.replaceOpWithNewOp<neura::AllocaOp>(alloca_op, result_type,
                                                   alloca_op.getDynamicSizes());
    } else {
      // For static dimensions, we can create the alloca without size arguments.
      rewriter.replaceOpWithNewOp<neura::AllocaOp>(alloca_op, result_type,
                                                   Value());
    }

    return success();
  }
};

struct LowerMemRefToNeuraPass
    : public PassWrapper<LowerMemRefToNeuraPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMemRefToNeuraPass)

  StringRef getArgument() const override { return "lower-memref-to-neura"; }
  StringRef getDescription() const override {
    return "Lower MemRef operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  RewritePatternSet populateMemRefToNeuraPatterns(MLIRContext *context) {
    RewritePatternSet patterns(context);
    patterns.add<MemRefLoadLowering>(context);
    patterns.add<MemRefStoreLowering>(context);
    patterns.add<MemRefAllocaToNeuraAlloca>(context);
    return patterns;
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    MLIRContext *context = &getContext();

    module_op.walk([&](func::FuncOp func_op) {
      if (func_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target =
            func_op->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (target && target.getValue() == mlir::accel::kNeuraTarget) {
          RewritePatternSet patterns = populateMemRefToNeuraPatterns(context);
          if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
            return signalPassFailure();
          }
        }
      }
    });

    module_op.walk([&](neura::KernelOp kernel_op) {
      if (kernel_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto accel_target =
            kernel_op->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (accel_target &&
            accel_target.getValue() == mlir::accel::kNeuraTarget) {
          Region &kernel_region = kernel_op.getBody();
          RewritePatternSet patterns = populateMemRefToNeuraPatterns(context);
          if (failed(
                  applyPatternsGreedily(kernel_region, std::move(patterns)))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerMemRefToNeuraPass() {
  return std::make_unique<LowerMemRefToNeuraPass>();
}
