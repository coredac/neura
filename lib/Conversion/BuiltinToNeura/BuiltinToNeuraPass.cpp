#include "Common/AcceleratorAttrs.h"
#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::neura;

namespace {

struct BuiltinUnrealizedConversionCastToNeuraCast
    : public OpRewritePattern<mlir::UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    // Only handles simple 1:1 casts.
    // TODO: Handle more complex casts if needed.
    if (op.getInputs().size() == 1 && op.getResults().size() == 1) {
      Value input = op.getInputs()[0];
      Type result_type = op.getResults()[0].getType();
      Type input_type = input.getType();

      StringRef cast_type;
      if (input_type.isIndex() && isa<IntegerType>(result_type)) {
        cast_type = "index_to_int";
      } else if (isa<IntegerType>(input_type) && result_type.isIndex()) {
        cast_type = "int_to_index";
      } else {
        return rewriter.notifyMatchFailure(op, "unsupported cast");
      }

      rewriter.replaceOpWithNewOp<neura::CastOp>(
          op, result_type, input, rewriter.getStringAttr(cast_type));
      return success();
    }
    return failure();
  }
};

struct LowerBuiltinToNeuraPass
    : public PassWrapper<LowerBuiltinToNeuraPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuiltinToNeuraPass)

  StringRef getArgument() const override { return "lower-builtin-to-neura"; }
  StringRef getDescription() const override {
    return "Lower Builtin operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  RewritePatternSet populateBuiltinToNeuraPatterns(MLIRContext *context) {
    RewritePatternSet patterns(context);
    patterns.add<BuiltinUnrealizedConversionCastToNeuraCast>(context);
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
          RewritePatternSet patterns = populateBuiltinToNeuraPatterns(context);
          if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
            return signalPassFailure();
          }
        }
      }
    });

    // Applies patterns to the neura.kernel regions.
    module_op.walk([&](neura::KernelOp kernel_op) {
      if (kernel_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto accel_target =
            kernel_op->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (accel_target &&
            accel_target.getValue() == mlir::accel::kNeuraTarget) {
          Region &kernel_region = kernel_op.getBody();
          RewritePatternSet patterns = populateBuiltinToNeuraPatterns(context);
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

std::unique_ptr<Pass> mlir::createLowerBuiltinToNeuraPass() {
  return std::make_unique<LowerBuiltinToNeuraPass>();
}
