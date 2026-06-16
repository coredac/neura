/**
 * @file ExpandMathToArithPass.cpp
 * @brief Emulates math.fpowi and math.tanh with arith/math.exp primitives
 *        so they survive pipeline transformations without type mismatches.
 *
 * @details
 * These emulations must run before --taskflow-conversion, because that pass
 * promotes constants (e.g. exponent 3 for x^3) to kernel block arguments.
 * Once promoted, the exponent is no longer a visible constant and can no
 * longer be emulated with a fixed chain of arith.mulf.
 */

#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Expand math.fpowi(x, const_n) → chain of arith.mulf.
/// Only constant positive integer exponents are supported;
/// negative/zero/unknown exponents are left unchanged.
struct ExpandFPowI : public OpRewritePattern<mlir::math::FPowIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::math::FPowIOp op,
                                PatternRewriter &rewriter) const override {
    Value exponent = op.getOperand(1);
    auto constExp = exponent.getDefiningOp<arith::ConstantOp>();
    if (!constExp)
      return failure();
    auto intAttr = mlir::dyn_cast<IntegerAttr>(constExp.getValue());
    if (!intAttr)
      return failure();
    int64_t expVal = intAttr.getInt();
    if (expVal < 1)
      return failure(); // n ≤ 0: leave for downstream canonicalization

    Type resultType = op.getType();
    Location loc = op.getLoc();
    Value input = op.getOperand(0);
    Value result = input;
    for (int64_t i = 1; i < expVal; ++i)
      result = rewriter.create<arith::MulFOp>(loc, result, input);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Expand math.tanh(x) → (exp(2x) - 1) / (exp(2x) + 1).
/// The intermediate math.exp will be handled downstream (either left as
/// math.exp or later lowered to neura.exp).
struct ExpandTanh : public OpRewritePattern<mlir::math::TanhOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::math::TanhOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getOperand();
    Type resultType = op.getType();
    Location loc = op.getLoc();

    auto two = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getFloatAttr(resultType, 2.0));
    auto one = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getFloatAttr(resultType, 1.0));

    auto twoX = rewriter.create<arith::MulFOp>(loc, input, two);
    auto exp2x = rewriter.create<mlir::math::ExpOp>(loc, twoX);
    auto numerator = rewriter.create<arith::SubFOp>(loc, exp2x, one);
    auto denominator = rewriter.create<arith::AddFOp>(loc, exp2x, one);
    auto result = rewriter.create<arith::DivFOp>(loc, numerator, denominator);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Module-level pass that greedily rewrites math.fpowi and math.tanh.
struct EmulateMathWithArithPass
    : public PassWrapper<EmulateMathWithArithPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmulateMathWithArithPass)

  StringRef getArgument() const override { return "emulate-math-with-arith"; }
  StringRef getDescription() const override {
    return "Emulate math.fpowi and math.tanh with arith/math.exp primitives";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandFPowI, ExpandTanh>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createEmulateMathWithArithPass() {
  return std::make_unique<EmulateMathWithArithPass>();
}
