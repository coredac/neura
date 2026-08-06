#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::neura;

namespace {

struct ArithConstantToNeuraConstant
    : public OpRewritePattern<mlir::arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    // Converts arith constant to Neura constant.
    Type result_type = op.getType();
    Attribute value = op.getValue();

    rewriter.replaceOpWithNewOp<neura::ConstantOp>(op, result_type, value);
    return success();
  }
};

struct ArithAddIToNeuraAdd : public OpRewritePattern<mlir::arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::AddOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct ArithFAddToNeuraFAdd : public OpRewritePattern<mlir::arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::FAddOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct ArithSubIToNeuraSub : public OpRewritePattern<mlir::arith::SubIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SubIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::SubOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct ArithSubFToNeuraFSub : public OpRewritePattern<mlir::arith::SubFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SubFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::FSubOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct ArithMulIToNeuraMul : public OpRewritePattern<mlir::arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::MulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct ArithMulFToNeuraFMul : public OpRewritePattern<mlir::arith::MulFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::FMulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct ArithDivSIToNeuraDiv : public OpRewritePattern<mlir::arith::DivSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    // Converts arith DivSIOp to Neura DivOp.
    rewriter.replaceOpWithNewOp<neura::DivOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct ArithFDivToNeuraFDiv : public OpRewritePattern<mlir::arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::FDivOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct ArithRemSIToNeuraOp : public OpRewritePattern<mlir::arith::RemSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    Location loc = op.getLoc();
    // Converts arith RemSIOp to basic Neura Op.

    Value div = rewriter.create<neura::DivOp>(loc, result_type, lhs, rhs);
    Value mul = rewriter.create<neura::MulOp>(loc, result_type, rhs, div);
    Value rem = rewriter.create<neura::SubOp>(loc, result_type, lhs, mul);

    rewriter.replaceOp(op, rem);
    return success();
  }
};

struct ArithCmpiToNeuraICmp : public OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    arith::CmpIPredicate arith_cmp_type = op.getPredicate();
    StringRef cmp_type;
    switch (arith_cmp_type) {
    case arith::CmpIPredicate::eq:
      cmp_type = "eq"; // ==
      break;
    case arith::CmpIPredicate::ne:
      cmp_type = "ne"; // !=
      break;
    case arith::CmpIPredicate::slt:
      cmp_type = "slt"; // <
      break;
    case arith::CmpIPredicate::sle:
      cmp_type = "sle"; // <=
      break;
    case arith::CmpIPredicate::sgt:
      cmp_type = "sgt"; // >
      break;
    case arith::CmpIPredicate::sge:
      cmp_type = "sge"; // >=
      break;
    case arith::CmpIPredicate::ult:
      cmp_type = "ult"; // unsigned <
      break;
    case arith::CmpIPredicate::ule:
      cmp_type = "ule"; // unsigned <=
      break;
    case arith::CmpIPredicate::ugt:
      cmp_type = "ugt"; // unsigned >
      break;
    case arith::CmpIPredicate::uge:
      cmp_type = "uge"; // unsigned >=
      break;
    default:
      return rewriter.notifyMatchFailure(op, "Unsupported arith CmpIOp type");
    }

    // Converts arith CmpIOp to Neura ICmpOp.

    rewriter.replaceOpWithNewOp<neura::ICmpOp>(
        op, result_type, lhs, rhs, rewriter.getStringAttr(cmp_type));
    return success();
  }
};

// arith.cmpf(a, b, pred) → neura.fcmp(a, b, pred). Mirrors ArithCmpiToNeuraICmp
// for floating-point comparisons (neura.fcmp already exists and is used by the
// min/max lowerings). Enables fp programs (e.g. softmax) through the pipeline.
struct ArithCmpfToNeuraFCmp : public OpRewritePattern<mlir::arith::CmpFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    arith::CmpFPredicate pred = op.getPredicate();
    StringRef cmp_type;
    switch (pred) {
    case arith::CmpFPredicate::OEQ: cmp_type = "oeq"; break;
    case arith::CmpFPredicate::OGT: cmp_type = "ogt"; break;
    case arith::CmpFPredicate::OGE: cmp_type = "oge"; break;
    case arith::CmpFPredicate::OLT: cmp_type = "olt"; break;
    case arith::CmpFPredicate::OLE: cmp_type = "ole"; break;
    case arith::CmpFPredicate::ONE: cmp_type = "one"; break;
    case arith::CmpFPredicate::ORD: cmp_type = "ord"; break;
    case arith::CmpFPredicate::UEQ: cmp_type = "ueq"; break;
    case arith::CmpFPredicate::UGT: cmp_type = "ugt"; break;
    case arith::CmpFPredicate::UGE: cmp_type = "uge"; break;
    case arith::CmpFPredicate::ULT: cmp_type = "ult"; break;
    case arith::CmpFPredicate::ULE: cmp_type = "ule"; break;
    case arith::CmpFPredicate::UNE: cmp_type = "une"; break;
    case arith::CmpFPredicate::UNO: cmp_type = "uno"; break;
    default:
      return rewriter.notifyMatchFailure(op, "Unsupported arith CmpFOp type");
    }
    rewriter.replaceOpWithNewOp<neura::FCmpOp>(
        op, result_type, lhs, rhs, rewriter.getStringAttr(cmp_type));
    return success();
  }
};

struct ArithSelectToNeuraSel : public OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value condition = op.getCondition();
    Value true_value = op.getTrueValue();
    Value false_value = op.getFalseValue();
    Type result_type = op.getType();

    // Converts arith SelectOp to Neura SelOp with consistent order: (cond,
    // ifTrue, ifFalse).
    rewriter.replaceOpWithNewOp<neura::SelOp>(op, result_type, condition,
                                              true_value, false_value);
    return success();
  }
};

struct ArithExtUIToNeuraCast : public OpRewritePattern<mlir::arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtUIOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getIn();
    Type result_type = op.getType();

    // Converts arith ExtUIOp to Neura cast operation.

    rewriter.replaceOpWithNewOp<neura::CastOp>(op, result_type, input,
                                               rewriter.getStringAttr("extui"));
    return success();
  }
};

struct ArithExtfToNeuraCast : public OpRewritePattern<mlir::arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getIn();
    Type result_type = op.getType();

    // Converts arith ExtFOp to Neura cast operation.

    rewriter.replaceOpWithNewOp<neura::CastOp>(op, result_type, input,
                                               rewriter.getStringAttr("extf"));
    return success();
  }
};

struct ArithIndexCastToNeuraCast
    : public OpRewritePattern<mlir::arith::IndexCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getIn();
    Type result_type = op.getType();
    Type in_type = input.getType();
    StringRef cast_string;

    // The isa<IntegerType> check is generic and handles any integer bit
    // width (e.g., i32, i64).
    if (in_type.isIndex() && isa<IntegerType>(result_type)) {
      cast_string = "index_to_int";
    } else if (isa<IntegerType>(in_type) && result_type.isIndex()) {
      cast_string = "int_to_index";
    } else {
      return rewriter.notifyMatchFailure(op, "index_cast");
    }

    // Converts arith IndexCastOp to Neura cast operation.

    rewriter.replaceOpWithNewOp<neura::CastOp>(
        op, result_type, input, rewriter.getStringAttr(cast_string));
    return success();
  }
};

struct ArithMinimumFToNeuraFCmpSel
    : public OpRewritePattern<mlir::arith::MinimumFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MinimumFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    Location loc = op.getLoc();

    // minimumf(a, b) → sel(fcmp(a, b, "olt"), a, b)
    // "olt" = ordered less-than: true when a < b (false if either is NaN).
    Value cmp = rewriter.create<neura::FCmpOp>(loc, result_type, lhs, rhs,
                                               rewriter.getStringAttr("olt"));
    rewriter.replaceOpWithNewOp<neura::SelOp>(op, result_type, cmp, lhs, rhs);
    return success();
  }
};

struct ArithMaximumFToNeuraFCmpSel
    : public OpRewritePattern<mlir::arith::MaximumFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MaximumFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    Location loc = op.getLoc();

    // maximumf(a, b) → sel(fcmp(a, b, "ogt"), a, b)
    // "ogt" = ordered greater-than: true when a > b (false if either is NaN).
    Value cmp = rewriter.create<neura::FCmpOp>(loc, result_type, lhs, rhs,
                                               rewriter.getStringAttr("ogt"));
    rewriter.replaceOpWithNewOp<neura::SelOp>(op, result_type, cmp, lhs, rhs);
    return success();
  }
};

// arith.andi(a, b) → neura.and(a, b)
struct ArithAndIToNeuraAnd : public OpRewritePattern<mlir::arith::AndIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AndIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    rewriter.replaceOpWithNewOp<neura::AndOp>(op, result_type, lhs, rhs);
    return success();
  }
};

// arith.ori(a, b) → neura.or(a, b)
struct ArithOrIToNeuraOr : public OpRewritePattern<mlir::arith::OrIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::OrIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    rewriter.replaceOpWithNewOp<neura::OrOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LowerArithToNeuraPass
    : public PassWrapper<LowerArithToNeuraPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerArithToNeuraPass)

  StringRef getArgument() const override { return "lower-arith-to-neura"; }
  StringRef getDescription() const override {
    return "Lower arith dialect operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  RewritePatternSet populateArithToNeuraPatterns(MLIRContext *context) {
    RewritePatternSet patterns(context);
    patterns.add<
        ArithFAddToNeuraFAdd, ArithConstantToNeuraConstant, ArithAddIToNeuraAdd,
        ArithCmpiToNeuraICmp, ArithCmpfToNeuraFCmp, ArithSelectToNeuraSel, ArithExtUIToNeuraCast,
        ArithIndexCastToNeuraCast, ArithFDivToNeuraFDiv, ArithExtfToNeuraCast,
        ArithMulFToNeuraFMul, ArithSubIToNeuraSub, ArithSubFToNeuraFSub,
        ArithMulIToNeuraMul, ArithDivSIToNeuraDiv, ArithRemSIToNeuraOp,
        ArithMinimumFToNeuraFCmpSel, ArithMaximumFToNeuraFCmpSel,
        ArithAndIToNeuraAnd, ArithOrIToNeuraOr>(context);
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
          RewritePatternSet patterns = populateArithToNeuraPatterns(context);
          // Apply patterns to the function, not the entire module
          if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
            signalPassFailure();
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
          RewritePatternSet patterns = populateArithToNeuraPatterns(context);
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

std::unique_ptr<mlir::Pass> mlir::createLowerArithToNeuraPass() {
  return std::make_unique<LowerArithToNeuraPass>();
}
