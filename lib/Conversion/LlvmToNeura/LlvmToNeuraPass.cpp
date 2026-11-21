#include "Common/AcceleratorAttrs.h"
#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::neura;

namespace {
// Lowers integer add from mlir.llvm.add to nuera.add. Provides the lowering
// here instead of tablegen due to that mlir.llvm.add uses an EnumProperty
// (IntegerOverflowFlags) defined via MLIR interfaces — which DRR cannot match
// on or extract from.
struct LlvmAddToNeuraAdd : public OpRewritePattern<mlir::LLVM::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::AddOp>(op, op.getType(), op.getLhs(),
                                              op.getRhs());
    return success();
  }
};

struct LlvmFAddToNeuraFAdd : public OpRewritePattern<mlir::LLVM::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FAddOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FAddOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmFSubToNeuraFSub : public OpRewritePattern<mlir::LLVM::FSubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FSubOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op.getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<neura::FSubOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmAndToNeuraAnd : public OpRewritePattern<mlir::LLVM::AndOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::AndOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::AndOp>(op, op.getType(), op.getLhs(),
                                              op.getRhs());
    return success();
  }
};

struct LlvmOrToNeuraOr : public OpRewritePattern<mlir::LLVM::OrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::OrOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::OrOp>(op, op.getType(), op.getLhs(),
                                             op.getRhs());
    return success();
  }
};

struct LlvmFMulToNeuraFMul : public OpRewritePattern<mlir::LLVM::FMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FMulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmSDivToNeuraDiv : public OpRewritePattern<LLVM::SDivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SDivOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::DivOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmSRemToNeuraRem : public OpRewritePattern<LLVM::SRemOp> {
  using OpRewritePattern<LLVM::SRemOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SRemOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Creates neura.rem operation to replace llvm.srem.
    rewriter.replaceOpWithNewOp<neura::RemOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmMaxNumToNeuraFMax : public OpRewritePattern<LLVM::MaxNumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MaxNumOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMaxOp>(
        op, result_type, lhs, rhs, rewriter.getStringAttr("maxnum"));
    return success();
  }
};

struct LlvmMaximumToNeuraFMax : public OpRewritePattern<LLVM::MaximumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MaximumOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMaxOp>(
        op, result_type, lhs, rhs, rewriter.getStringAttr("maximum"));
    return success();
  }
};

struct LlvmMinNumToNeuraFMin : public OpRewritePattern<LLVM::MinNumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MinNumOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMinOp>(
        op, result_type, lhs, rhs, rewriter.getStringAttr("minnum"));
    return success();
  }
};

struct LlvmMinimumToNeuraFMin : public OpRewritePattern<LLVM::MinimumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MinimumOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMinOp>(
        op, result_type, lhs, rhs, rewriter.getStringAttr("minimum"));
    return success();
  }
};

struct LlvmFDivToNeuraFDiv : public OpRewritePattern<mlir::LLVM::FDivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FDivOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FDivOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmFPToSIToNeuraCast : public OpRewritePattern<mlir::LLVM::FPToSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FPToSIOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getArg();
    Type result_type = op.getType();

    // Creates a cast operation with "fptosi" as the cast type.
    rewriter.replaceOpWithNewOp<neura::CastOp>(
        op, result_type, input, rewriter.getStringAttr("fptosi"));
    return success();
  }
};

struct LlvmSelectToNeuraSel : public OpRewritePattern<LLVM::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value cond = op.getCondition();
    Value true_value = op.getTrueValue();
    Value false_value = op.getFalseValue();
    Type result_type = op.getType();

    // neura.sel now follows the same order as llvm.select: (cond, ifTrue,
    // ifFalse)
    rewriter.replaceOpWithNewOp<neura::SelOp>(op, result_type, cond, true_value,
                                              false_value);
    return success();
  }
};

struct LlvmFMulAddToNeuraFMulFAdd
    : public OpRewritePattern<mlir::LLVM::FMulAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FMulAddOp op,
                                PatternRewriter &rewriter) const override {
    Value a = op->getOperand(0);
    Value b = op->getOperand(1);
    Value c = op->getOperand(2);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMulFAddOp>(op, result_type, a, b, c);
    return success();
  }
};

// Handles LLVM intrinsic memset operations.
struct LlvmMemsetToNeuraOps : public OpRewritePattern<LLVM::MemsetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MemsetOp op,
                                PatternRewriter &rewriter) const override {
    // Gets all operands: dest, value, len, is_volatile.
    auto dest = op.getDst();
    auto value = op.getVal();
    auto len = op.getLen();
    auto is_volatile = op.getIsVolatile();

    // Creates neura.memset operation with full semantics.
    // Passes all operands to the hardware-specific operation.
    // The RTL layer can implement this as appropriate for the target hardware.
    rewriter.replaceOpWithNewOp<neura::MemsetOp>(op, dest, value, len,
                                                 is_volatile);
    return success();
  }
};

struct LlvmVFMulToNeuraVFMul : public OpRewritePattern<mlir::LLVM::FMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FMulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xf32>.
    auto vec_type = mlir::dyn_cast<VectorType>(result_type);
    if (!vec_type || !mlir::isa<FloatType>(vec_type.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VFMulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmVMulToNeuraVMul : public OpRewritePattern<mlir::LLVM::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::MulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xInt>.
    auto vec_type = mlir::dyn_cast<VectorType>(result_type);
    if (!vec_type || !mlir::isa<IntegerType>(vec_type.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VMulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmVAddToNeuraVAdd : public OpRewritePattern<mlir::LLVM::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::AddOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xInt>.
    auto vec_type = mlir::dyn_cast<VectorType>(result_type);
    if (!vec_type || !mlir::isa<IntegerType>(vec_type.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VAddOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmVFAddToNeuraVFAdd : public OpRewritePattern<mlir::LLVM::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FAddOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xf32>.
    auto vec_type = mlir::dyn_cast<VectorType>(result_type);
    if (!vec_type || !mlir::isa<FloatType>(vec_type.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VFAddOp>(op, result_type, lhs, rhs);
    return success();
  }
};

// Handles LLVM intrinsic operations like llvm.intr.vector.reduce.add.
// These are generic intrinsic calls, not specific op types.
struct LlvmVectorReduceAddToNeuraVectorReduceAdd : public RewritePattern {
  LlvmVectorReduceAddToNeuraVectorReduceAdd(MLIRContext *context)
      : RewritePattern("llvm.intr.vector.reduce.add", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Checks that we have exactly one operand and one result.
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    Value input = op->getOperand(0);
    Type result_type = op->getResult(0).getType();

    rewriter.replaceOpWithNewOp<neura::VectorReduceAddOp>(op, result_type,
                                                          input);
    return success();
  }
};

struct LlvmICmpToNeuraICmp : public OpRewritePattern<LLVM::ICmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ICmpOp op,
                                PatternRewriter &rewriter) const override {
    auto pred = op.getPredicate();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::ICmpOp>(
        op, result_type, lhs, rhs,
        rewriter.getStringAttr(LLVM::stringifyICmpPredicate(pred)));
    return success();
  }
};

struct LlvmFCmpToNeuraFCmp : public OpRewritePattern<LLVM::FCmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FCmpOp op,
                                PatternRewriter &rewriter) const override {
    auto pred = op.getPredicate();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::FCmpOp>(
        op, result_type, lhs, rhs,
        rewriter.getStringAttr(LLVM::stringifyFCmpPredicate(pred)));
    return success();
  }
};

struct LlvmGEPToNeuraGEP : public OpRewritePattern<mlir::LLVM::GEPOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::GEPOp op,
                                PatternRewriter &rewriter) const override {
    Value base = op.getBase();
    SmallVector<Value> index_values;

    for (auto gep_index : op.getIndices()) {
      if (auto val = gep_index.dyn_cast<Value>()) {
        index_values.push_back(val);
      } else if (auto int_attr = gep_index.dyn_cast<IntegerAttr>()) {
        // Creates constant operation state manually.
        OperationState state(op.getLoc(),
                             neura::ConstantOp::getOperationName());
        state.addAttribute("value", int_attr);
        state.addTypes(rewriter.getIndexType());
        Value cst = rewriter.create(state)->getResult(0);
        index_values.push_back(cst);
      } else {
        return op.emitOpError("Unsupported GEP index kind");
      }
    }

    rewriter.replaceOpWithNewOp<neura::GEP>(op, op.getType(), base,
                                            index_values);
    return success();
  }
};

struct LlvmLoadToNeuraLoad : public OpRewritePattern<mlir::LLVM::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::LoadOp op,
                                PatternRewriter &rewriter) const override {
    Value ptr = op.getAddr(); // getPointer() is deprecated.
    Type result_type = op.getResult().getType();
    rewriter.replaceOpWithNewOp<neura::LoadOp>(op, result_type, ptr);
    return success();
  }
};

struct LlvmStoreToNeuraStore : public OpRewritePattern<mlir::LLVM::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value value = op.getValue();
    Value addr = op.getAddr(); // getPointer() is deprecated
    rewriter.replaceOpWithNewOp<neura::StoreOp>(op, value, addr);
    return success();
  }
};

struct LlvmCondBrToNeuraCondBr : public OpRewritePattern<LLVM::CondBrOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CondBrOp op,
                                PatternRewriter &rewriter) const override {
    // Gets the source operation's successors (basic blocks).
    Block *true_dest = op.getTrueDest();
    Block *false_dest = op.getFalseDest();

    // Gets the operands for each destination.
    ValueRange true_operands = op.getTrueDestOperands();
    ValueRange false_operands = op.getFalseDestOperands();

    // Creates the new operation with proper successors.
    auto new_op = rewriter.create<neura::CondBr>(
        op.getLoc(),       // Location
        op.getCondition(), // Condition
        true_operands,     // True destination operands
        false_operands,    // False destination operands
        true_dest,         // True destination block
        false_dest         // False destination block
    );

    // Replaces the old op with the new one.
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }
};

struct LlvmBrToNeuraBr : public OpRewritePattern<LLVM::BrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::BrOp op,
                                PatternRewriter &rewriter) const override {
    // Gets the destination block and its operands.
    Block *dest = op.getDest();
    ValueRange dest_operands = op.getDestOperands();

    // Creates the new Neura_Br operation.
    rewriter.replaceOpWithNewOp<neura::Br>(op, dest_operands, dest);

    return success();
  }
};

struct LlvmReturnToNeuraReturn : public OpRewritePattern<LLVM::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::ReturnOp>(op, op.getOperands());
    return success();
  }
};

struct LlvmFNegToNeuraFNeg : public OpRewritePattern<LLVM::FNegOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FNegOp op,
                                PatternRewriter &rewriter) const override {
    // Gets operand.
    Value operand = op.getOperand();
    Type result_type = op.getType();

    // Replaces with neura.fneg operation.
    rewriter.replaceOpWithNewOp<neura::FNegOp>(op, result_type, operand);
    return success();
  }
};

struct LlvmSubToNeuraSub : public OpRewritePattern<LLVM::SubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SubOp op,
                                PatternRewriter &rewriter) const override {
    // Gets operands.
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Replaces with neura.sub.
    rewriter.replaceOpWithNewOp<neura::SubOp>(op, result_type, lhs, rhs);
    return success();
  }
};

// TODO: Implements LlvmXOrToNeuraOr. Used in ADPCM coder and FFT kernels.
//       llvm.xor operations appear in:
//       - adpcm_coder-kernel.mlir (line 104: %87 = llvm.xor %29, %19 : i1)
//       - fft_kernel.mlir (line 19: %11 = llvm.xor %10, %3 : i32)
//       Implementation: xor(a, b) = or(a, b) for boolean values.

// TODO: Implements LlvmAndToNeuraMul. Used in ADPCM coder and MVT kernels.
//       llvm.and operations appear in:
//       - adpcm_coder-kernel.mlir (lines 55, 94: bitwise AND operations)
//       - mvt-kernel.mlir (lines 44, 47, 50, 53: vector and scalar AND
//       operations) Implementation: and(a, b) = mul(a, b) for boolean values.

// TODO: Implements LlvmAllocaToNeuraOps. Used in DTW kernel.
//       llvm.alloca operations appear in:
//       - dtw-kernel-O0.mlir (lines 19-23: multiple stack allocations)
//       Implementation: For CGRA, erases alloca or converts to register
//       allocation.

// TODO: Implements LlvmLShrToNeuraShl. Used in ADPCM coder/decoder and FFT
// kernels.
//       llvm.lshr operations appear in:
//       - adpcm_coder-kernel.mlir (line 54: %42 = llvm.lshr %40, %7 : i32)
//       - adpcm_decoder-kernel.ll (line 35: %30 = lshr i32 %29, 4)
//       - fft_kernel.mlir (line 67: %49 = llvm.lshr %7, %1 : i32)
//       Implementation: Needs proper logical right shift (lshr(x,n) !=
//       shl(x,-n)).

// TODO: Implements LlvmAShrToNeuraAShr. Used in ADPCM coder/decoder kernels.
//       llvm.ashr operations appear in:
//       - adpcm_coder-kernel.mlir (lines 57, 63, 70: multiple ashr operations)
//       - adpcm_decoder-kernel.ll (lines 49, 56, 61: ashr i32 %20, 3/1/2)
//       Implementation: Needs proper arithmetic right shift (preserves sign
//       bit).

struct LlvmSMaxToNeuraSMax : public OpRewritePattern<LLVM::SMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SMaxOp op,
                                PatternRewriter &rewriter) const override {
    // Gets operands.
    Value lhs = op.getA();
    Value rhs = op.getB();
    Type result_type = op.getType();
    Location loc = op.getLoc();

    // Implements smax(a, b) = a >= b ? a : b.
    auto cmp = rewriter.create<neura::ICmpOp>(
        loc, rewriter.getI1Type(), lhs, rhs, rewriter.getStringAttr("sge"));

    // Selects: a >= b ? a : b.
    rewriter.replaceOpWithNewOp<neura::SelOp>(op, result_type, cmp, lhs, rhs);
    return success();
  }
};

// TODO: Implements LlvmAbsToNeuraAbs. Used in ADPCM coder kernel.
//       llvm.intr.abs operations appear in adpcm_coder-kernel.mlir.
//       Implementation: abs(x) = x >= 0 ? x : -x (using ICmpOp + SelOp).

struct FuncReturnToNeuraReturn : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::ReturnOp>(op, op.getOperands());
    return success();
  }
};

struct LlvmConstantToNeuraConstant : public OpRewritePattern<LLVM::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto attr = op.getValue();

    // Creates operation state manually.
    OperationState state(op.getLoc(), neura::ConstantOp::getOperationName());
    state.addAttribute("value", attr);
    state.addTypes(op.getType());

    // Creates the operation and replaces.
    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op->getResults());
    return success();
  }
};

struct LlvmAddressOfToNeuraConstant
    : public OpRewritePattern<LLVM::AddressOfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AddressOfOp op,
                                PatternRewriter &rewriter) const override {
    // Represents address-of as a Neura constant carrying the referenced
    // global symbol.
    Attribute global_symbol_attr = op->getAttr("global_name");
    if (!global_symbol_attr) {
      return op.emitOpError("expects global_name attribute");
    }

    OperationState state(op.getLoc(), neura::ConstantOp::getOperationName());
    state.addAttribute("value", global_symbol_attr);
    state.addTypes(op.getType());

    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op->getResults());
    return success();
  }
};

struct LlvmAllocaToNeuraAlloca : public OpRewritePattern<LLVM::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    Value size = op.getArraySize();
    Type result_type = op.getType();

    // Converts the size to neura.data<i32, i1> if it's not already.
    // Assumes the size is already in the right format.
    // Handles type conversion here.

    rewriter.replaceOpWithNewOp<neura::AllocaOp>(op, result_type, size);
    return success();
  }
};

struct LlvmSExtToNeuraSExt : public OpRewritePattern<LLVM::SExtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SExtOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getArg();
    Type result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::SExtOp>(op, result_type, input);
    return success();
  }
};

struct LlvmZExtToNeuraZExt : public OpRewritePattern<LLVM::ZExtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ZExtOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getArg();
    Type result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::ZExtOp>(op, result_type, input);
    return success();
  }
};

struct LlvmTruncToNeuraCast : public OpRewritePattern<LLVM::TruncOp> {
  using OpRewritePattern<LLVM::TruncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::TruncOp op,
                                PatternRewriter &rewriter) const override {
    // Trunc is a simple cast operation.
    auto result =
        rewriter.create<neura::CastOp>(op.getLoc(), op.getType(), op.getArg(),
                                       rewriter.getStringAttr("trunc"));
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

struct LlvmUDivToNeuraDiv : public OpRewritePattern<LLVM::UDivOp> {
  using OpRewritePattern<LLVM::UDivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::UDivOp op,
                                PatternRewriter &rewriter) const override {
    // UDiv is unsigned division.
    auto result = rewriter.create<neura::DivOp>(op.getLoc(), op.getType(),
                                                op.getLhs(), op.getRhs());
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

struct LlvmURemToNeuraRem : public OpRewritePattern<LLVM::URemOp> {
  using OpRewritePattern<LLVM::URemOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::URemOp op,
                                PatternRewriter &rewriter) const override {
    // URem is unsigned remainder.
    auto result = rewriter.create<neura::RemOp>(op.getLoc(), op.getType(),
                                                op.getLhs(), op.getRhs());
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

struct LlvmMulToNeuraMul : public OpRewritePattern<LLVM::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::MulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmShlToNeuraShl : public OpRewritePattern<LLVM::ShlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ShlOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    rewriter.replaceOpWithNewOp<neura::ShlOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmFuncToNeuraFunc : public OpRewritePattern<LLVM::LLVMFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp op,
                                PatternRewriter &rewriter) const override {

    auto target = op->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
    if (!target || target.getValue() != mlir::accel::kNeuraTarget) {
      return failure();
    }

    // Converts LLVMFunctionType to FunctionType.
    auto llvm_func_type = op.getFunctionType();
    auto func_type = rewriter.getFunctionType(llvm_func_type.getParams(),
                                              llvm_func_type.getReturnType());

    // Creates the new func.func operation using OperationState to have full
    // control.
    OperationState state(op.getLoc(), func::FuncOp::getOperationName());
    state.addAttribute("sym_name", rewriter.getStringAttr(op.getName()));
    state.addAttribute("function_type", TypeAttr::get(func_type));

    // Copies ALL attributes from the original llvm.func exactly as they are.
    // Skips function type and name attributes as they are handled separately.
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "function_type" || attr.getName() == "sym_name") {
        continue;
      }
      attrs.push_back(attr);
    }
    state.addAttributes(attrs);

    // Adds the function body region.
    state.addRegion();

    auto new_func = cast<func::FuncOp>(rewriter.create(state));

    // Moves the function body.
    rewriter.inlineRegionBefore(op.getBody(), new_func.getBody(),
                                new_func.getBody().end());

    // Replaces the old function.
    rewriter.replaceOp(op, new_func);
    return success();
  }
};

struct LlvmCallToFuncCall : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rewriter) const override {
    // Gets the callee name.
    auto callee = op.getCallee();
    if (!callee) {
      return failure();
    }

    // Checks if the callee function exists as func.func in the module.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) {
      return failure();
    }

    // Looks for a func.func with the same name.
    func::FuncOp func_op = module.lookupSymbol<func::FuncOp>(callee.value());
    if (!func_op) {
      return failure();
    }

    // Gets the result types from the function signature.
    auto result_types = func_op.getFunctionType().getResults();

    // Converts the call to func.call.
    auto new_call = rewriter.create<func::CallOp>(
        op.getLoc(), result_types, callee.value(), op.getArgOperands());

    // Replaces the old call with the new one.
    // Handles both cases: calls with results and calls without results.
    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, new_call->getResults());
    }

    return success();
  }
};

struct LowerLlvmToNeuraPass
    : public PassWrapper<LowerLlvmToNeuraPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLlvmToNeuraPass)

  StringRef getArgument() const override { return "lower-llvm-to-neura"; }
  StringRef getDescription() const override {
    return "Lower LLVM operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  RewritePatternSet populateLlvmToNeuraPatterns(MLIRContext *context) {
    RewritePatternSet patterns(context);
    patterns.add<LlvmConstantToNeuraConstant>(&getContext());
    patterns.add<LlvmAddressOfToNeuraConstant>(&getContext());
    // Vector operations must be registered before scalar operations
    // to ensure vector types are matched first.
    patterns.add<LlvmVMulToNeuraVMul>(&getContext());
    patterns.add<LlvmVAddToNeuraVAdd>(&getContext());
    patterns.add<LlvmVFMulToNeuraVFMul>(&getContext());
    patterns.add<LlvmVFAddToNeuraVFAdd>(&getContext());
    patterns.insert<LlvmVectorReduceAddToNeuraVectorReduceAdd>(&getContext());
    // Scalar operations.
    patterns.add<LlvmAddToNeuraAdd>(&getContext());
    patterns.add<LlvmAndToNeuraAnd>(&getContext());
    patterns.add<LlvmOrToNeuraOr>(&getContext());
    patterns.add<LlvmFAddToNeuraFAdd>(&getContext());
    patterns.add<LlvmFMulToNeuraFMul>(&getContext());
    patterns.add<LlvmICmpToNeuraICmp>(&getContext());
    patterns.add<LlvmFCmpToNeuraFCmp>(&getContext());
    patterns.add<LlvmGEPToNeuraGEP>(&getContext());
    patterns.add<LlvmLoadToNeuraLoad>(&getContext());
    patterns.add<LlvmStoreToNeuraStore>(&getContext());
    patterns.add<LlvmCondBrToNeuraCondBr>(&getContext());
    patterns.add<LlvmBrToNeuraBr>(&getContext());
    patterns.add<LlvmReturnToNeuraReturn>(&getContext());
    patterns.add<FuncReturnToNeuraReturn>(&getContext());
    patterns.add<LlvmFSubToNeuraFSub>(&getContext());
    patterns.add<LlvmAllocaToNeuraAlloca>(&getContext());
    patterns.add<LlvmSExtToNeuraSExt>(&getContext());
    patterns.add<LlvmZExtToNeuraZExt>(&getContext());
    patterns.add<LlvmMulToNeuraMul>(&getContext());
    patterns.add<LlvmFuncToNeuraFunc>(&getContext());
    patterns.add<LlvmCallToFuncCall>(&getContext());
    patterns.add<LlvmShlToNeuraShl>(&getContext());
    patterns.add<LlvmSDivToNeuraDiv>(&getContext());
    patterns.add<LlvmSRemToNeuraRem>(&getContext());
    patterns.add<LlvmMaxNumToNeuraFMax>(&getContext());
    patterns.add<LlvmMaximumToNeuraFMax>(&getContext());
    patterns.add<LlvmMinNumToNeuraFMin>(&getContext());
    patterns.add<LlvmMinimumToNeuraFMin>(&getContext());
    patterns.add<LlvmFDivToNeuraFDiv>(&getContext());
    patterns.add<LlvmFPToSIToNeuraCast>(&getContext());
    patterns.add<LlvmFMulAddToNeuraFMulFAdd>(&getContext());
    patterns.add<LlvmSelectToNeuraSel>(&getContext());
    patterns.add<LlvmMemsetToNeuraOps>(&getContext());
    patterns.add<LlvmFNegToNeuraFNeg>(&getContext());
    patterns.add<LlvmSubToNeuraSub>(&getContext());
    patterns.add<LlvmTruncToNeuraCast>(&getContext());
    patterns.add<LlvmUDivToNeuraDiv>(&getContext());
    patterns.add<LlvmURemToNeuraRem>(&getContext());
    patterns.add<LlvmSMaxToNeuraSMax>(&getContext());
    // TODO: Adds more LLVM to Neura conversion patterns as needed.
    // patterns.add<LlvmXOrToNeuraOr>(&getContext());     // TODO: Uses in ADPCM
    // coder + FFT kernels. patterns.add<LlvmAndToNeuraMul>(&getContext()); //
    // TODO: Uses in ADPCM coder + MVT kernels.
    // patterns.add<LlvmAllocaToNeuraOps>(&getContext()); // TODO: Uses in DTW
    // kernel.
    // TODO: Fixes right shift implementations. Current implementations are
    // incorrect. patterns.add<LlvmLShrToNeuraShl>(&getContext());  // TODO:
    // Uses in ADPCM coder/decoder + FFT kernels.
    // patterns.add<LlvmAShrToNeuraAShr>(&getContext()); // TODO: Uses in ADPCM
    // coder/decoder kernels. patterns.add<LlvmAbsToNeuraAbs>(&getContext()); //
    // TODO: Uses in ADPCM coder kernel.
    return patterns;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module_op = getOperation();

    // Performs the llvm.func -> func.func conversion first.
    RewritePatternSet func_patterns(context);
    func_patterns.add<LlvmFuncToNeuraFunc>(context);
    func_patterns.add<LlvmCallToFuncCall>(context);

    if (failed(applyPatternsGreedily(module_op, std::move(func_patterns)))) {
      signalPassFailure();
    }

    // Performs operation-level conversions for func::FuncOp.
    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](FunctionOpInterface func) {
      if (func->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target =
            func->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (target && target.getValue() == mlir::accel::kNeuraTarget) {
          for (Region &region : func->getRegions()) {
            RewritePatternSet patterns = populateLlvmToNeuraPatterns(context);
            if (failed(applyPatternsGreedily(region, std::move(patterns)))) {
              signalPassFailure();
            }
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
          RewritePatternSet patterns = populateLlvmToNeuraPatterns(context);
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

std::unique_ptr<Pass> mlir::createLowerLlvmToNeuraPass() {
  return std::make_unique<LowerLlvmToNeuraPass>();
}
