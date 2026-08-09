#include "Common/AcceleratorAttrs.h"
#include "Conversion/AffineToNeura/LoopNestAnalysis.h"
#include "Conversion/NeuraConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mlir::neura;
using namespace mlir::func;

#define GEN_PASS_DEF_LOWERAFFINETONEURA
#include "Conversion/NeuraConversionPasses.h.inc"

namespace {
// Converts an AffineMap to explicit index computations using Neura operations.
// This function handles the expansion of affine expressions into arithmetic
// ops.
//
// Example 1 - Simple dimension access:
// Before: affine_map<(d0, d1) -> (d0, d1)> with operands (%i, %j)
// After:  Returns [%i, %j] directly
//
// Example 2 - Constant offset:
// Before: affine_map<(d0) -> (d0 + 5)> with operand %i
// After:  %c5 = neura.constant 5 : index
//         %result = neura.add %i, %c5 : index
//         Returns [%result]
//
// Example 3 - Complex expression:
// Before: affine_map<(d0, d1) -> (d0 * 2 + d1)> with operands (%i, %j)
// After:  %c2 = neura.constant 2 : index
//         %mul = neura.mul %i, %c2 : index
//         %result = neura.add %mul, %j : index
//         Returns [%result]
LogicalResult convertAffineMapToIndices(AffineMap map, ValueRange map_operands,
                                        Location loc, PatternRewriter &rewriter,
                                        SmallVector<Value> &new_indices) {
  new_indices.clear();
  new_indices.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    if (AffineConstantExpr const_expr = dyn_cast<AffineConstantExpr>(expr)) {
      IndexType index_type = rewriter.getIndexType();
      IntegerAttr value_attr =
          rewriter.getIntegerAttr(index_type, const_expr.getValue());
      new_indices.push_back(
          rewriter.create<neura::ConstantOp>(loc, index_type, value_attr));
    } else if (AffineDimExpr dim_expr = dyn_cast<AffineDimExpr>(expr)) {
      if (dim_expr.getPosition() >= map.getNumDims() ||
          dim_expr.getPosition() >=
              map_operands
                  .size()) { // Checks against mapOperands size for safety.
        return failure();
      }
      new_indices.push_back(map_operands[dim_expr.getPosition()]);
    } else if (AffineSymbolExpr sym_expr = dyn_cast<AffineSymbolExpr>(expr)) {
      unsigned symbol_operand_index = map.getNumDims() + sym_expr.getPosition();
      if (symbol_operand_index >= map_operands.size()) {
        return failure();
      }
      new_indices.push_back(map_operands[symbol_operand_index]);
    } else {
      // For more complex affine expressions (e.g., d0 + c1),
      // expands them into explicit Neura arithmetic operations.
      // Supports: Add, Mul, Mod, FloorDiv, CeilDiv.
      llvm::errs() << "[affine2neura] Expanding complex affine expression: "
                   << expr << "\n";

      // Helper lambda: recursively expands AffineExpr to Value.
      std::function<Value(AffineExpr)> expandExpr = [&](AffineExpr e) -> Value {
        // Constant expression.
        if (auto const_expr = dyn_cast<AffineConstantExpr>(e)) {
          return rewriter.create<neura::ConstantOp>(
              loc, rewriter.getIndexType(),
              rewriter.getIntegerAttr(rewriter.getIndexType(),
                                      const_expr.getValue()));
        }
        // Dimension expression.
        else if (auto dim_expr = dyn_cast<AffineDimExpr>(e)) {
          return map_operands[dim_expr.getPosition()];
        }
        // Symbol expression.
        else if (auto sym_expr = dyn_cast<AffineSymbolExpr>(e)) {
          unsigned symbol_operand_index =
              map.getNumDims() + sym_expr.getPosition();
          return map_operands[symbol_operand_index];
        }
        // Binary operation expression.
        else if (auto bin_expr = dyn_cast<AffineBinaryOpExpr>(e)) {
          Value lhs = expandExpr(bin_expr.getLHS());
          Value rhs = expandExpr(bin_expr.getRHS());

          switch (bin_expr.getKind()) {
          case AffineExprKind::Add:
            return rewriter
                .create<neura::AddOp>(loc, rewriter.getIndexType(), lhs, rhs)
                .getResult();
          case AffineExprKind::Mul:
            return rewriter
                .create<neura::MulOp>(loc, rewriter.getIndexType(), lhs, rhs)
                .getResult();
          case AffineExprKind::Mod:
            return rewriter
                .create<neura::RemOp>(loc, rewriter.getIndexType(), lhs, rhs)
                .getResult();
          case AffineExprKind::FloorDiv:
            return rewriter
                .create<neura::DivOp>(loc, rewriter.getIndexType(), lhs, rhs)
                .getResult();
          case AffineExprKind::CeilDiv: {
            // ceildiv(a, b) = floordiv(a + b - 1, b).
            Value one = rewriter.create<neura::ConstantOp>(
                loc, rewriter.getIndexType(),
                rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
            Value b_minus_1 = rewriter
                                  .create<neura::SubOp>(
                                      loc, rewriter.getIndexType(), rhs, one)
                                  .getResult();
            Value numerator =
                rewriter
                    .create<neura::AddOp>(loc, rewriter.getIndexType(), lhs,
                                          b_minus_1)
                    .getResult();
            return rewriter
                .create<neura::DivOp>(loc, rewriter.getIndexType(), numerator,
                                      rhs)
                .getResult();
          }
          default:
            llvm::errs() << "[affine2neura] Unsupported binary op kind: "
                         << static_cast<int>(bin_expr.getKind()) << "\n";
            return Value();
          }
        }

        llvm::errs() << "[affine2neura] Unsupported affine expression type\n";
        return Value();
      };

      Value expanded = expandExpr(expr);
      if (!expanded) {
        // Fallback: if expansion fails, use affine.apply (ensures correctness).
        llvm::errs() << "[affine2neura] Failed to expand, using affine.apply\n";
        AffineMap single_result_map = AffineMap::get(
            map.getNumDims(), map.getNumSymbols(), expr, rewriter.getContext());
        expanded = rewriter.create<affine::AffineApplyOp>(
            loc, single_result_map, map_operands);
      }
      new_indices.push_back(expanded);
    }
  }
  return success();
}

// Converts affine.load to neura.load_indexed.
// Expands the affine map into explicit index computations.
//
// Example 1 - Simple 2D array access:
// Before: %val = affine.load %A[%i, %j] : memref<10x20xf32>
// After:  %val = neura.load_indexed %A[%i, %j : index, index] memref<10x20xf32>
// : f32
//
// Example 2 - With affine expression:
// Before: %val = affine.load %A[%i * 2 + 1, %j] : memref<100x100xf32>
// After:  %c2 = neura.constant 2 : index
//         %c1 = neura.constant 1 : index
//         %mul = neura.mul %i, %c2 : index
//         %idx0 = neura.add %mul, %c1 : index
//         %val = neura.load_indexed %A[%idx0, %j : index, index]
//         memref<100x100xf32> : f32
struct AffineLoadLowering : public OpRewritePattern<affine::AffineLoadOp> {
  AffineLoadLowering(MLIRContext *context)
      : OpRewritePattern<affine::AffineLoadOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(affine::AffineLoadOp load_op,
                                PatternRewriter &rewriter) const override {
    Location loc = load_op.getLoc();
    auto memref = load_op.getMemref();
    AffineMap map = load_op.getAffineMap();
    ValueRange map_operands = load_op.getMapOperands();
    // Gets the indices for the load operation.
    SmallVector<Value> new_indices;
    if (failed(convertAffineMapToIndices(map, map_operands, loc, rewriter,
                                         new_indices))) {
      return load_op.emitError(
          "[affine2neura] Failed to convert affine map to indices");
    }

    MemRefType memref_type = dyn_cast<MemRefType>(memref.getType());
    if (!memref_type) {
      return load_op.emitError(
          "[affine2neura] Base of load is not a MemRefType");
    }
    if (new_indices.size() != static_cast<size_t>(memref_type.getRank())) {
      return load_op.emitError(
                 "[affine2neura] Number of indices from affine map (")
             << new_indices.size() << ") does not match memref rank ("
             << memref_type.getRank() << ")";
    }

    // NOTE: No explicit dimension limit is enforced here. The lowering supports
    // arbitrary dimensions theoretically. For CGRA hardware with limited
    // address generation units, dimension constraints should be handled at a
    // later stage (e.g., during mapping or hardware-specific lowering passes).

    // Creates the neura.load_indexed operation.
    LoadIndexedOp new_load_op = rewriter.create<neura::LoadIndexedOp>(
        loc, load_op.getType(), memref, ValueRange{new_indices});

    rewriter.replaceOp(load_op, new_load_op.getResult());
    return success();
  }
};

// Converts affine.store to neura.store_indexed.
// Similar to AffineLoadLowering, expands affine maps into explicit indices.
//
// Example 1 - Simple store:
// Before: affine.store %val, %A[%i, %j] : memref<10x20xf32>
// After:  neura.store_indexed %val to %A[%i, %j : index, index]
// memref<10x20xf32> : f32
//
// Example 2 - With affine expression:
// Before: affine.store %val, %A[%i + 1, %j * 2] : memref<100x100xf32>
// After:  %c1 = neura.constant 1 : index
//         %c2 = neura.constant 2 : index
//         %idx0 = neura.add %i, %c1 : index
//         %idx1 = neura.mul %j, %c2 : index
//         neura.store_indexed %val to %A[%idx0, %idx1 : index, index]
//         memref<100x100xf32> : f32
struct AffineStoreLowering : public OpRewritePattern<affine::AffineStoreOp> {
  AffineStoreLowering(MLIRContext *context)
      : OpRewritePattern<affine::AffineStoreOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(affine::AffineStoreOp store_op,
                                PatternRewriter &rewriter) const override {
    Location loc = store_op.getLoc();
    auto memref = store_op.getMemref();
    Value value = store_op.getValueToStore();
    AffineMap map = store_op.getAffineMap();
    ValueRange mapOperands = store_op.getMapOperands();

    SmallVector<Value> newIndices;
    if (failed(convertAffineMapToIndices(map, mapOperands, loc, rewriter,
                                         newIndices))) {
      return store_op.emitError(
          "[affine2neura] Failed to convert affine map to indices");
    }

    MemRefType memRefType = dyn_cast<MemRefType>(memref.getType());
    if (!memRefType) {
      return store_op.emitError(
          "[affine2neura] Base of store is not a MemRefType");
    }
    if (newIndices.size() != static_cast<size_t>(memRefType.getRank())) {
      return store_op.emitError(
                 "[affine2neura] Number of indices from affine map (")
             << newIndices.size() << ") does not match memref rank ("
             << memRefType.getRank() << ")";
    }

    rewriter.create<neura::StoreIndexedOp>(loc, value, memref,
                                           ValueRange{newIndices});
    rewriter.eraseOp(store_op);
    return success();
  }
};

// Converts affine.apply to explicit Neura arithmetic operations.
// Recursively expands the affine expression tree into primitive operations.
//
// Example 1 - Linear expression:
// Before: %result = affine.apply affine_map<(d0) -> (d0 + 5)>(%i)
// After:  %c5 = neura.constant 5 : index
//         %result = neura.add %i, %c5 : index
//
// Example 2 - Multiply-add:
// Before: %result = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%i, %j)
// After:  %c2 = neura.constant 2 : index
//         %mul = neura.mul %i, %c2 : index
//         %result = neura.add %mul, %j : index
//
// Example 3 - Modulo operation:
// Before: %result = affine.apply affine_map<(d0) -> (d0 mod 8)>(%i)
// After:  %c8 = neura.constant 8 : index
//         %result = neura.rem %i, %c8 : index
//
// Example 4 - Complex nested expression:
// Before: %result = affine.apply affine_map<(d0, d1) -> ((d0 + 1) * d1)>(%i,
// %j) After:  %c1 = neura.constant 1 : index
//         %add = neura.add %i, %c1 : index
//         %result = neura.mul %add, %j : index
struct AffineApplyLowering : public OpRewritePattern<affine::AffineApplyOp> {
  AffineApplyLowering(MLIRContext *context)
      : OpRewritePattern<affine::AffineApplyOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(affine::AffineApplyOp apply_op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = apply_op.getAffineMap();
    ValueRange operands = apply_op.getMapOperands();
    Location loc = apply_op.getLoc();

    // Note: AffineMap can have multiple results in general MLIR contexts
    // (e.g., affine_map<(d0, d1) -> (d0 + 1, d1 * 2)> returns two values).
    // However, AffineApplyOp specifically enforces single-result maps at
    // construction time. This check serves as a safety guard.
    //
    // Example transformation:
    // Before: %result = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%i,
    // %j) After:  %c2 = arith.constant 2 : index
    //         %mul = arith.muli %i, %c2 : index
    //         %result = arith.addi %mul, %j : index
    if (map.getNumResults() != 1) {
      return apply_op.emitError(
          "[affine2neura] AffineApplyOp must have a single result");
    }

    AffineExpr expr = map.getResult(0);
    llvm::errs() << "[affine2neura] Expanding affine.apply expression: " << expr
                 << "\n";

    // Helper lambda: recursively expands AffineExpr to Value.
    std::function<Value(AffineExpr)> expandExpr = [&](AffineExpr e) -> Value {
      // Constant expression.
      if (auto const_expr = dyn_cast<AffineConstantExpr>(e)) {
        return rewriter.create<neura::ConstantOp>(
            loc, rewriter.getIndexType(),
            rewriter.getIntegerAttr(rewriter.getIndexType(),
                                    const_expr.getValue()));
      }
      // Dimension expression.
      else if (auto dim_expr = dyn_cast<AffineDimExpr>(e)) {
        return operands[dim_expr.getPosition()];
      }
      // Symbol expression.
      else if (auto sym_expr = dyn_cast<AffineSymbolExpr>(e)) {
        unsigned symbol_operand_index =
            map.getNumDims() + sym_expr.getPosition();
        return operands[symbol_operand_index];
      }
      // Binary operation expression.
      else if (auto bin_expr = dyn_cast<AffineBinaryOpExpr>(e)) {
        Value lhs = expandExpr(bin_expr.getLHS());
        Value rhs = expandExpr(bin_expr.getRHS());

        if (!lhs || !rhs) {
          return Value();
        }

        switch (bin_expr.getKind()) {
        case AffineExprKind::Add:
          return rewriter
              .create<neura::AddOp>(loc, rewriter.getIndexType(), lhs, rhs)
              .getResult();
        case AffineExprKind::Mul:
          return rewriter
              .create<neura::MulOp>(loc, rewriter.getIndexType(), lhs, rhs)
              .getResult();
        case AffineExprKind::Mod:
          return rewriter
              .create<neura::RemOp>(loc, rewriter.getIndexType(), lhs, rhs)
              .getResult();
        case AffineExprKind::FloorDiv:
          return rewriter
              .create<neura::DivOp>(loc, rewriter.getIndexType(), lhs, rhs)
              .getResult();
        case AffineExprKind::CeilDiv: {
          // ceildiv(a, b) = floordiv(a + b - 1, b).
          Value one = rewriter.create<neura::ConstantOp>(
              loc, rewriter.getIndexType(),
              rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
          Value b_minus_1 =
              rewriter
                  .create<neura::SubOp>(loc, rewriter.getIndexType(), rhs, one)
                  .getResult();
          Value numerator =
              rewriter
                  .create<neura::AddOp>(loc, rewriter.getIndexType(), lhs,
                                        b_minus_1)
                  .getResult();
          return rewriter
              .create<neura::DivOp>(loc, rewriter.getIndexType(), numerator,
                                    rhs)
              .getResult();
        }
        default:
          llvm::errs() << "[affine2neura] Unsupported binary op kind: "
                       << static_cast<int>(bin_expr.getKind()) << "\n";
          return Value();
        }
      }

      llvm::errs() << "[affine2neura] Unsupported affine expression type\n";
      return Value();
    };

    Value expanded = expandExpr(expr);
    if (!expanded) {
      return apply_op.emitError(
          "[affine2neura] Failed to expand affine.apply expression");
    }

    rewriter.replaceOp(apply_op, expanded);
    return success();
  }
};

// Converts affine.for loops to neura.loop_control with dataflow semantics.
// Creates constant true for top-level loops, reuses parent's valid signal for
// nested loops.
//
// Example 1 - Simple single loop:
// Before: affine.for %i = 0 to 10 {
//           %val = affine.load %A[%i] : memref<10xf32>
//         }
// After:  %c_true = neura.constant 1 : i1
//         %i, %valid1 = "neura.loop_control"(%c_true) <{end = 10, start = 0,
//         step = 1}> : (i1) -> (index, i1) %val = neura.load_indexed %A[%i :
//         index] memref<10xf32> : f32
//
// Example 2 - Nested loops (demonstrates valid signal reuse):
// Before: affine.for %i = 0 to 10 {
//           affine.for %j = 0 to 20 {
//             %val = affine.load %A[%i, %j] : memref<10x20xf32>
//           }
//         }
// After:  %c_true = neura.constant 1 : i1
//         %i, %valid_i = "neura.loop_control"(%c_true) <{end = 10, start = 0,
//         step = 1}> : (i1) -> (index, i1) %j, %valid_j =
//         "neura.loop_control"(%valid_i) <{end = 20, start = 0, step = 1}> :
//         (i1) -> (index, i1) %val = neura.load_indexed %A[%i, %j : index,
//         index] memref<10x20xf32> : f32 (Note: Inner loop reuses outer loop's
//         valid_i signal, no second constant)
//
// Example 3 - Non-zero bounds and step:
// Before: affine.for %i = 5 to 100 step 2 {
//           %val = affine.load %A[%i] : memref<100xf32>
//         }
// After:  %c_true = neura.constant 1 : i1
//         %i, %valid1 = "neura.loop_control"(%c_true) <{end = 100, start = 5,
//         step = 2}> : (i1) -> (index, i1) %val = neura.load_indexed %A[%i :
//         index] memref<100xf32> : f32
struct AffineForLowering : public OpRewritePattern<affine::AffineForOp> {
  const LoopNestAnalysis &analysis;
  llvm::DenseMap<Operation *, Value> &loopValidSignals;

  AffineForLowering(MLIRContext *context, const LoopNestAnalysis &analysis,
                    llvm::DenseMap<Operation *, Value> &loopValidSignals)
      : OpRewritePattern<affine::AffineForOp>(context, /*benefit=*/1),
        analysis(analysis), loopValidSignals(loopValidSignals) {}

  LogicalResult matchAndRewrite(affine::AffineForOp for_op,
                                PatternRewriter &rewriter) const override {
    Location loc = for_op.getLoc();

    // Extracts loop bounds - must be constant.
    // Dynamic bounds are not supported as neura.loop_control requires
    // compile-time constant attributes for hardware configuration.
    if (!for_op.hasConstantLowerBound() || !for_op.hasConstantUpperBound()) {
      return for_op.emitError(
          "[affine2neura] Non-constant loop bounds not supported. "
          "Loop bounds must be compile-time constants for CGRA configuration");
    }

    int64_t lower_bound = for_op.getConstantLowerBound();
    int64_t upper_bound = for_op.getConstantUpperBound();
    int64_t step = for_op.getStepAsInt();

    // Get loop nesting information
    LoopInfo *loopInfo = analysis.getLoopInfo(for_op);
    Type i1_type = rewriter.getI1Type();
    Value parent_valid;

    // Optimization: Reuse parent loop's valid signal for nested loops.
    // This avoids creating redundant initialization for each nested loop.
    if (loopInfo && loopInfo->parent) {
      // This is a nested loop - try to reuse parent's loop_valid signal
      auto it = loopValidSignals.find(loopInfo->parent->loop.getOperation());
      if (it != loopValidSignals.end()) {
        parent_valid = it->second;
        llvm::errs() << "[affine2neura] Reusing parent valid signal for "
                     << "nested loop (depth=" << loopInfo->depth << ")\n";
      } else {
        // Fallback: parent not yet converted, create constant true
        IntegerAttr true_attr = rewriter.getIntegerAttr(i1_type, 1);
        parent_valid =
            rewriter.create<neura::ConstantOp>(loc, i1_type, true_attr);
        llvm::errs() << "[affine2neura] Parent valid not available, "
                     << "creating constant true for nested loop\n";
      }
    } else {
      // Top-level loop - create constant true to ensure it's always valid
      IntegerAttr true_attr = rewriter.getIntegerAttr(i1_type, 1);
      parent_valid =
          rewriter.create<neura::ConstantOp>(loc, i1_type, true_attr);
      if (loopInfo) {
        llvm::errs()
            << "[affine2neura] Created constant true for top-level loop "
            << "(depth=" << loopInfo->depth << ")\n";
      }
    }

    // Creates loop_control operation.
    auto index_type = rewriter.getIndexType();

    auto loop_control = rewriter.create<neura::LoopControlOp>(
        loc,
        /*resultTypes=*/TypeRange{index_type, i1_type},
        /*parentValid=*/parent_valid,
        /*iterationType=*/rewriter.getStringAttr("increment"),
        /*start=*/rewriter.getI64IntegerAttr(lower_bound),
        /*end=*/rewriter.getI64IntegerAttr(upper_bound),
        /*step=*/rewriter.getI64IntegerAttr(step));

    Value loop_index = loop_control.getResult(0);
    Value loop_valid = loop_control.getResult(1);

    // Store the loop_valid signal for child loops to use.
    // This enables the optimization for nested loops.
    loopValidSignals[for_op.getOperation()] = loop_valid;

    // Inlines the body operations before the for_op.
    Block &body_block = for_op.getRegion().front();
    Operation *terminator = body_block.getTerminator();
    rewriter.eraseOp(terminator); // Removes affine.yield first.

    // Merge the loop body into the parent block before the for_op.
    // Pass the loop_index as replacement for the induction variable block
    // argument.
    rewriter.inlineBlockBefore(&body_block, for_op.getOperation(),
                               {loop_index});

    // Erases the for_op.
    rewriter.eraseOp(for_op);

    return success();
  }
};

struct LowerAffineToNeuraPass
    : public PassWrapper<LowerAffineToNeuraPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAffineToNeuraPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<neura::NeuraDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect>();
  }

  StringRef getArgument() const override { return "lower-affine-to-neura"; }
  StringRef getDescription() const override {
    return "Lower affine operations to Neura dialect operations";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    MLIRContext *context = module_op.getContext();

    module_op.walk([&](func::FuncOp func_op) {
      // Checks if function targets neura accelerator, or applies to all if no
      // attribute.
      if (func_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target =
            func_op->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (!target || target.getValue() != mlir::accel::kNeuraTarget) {
          return; // Skips this function.
        }
      }
      // If no accelerator attribute, applies the pass anyway (for testing).

      // Step 1: Perform loop nest analysis
      // This builds the loop hierarchy and identifies perfect/imperfect nests
      llvm::errs() << "[affine2neura] Analyzing loop nests in function: "
                   << func_op.getName() << "\n";
      LoopNestAnalysis analysis(func_op);
      analysis.dump(); // Print analysis results for debugging

      // Step 2: Create a map to store loop_valid signals
      // This allows nested loops to reuse parent's valid signal
      llvm::DenseMap<Operation *, Value> loopValidSignals;

      // Step 3: Set up dialect conversion
      // We use Dialect Conversion instead of Greedy Pattern Rewriter because:
      // 1. It provides better error reporting when conversion fails
      // 2. It explicitly defines which operations are legal/illegal
      // 3. It's the standard approach for dialect lowering passes
      ConversionTarget target(*context);
      target.addLegalDialect<neura::NeuraDialect, arith::ArithDialect,
                             memref::MemRefDialect, func::FuncDialect>();
      target.addIllegalDialect<affine::AffineDialect>();

      // Step 4: Register rewrite patterns with analysis
      RewritePatternSet patterns(context);
      patterns
          .add<AffineLoadLowering, AffineStoreLowering, AffineApplyLowering>(
              context);
      // Pass references to the analysis and loopValidSignals map
      patterns.add<AffineForLowering>(context, std::cref(analysis),
                                      std::ref(loopValidSignals));

      if (failed(
              applyPartialConversion(func_op, target, std::move(patterns)))) {
        func_op.emitError("[affine2neura] Failed to lower affine "
                          "operations to Neura dialect");
        signalPassFailure();
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerAffineToNeuraPass() {
  return std::make_unique<LowerAffineToNeuraPass>();
}