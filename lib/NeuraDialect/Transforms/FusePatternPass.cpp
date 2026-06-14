#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_DEF_FUSEPATTERN
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct FuseFAddFAddPattern : public OpRewritePattern<neura::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FAddOp second,
                                PatternRewriter &rewriter) const override {
    // Checks if rhs exists before trying to get its defining op.
    if (!second.getRhs()) {
      return failure();
    }

    Value lhs = second.getLhs();
    Value rhs = second.getRhs();

    auto lhs_op = lhs.getDefiningOp<neura::FAddOp>();
    auto rhs_op = rhs.getDefiningOp<neura::FAddOp>();

    neura::FAddOp first = nullptr;
    Value tail;

    // Case 1: LHS is another fadd.
    if (lhs_op && second.getRhs()) {
      first = lhs_op;
      tail = second.getRhs();
    }
    // Case 2: RHS is another fadd.
    else if (rhs_op && second.getLhs()) {
      first = rhs_op;
      tail = second.getLhs();
    }

    if (!first) {
      return failure();
    }

    // Only fuses if the first fadd is not reused elsewhere.
    if (!first->hasOneUse()) {
      return failure();
    }

    // Bail out if the first fadd's rhs is attribute-sourced (null Value).
    if (!first.getRhs()) {
      return failure();
    }

    Location loc = second.getLoc();
    Type type = second.getType();

    auto fused = rewriter.create<neura::FAddFAddOp>(loc, type, first.getLhs(),
                                                    first.getRhs(), tail);

    rewriter.replaceOp(second, fused.getResult());
    rewriter.eraseOp(first);
    return success();
  }
};

struct FuseFMulFAddPattern : public OpRewritePattern<neura::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FAddOp add,
                                PatternRewriter &rewriter) const override {
    // Checks if rhs exists before trying to get its defining op.
    if (!add.getRhs()) {
      return failure();
    }

    auto lhs_op = add.getLhs().getDefiningOp<neura::FMulOp>();
    auto rhs_op = add.getRhs().getDefiningOp<neura::FMulOp>();

    neura::FMulOp fmul = nullptr;
    Value other;

    // Case 1: fmul is on the LHS.
    if (lhs_op && add.getRhs()) {
      fmul = lhs_op;
      other = add.getRhs();
    }
    // Case 2: fmul is on the RHS.
    else if (rhs_op && add.getLhs()) {
      fmul = rhs_op;
      other = add.getLhs();
    }

    if (!fmul) {
      return failure();
    }

    // Optionally fuses if fmul has a single use.
    if (!fmul->hasOneUse()) {
      return failure();
    }

    // Bail out if the fmul's rhs is attribute-sourced (null Value).
    if (!fmul.getRhs()) {
      return failure();
    }

    Location loc = add.getLoc();
    Type type = add.getType();

    auto fused = rewriter.create<neura::FMulFAddOp>(loc, type, fmul.getLhs(),
                                                    fmul.getRhs(), other);

    rewriter.replaceOp(add, fused.getResult());
    rewriter.eraseOp(fmul);
    return success();
  }
};

struct FuseGepLoadPattern : public OpRewritePattern<neura::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::LoadOp load,
                                PatternRewriter &rewriter) const override {
    Value addr = load.getAddr();
    auto gep_op = addr.getDefiningOp<neura::GEP>();

    if (!gep_op)
      return failure();

    // Only fuses if the gep has a single use.
    if (!gep_op->hasOneUse())
      return failure();

    Location loc = load.getLoc();
    Type type = load.getType();

    // Creates the fused operation with base and indices from gep.
    SmallVector<Value> indexValues;
    for (auto gepIndex : gep_op.getIndices()) {
      indexValues.push_back(gepIndex);
    }

    auto fused = rewriter.create<neura::LoadIndexedOp>(
        loc, type, gep_op.getBase(), indexValues);

    rewriter.replaceOp(load, fused.getResult());
    rewriter.eraseOp(gep_op);
    return success();
  }
};

struct FuseGepStorePattern : public OpRewritePattern<neura::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::StoreOp store,
                                PatternRewriter &rewriter) const override {
    Value addr = store.getAddr();
    auto gep_op = addr.getDefiningOp<neura::GEP>();

    if (!gep_op)
      return failure();

    // Only fuses if the gep has a single use.
    if (!gep_op->hasOneUse())
      return failure();

    Location loc = store.getLoc();

    // Creates the fused operation with base and indices from gep.
    SmallVector<Value> indexValues;
    for (auto gepIndex : gep_op.getIndices()) {
      indexValues.push_back(gepIndex);
    }

    rewriter.create<neura::StoreIndexedOp>(loc, store.getValue(),
                                           gep_op.getBase(), indexValues);

    rewriter.eraseOp(store);
    rewriter.eraseOp(gep_op);
    return success();
  }
};

struct FuseMulAddPattern : public OpRewritePattern<neura::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::AddOp add,
                                PatternRewriter &rewriter) const override {
    // Checks if rhs exists before trying to get its defining op.
    if (!add.getRhs()) {
      return failure();
    }

    auto lhs_op = add.getLhs().getDefiningOp<neura::MulOp>();
    auto rhs_op = add.getRhs().getDefiningOp<neura::MulOp>();

    neura::MulOp mul = nullptr;
    Value other;

    // Case 1: mul is on the LHS.
    if (lhs_op && add.getRhs()) {
      mul = lhs_op;
      other = add.getRhs();
    }
    // Case 2: mul is on the RHS.
    else if (rhs_op && add.getLhs()) {
      mul = rhs_op;
      other = add.getLhs();
    }

    if (!mul) {
      return failure();
    }

    // Only fuses if mul has a single use.
    if (!mul->hasOneUse()) {
      return failure();
    }

    // Bail out if the mul's rhs is attribute-sourced (null Value).
    if (!mul.getRhs()) {
      return failure();
    }

    Location loc = add.getLoc();
    Type type = add.getType();

    auto fused = rewriter.create<neura::MulAddOp>(loc, type, mul.getLhs(),
                                                  mul.getRhs(), other, Value());

    rewriter.replaceOp(add, fused.getResult());
    rewriter.eraseOp(mul);
    return success();
  }
};

struct FusePatternPass
    : public PassWrapper<FusePatternPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePatternPass)

  StringRef getArgument() const override { return "fuse-pattern"; }
  StringRef getDescription() const override {
    return "Apply Neura fusion patterns.";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseFAddFAddPattern>(&getContext(), 2);
    patterns.add<FuseFMulFAddPattern>(&getContext(), 3);
    patterns.add<FuseGepLoadPattern>(&getContext(), 4);
    patterns.add<FuseGepStorePattern>(&getContext(), 5);
    patterns.add<FuseMulAddPattern>(&getContext(), 6);
    FrozenRewritePatternSet frozen(std::move(patterns));

    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](Operation *op) {
      if (!op->getRegions().empty()) {
        for (Region &region : op->getRegions()) {
          if (failed(applyPatternsGreedily(region, frozen))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFusePatternPass() {
  return std::make_unique<FusePatternPass>();
}
} // namespace mlir::neura
