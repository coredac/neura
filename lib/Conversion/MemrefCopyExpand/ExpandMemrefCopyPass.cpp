#include "Conversion/NeuraConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

// Expands memref.copy(src, dst) into an explicit nested affine.for loop that
// copies src to dst element by element. Bufferization (e.g.
// -buffer-results-to-out-params) commonly leaves a trailing memref.copy that
// no Neura lowering pass understands; rewriting it into plain affine
// load/store lets it flow through the same, already-supported
// --lower-affine-to-neura path as the rest of the kernel body, instead of
// needing a dedicated Neura-side lowering for memref.copy itself.
struct MemrefCopyToAffineLoop : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy_op,
                                PatternRewriter &rewriter) const override {
    // If the copy's target has no other users, the copy may be dead (nothing
    // in this region ever reads the destination), so it can be dropped
    // instead of materializing a loop nest for it. This only holds for a
    // value whose lifetime is local to this region (e.g. a memref.alloc or
    // memref.subview) -- a block argument (a function parameter, most
    // commonly an out-param from -buffer-results-to-out-params) escapes to
    // the caller, so "no readers in this function" does not mean "unread":
    // erasing the copy there would silently drop the function's actual
    // output.
    Value target = copy_op.getTarget();
    if (!isa<BlockArgument>(target)) {
      bool has_other_users = false;
      for (Operation *user : target.getUsers()) {
        if (user != copy_op.getOperation()) {
          has_other_users = true;
          break;
        }
      }
      if (!has_other_users) {
        rewriter.eraseOp(copy_op);
        return success();
      }
    }

    Value src = copy_op.getSource();
    auto memref_type = dyn_cast<MemRefType>(src.getType());
    if (!memref_type) {
      return rewriter.notifyMatchFailure(copy_op, "expects a memref source");
    }

    Location loc = copy_op.getLoc();

    // Builds the affine bound for one dimension: a constant map for a
    // statically-sized dimension, or a single-symbol map reading the
    // dimension's runtime size (via memref.dim) for a dynamic one.
    auto makeBound =
        [&](int64_t dim_size) -> std::pair<AffineMap, SmallVector<Value>> {
      if (ShapedType::isDynamic(dim_size)) {
        return {AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                               rewriter.getAffineSymbolExpr(0)),
                {}};
      }
      return {AffineMap::getConstantMap(dim_size, rewriter.getContext()), {}};
    };

    // Builds one affine.for per dimension, nesting each new loop inside the
    // previous one's body, then loads from src and stores to dst at the
    // innermost level using the collected induction variables as indices.
    SmallVector<Value> indices;
    for (int i = 0; i < memref_type.getRank(); ++i) {
      int64_t dim_size = memref_type.getShape()[i];
      auto [ub_map, ub_operands] = makeBound(dim_size);
      if (ShapedType::isDynamic(dim_size)) {
        Value dim_index = rewriter.create<arith::ConstantIndexOp>(loc, i);
        ub_operands.push_back(
            rewriter.create<memref::DimOp>(loc, src, dim_index));
      }
      auto for_op = rewriter.create<affine::AffineForOp>(
          loc, /*lbOperands=*/ValueRange{},
          AffineMap::getConstantMap(0, rewriter.getContext()), ub_operands,
          ub_map, /*step=*/1);
      rewriter.setInsertionPointToStart(for_op.getBody());
      indices.push_back(for_op.getInductionVar());
    }
    Value loaded = rewriter.create<affine::AffineLoadOp>(loc, src, indices);
    rewriter.create<affine::AffineStoreOp>(loc, loaded, target, indices);

    rewriter.eraseOp(copy_op);
    return success();
  }
};

struct ExpandMemrefCopyPass
    : public PassWrapper<ExpandMemrefCopyPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandMemrefCopyPass)

  StringRef getArgument() const override { return "expand-memref-copy"; }

  StringRef getDescription() const override {
    return "Expands memref.copy into an explicit nested affine.for loop";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<MemrefCopyToAffineLoop>(context);
    if (failed(applyPatternsGreedily(module_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createExpandMemrefCopyPass() {
  return std::make_unique<ExpandMemrefCopyPass>();
}
