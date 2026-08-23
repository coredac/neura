#include "Conversion/NeuraConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
    Value src = copy_op.getSource();
    Value dst = copy_op.getTarget();
    auto memref_type = dyn_cast<MemRefType>(src.getType());
    if (!memref_type || !memref_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          copy_op, "expects a statically-shaped memref");
    }

    Location loc = copy_op.getLoc();
    ArrayRef<int64_t> shape = memref_type.getShape();

    // Builds one affine.for per dimension, nesting each new loop inside the
    // previous one's body, then loads from src and stores to dst at the
    // innermost level using the collected induction variables as indices.
    SmallVector<Value> indices;
    for (int64_t dim_size : shape) {
      auto for_op = rewriter.create<affine::AffineForOp>(loc, 0, dim_size);
      rewriter.setInsertionPointToStart(for_op.getBody());
      indices.push_back(for_op.getInductionVar());
    }
    Value loaded = rewriter.create<affine::AffineLoadOp>(loc, src, indices);
    rewriter.create<affine::AffineStoreOp>(loc, loaded, dst, indices);

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
    registry.insert<affine::AffineDialect>();
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
