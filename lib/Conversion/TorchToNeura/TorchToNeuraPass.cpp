#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::neura;

namespace {

// Names the generic marker op produced by the frontend.
// The frontend keeps the custom op opaque as a torch.operator, and the Python
// bridge strips the torch types so it appears as a generic op with this name.
static constexpr llvm::StringRef kTorchGatherName = "torch.neura.gather";

// Rewrites the frontend marker op "torch.neura.gather" into a native
// neura.gather operation.
struct TorchGatherToNeuraGather : public RewritePattern {
  TorchGatherToNeuraGather(MLIRContext *context)
      : RewritePattern(kTorchGatherName, /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Checks the signature: gather takes a table and an indices operand and
    // produces a single result.
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expects exactly two operands and one result");
    }

    Value table = op->getOperand(0);
    Value indices = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Replaces the op in place: the new op reuses the same operands and its
    // result inherits the original users, so the data-flow connections
    // (def-use edges) stay unchanged.
    rewriter.replaceOpWithNewOp<neura::GatherOp>(op, result_type, table,
                                                 indices);
    return success();
  }
};

// Lowers the torch frontend marker op to the Neura dialect as a dedicated
// conversion pass.
struct LowerTorchToNeuraPass
    : public PassWrapper<LowerTorchToNeuraPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTorchToNeuraPass)

  StringRef getArgument() const override { return "lower-torch-to-neura"; }

  StringRef getDescription() const override {
    return "Lower torch frontend marker operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    MLIRContext *context = &getContext();

    // Applies the patterns across the whole module without gating on the
    // accelerator attribute, because a freshly imported frontend module does
    // not carry that attribute yet.
    RewritePatternSet patterns(context);
    patterns.add<TorchGatherToNeuraGather>(context);

    if (failed(applyPatternsGreedily(module_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerTorchToNeuraPass() {
  return std::make_unique<LowerTorchToNeuraPass>();
}
