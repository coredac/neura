#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

#define GEN_PASS_DEF_INSERTDATAMOV
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct InsertDataMovForNeuraOps : public RewritePattern {
  InsertDataMovForNeuraOps(MLIRContext *context)
      : RewritePattern(/*matchAnyOpTypeTag=*/MatchAnyOpTypeTag(), /*benefit=*/1,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only processes operations from the neura dialect. Operations from
    // other dialects (arith, math, etc.) should have been lowered to neura
    // ops by earlier passes (LowerArithToNeura, etc.) before this pass runs.
    if (op->getDialect()->getNamespace() != accel::kNeuraTarget ||
        isa<neura::DataMovOp>(op) ||
        // ReserveOp creates a loop-carried placeholder in the dataflow
        // recurrence cycle: %v = neura.reserve; neura.ctrl_mov %next -> %v.
        // Its result must NOT be wrapped in DataMovOp, because ctrl_mov needs
        // a direct reference to the same SSA value used by phi_start.
        // Inserting a DataMovOp between reserve and its consumers would break
        // the ctrl_mov→reserve back-edge and corrupt the recurrence cycle.
        isa<neura::ReserveOp>(op) || isa<neura::KernelOp>(op) ||
        isa<neura::FusedOp>(op)) {
      return failure();
    }

    // Skips operations inside fused_op regions.
    Operation *parent_op = op->getParentOp();
    while (parent_op) {
      if (isa<neura::FusedOp>(parent_op)) {
        return failure();
      }
      parent_op = parent_op->getParentOp();
    }

    bool all_inputs_are_mov_except_reserve =
        llvm::all_of(op->getOperands(), [](Value v) {
          Operation *def_op = v.getDefiningOp();
          return isa<BaseMemRefType>(v.getType()) ||
                 isa_and_nonnull<neura::DataMovOp>(def_op) ||
                 isa_and_nonnull<neura::ReserveOp>(def_op);
        });

    if (all_inputs_are_mov_except_reserve) {
      return failure(); // All operands are already handled
    }

    // // Skips ops that already being inserted mov on the operands.
    // bool all_inputs_are_mov = llvm::all_of(op->getOperands(), [](Value v) {
    //   return isa_and_nonnull<neura::DataMovOp>(v.getDefiningOp());
    // });
    // if (all_inputs_are_mov) {
    //   return failure();
    // }

    // // Special case: skips rewriting phi if any operand is from reserve.
    // if (isa<neura::PhiOp>(op)) {
    //   bool has_reserved_input = llvm::any_of(op->getOperands(), [](Value v)
    //   {
    //     return isa_and_nonnull<neura::ReserveOp>(v.getDefiningOp());
    //   });

    //   if (has_reserved_input)
    //     return failure();  // Skip entire phi if any operand is reserved.
    // }

    // Makes sure none of the operand has being processed.
    bool has_any_mov_input = llvm::any_of(op->getOperands(), [](Value v) {
      return isa_and_nonnull<neura::DataMovOp>(v.getDefiningOp());
    });
    if (has_any_mov_input) {
      llvm::errs() << "Warning: Operand already wrapped in neura.data_mov: "
                   << *op << "\n";
    }
    // assert(!has_any_mov_input &&
    //        "Unexpected: operand already wrapped in neura.mov");

    Location loc = op->getLoc();

    // Skips adding mov if the consumer is ctrl_mov.
    if (isa<neura::CtrlMovOp>(op)) {
      return failure(); // do not rewrite
    }

    // Wraps operands in mov, but skip those already wrapped or from reserve.
    SmallVector<Value> new_operands;
    bool any_change = false;
    for (Value operand : op->getOperands()) {
      Operation *producer = operand.getDefiningOp();

      // Memrefs are host-memory address handles.  A Neura load/store consumes
      // one directly; only payload values need a routed DataMov edge.
      if (isa<BaseMemRefType>(operand.getType())) {
        new_operands.push_back(operand);
        continue;
      }

      // Does NOT wrap operands that come from reserve: the reserve result
      // is the recurrence back-edge target for ctrl_mov.  Wrapping it would
      // produce a new SSA value, breaking the ctrl_mov→reserve cycle.
      if (producer && (isa<neura::ReserveOp>(producer) ||
                       isa<neura::DataMovOp>(producer))) {
        new_operands.push_back(operand);
        continue;
      }

      auto mov =
          rewriter.create<neura::DataMovOp>(loc, operand.getType(), operand);
      new_operands.push_back(mov);
      any_change = true;
    }

    // If no changes were made, skip rewriting
    if (!any_change) {
      return failure();
    }

    // Clones op with new operands.
    OperationState state(loc, op->getName());
    state.addOperands(new_operands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());

    // Copies successors for terminator operations.
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      for (Block *successor : op->getSuccessors()) {
        state.addSuccessors(successor);
      }
    }

    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op->getResults());
    return success();
  }
};

// Wraps all fused_op's inputs and outputs with data_mov operations in the
// module.
void wrapFusedOpsWithDataMov(ModuleOp module_op) {
  SmallVector<neura::FusedOp> fused_ops_to_process;
  module_op.walk([&](neura::FusedOp fused_op) {
    fused_ops_to_process.push_back(fused_op);
  });

  for (neura::FusedOp fused_op : fused_ops_to_process) {
    OpBuilder rewriter(fused_op->getContext());
    rewriter.setInsertionPoint(fused_op);
    Location loc = fused_op->getLoc();

    // Wrap inputs with data_mov
    SmallVector<Value> new_operands;
    for (Value operand : fused_op->getOperands()) {
      Operation *producer = operand.getDefiningOp();

      if (isa<BaseMemRefType>(operand.getType())) {
        new_operands.push_back(operand);
        continue;
      }

      // Skip if already wrapped in data_mov or from reserve
      if (isa_and_nonnull<neura::DataMovOp>(producer) ||
          isa_and_nonnull<neura::ReserveOp>(producer)) {
        new_operands.push_back(operand);
      } else {
        auto mov =
            rewriter.create<neura::DataMovOp>(loc, operand.getType(), operand);
        new_operands.push_back(mov);
      }
    }

    // Clone fused_op with new operands using IRMapping
    IRMapping mapper;
    for (size_t i = 0; i < fused_op->getNumOperands(); ++i) {
      mapper.map(fused_op->getOperand(i), new_operands[i]);
    }

    Operation *new_fused_op = rewriter.clone(*fused_op.getOperation(), mapper);

    // Update the operands of the cloned operation
    for (size_t i = 0; i < new_operands.size(); ++i) {
      new_fused_op->setOperand(i, new_operands[i]);
    }

    // Wrap outputs with data_mov - create separate data_mov for each user
    rewriter.setInsertionPointAfter(new_fused_op);

    // For each result of the fused_op, create a separate data_mov for each user
    for (size_t result_idx = 0; result_idx < fused_op->getNumResults();
         ++result_idx) {
      Value old_result = fused_op->getResult(result_idx);
      Value new_result = new_fused_op->getResult(result_idx);

      // Collect all users first (to avoid iterator invalidation)
      SmallVector<OpOperand *> users_to_update;
      for (OpOperand &use : old_result.getUses()) {
        users_to_update.push_back(&use);
      }

      // Create a separate data_mov for each user
      for (OpOperand *use : users_to_update) {
        Operation *user_op = use->getOwner();

        // If the user is already a data_mov (created by another fused_op's
        // input wrapping), just update its operand to avoid nested data_mov
        if (auto existing_mov = llvm::dyn_cast<neura::DataMovOp>(user_op)) {
          if (use->getOperandNumber() == 0) { // data_mov only has one operand
            existing_mov->setOperand(0, new_result);
            continue;
          }
        }

        // Otherwise, create a new data_mov for this user
        auto mov = rewriter.create<neura::DataMovOp>(loc, new_result.getType(),
                                                     new_result);
        use->set(mov);
      }
    }

    fused_op->erase();
  }
}

struct InsertDataMovPass
    : public PassWrapper<InsertDataMovPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertDataMovPass)

  StringRef getArgument() const override { return "insert-data-mov"; }
  StringRef getDescription() const override {
    return "Insert neura.data_mov before all neura dialect operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InsertDataMovForNeuraOps>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));

    ModuleOp module_op = getOperation();

    // Step 1, handles fused_op operations specially.
    wrapFusedOpsWithDataMov(module_op);

    // Then applies patterns to every region inside the module, excluding
    // fused_op regions.
    // module_op.walk([&](Operation *op) {
    //   if (!op->getRegions().empty() && !llvm::isa<neura::FusedOp>(op)) {
    //     for (Region &region : op->getRegions()) {
    //       if (failed(applyPatternsGreedily(region, frozen))) {
    //         signalPassFailure();
    //       }
    //     }
    //   }
    // });

    // Step 2: Processes functions with neura accelerator attribute.
    llvm::errs() << "[InsertDataMovPass] Processing functions...\n";
    module_op.walk([&](func::FuncOp func_op) {
      auto accel_attr =
          func_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);

      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        llvm::errs() << "[InsertDataMovPass]   Skipping function: "
                     << func_op.getName() << " (not neura target)\n";
        return;
      }

      llvm::errs() << "[InsertDataMovPass]   Processing function: "
                   << func_op.getName() << "\n";

      Region &func_region = func_op.getBody();
      if (failed(applyPatternsGreedily(func_region, frozen))) {
        llvm::errs() << "[InsertDataMovPass]   ❌ Failed to apply patterns\n";
        signalPassFailure();
      } else {
        llvm::errs() << "[InsertDataMovPass]   ✅ Successfully processed\n";
      }
    });

    // Step 3: Processes kernels with neura accelerator attributes.
    llvm::errs() << "[InsertDataMovPass] Processing kernels...\n";
    module_op.walk([&](neura::KernelOp kernel_op) {
      auto accel_attr =
          kernel_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);

      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        llvm::errs()
            << "[InsertDataMovPass]   Skipping kernel (not neura target)\n";
        return;
      }

      llvm::errs() << "[InsertDataMovPass]   Processing kernel...\n";

      Region &kernel_region = kernel_op.getBody();
      if (failed(applyPatternsGreedily(kernel_region, frozen))) {
        llvm::errs() << "[InsertDataMovPass]   ❌ Failed to apply patterns\n";
        signalPassFailure();
      } else {
        llvm::errs() << "[InsertDataMovPass]   ✅ Successfully processed\n";
      }
    });

    llvm::errs() << "[InsertDataMovPass] ✅ Pass complete\n";
  }
};
} // namespace

namespace mlir {
namespace neura {

std::unique_ptr<Pass> createInsertDataMovPass() {
  return std::make_unique<InsertDataMovPass>();
}

} // namespace neura
} // namespace mlir
