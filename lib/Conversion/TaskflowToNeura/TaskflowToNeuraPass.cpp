#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===============================================================================
// Converts taskflow.hyperblock to neura.kernel.
//===============================================================================
// Pattern to convert taskflow.hyperblock to neura.kernel.
//
// Hyperblock structure:
//   %result = taskflow.hyperblock(%idx, %iter_init) {
//   ^bb0(%idx_arg: index, %iter_arg: T):
//     ... body ...
//     taskflow.hyperblock.yield outputs(%next_iter : T)
//   } : (index, T) -> T
//
// Kernel structure:
//   %result = neura.kernel ins(%idx, %live_in...) iter_args(%iter_init) {
//   ^bb0(%idx_arg: index, %live_in_args..., %iter_arg: T):
//     ... body ...
//     neura.yield iter_args(%next_iter) results(%next_iter)
//   } -> T
//
// Block argument order must match:
//   Hyperblock: [indices..., iter_args...]
//   Kernel:     [inputs (indices + live_ins)..., iter_args...]
struct HyperblockToKernelPattern
    : public OpRewritePattern<TaskflowHyperblockOp> {
  using OpRewritePattern<TaskflowHyperblockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskflowHyperblockOp hyperblock_op,
                                PatternRewriter &rewriter) const override {
    Location loc = hyperblock_op.getLoc();

    // Finds the parent task to access task's block arguments.
    TaskflowTaskOp task_op = hyperblock_op->getParentOfType<TaskflowTaskOp>();
    if (!task_op) {
      return failure();
    }

    // Asserts that each task contains only one hyperblock.
    // (Fused tasks may contain multiple hyperblocks, which is valid.)
    int hyperblock_count = 0;
    task_op.walk([&](TaskflowHyperblockOp op) { hyperblock_count++; });

    Block &hb_block = hyperblock_op.getBody().front();
    Block &task_block = task_op.getBody().front();

    // Gets hyperblock operands.
    SmallVector<Value> indices(hyperblock_op.getIndices());
    DenseSet<Value> indices_set(indices.begin(), indices.end());
    SmallVector<Value> iter_args_init(hyperblock_op.getIterArgs());
    DenseSet<Value> iter_args_init_set(iter_args_init.begin(),
                                       iter_args_init.end());
    size_t num_indices = indices.size();
    size_t num_iter_args_init = iter_args_init.size();

    // Collects live-in values of the hyperblock: task block arguments used in
    // the hyperblock body.
    llvm::DenseSet<Value> live_in_set;
    SmallVector<Value> live_in_values;

    // Collects constants that should be internalized in the kernel.
    SmallVector<Operation *> constant_ops_to_internalize;

    hyperblock_op.walk([&](Operation *op) {
      if (op == hyperblock_op.getOperation()) {
        // Skips the hyperblock op itself.
        return;
      }
      for (Value operand : op->getOperands()) {
        if (auto block_arg = dyn_cast<BlockArgument>(operand)) {
          Block *owner_block = block_arg.getOwner();

          if (block_arg.getOwner() == &task_block) {
            if (iter_args_init_set.contains(operand) ||
                indices_set.contains(operand)) {
              // Skips iter args and indices.
              continue;
            }
            if (live_in_set.insert(operand).second) {
              live_in_values.push_back(operand);
            }
          } else if (block_arg.getOwner() == &hb_block) {
            // Block argument from hyperblock - already handled.
            continue;
          } else {
            // Block argument from another block.
            Operation *owner_op = owner_block->getParentOp();

            // Checks if the owner op is INSIDE the hyperblock.
            bool is_inside_hyperblock = hyperblock_op->isAncestor(owner_op);

            if (is_inside_hyperblock) {
              // This is a block argument from an operation inside the
              // hyperblock (e.g., scf.for induction variable) These should NOT
              // be added as live-ins.
              continue;
            }

            // Checks if it's from an operation OUTSIDE hyperblock but INSIDE
            // task.
            bool is_in_task = task_op->isAncestor(owner_op);

            if (is_in_task) {
              // This is a block argument from an outer affine.for loop
              // Adds as live-in value.
              if (live_in_set.insert(operand).second) {
                live_in_values.push_back(operand);
              }
            } else {
              llvm::errs() << "ERROR: Block argument from outside task\n";
              llvm::errs() << "  Operand: " << operand << "\n";
              llvm::errs() << "  Owner block: " << *owner_block << "\n";
              llvm::errs() << "  Owner op: " << *owner_op << "\n";
              assert(false && "Unexpected block argument from outside task");
            }
          }
        } else if (operand.getDefiningOp()) {
          Operation *def_op = operand.getDefiningOp();

          // Checks three regions:
          // 1. Inside hyperblock
          // 2. Inside task body (but outside hyperblock)
          // 3. Outside task body (error)
          bool is_in_hyperblock = hyperblock_op->isProperAncestor(def_op);
          bool is_in_task_body = task_op->isProperAncestor(def_op);

          if (is_in_hyperblock) {
            // Defined inside hyperblock - do nothing.
            continue;
          } else if (is_in_task_body && !is_in_hyperblock) {
            // If it is a constant in task body, marks it for internalization.
            if (def_op->hasTrait<OpTrait::ConstantLike>()) {
              // Don't add to live_in.
              constant_ops_to_internalize.push_back(def_op);
              continue;
            } else {
              // Non-constant value from outer loop body.
              // Adds as live-in (will be passed from outer scope).
              if (live_in_set.insert(operand).second) {
                live_in_values.push_back(operand);
                llvm::errs()
                    << "[taskflow2neura] Added live-in from outer loop body: "
                    << operand << " from op: " << *def_op << "\n";
              }
              continue;
            }
          } else {
            // Defined outside task - ERROR.
            llvm::errs() << "ERROR: Value from outside task\n";
            llvm::errs() << "  Operand: " << operand << "\n";
            llvm::errs() << "  Defining op: " << *def_op << "\n";
            assert(false && "Operand defined outside task");
          }
        }
      }
    });

    // Builds the neura.kernel inputs: [indices..., live_ins...].
    SmallVector<Value> kernel_inputs;
    kernel_inputs.append(indices);
    kernel_inputs.append(live_in_values);

    // Result types from hyperblock.
    SmallVector<Type> resultTypes(hyperblock_op.getResultTypes());

    // Creates neura.kernel.
    neura::KernelOp kernelOp = rewriter.create<neura::KernelOp>(
        loc, resultTypes, kernel_inputs, iter_args_init,
        /*Optional cgra_id*/ nullptr, /*Optional kernel_name*/ nullptr,
        /*Optional accelerator*/ nullptr);

    // Creates the entry block for kernel.
    Region &kernel_region = kernelOp.getBody();
    Block *entry_block = rewriter.createBlock(&kernel_region);

    IRMapping mapping;

    // Kernel block argument layout: [inputs..., iter_args...]
    // Where inputs = [indices..., live_ins...]
    //
    // Hyperblock block argument layout: [indices..., iter_args...]

    // 1. Adds block arguments for indices and map to hyperblock's index args.
    for (size_t i = 0; i < num_indices; ++i) {
      BlockArgument kernel_indices_arg =
          entry_block->addArgument(indices[i].getType(), loc);
      BlockArgument hb_arg = hb_block.getArgument(i);
      mapping.map(hb_arg, kernel_indices_arg);
    }

    // 2. Adds block arguments for live-in values and map to task block args.
    for (Value live_in : live_in_values) {
      BlockArgument kernel_live_in_arg =
          entry_block->addArgument(live_in.getType(), loc);
      mapping.map(live_in, kernel_live_in_arg);
    }

    // 3. Adds block arguments for iter_args and map to hyperblock's iter_args.
    for (size_t i = 0; i < num_iter_args_init; ++i) {
      BlockArgument kernel_iter_arg =
          entry_block->addArgument(iter_args_init[i].getType(), loc);
      BlockArgument hb_arg = hb_block.getArgument(num_indices + i);
      mapping.map(hb_arg, kernel_iter_arg);
    }

    // Clones constants into kernel.
    rewriter.setInsertionPointToStart(entry_block);
    for (Operation *const_op : constant_ops_to_internalize) {
      Operation *cloned = rewriter.clone(*const_op);
      // Maps the original constant to the cloned one.
      for (size_t i = 0; i < const_op->getNumResults(); ++i) {
        mapping.map(const_op->getResult(i), cloned->getResult(i));
      }
    }

    // Clones hyperblock body into kernel.
    rewriter.setInsertionPointToEnd(entry_block);
    for (Operation &op : hb_block.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Converts hyperblock.yield to neura.yield.
    TaskflowHyperblockYieldOp hb_yield_op =
        cast<TaskflowHyperblockYieldOp>(hb_block.getTerminator());

    SmallVector<Value> iter_args_next;
    SmallVector<Value> results;

    // Maps yield outputs.
    for (Value out : hb_yield_op.getResults()) {
      Value mapped = mapping.lookupOrDefault(out);
      results.push_back(mapped);
    }

    for (Value iter_arg : hb_yield_op.getIterArgsNext()) {
      Value mapped = mapping.lookupOrDefault(iter_arg);
      iter_args_next.push_back(mapped);
    }

    rewriter.create<neura::YieldOp>(loc, iter_args_next, results);

    // Replaces hyperblock with kernel.
    rewriter.replaceOp(hyperblock_op, kernelOp.getResults());

    return success();
  }
};

struct InternalizeCounterPattern : public OpRewritePattern<neura::KernelOp> {
  using OpRewritePattern<neura::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::KernelOp kernel_op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs(kernel_op.getInputs());
    SmallVector<Value> iter_args_init(kernel_op.getIterArgsInit());

    // Finds counter inputs: inputs defined by taskflow.counter ops.
    SmallVector<std::pair<size_t, TaskflowCounterOp>> counter_inputs;

    for (size_t i = 0; i < inputs.size(); i++) {
      if (TaskflowCounterOp counter_op =
              inputs[i].getDefiningOp<TaskflowCounterOp>()) {
        counter_inputs.push_back({i, counter_op});
      }
    }

    // If there is no counter inputs, nothing to do.
    if (counter_inputs.empty()) {
      return failure();
    }

    Location loc = kernel_op.getLoc();
    Block &old_block = kernel_op.getBody().front();

    // Is this value produced by a constant-like op?
    auto isConstantLike = [](Value val) -> bool {
      return val.getDefiningOp() &&
             val.getDefiningOp()->hasTrait<OpTrait::ConstantLike>();
    };

    // Builds counter index set.
    DenseSet<size_t> counter_idx_set;
    for (auto &[idx, _] : counter_inputs) {
      counter_idx_set.insert(idx);
    }

    // Builds new_inputs:
    //    1) original non-counter inputs (preserved)
    //    2) dynamic bound values from counters (deduplicated)
    llvm::MapVector<Value, size_t> dynamic_bound_to_input_idx;
    SmallVector<Value> new_inputs;

    for (size_t i = 0; i < inputs.size(); i++) {
      if (!counter_idx_set.contains(i)) {
        new_inputs.push_back(inputs[i]);
      }
    }
    // Collects dynamic (non-constant) bound values that need to be passed into
    // the neura.kernel as additional inputs.
    for (auto &[idx, counter_op] : counter_inputs) {
      for (Value bound : {counter_op.getLowerBound(),
                          counter_op.getUpperBound(), counter_op.getStep()}) {
        if (!isConstantLike(bound) &&
            !dynamic_bound_to_input_idx.count(bound)) {
          dynamic_bound_to_input_idx.insert({bound, new_inputs.size()});
          new_inputs.push_back(bound);
        }
      }
    }

    // Creates new kernel with updated inputs.
    SmallVector<Type> result_types(kernel_op.getResultTypes());
    neura::KernelOp new_kernel_op = rewriter.create<neura::KernelOp>(
        loc, result_types, new_inputs, iter_args_init,
        /*cgra_id=*/kernel_op.getCgraIdAttr(),
        /*kernel_name=*/kernel_op.getKernelNameAttr(),
        /*accelerator=*/kernel_op.getAcceleratorAttr());

    // Creates the entry block for new kernel.
    Region &new_region = new_kernel_op.getBody();
    Block *new_block = rewriter.createBlock(&new_region);

    IRMapping mapping;

    // Block args for non-counter original inputs.
    for (size_t i = 0; i < inputs.size(); i++) {
      BlockArgument old_arg = old_block.getArgument(i);
      if (!counter_idx_set.contains(i)) {
        BlockArgument new_arg = new_block->addArgument(old_arg.getType(), loc);
        mapping.map(old_arg, new_arg);
      }
    }

    // Block args for iter_args.
    DenseMap<Value, Value> dynamic_bound_to_block_arg;
    for (auto &[bound_val, _] : dynamic_bound_to_input_idx) {
      Value arg = new_block->addArgument(bound_val.getType(), loc);
      dynamic_bound_to_block_arg[bound_val] = arg;
    }

    // Block args for iter_args.
    size_t num_inputs = inputs.size();
    for (size_t i = 0; i < iter_args_init.size(); i++) {
      BlockArgument old_arg = old_block.getArgument(num_inputs + i);
      BlockArgument new_arg = new_block->addArgument(old_arg.getType(), loc);
      mapping.map(old_arg, new_arg);
    }

    // Inserts neura.counter ops at the start of the new block.
    rewriter.setInsertionPointToStart(new_block);

    // Resolves a bound value into somthing valid inside new_block:
    //   constant -> clone the defining op
    //   dynamic bound -> use the corresponding block argument
    auto resolveBoundValue = [&](Value val) -> Value {
      if (isConstantLike(val)) {
        return rewriter.clone(*val.getDefiningOp())->getResult(0);
      }
      return dynamic_bound_to_block_arg.lookup(val);
    };

    for (auto &[old_idx, source_counter] : counter_inputs) {
      BlockArgument old_counter_arg = old_block.getArgument(old_idx);
      Value lb = resolveBoundValue(source_counter.getLowerBound());
      Value ub = resolveBoundValue(source_counter.getUpperBound());
      Value step = resolveBoundValue(source_counter.getStep());

      // Creates neura.counter op, forwarding hierarchy and dynamism attrs.
      neura::CounterOp new_counter_op = rewriter.create<neura::CounterOp>(
          source_counter.getLoc(), old_counter_arg.getType(), lb, ub, step,
          source_counter.getCounterHierarchyAttr(),
          source_counter.getCounterDynamismAttr(),
          source_counter.getCounterIdAttr());
      mapping.map(old_counter_arg, new_counter_op.getCurrentIndex());
    }

    // Clones rest of the body.
    rewriter.setInsertionPointToEnd(new_block);
    for (Operation &op : old_block.getOperations()) {
      rewriter.clone(op, mapping);
    }

    // Replaces old kernel with new kernel.
    rewriter.replaceOp(kernel_op, new_kernel_op.getResults());

    return success();
  }
};

struct ConvertTaskflowToNeuraPass
    : public PassWrapper<ConvertTaskflowToNeuraPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTaskflowToNeuraPass)

  StringRef getArgument() const override { return "convert-taskflow-to-neura"; }
  StringRef getDescription() const override {
    return "Convert taskflow operations to neura.kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::taskflow::TaskflowDialect>();
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Converts entire hyperblock to neura.kernel.
    // Phase 1: Converts hyperblocks to kernels.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<HyperblockToKernelPattern>(ctx);

      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 2: Internalizes counters into kernels.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<InternalizeCounterPattern>(ctx);

      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
  // }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertTaskflowToNeuraPass() {
  return std::make_unique<ConvertTaskflowToNeuraPass>();
}