#include "Conversion/ConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//------------------------------------------------------------------------------
// Helper Functions.
//------------------------------------------------------------------------------
// Collects memrefs that are loaded (read) within a given operation scope.
static void collectReadMemrefs(Operation *op, SetVector<Value> &read_memrefs) {
  op->walk([&](Operation *nested_op) {
    if (auto load_op = dyn_cast<affine::AffineLoadOp>(nested_op)) {
      read_memrefs.insert(load_op.getMemRef());
    } else if (auto load_op = dyn_cast<memref::LoadOp>(nested_op)) {
      read_memrefs.insert(load_op.getMemRef());
    }
  });
}

// Collects memrefs that are stored (written) within a given operation scope.
static void collectWrittenMemrefs(Operation *op,
                                  SetVector<Value> &written_memrefs) {
  op->walk([&](Operation *nested_op) {
    if (auto store_op = dyn_cast<affine::AffineStoreOp>(nested_op)) {
      written_memrefs.insert(store_op.getMemRef());
    } else if (auto store_op = dyn_cast<memref::StoreOp>(nested_op)) {
      written_memrefs.insert(store_op.getMemRef());
    }
  });
}

// Collects external values used within a given scope of operations.
static void collectExternalValues(Operation *root_op,
                                  const DenseSet<Operation *> &scope_ops,
                                  SetVector<Value> &external_values) {
  for (Value operand : root_op->getOperands()) {
    // Skips memref types (handled separately as memory dependencies).
    if (isa<MemRefType>(operand.getType())) {
      continue;
    }

    // Checks if it's a block argument.
    if (auto block_arg = dyn_cast<BlockArgument>(operand)) {
      // Only adds if the block argument is not from within the scope.
      Operation *parent_op = block_arg.getOwner()->getParentOp();
      if (!scope_ops.contains(parent_op)) {
        external_values.insert(operand);
      }
      continue;
    }

    // Checks if the operand is defined outside the scope.
    Operation *def_op = operand.getDefiningOp();
    if (def_op && !scope_ops.contains(def_op)) {
      external_values.insert(operand);
    }
  }

  // Recursively processes nested operations.
  for (Region &region : root_op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        collectExternalValues(&op, scope_ops, external_values);
      }
    }
  }
}

// Updates operands of a non-loop operation after preceding loops have been
// converted into taskflow.task ops. Memref operands consume the latest write
// state for that original memref; scalar SSA values use value_mapping.
static void
updateOperationOperands(Operation *op,
                        const DenseMap<Value, Value> &latest_write_out,
                        const DenseMap<Value, Value> &value_mapping) {
  for (OpOperand &operand : op->getOpOperands()) {
    Value original_value = operand.get();
    if (isa<MemRefType>(original_value.getType())) {
      auto it = latest_write_out.find(original_value);
      if (it != latest_write_out.end())
        operand.set(it->second);
      continue;
    }

    auto it = value_mapping.find(original_value);
    if (it != value_mapping.end()) {
      operand.set(it->second);
    }
  }
}

//------------------------------------------------------------------------------
// Analyzes all the original memory access info before conversion.
//------------------------------------------------------------------------------
struct MemrefAccessInfo {
  SetVector<Value> read_memrefs;
  SetVector<Value> write_memrefs;
};

static DenseMap<Operation *, MemrefAccessInfo>
analyzeMemrefAccesses(func::FuncOp func_op) {
  DenseMap<Operation *, MemrefAccessInfo> loop_to_memref_info;

  func_op.walk([&](affine::AffineForOp for_op) {
    llvm::errs() << "\nAnalyzing memref accesses for loop:\n" << for_op << "\n";
    MemrefAccessInfo access_info;

    collectReadMemrefs(for_op.getOperation(), access_info.read_memrefs);
    collectWrittenMemrefs(for_op.getOperation(), access_info.write_memrefs);

    loop_to_memref_info[for_op] = access_info;
  });

  return loop_to_memref_info;
}

//------------------------------------------------------------------------------
// Task Conversion
//------------------------------------------------------------------------------

// Converts a top-level affine.for to a taskflow.task operation.
static TaskflowTaskOp convertLoopToTask(
    OpBuilder &builder, affine::AffineForOp for_op,
    DenseMap<Value, Value> &latest_write_out,
    DenseMap<Value, SmallVector<Value>> &pending_read_outs,
    DenseMap<Value, Value> &value_mapping,
    const DenseMap<Operation *, MemrefAccessInfo> &loop_to_original_memref_info,
    const SetVector<Value> &read_output_memrefs, int task_id) {
  Location loc = for_op.getLoc();
  std::string task_name = "Task_" + std::to_string(task_id);

  // Collects all operations in the loop scope.
  DenseSet<Operation *> scope_ops;
  scope_ops.insert(for_op.getOperation());
  for_op.walk([&](Operation *op) { scope_ops.insert(op); });

  //-------------------------------------------------------------------
  // Step 1: Collects read and written memrefs.
  //-------------------------------------------------------------------
  SetVector<Value> read_memrefs;
  SetVector<Value> write_memrefs;
  collectReadMemrefs(for_op.getOperation(), read_memrefs);
  collectWrittenMemrefs(for_op.getOperation(), write_memrefs);

  llvm::errs() << "Read memrefs for loop:\n" << for_op << "\n";
  for (Value memref : read_memrefs) {
    llvm::errs() << memref << "\n";
  }

  llvm::errs() << "Written memrefs for loop:\n" << for_op << "\n";
  for (Value memref : write_memrefs) {
    llvm::errs() << memref << "\n";
  }

  // Collects original memref access info.
  auto it = loop_to_original_memref_info.find(for_op.getOperation());
  assert(it != loop_to_original_memref_info.end() &&
         "Original memref access info not found for the loop");
  const MemrefAccessInfo &original_memref_info = it->second;
  SetVector<Value> original_read_memrefs = original_memref_info.read_memrefs;
  SetVector<Value> original_write_memrefs = original_memref_info.write_memrefs;

  //-------------------------------------------------------------------
  // Step 2: Determines memory inputs and outputs.
  //-------------------------------------------------------------------
  // Memory outputs: ONLY memrefs that are written.
  // This ensures RAW and WAW dependencies are respected.
  SetVector<Value> output_memrefs;
  output_memrefs.insert(write_memrefs.begin(), write_memrefs.end());

  //-------------------------------------------------------------------
  // Step 3: Collects external SSA values (non-memref).
  //-------------------------------------------------------------------
  SetVector<Value> external_values;
  collectExternalValues(for_op.getOperation(), scope_ops, external_values);

  llvm::errs() << "External values for loop:\n" << for_op << "\n";
  for (Value val : external_values) {
    llvm::errs() << val << "\n";
  }

  //-------------------------------------------------------------------
  // Step 4: Resolves inputs through value mapping.
  //-------------------------------------------------------------------
  SmallVector<Value> read_inputs;
  SmallVector<Value> write_inputs;
  SmallVector<Value> write_input_states;
  SmallVector<Value> write_body_state_memrefs;
  SmallVector<Value> value_inputs;
  IRMapping mapping;

  // Resolves read inputs from the latest write state only. Read outputs from
  // previous readers are pending WAR dependencies for future writers and must
  // not serialize independent readers.
  for (Value memref : read_memrefs) {
    Value resolved_memref = latest_write_out.lookup(memref);
    if (!resolved_memref) {
      resolved_memref = memref;
    }
    read_inputs.push_back(resolved_memref);
    mapping.map(memref, resolved_memref);
  }

  // Resolves write inputs. A writer must consume all pending read states for
  // the same original memref before it writes, then it consumes the latest
  // write state if there is no pending read state. This keeps WAR ordering in
  // the memref dependence-state chain without introducing read-read edges.
  for (Value memref : write_memrefs) {
    auto pending_it = pending_read_outs.find(memref);
    if (pending_it != pending_read_outs.end() && !pending_it->second.empty()) {
      for (Value pending_read : pending_it->second) {
        write_inputs.push_back(pending_read);
        write_input_states.push_back(pending_read);
      }
      write_body_state_memrefs.push_back(pending_it->second.back());
    } else {
      Value resolved_memref = latest_write_out.lookup(memref);
      if (!resolved_memref) {
        resolved_memref = memref;
      }
      write_inputs.push_back(resolved_memref);
      write_input_states.push_back(resolved_memref);
      write_body_state_memrefs.push_back(resolved_memref);
    }
  }

  // Resolves external SSA value inputs.
  for (Value external_val : external_values) {
    Value resolved_val = value_mapping.lookup(external_val);
    if (!resolved_val) {
      resolved_val = external_val;
    }
    value_inputs.push_back(resolved_val);
    mapping.map(external_val, resolved_val);
  }

  //-------------------------------------------------------------------
  // Step 5: Prepares output types.
  //-------------------------------------------------------------------
  // Read output types: sparse read states for WAR ordering. These are produced
  // only when a later writer must depend on this task's read.
  SmallVector<Type> read_output_types;
  for (Value memref : read_output_memrefs) {
    read_output_types.push_back(memref.getType());
  }

  SmallVector<Type> memory_output_types;
  for (Value memref : output_memrefs) {
    memory_output_types.push_back(memref.getType());
  }

  SmallVector<Type> value_output_types;
  for (Type result_type : for_op.getResultTypes()) {
    value_output_types.push_back(result_type);
  }

  //-------------------------------------------------------------------
  // Step 6: Creates the taskflow.task operation.
  //-------------------------------------------------------------------
  TaskflowTaskOp task_op = builder.create<TaskflowTaskOp>(
      loc,
      /*read_outputs=*/read_output_types,
      /*write_outputs=*/memory_output_types,
      /*value_outputs=*/value_output_types,
      /*read_inputs=*/read_inputs,
      /*write_inputs=*/write_inputs,
      /*value_inputs=*/value_inputs,
      /*task_name=*/builder.getStringAttr(task_name),
      /*original_read_memrefs=*/original_read_memrefs.getArrayRef(),
      /*original_write_memrefs=*/original_write_memrefs.getArrayRef());

  //-------------------------------------------------------------------
  // Step 7: Builds the task body.
  //-------------------------------------------------------------------
  Block *task_body = new Block();
  task_op.getBody().push_back(task_body);

  // Adds block arguments (memory inputs first, then value inputs).
  DenseMap<Value, BlockArgument> input_to_block_arg;
  // Memory read input arguments.
  for (Value memref : read_memrefs) {
    BlockArgument arg = task_body->addArgument(memref.getType(), loc);
    mapping.map(memref, arg);
    input_to_block_arg[memref] = arg;
  }

  // Memory write input arguments. There can be multiple dependence states for
  // one original write memref when several previous readers are pending; the
  // last argument is used inside the task body as the writable memref state.
  for (Value state_memref : write_input_states) {
    BlockArgument arg = task_body->addArgument(state_memref.getType(), loc);
    mapping.map(state_memref, arg);
    input_to_block_arg[state_memref] = arg;
  }
  for (auto [original_memref, state_memref] :
       llvm::zip(write_memrefs, write_body_state_memrefs)) {
    auto it = input_to_block_arg.find(state_memref);
    assert(it != input_to_block_arg.end() && "write state has no block arg");
    mapping.map(original_memref, it->second);
    input_to_block_arg[original_memref] = it->second;
  }

  // Value input arguments.
  for (Value val : external_values) {
    BlockArgument arg = task_body->addArgument(val.getType(), loc);
    mapping.map(val, arg);
    input_to_block_arg[val] = arg;
  }

  // Clones loop into the task body.
  OpBuilder task_builder(task_body, task_body->begin());
  Operation *cloned_loop = task_builder.clone(*for_op.getOperation(), mapping);

  //---------------------------------------------------------------
  // Step 8: Creates the yield operation.
  //---------------------------------------------------------------
  task_builder.setInsertionPointToEnd(task_body);
  SmallVector<Value> yield_for_done_reads;
  SmallVector<Value> yield_for_done_writes;
  SmallVector<Value> value_yield_operands;

  // Read yield outputs: passthrough only sparse read states needed by later
  // writers.
  for (Value memref : read_output_memrefs) {
    if (input_to_block_arg.count(memref)) {
      yield_for_done_reads.push_back(input_to_block_arg[memref]);
    } else {
      assert(false && "Read memref not in inputs!");
    }
  }

  // Memory yield outputs: yield the written memrefs.
  for (Value memref : output_memrefs) {
    if (input_to_block_arg.count(memref)) {
      yield_for_done_writes.push_back(input_to_block_arg[memref]);
    } else {
      assert(false && "Written memref not in inputs!");
    }
  }

  // Value yield outputs: yield the loop results.
  for (Value result : cloned_loop->getResults()) {
    value_yield_operands.push_back(result);
  }
  task_builder.create<TaskflowYieldOp>(loc, yield_for_done_reads,
                                       yield_for_done_writes,
                                       value_yield_operands);

  //-------------------------------------------------------------------
  // Step 9 : Updates value mapping with task outputs for subsequent tasks
  // conversion.
  //-------------------------------------------------------------------
  // Read outputs: pending WAR states for future writers. These do not update
  // latest_write_out, so later readers of the same memref remain independent.
  for (auto [memref, task_read_output] :
       llvm::zip(read_output_memrefs, task_op.getDoneReads())) {
    pending_read_outs[memref].push_back(task_read_output);
  }

  // Memory outputs (write): establishes RAW/WAW dependency chain and consumes
  // pending reads for that memref.
  for (auto [memref, task_output] :
       llvm::zip(output_memrefs, task_op.getDoneWrites())) {
    latest_write_out[memref] = task_output;
    pending_read_outs[memref].clear();
  }

  return task_op;
}

//------------------------------------------------------------------------------
// Main Conversion Process.
//------------------------------------------------------------------------------
// Converts a single function to TaskFlow operations.
static LogicalResult convertFuncToTaskflow(func::FuncOp func_op) {

  llvm::errs() << "\n===Converting function: " << func_op.getName() << "===\n";

  DenseMap<Operation *, MemrefAccessInfo> loop_to_original_memref_info =
      analyzeMemrefAccesses(func_op);
  OpBuilder builder(func_op.getContext());
  SmallVector<affine::AffineForOp> loops_to_erase;
  DenseMap<Value, Value> latest_write_out;
  DenseMap<Value, SmallVector<Value>> pending_read_outs;
  DenseMap<Value, Value> value_mapping;
  int task_id_counter = 0;

  // Processes each block in the function.
  for (Block &block : func_op.getBlocks()) {
    // Collects operations to process (to avoid iterator invalidation).
    SmallVector<Operation *> ops_to_process;
    for (Operation &op : block) {
      ops_to_process.push_back(&op);
    }

    llvm::errs() << "ops_to_process:\n";
    for (Operation *op : ops_to_process) {
      llvm::errs() << *op << "\n";
    }

    // Computes sparse read-out liveness. A task only needs to yield a read
    // state when a later task in the same block writes the same original
    // memref. Pure read-read sharing must not create dependence edges.
    DenseMap<Operation *, SetVector<Value>> loop_to_read_output_memrefs;
    DenseSet<Value> future_write_memrefs;
    for (Operation *op : llvm::reverse(ops_to_process)) {
      auto for_op = dyn_cast<affine::AffineForOp>(op);
      if (!for_op) {
        continue;
      }

      auto info_it = loop_to_original_memref_info.find(for_op.getOperation());
      assert(info_it != loop_to_original_memref_info.end() &&
             "Original memref access info not found for the loop");

      const MemrefAccessInfo &access_info = info_it->second;
      SetVector<Value> read_outputs;
      for (Value memref : access_info.read_memrefs) {
        if (future_write_memrefs.contains(memref) &&
            !access_info.write_memrefs.contains(memref)) {
          read_outputs.insert(memref);
        }
      }
      loop_to_read_output_memrefs[for_op.getOperation()] = read_outputs;

      for (Value memref : access_info.write_memrefs) {
        future_write_memrefs.insert(memref);
      }
    }

    // Processes each operation in order (top to bottom).
    for (Operation *op : ops_to_process) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
        // Converts affine.for to taskflow.task.
        OpBuilder builder(for_op);
        TaskflowTaskOp task_op = convertLoopToTask(
            builder, for_op, latest_write_out, pending_read_outs, value_mapping,
            loop_to_original_memref_info,
            loop_to_read_output_memrefs[for_op.getOperation()],
            task_id_counter++);

        // Replaces uses of loop results with task value outputs.
        for (auto [loop_result, task_value_output] :
             llvm::zip(for_op.getResults(), task_op.getValueOutputs())) {
          loop_result.replaceAllUsesWith(task_value_output);
        }
        loops_to_erase.push_back(for_op);
      } else {
        // Updates operands of non-loop operations based on the taskflow values
        // produced by earlier converted loops.
        updateOperationOperands(op, latest_write_out, value_mapping);
      }
    }
  }

  // Erases the original loops after conversion.
  for (affine::AffineForOp for_op : loops_to_erase) {
    for_op.erase();
  }

  return success();
}

class ConvertAffineToTaskflowPass
    : public PassWrapper<ConvertAffineToTaskflowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAffineToTaskflowPass)

  StringRef getArgument() const final { return "convert-affine-to-taskflow"; }

  StringRef getDescription() const final {
    return "Convert Affine operations to Taskflow operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TaskflowDialect, affine::AffineDialect, func::FuncDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    WalkResult result = module.walk([](func::FuncOp func_op) {
      if (failed(convertFuncToTaskflow(func_op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertAffineToTaskflowPass() {
  return std::make_unique<ConvertAffineToTaskflowPass>();
}
