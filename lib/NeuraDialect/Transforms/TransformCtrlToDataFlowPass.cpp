#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>

using namespace mlir;

#define GEN_PASS_DEF_TRANSFORMCTRLTODATAFLOW
#include "NeuraDialect/NeuraPasses.h.inc"

// Attribute name to mark iter_arg init constants.
constexpr const char *kIterArgInitAttr = "is_iter_arg_init";

//---------------------------------------------------------------------------
// Checks if task has counter (root counter).
//---------------------------------------------------------------------------
bool taskHasCounter(neura::KernelOp kernel_op) {
  if (!kernel_op) {
    return false;
  }

  bool found_counter = false;
  kernel_op.walk([&](neura::CounterOp counter_op) {
    if (counter_op) {
      found_counter = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return found_counter;
}

//---------------------------------------------------------------------------
// Finds root counter in the neura.kernel operation.
//---------------------------------------------------------------------------
neura::CounterOp findRootCounterInKernel(neura::KernelOp kernel_op) {
  if (!kernel_op) {
    return nullptr;
  }

  neura::CounterOp root_counter = nullptr;
  neura::CounterOp leaf_counter = nullptr;

  // Walks throught kernel body to find counter.
  kernel_op.walk([&](neura::CounterOp counter_op) {
    StringAttr hierarchy = counter_op.getCounterHierarchyAttr();
    if (!hierarchy) {
      return;
    }
    if (hierarchy.getValue() == "root") {
      root_counter = counter_op;
    } else if (hierarchy.getValue() == "leaf") {
      leaf_counter = counter_op;
    }
  });

  if (root_counter) {
    return root_counter;
  } else if (leaf_counter) {
    return leaf_counter;
  }

  return nullptr;
}

// Inserts `grant_once` for every predicated value defined in the entry block
// that is used outside of the block (i.e., a live-out).
void GrantPredicateInEntryBlock(Block *entry_block, OpBuilder &builder,
                                bool has_task_counter = false) {
  if (has_task_counter) {
    return;
  }
  SmallVector<Value> live_out_arg_values;

  // Step 1: Collects all live-out values first.
  for (Operation &op : *entry_block) {
    for (Value result : op.getResults()) {
      if (!isa<neura::PredicatedValue>(result.getType())) {
        continue;
      }

      bool used_in_branch = false;

      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();

        // Case 1: Operand of a branch/cond_br → grant_once
        // Since we add the --cononicalize-live-in pass, all the live-out values
        // in entry block must be passed to other blocks using branch/cond_br.
        if (isa<neura::Br, neura::CondBr>(user)) {
          used_in_branch = true;
        }

        if (!isa<neura::Br, neura::CondBr>(user) &&
            user->getBlock() != entry_block) {
          assert(
              false &&
              "Live-out value in entry block must be used in a branch/cond_br "
              "operation.");
        }
      }

      if (used_in_branch) {
        live_out_arg_values.push_back(result);
      }
    }
  }

  // Step 2: Inserts grant_once for each candidate.
  // Inserts grant_once.
  for (Value val : live_out_arg_values) {
    Operation *def_op = val.getDefiningOp();
    if (!def_op) {
      continue;
    }

    builder.setInsertionPointAfter(def_op);
    auto granted = builder.create<neura::GrantOnceOp>(def_op->getLoc(),
                                                      val.getType(), val);

    // Replaces uses in branch ops.
    for (OpOperand &use : llvm::make_early_inc_range(val.getUses())) {
      Operation *user = use.getOwner();
      if (isa<neura::Br, neura::CondBr>(user)) {
        use.set(granted.getResult());
      }
    }
  }
}

//---------------------------------------------------------------------------
// Iter_args handling (always grant_once for now).
//---------------------------------------------------------------------------
void handleKernelIterArgs(neura::KernelOp kernel_op, Block *entry_block,
                          OpBuilder &builder,
                          SmallVector<Value> &iter_arg_phi_values) {
  llvm::errs() << "[iter_args] Handling kernel iter_args...\n";

  SmallVector<neura::ConstantOp> iter_arg_init_ops;
  for (Operation &op : kernel_op.getOps()) {
    if (auto const_op = dyn_cast<neura::ConstantOp>(op)) {
      if (const_op->hasAttr(kIterArgInitAttr) &&
          const_op->getAttrOfType<BoolAttr>(kIterArgInitAttr).getValue()) {
        iter_arg_init_ops.push_back(const_op);
      }
    }
  }

  if (iter_arg_init_ops.empty()) {
    llvm::errs() << "[iter_args]   No iter_args\n";
    return;
  }

  neura::YieldOp yield_op = nullptr;
  for (Operation &op : kernel_op.getOps()) {
    if (auto yld = dyn_cast<neura::YieldOp>(op)) {
      yield_op = yld;
      break;
    }
  }

  if (!yield_op || yield_op.getIterArgsNext().empty()) {
    llvm::errs() << "[iter_args]   No iter_args_next in yield\n";
    return;
  }

  for (size_t i = 0; i < iter_arg_init_ops.size(); ++i) {
    neura::ConstantOp init_const = iter_arg_init_ops[i];
    Value feedback_value = yield_op.getIterArgsNext()[i];

    llvm::errs() << "[iter_args]   Processing iter_arg " << i << "\n";

    // Grants once the init.
    builder.setInsertionPointAfter(init_const);
    neura::GrantOnceOp granted_init = builder.create<neura::GrantOnceOp>(
        init_const.getLoc(), init_const.getType(), init_const.getResult());

    // Creates reserve for feedback value.
    builder.setInsertionPointAfter(granted_init);
    neura::ReserveOp reserve_op = builder.create<neura::ReserveOp>(
        init_const.getLoc(), init_const.getType());

    // Creates phi for init and feedback.
    builder.setInsertionPointAfter(reserve_op);
    neura::PhiOp phi = builder.create<neura::PhiOp>(
        init_const.getLoc(), init_const.getType(),
        ValueRange{granted_init.getResult(), reserve_op.getResult()});

    // Replaces uses.
    init_const.getResult().replaceUsesWithIf(
        phi.getResult(), [&](OpOperand &use) {
          Operation *user = use.getOwner();
          return user != granted_init && !isa<neura::CtrlMovOp>(user);
        });

    // Creates ctrl_mov.
    builder.setInsertionPoint(yield_op);
    builder.create<neura::CtrlMovOp>(yield_op.getLoc(), feedback_value,
                                     reserve_op.getResult());

    iter_arg_phi_values.push_back(phi.getResult());

    init_const->removeAttr(kIterArgInitAttr);
    llvm::errs() << "[iter_args]     Created iter_arg with grant_once\n";
  }

  llvm::errs() << "[iter_args] Iter_args complete\n\n";
}

//---------------------------------------------------------------------------
// Handles kernel yield with counter-based gating.
//---------------------------------------------------------------------------
void handleKernelYieldTermination(
    neura::KernelOp kernel_op, Block *entry_block, OpBuilder &builder,
    bool has_task_counter, const SmallVector<Value> &iter_arg_phi_values) {
  llvm::errs() << "[yield] ========================================\n";
  llvm::errs() << "[yield] Handling Yield Termination\n";
  llvm::errs() << "[yield] ========================================\n";

  neura::YieldOp yield_op = nullptr;
  for (Operation &op : kernel_op.getOps()) {
    if (auto yld = dyn_cast<neura::YieldOp>(op)) {
      yield_op = yld;
      break;
    }
  }

  if (!yield_op) {
    llvm::errs() << "[yield] No yield operation found\n";
    return;
  }

  builder.setInsertionPoint(yield_op);

  if (!yield_op->hasAttr("yield_type")) {
    llvm::errs() << "[yield] No yield_type attribute\n";
    yield_op.erase();
    return;
  }

  StringRef yield_type =
      yield_op->getAttrOfType<StringAttr>("yield_type").getValue();

  //--------------------------------------------------------------------------
  // Case 1: VALUE yield
  //--------------------------------------------------------------------------
  if (yield_type == "value") {
    llvm::errs() << "[yield] Processing VALUE yield\n";

    if (has_task_counter) {
      llvm::errs()
          << "[yield]   Has counter → Gate with NOT(counter predicate)\n";

      // Finds counter in kernel that defines the predicate.
      neura::CounterOp counter_op = findRootCounterInKernel(kernel_op);

      assert(counter_op &&
             "Kernel has outer task counter but no neura::CounterOp found.");

      // Extracts predicate and negates it.
      Value counter_value = counter_op.getCurrentIndex();

      auto pred_type = builder.getType<neura::PredicatedValue>(
          builder.getI1Type(), builder.getI1Type());

      auto extract_pred = builder.create<neura::ExtractPredicateOp>(
          counter_op.getLoc(), pred_type, counter_value);

      Value counter_pred = extract_pred.getPredicate();

      // When the counter predicate is false, we want to trigger the return.
      auto not_op = builder.create<neura::NotOp>(
          counter_op.getLoc(), counter_pred.getType(), counter_pred);

      Value return_gate = not_op.getResult();

      llvm::errs() << "[yield]     Extracted counter predicate\n";
      llvm::errs() << "[yield]     Created NOT gate for return\n";

      // Gates all results with NOT (counter predicate).
      SmallVector<Value> gated_results;
      for (size_t i = 0; i < yield_op.getResults().size(); ++i) {
        Value result_to_gate = iter_arg_phi_values[i];

        auto gated = builder.create<neura::GrantPredicateOp>(
            yield_op.getLoc(), result_to_gate.getType(), result_to_gate,
            return_gate);
        gated_results.push_back(gated.getResult());

        llvm::errs() << "[yield]     Gated result with NOT(counter_pred)\n";
      }

      auto return_val = builder.create<neura::ReturnValueOp>(yield_op.getLoc(),
                                                             gated_results);
      llvm::errs() << "[yield]   Created return_value with counter gating\n";
      builder.setInsertionPointAfter(return_val);
      builder.create<neura::YieldOp>(builder.getUnknownLoc());

    } else {
      llvm::errs() << "[yield]   No counter, handled as normal case.\n";
    }
    yield_op.erase();
  }
  //--------------------------------------------------------------------------
  // Case 2: VOID yield
  //--------------------------------------------------------------------------
  else if (yield_type == "void") {
    llvm::errs() << "[yield] Processing VOID yield\n";
  }
  llvm::errs() << "[yield] ========================================\n\n";
}

// Control flow struct.
struct ControlFlowInfo {
  struct Edge {
    Block *source;
    Block *target;
    Value condition; // Optional condition for the edge.
    bool is_not_condition;
    SmallVector<Value> passed_values; // Values passed to the target block.
    bool is_back_edge;
  };
  SmallVector<std::unique_ptr<Edge>> all_edges; // All edges in the function.
  llvm::MapVector<Block *, SmallVector<Edge *>>
      incoming_edges; // Incoming edges for each block.
  llvm::MapVector<Block *, SmallVector<Edge *>>
      outgoing_edges; // Outgoing edges for each block.

  llvm::MapVector<Block *, SmallVector<Edge *>> back_edges;
  llvm::MapVector<Block *, SmallVector<Edge *>> forward_edges;
  llvm::SmallVector<Block *>
      blocks_with_back_edges; // Blocks with backward edges.

  Edge *createEdge() {
    all_edges.push_back(std::make_unique<Edge>());
    return all_edges.back().get();
  }
};

// Checks if all the live-out values in a block are dominated by the block's
// arguments.
void assertLiveOutValuesDominatedByBlockArgs(Region &region) {
  llvm::errs()
      << "[ctrl2data] Asserting live-out values dominated by block arguments\n";
  for (Block &block : region) {
    if (&block == &region.front()) {
      continue;
    }

    DenseSet<Value> live_out_values;
    for (Operation &op : block) {
      for (Value result : op.getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (use.getOwner()->getBlock() != &block) {
            live_out_values.insert(result);
            break;
          }
        }
      }
    }

    // Skips blocks with no live-out values.
    if (live_out_values.empty()) {
      continue;
    }

    DenseSet<Value> dominated_values;

    if (block.getNumArguments() == 0 && !live_out_values.empty()) {
      assert(false && "Block without arguments has live-out values, please "
                      "enable the --canonicalize-live-in pass.");
    }

    for (BlockArgument arg : block.getArguments()) {
      dominated_values.insert(arg);
    }

    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : block) {
        for (Value result : op.getResults()) {
          if (dominated_values.count(result)) {
            continue;
          }

          for (Value operand : op.getOperands()) {
            if (dominated_values.count(operand)) {
              dominated_values.insert(result);
              changed = true;
              break;
            }
          }
        }
      }
    }
    for (Value live_out : live_out_values) {
      if (!dominated_values.count(live_out)) {
        assert(
            false &&
            "Live-out value not dominated by block arguments or "
            "live-in values, please enable the --canonicalize-live-in pass.");
      }
    }
  }

  llvm::errs() << "[ctrl2data] All live-out values are dominated by block "
                  "arguments or live-in values.\n";
}

// Builds control flow info for the given function.
void buildControlFlowInfo(Region &region, ControlFlowInfo &ctrl_info,
                          DominanceInfo &dom_info) {
  for (Block &block : region) {
    Operation *terminator = block.getTerminator();

    if (auto cond_br = dyn_cast<neura::CondBr>(terminator)) {
      Block *true_dest = cond_br.getTrueDest();
      Block *false_dest = cond_br.getFalseDest();

      // Creates an edge for true destination.
      ControlFlowInfo::Edge *true_edge = ctrl_info.createEdge();
      true_edge->source = &block;
      true_edge->target = true_dest;
      true_edge->condition = cond_br.getCondition();
      true_edge->is_not_condition = false;
      true_edge->is_back_edge = dom_info.dominates(true_dest, &block);
      for (Value passed_value : cond_br.getTrueArgs()) {
        true_edge->passed_values.push_back(passed_value);
      }

      // Creates an edge for false destination.
      ControlFlowInfo::Edge *false_edge = ctrl_info.createEdge();
      false_edge->source = &block;
      false_edge->target = false_dest;
      false_edge->condition = cond_br.getCondition();
      false_edge->is_not_condition = true;
      false_edge->is_back_edge = dom_info.dominates(false_dest, &block);
      for (Value passed_value : cond_br.getFalseArgs()) {
        false_edge->passed_values.push_back(passed_value);
      }

      // Creates the blocks to edges mapping.
      ctrl_info.outgoing_edges[&block].push_back(true_edge);
      ctrl_info.outgoing_edges[&block].push_back(false_edge);
      ctrl_info.incoming_edges[true_dest].push_back(true_edge);
      ctrl_info.incoming_edges[false_dest].push_back(false_edge);

      // Handles back edges.
      if (true_edge->is_back_edge) {
        ctrl_info.back_edges[&block].push_back(true_edge);
        ctrl_info.blocks_with_back_edges.push_back(&block);
      } else {
        ctrl_info.forward_edges[&block].push_back(true_edge);
      }
      if (false_edge->is_back_edge) {
        ctrl_info.back_edges[&block].push_back(false_edge);
        ctrl_info.blocks_with_back_edges.push_back(&block);
      } else {
        ctrl_info.forward_edges[&block].push_back(false_edge);
      }

    } else if (auto br = dyn_cast<neura::Br>(terminator)) {
      Block *dest = br.getDest();

      // Creates unconditional edge to the destination block.
      ControlFlowInfo::Edge *edge = ctrl_info.createEdge();
      edge->source = &block;
      edge->target = dest;
      edge->condition = nullptr; // No condition for Br.
      edge->is_not_condition = false;
      edge->is_back_edge = dom_info.dominates(dest, &block);
      for (Value passed_value : br.getArgs()) {
        edge->passed_values.push_back(passed_value);
      }

      // Updates the blocks to edges mapping.
      ctrl_info.outgoing_edges[&block].push_back(edge);
      ctrl_info.incoming_edges[dest].push_back(edge);

      // Handles back edges.
      if (edge->is_back_edge) {
        ctrl_info.back_edges[&block].push_back(edge);
        ctrl_info.blocks_with_back_edges.push_back(&block);
      } else {
        ctrl_info.forward_edges[&block].push_back(edge);
      }

    } else if (auto rt = dyn_cast<neura::ReturnOp>(terminator)) {
      llvm::errs() << "[ctrl2data] ReturnOp found: " << *rt << "\n";
    } else if (auto yield = dyn_cast<neura::YieldOp>(terminator)) {
      llvm::errs() << "[ctrl2data] YieldOp found: " << *yield << "\n";
    } else {
      assert(false && "Unknown terminator operation in control flow graph.");
    }
  }
}

Value getProcessedCondition(Value condition, bool is_not_condition,
                            llvm::MapVector<Value, Value> &condition_cache,
                            OpBuilder &builder) {
  if (!is_not_condition) {
    return condition;
  }

  // First, checks if we already have a cached negated condition.
  auto it = condition_cache.find(condition);
  if (it != condition_cache.end()) {
    return it->second;
  }

  // Second, searches for an existing not operation that uses this condition.
  // This handles the not operations created by CanonicalizeReturnPass.
  for (OpOperand &use : condition.getUses()) {
    if (neura::NotOp not_op = dyn_cast<neura::NotOp>(use.getOwner())) {
      if (not_op.getOperand() == condition) {
        Value not_result = not_op.getResult();
        condition_cache[condition] = not_result;
        return not_result;
      }
    }
  }

  // Third, creates a new not operation to negate the condition.
  Block *source = condition.getDefiningOp()->getBlock();
  builder.setInsertionPoint(source->getTerminator());
  Value not_condition = builder.create<neura::NotOp>(
      condition.getLoc(), condition.getType(), condition);
  condition_cache[condition] = not_condition;
  return not_condition;
}

void createReserveAndPhiOps(
    Region &region, ControlFlowInfo &ctrl_info,
    llvm::MapVector<BlockArgument, Value> &arg_to_reserve,
    llvm::MapVector<BlockArgument, Value> &arg_to_phi_result,
    OpBuilder &builder) {
  DominanceInfo dom_info(region.getParentOp());

  // ================================================
  // Step 1: Categorizes edges into six types.
  // ================================================
  // Type 1: Backward cond_br edges with values.
  // Type 2: Backward br edges with values.
  // Type 3: Forward cond_br edges with values.
  // Type 4: Forward br edges with values.
  // Type 5: Forward cond_br edges without values.
  // Type 6: Forward br edges without values.
  // Type 7: Backward cond_br edges without values.
  // Type 8: Backward br edges without values.

  // After --canonicalize-live-in pass, all live-in values are promoted to block
  // arguments. This means that any edge without values can now be
  // treated as a edge with values.

  // ***************************************************************************
  // * So we only need to handle edges with values, i.e., Type 1, 2, 3, and 4. *
  // ***************************************************************************

  // Uses llvm::MapVector instead of DenseMap to maintain insertion order.
  llvm::MapVector<BlockArgument, SmallVector<ControlFlowInfo::Edge *>>
      backward_value_edges;
  llvm::MapVector<BlockArgument, SmallVector<ControlFlowInfo::Edge *>>
      forward_value_edges;
  llvm::MapVector<Block *, SmallVector<ControlFlowInfo::Edge *>>
      block_conditional_edges;

  llvm::MapVector<Value, Value> condition_cache;

  llvm::MapVector<BlockArgument, SmallVector<Value>> arg_to_phi_operands;

  // Tracks the mapping of live-out values.
  llvm::MapVector<Value, Value> value_to_predicated_value;

  for (auto &edge : ctrl_info.all_edges) {
    Block *target = edge->target;

    // Type 1 & 2: Backward cond_br/br edges with values.
    if (edge->is_back_edge && !edge->passed_values.empty()) {
      if (edge->passed_values.size() != target->getNumArguments()) {
        llvm::errs()
            << "[ctrl2data] Error: Number of passed values does not match "
               "target block arguments.\n";
        assert(false);
      }
      for (BlockArgument arg : target->getArguments()) {
        backward_value_edges[arg].push_back(edge.get());
      }
    }
    // Type 3 & 4: Forward cond_br/br edges with values.
    else if (!edge->is_back_edge && !edge->passed_values.empty()) {
      ;
      if (edge->passed_values.size() != target->getNumArguments()) {
        llvm::errs()
            << "[ctrl2data] Error: Number of passed values does not match "
               "target block arguments.\n";
        assert(false);
      }
      for (BlockArgument arg : target->getArguments()) {
        forward_value_edges[arg].push_back(edge.get());
      }
    }
  }

  // ================================================
  // Step 3: Creates reserve and ctrl_mov operations for needed blockarguments.
  // ================================================
  // Handles Type 1 & 2 edges.
  for (auto &backward_pair : backward_value_edges) {
    BlockArgument arg = backward_pair.first;
    auto &edges = backward_pair.second;
    Block *block = arg.getOwner();
    builder.setInsertionPointToStart(block);
    neura::ReserveOp reserve_op =
        builder.create<neura::ReserveOp>(arg.getLoc(), arg.getType());
    arg_to_reserve[arg] = reserve_op.getResult();
    arg_to_phi_operands[arg].push_back(reserve_op.getResult());

    // Creates ctrl_mov operations for reserved values.
    for (ControlFlowInfo::Edge *edge : edges) {
      Value val = edge->passed_values[arg.getArgNumber()];
      builder.setInsertionPoint(edge->source->getTerminator());

      Value predicated_val = val;
      if (edge->condition) {
        Value processed_condition = getProcessedCondition(
            edge->condition, edge->is_not_condition, condition_cache, builder);
        predicated_val = builder.create<neura::GrantPredicateOp>(
            edge->condition.getLoc(), val.getType(), predicated_val,
            processed_condition);
      }

      // Creates ctrl_mov operation.
      builder.create<neura::CtrlMovOp>(val.getLoc(), predicated_val,
                                       reserve_op);
    }
  }

  // ================================================
  // Step 4: Prepares for creating phi operations.
  // ================================================
  // Handles Type 3 & 4 edges.

  for (auto &forward_pair : forward_value_edges) {
    BlockArgument arg = forward_pair.first;
    auto &edges = forward_pair.second;

    for (ControlFlowInfo::Edge *edge : edges) {
      Value val = edge->passed_values[arg.getArgNumber()];

      Value predicated_val = val;
      if (edge->condition) {
        builder.setInsertionPoint(edge->source->getTerminator());
        Value processed_condition = getProcessedCondition(
            edge->condition, edge->is_not_condition, condition_cache, builder);

        predicated_val = builder.create<neura::GrantPredicateOp>(
            edge->condition.getLoc(), predicated_val.getType(), predicated_val,
            processed_condition);
      }

      arg_to_phi_operands[arg].push_back(predicated_val);
    }
  }

  // ================================================
  // Step 5: Creates phi operations for each block argument.
  // ================================================
  for (auto &arg_to_phi_pair : arg_to_phi_operands) {
    BlockArgument arg = arg_to_phi_pair.first;
    auto &phi_operands = arg_to_phi_pair.second;

    if (phi_operands.size() <= 1) {
      // No need to create a phi operation if there's only one operand.

      if (phi_operands.size() == 1) {
        arg_to_phi_result[arg] = phi_operands[0];
        arg.replaceAllUsesWith(phi_operands[0]);
      }
      continue;
    }

    // Handles the blockargument with/without reserve separately (different
    // insertion points).
    if (arg_to_reserve.count(arg)) {
      Value reserve_value = arg_to_reserve[arg];
      builder.setInsertionPointAfter(reserve_value.getDefiningOp());

      // Creates phi operation for reserved values.
      auto phi = builder.create<neura::PhiOp>(arg.getLoc(), arg.getType(),
                                              phi_operands);
      arg_to_phi_result[arg] = phi;
    } else {
      Block *placement = arg.getParentBlock();

      builder.setInsertionPointToStart(placement);

      // Creates phi operation for block argument without reserve.
      auto phi = builder.create<neura::PhiOp>(arg.getLoc(), arg.getType(),
                                              phi_operands);
      arg.replaceAllUsesWith(phi);
      arg_to_phi_result[arg] = phi;
    }
  }
}

// Transforms control flow into data flow.
void transformControlFlowToDataFlow(Region &region, ControlFlowInfo &ctrl_info,
                                    DominanceInfo &dom_info, OpBuilder &builder,
                                    bool has_task_counter = false) {

  // Asserts that all live-out values are dominated by block arguments.
  assertLiveOutValuesDominatedByBlockArgs(region);

  // Creates reserve and phi operations for each block argument.
  llvm::MapVector<BlockArgument, Value> arg_to_reserve;
  llvm::MapVector<BlockArgument, Value> arg_to_phi_result;
  createReserveAndPhiOps(region, ctrl_info, arg_to_reserve, arg_to_phi_result,
                         builder);

  // Replaces blockarguments with phi results.
  for (auto &arg_to_phi_pair : arg_to_phi_result) {
    BlockArgument arg = arg_to_phi_pair.first;
    Value phi_result = arg_to_phi_pair.second;
    arg.replaceAllUsesWith(phi_result);
  }

  // Flattens blocks into the entry block.
  // Sorts blocks by reverse post-order traversal to maintain SSA dominance.
  Block *entry_block = &region.front();
  SmallVector<Block *> blocks_to_flatten;

  // Uses reverse post-order: visit successors before predecessors.
  // This ensures that when we move blocks, definitions come before uses.
  llvm::SetVector<Block *> visited;
  // Post-order traversal result, used for sorting blocks.
  SmallVector<Block *> po_order;

  std::function<void(Block *)> po_traverse = [&](Block *block) {
    // Records visited block and skips if already visited.
    if (!visited.insert(block)) {
      return;
    }

    // Visits successors first (post-order).
    Operation *terminator = block->getTerminator();
    if (auto br = dyn_cast<neura::Br>(terminator)) {
      po_traverse(br.getDest());
    } else if (auto cond_br = dyn_cast<neura::CondBr>(terminator)) {
      po_traverse(cond_br.getTrueDest());
      po_traverse(cond_br.getFalseDest());
    }

    // Adds to post-order.
    po_order.push_back(block);
  };

  po_traverse(entry_block);

  // Reverses post-order for forward traversal.
  SmallVector<Block *> rpo_order(po_order.rbegin(), po_order.rend());

  // Collects non-entry blocks in RPO order.
  for (Block *block : rpo_order) {
    if (block != entry_block) {
      blocks_to_flatten.push_back(block);
    }
  }

  // Erases terminators before moving ops into entry block.
  for (Block *block : blocks_to_flatten) {
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (isa<neura::Br>(op) || isa<neura::CondBr>(op)) {
        op.erase();
      }
    }
  }

  // Moves all operations from blocks to the entry block before the
  // terminator.
  for (Block *block : blocks_to_flatten) {
    auto &ops = block->getOperations();
    while (!ops.empty()) {
      Operation &op = ops.front();
      op.moveBefore(&entry_block->back());
    }
  }

  // Erases any remaining br/cond_br that were moved into the entry block.
  for (Operation &op : llvm::make_early_inc_range(*entry_block)) {
    if (isa<neura::Br>(op) || isa<neura::CondBr>(op)) {
      op.erase();
    }
  }

  // Erases now-empty blocks.
  for (Block *block : blocks_to_flatten) {
    block->erase();
  }

  if (has_task_counter) {
    return;
  }

  // Converts neura.return to return_void or return_value.
  SmallVector<neura::ReturnOp> return_ops;
  for (Operation &op : llvm::make_early_inc_range(*entry_block)) {
    if (auto return_op = dyn_cast<neura::ReturnOp>(op)) {
      return_ops.push_back(return_op);
    }
  }

  llvm::errs() << "[ctrl2data] Converting neura.return operations to "
                  "return_void/value...\n";

  for (neura::ReturnOp return_op : return_ops) {
    builder.setInsertionPoint(return_op);

    if (return_op->hasAttr("return_type") &&
        dyn_cast<StringAttr>(return_op->getAttr("return_type")).getValue() ==
            "void") {
      llvm::errs() << "[ctrl2data] Converting to neura.return_void.\n";
      Value trigger = return_op->getOperand(0);
      builder.create<neura::ReturnVoidOp>(return_op.getLoc(), trigger);
    } else if (return_op->hasAttr("return_type") &&
               dyn_cast<StringAttr>(return_op->getAttr("return_type"))
                       .getValue() == "value") {
      builder.create<neura::ReturnValueOp>(return_op.getLoc(),
                                           return_op.getValues());
    } else {
      assert(false && "Unknown return type attribute in neura.return.");
    }
    return_op.erase();
  }

  llvm::errs()
      << "[ctrl2data] All neura.return operations converted successfully.\n";
  // Adds neura.yield at the end of the entry block as terminator.
  builder.setInsertionPointToEnd(entry_block);
  builder.create<neura::YieldOp>(builder.getUnknownLoc());

  // Sets the "dataflow_mode" attribute to "predicate" for the parent
  // function/kernel.
  Operation *parent_op = region.getParentOp();
  llvm::errs() << "[ctrl2data] Parent operation: " << *parent_op << "\n";
  if (auto func = dyn_cast<func::FuncOp>(parent_op)) {
    func->setAttr(
        neura::attr::kDataflowMode,
        StringAttr::get(func.getContext(), neura::attr::val::kModePredicate));
    llvm::errs() << "[ctrl2data] Set dataflow mode to predicate for function: "
                 << func.getName() << "\n";
  } else if (auto kernel = dyn_cast<neura::KernelOp>(parent_op)) {
    // Parent is a kernel.
    kernel->setAttr(
        neura::attr::kDataflowMode,
        StringAttr::get(kernel.getContext(), neura::attr::val::kModePredicate));
    llvm::errs() << "[ctrl2data] Set dataflow mode to predicate for kernel.\n";
  } else {
    assert(false && "[ctrl2data] Warning: Parent operation is neither a "
                    "func::FuncOp nor a neura::KernelOp.\n");
  }
}

// Converts phi operations with reserve operands to phi_start operations.
// Converts only canonical loop-init phi operations to phi_start.
// Required pattern:
//   one operand is produced by neura.grant_once
//   the other operand is produced by neura.reserve
// Operand order is not fixed. Phi ops that do not match this pattern are
// left unchanged (they carry purely predicated values and must stay as phi).
void convertPhiToPhiStart(Region &region, OpBuilder &builder) {
  llvm::errs() << "[ctrl2data] Converting phi operations to phi_start...\n";

  Block *entry_block = &region.front();
  SmallVector<neura::PhiOp> phi_ops_to_convert;

  // Step 1: Collects only phi ops matching (grant_once, reserve), order-free.
  entry_block->walk([&](neura::PhiOp phi_op) {
    ValueRange inputs = phi_op.getInputs();
    if (inputs.size() != 2) {
      return;
    }

    Operation *op0_def = inputs[0].getDefiningOp();
    Operation *op1_def = inputs[1].getDefiningOp();

    if (!op0_def || !op1_def) {
      return;
    }

    bool is_grant_reserve =
        isa<neura::GrantOnceOp>(op0_def) && isa<neura::ReserveOp>(op1_def);
    bool is_reserve_grant =
        isa<neura::ReserveOp>(op0_def) && isa<neura::GrantOnceOp>(op1_def);

    if (is_grant_reserve || is_reserve_grant) {
      phi_ops_to_convert.push_back(phi_op);
    }
  });

  // Step 2: Converts each collected phi operation to phi_start.
  for (neura::PhiOp phi_op : phi_ops_to_convert) {
    Value op0 = phi_op.getInputs()[0];
    Value op1 = phi_op.getInputs()[1];
    Operation *op0_def = op0.getDefiningOp();
    Operation *op1_def = op1.getDefiningOp();

    Value init_operand;
    Value reserve_operand;
    if (op0_def && op1_def && isa<neura::GrantOnceOp>(op0_def) &&
        isa<neura::ReserveOp>(op1_def)) {
      init_operand = op0;
      reserve_operand = op1;
    } else if (op0_def && op1_def && isa<neura::ReserveOp>(op0_def) &&
               isa<neura::GrantOnceOp>(op1_def)) {
      init_operand = op1;
      reserve_operand = op0;
    } else {
      continue;
    }

    // Creates phi_start operation.
    builder.setInsertionPoint(phi_op);
    neura::PhiStartOp phi_start_op = builder.create<neura::PhiStartOp>(
        phi_op.getLoc(), phi_op.getResult().getType(), init_operand,
        reserve_operand);

    // Replaces uses and erases the original phi operation.
    phi_op.getResult().replaceAllUsesWith(phi_start_op.getResult());
    phi_op.erase();
  }
}

namespace {
struct TransformCtrlToDataFlowPass
    : public PassWrapper<TransformCtrlToDataFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformCtrlToDataFlowPass)

  StringRef getArgument() const override {
    return "transform-ctrl-to-data-flow";
  }
  StringRef getDescription() const override {
    return "Transforms control flow into data flow using predicated "
           "execution";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Step 1: Processes each function with neura target in the module.
    module.walk([&](func::FuncOp func_op) {
      auto accel_attr =
          func_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        return;
      }

      Region &region = func_op.getBody();
      DominanceInfo dom_info(func_op);
      OpBuilder builder(func_op.getContext());
      GrantPredicateInEntryBlock(&region.front(), builder);
      assertLiveOutValuesDominatedByBlockArgs(region);

      ControlFlowInfo ctrl_info;
      buildControlFlowInfo(region, ctrl_info, dom_info);
      transformControlFlowToDataFlow(region, ctrl_info, dom_info, builder);

      // Converts phi operations to phi_start operations.
      convertPhiToPhiStart(region, builder);
    });

    // Step 2: Processes neura.kernel operation.
    // For neura.kernel operations, we need to handle three cases:
    // Case 1: outer task has counter, no return value
    //     - Skips grant predicate in entry block
    //     - Outer counter (root counter) gates the return
    // Case 2: outer task has counter, with return value
    //     - Skips grant predicate in entry block
    //     - Outer counter (root counter) gates the return
    //     - Inserts extract_predicate from outer counter (root counter) to
    //     gate return values
    // Case 3: outer task has no counter, with/without return value
    //     - Normal grant predicate in entry block
    //     - Normal transfrom-ctrl-to-data-flow process
    module.walk([&](neura::KernelOp kernel_op) {
      auto accel_attr =
          kernel_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        return;
      }

      llvm::errs()
          << "\n[ctrl2data] ========================================\n";
      llvm::errs() << "[ctrl2data] Processing KERNEL\n";
      llvm::errs() << "[ctrl2data] ========================================\n";

      Region &kernel_region = kernel_op.getBody();
      Block *entry_block = &kernel_region.front();
      OpBuilder builder(kernel_op.getContext());
      DominanceInfo dom_info(kernel_op);

      // STEP 0: Checks if the kernel has root counter.
      bool has_task_counter = taskHasCounter(kernel_op);

      llvm::errs() << "[ctrl2data] Task has counter: "
                   << (has_task_counter ? "YES" : "NO") << "\n\n";
      if (!has_task_counter) {
        llvm::errs() << "[ctrl2data] === Kernel WITHOUT counter ===\n";
        llvm::errs() << "[ctrl2data] Using standard function lowering flow\n\n";

        // Step 1: Grant predicates in entry block
        llvm::errs() << "[ctrl2data] STEP 1: Grant predicates\n";
        GrantPredicateInEntryBlock(entry_block, builder, false);

        // Step 2: Assert live-out values
        llvm::errs() << "[ctrl2data] STEP 2: Assert live-out values\n";
        assertLiveOutValuesDominatedByBlockArgs(kernel_region);

        // Step 3: Build control flow info
        llvm::errs() << "[ctrl2data] STEP 3: Build control flow info\n";
        ControlFlowInfo ctrl_info;
        buildControlFlowInfo(kernel_region, ctrl_info, dom_info);

        // Step 4: Transform control flow to data flow
        llvm::errs() << "[ctrl2data] STEP 4: Transform control flow\n";
        transformControlFlowToDataFlow(kernel_region, ctrl_info, dom_info,
                                       builder,
                                       false); // ✅ false = use function logic

        // Step 5: Convert phi to phi_start
        llvm::errs() << "[ctrl2data] STEP 5: Convert phi to phi_start\n";
        convertPhiToPhiStart(kernel_region, builder);

        llvm::errs() << "[ctrl2data] ✅ Kernel WITHOUT counter complete\n";
        llvm::errs()
            << "[ctrl2data] ========================================\n\n";

        // Set dataflow mode attribute
        kernel_op->setAttr(neura::attr::kDataflowMode,
                           StringAttr::get(kernel_op.getContext(),
                                           neura::attr::val::kModePredicate));
        return;
      }

      SmallVector<Value> iter_arg_phi_values;

      // STEP 1: Handles iter_args of the neura.kernel.
      llvm::errs() << "[ctrl2data] === STEP 1: Handle iter_args ===\n";
      handleKernelIterArgs(kernel_op, entry_block, builder,
                           iter_arg_phi_values);

      // STEP 2: Grants predicates (only if NO task counter).
      llvm::errs() << "[ctrl2data] === STEP 2: Grant predicates ===\n";
      GrantPredicateInEntryBlock(entry_block, builder, has_task_counter);

      // STEP 3: Transforms control flow (if multi-block).
      if (kernel_region.getBlocks().size() > 1) {
        llvm::errs() << "[ctrl2data] === STEP 3: Transform control flow ===\n";
        assertLiveOutValuesDominatedByBlockArgs(kernel_region);
        ControlFlowInfo ctrl_info;
        buildControlFlowInfo(kernel_region, ctrl_info, dom_info);
        transformControlFlowToDataFlow(kernel_region, ctrl_info, dom_info,
                                       builder, true);
      } else {
        llvm::errs() << "[ctrl2data] === STEP 3: Single block (skip) ===\n";
      }
      // STEP 4: Handles yield termination in neura.kernel.
      llvm::errs() << "[ctrl2data] === STEP 4: Handle yield ===\n";
      handleKernelYieldTermination(kernel_op, entry_block, builder,
                                   has_task_counter, iter_arg_phi_values);

      convertPhiToPhiStart(kernel_region, builder);

      kernel_op->setAttr(neura::attr::kDataflowMode,
                         StringAttr::get(kernel_op.getContext(),
                                         neura::attr::val::kModePredicate));

      llvm::errs() << "[ctrl2data] ========================================\n";
      llvm::errs() << "[ctrl2data] ✅ KERNEL Complete\n";
      llvm::errs()
          << "[ctrl2data] ========================================\n\n";
      llvm::errs() << "transformed kernel op:\n" << kernel_op << "\n";
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass() {
  return std::make_unique<TransformCtrlToDataFlowPass>();
}
} // namespace mlir::neura