#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <string>

using namespace mlir;

#define GEN_PASS_DEF_CANONICALIZELIVEIN
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct DirectDominatingLiveIn {
  // The live-in value.
  Value value;
  // The block where the live-in value is defined.
  Block *defining_block;
  // The block where the live-in value is used.
  Block *using_block;
};

bool canReachBlock(Block *from, Block *to) {
  if (from == to) {
    return true;
  }

  SmallVector<Block *> worklist;
  llvm::DenseSet<Block *> visited;
  worklist.push_back(from);

  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    if (!visited.insert(block).second) {
      continue;
    }

    for (Block *succ : block->getSuccessors()) {
      if (succ == to) {
        return true;
      }
      worklist.push_back(succ);
    }
  }

  return false;
}

bool hasBackEdgeBeforeTarget(Block *from, Block *to, DominanceInfo &dom_info) {
  Region *region = from->getParent();
  for (Block &block : region->getBlocks()) {
    // A back edge that leaves the use block is after this live-in is consumed.
    // Rejecting it is too conservative for loop latch / merge blocks.
    if (&block == to) {
      continue;
    }

    if (!canReachBlock(from, &block) || !canReachBlock(&block, to)) {
      continue;
    }

    for (Block *succ : block.getSuccessors()) {
      if (succ == &block || dom_info.dominates(succ, &block)) {
        return true;
      }
    }
  }

  return false;
}

// Checks if two blocks form a single-source-single-sink pattern with
// conditional control flow between them.
//
// Pattern Structure:
//        [ Source Block A ]
//           /           \
//          /             \
//    [ Block B ]     [ Block C ]
//          \             /
//           \           /
//        [ Sink Block D ]
//
// Key Properties:
// 1. Source block A dominates sink block D
//    - All paths to D must go through A
// 2. Sink block D post-dominates source block A
//    - All paths from A eventually reach D
// 3. There exists at least one conditional branch (cond_br) between A and D
//    - Control flow diverges and then converges
// 4. No back edges before the sink (loop-free before the use)
//    - Neither branch target of any cond_br dominates the cond_br block itself
//    - A nested loop/backedge on either branch before reaching D is invalid
//    - A back edge that leaves D itself is allowed because the value has already
//      been consumed at D
//
// Examples of Valid Patterns:
//
// 1. Simple if-else:
//        [ A: cond_br ]
//         /          \
//    [ B: then ]  [ C: else ]
//         \          /
//          [ D: merge ]
//
// 2. Asymmetric branches:
//        [ A: cond_br ]
//         /          \
//    [ B ]            |
//         \          /
//          [ D: merge ]
//
// Counter-examples (Not Valid):
//
// 1. Direct loop structure (has back edge):
//        [ A: cond_br ]  <---+
//         /          \       |
//    [ B: exit ]  [ C ]      |
//                     \------+
//
// 2. Nested loop hidden in an asymmetric branch before the sink:
//        [ A: cond_br ]
//         /          \
//        |       [ D: sink ]
//        v
//    [ B: loop header ] <---+
//        |                  |
//        v                  |
//    [ C: loop latch ] -----+
//        |
//        v
//    [ D: sink ]
//
// 3. Entry block as source:
//        [ Entry Block ]  <- Excluded to maintain compatibility
//              |              with TransformCtrlToDataFlowPass
//           [ cond_br ]
//
// This pattern is used to identify direct dominating live-ins that cross
// conditional branches, enabling specialized optimization for values that
// flow through divergent-convergent control flow regions.
bool isSingleSourceSingleSinkPattern(Block *defining_block, Block *using_block,
                                     DominanceInfo &dom_info,
                                     PostDominanceInfo &post_dom_info) {
  // 1. If defining_block and using_block are the same, then there are no
  // conditional branches on the path.
  if (defining_block == using_block) {
    return false;
  }

  // 2. defining_block must dominate using_block.
  // This ensures that all paths to using_block go through defining_block.
  if (!dom_info.dominates(defining_block, using_block)) {
    return false;
  }

  // 3. using_block must post-dominate defining_block.
  // This ensures that all paths from defining_block eventually reach
  // using_block.
  if (!post_dom_info.postDominates(using_block, defining_block)) {
    return false;
  }

  // 4. If defining_block is the entry block of the region, it is not considered
  // as crossing a conditional branch.
  // Avoids violating assertions in TransformCtrlToDataFlowPass.cpp.
  if (defining_block == &defining_block->getParent()->front()) {
    return false;
  }

  // 5. Checks if using_block is a direct successor (no intermediate blocks) of
  // defining_block.
  for (Block *succ : defining_block->getSuccessors()) {
    if (succ == using_block) {
      Operation *term_op = defining_block->getTerminator();
      // If the terminator is an unconditional branch, then no conditional
      // branch exists on the path.
      if (isa<neura::Br>(term_op)) {
        return false;
      }
      // If it is a conditional branch, but both targets are using_block, it is
      // also considered no real branch.
      if (auto cond_br = dyn_cast<neura::CondBr>(term_op)) {
        if (cond_br.getTrueDest() == using_block &&
            cond_br.getFalseDest() == using_block) {
          return false;
        }
      }
    }
  }

  // 6. Finds any conditional branch on the paths from defining_block to
  // using_block. This is to find any conditional branch divergence between the
  // defining_block and using_block.
  // Because we also support the case where defining_block itself does not
  // contain cond_br (e.g., E in this example).
  //          [ E: br ]
  //              |
  //        [ A: cond_br ]
  //         /          \
  //    [ B: then ]  [ C: else ]
  //         \          /
  //          [ D: merge ]
  bool found_conditional_branch = false;
  Block *conditional_branch_block = nullptr;

  Region *region = defining_block->getParent();
  for (Block &block : region->getBlocks()) {
    if (&block == defining_block || &block == using_block) {
      continue;
    }

    // Checks if this block is on the path from defining_block to using_block.
    if (dom_info.dominates(defining_block, &block) &&
        dom_info.dominates(&block, using_block)) {

      // Checks if this block's terminator is a conditional branch.
      Operation *term_op = block.getTerminator();
      if (auto cond_br = dyn_cast<neura::CondBr>(term_op)) {
        Block *true_dest = cond_br.getTrueDest();
        Block *false_dest = cond_br.getFalseDest();

        // Ensures both branch targets are different (true conditional branch).
        if (true_dest != false_dest) {
          found_conditional_branch = true;
          conditional_branch_block = &block;
          break;
        }
      }
    }
  }

  // 7. Checks the terminator of defining_block itself.
  Operation *defining_term = defining_block->getTerminator();
  if (auto cond_br = dyn_cast<neura::CondBr>(defining_term)) {
    Block *true_dest = cond_br.getTrueDest();
    Block *false_dest = cond_br.getFalseDest();
    if (true_dest != false_dest) {
      found_conditional_branch = true;
      conditional_branch_block = defining_block;
    }
  }

  if (!found_conditional_branch) {
    return false;
  }

  if (hasBackEdgeBeforeTarget(defining_block, using_block, dom_info)) {
    return false;
  }

  // 8. Key Constraint: Verifies that BOTH branches eventually reach using_block
  // WITHOUT creating a loop back to conditional_branch_block or earlier.
  assert(conditional_branch_block &&
         "Must have found a conditional branch block");

  Operation *cond_term = conditional_branch_block->getTerminator();
  auto cond_br = dyn_cast<neura::CondBr>(cond_term);
  assert(cond_br && "Must be a conditional branch");

  Block *true_dest = cond_br.getTrueDest();
  Block *false_dest = cond_br.getFalseDest();

  // Checks loop back edge: If either branch goes back to the conditional branch
  // block or any of its dominators, it creates a loop.
  if (true_dest == conditional_branch_block ||
      dom_info.dominates(true_dest, conditional_branch_block)) {
    llvm::errs()
        << "[CanoLiveIn] True branch creates a back edge (loop pattern)\n";
    return false;
  }

  if (false_dest == conditional_branch_block ||
      dom_info.dominates(false_dest, conditional_branch_block)) {
    llvm::errs()
        << "[CanoLiveIn] False branch creates a back edge (loop pattern)\n";
    return false;
  }

  // Checks if both branches can reach using_block.
  bool true_reaches = (true_dest == using_block);
  if (!true_reaches) {
    if (dom_info.dominates(true_dest, using_block)) {
      true_reaches = true;
    } else {
      for (Block *pred : using_block->getPredecessors()) {
        if (pred == true_dest || dom_info.dominates(true_dest, pred)) {
          true_reaches = true;
          break;
        }
      }
    }
  }

  bool false_reaches = (false_dest == using_block);
  if (!false_reaches) {
    if (dom_info.dominates(false_dest, using_block)) {
      false_reaches = true;
    } else {
      for (Block *pred : using_block->getPredecessors()) {
        if (pred == false_dest || dom_info.dominates(false_dest, pred)) {
          false_reaches = true;
          break;
        }
      }
    }
  }

  if (!true_reaches || !false_reaches) {
    return false;
  }

  return true;
}

// Checks if there's a direct unconditional path from defining_block to
// using_block without crossing any conditional branches.
//
// Pattern Structure:
//    [ Defining Block A ]
//             |  (br)
//             v
//       [ Block B ]
//             |  (br)
//             v
//       [ Block C ]
//             |  (br)
//             v
//    [ Using Block D ]
//
// Key Properties:
// 1. Defining block dominates using block
//    - All paths to using_block go through defining_block
// 2. Using block post-dominates defining block
//    - All paths from defining_block eventually reach using_block
//    - This ensures there's a unique path
// 3. All intermediate blocks only have unconditional branches (br)
//    - No conditional branches (cond_br) on the path
// 4. No loops (no back edges)
//
// Examples of Valid Patterns:
//
// 1. Direct successor:
//    [ A: br ]
//       |
//    [ B ]
//
// 2. Chain of unconditional branches:
//    [ A: br ]
//       |
//    [ B: br ]
//       |
//    [ C: br ]
//       |
//    [ D ]
//
// Counter-examples (Not Valid):
//
// 1. Has conditional branch:
//    [ A: br ]
//       |
//    [ B: cond_br ]  <- Has cond_br
//      /    \
//    ...    ...
//
// 2. Entry block as defining:
//    [ Entry: br ]  <- Excluded
//       |
//    [ B ]
//
// 3. Loop structure:
//    [ A: br ]  <--+
//       |          |
//    [ B: br ]-----+
//
// This pattern identifies the simplest form of direct dominating live-ins where
// values flow through a linear sequence of blocks without any control flow
// divergence.
bool isDirectUnconditionalPattern(Block *defining_block, Block *using_block,
                                  DominanceInfo &dom_info,
                                  PostDominanceInfo &post_dom_info) {
  // 1. If blocks are the same, not a valid pattern.
  if (defining_block == using_block) {
    return false;
  }

  // 2. Defing block must dominate using block.
  if (!dom_info.dominates(defining_block, using_block)) {
    return false;
  }

  // 3. Using block must post-dominate defining block.
  if (!post_dom_info.postDominates(using_block, defining_block)) {
    return false;
  }

  if (hasBackEdgeBeforeTarget(defining_block, using_block, dom_info)) {
    return false;
  }

  // 4. Entry block cannot be the defining block.
  if (defining_block == &defining_block->getParent()->front()) {
    return false;
  }

  // 5. Checks all blocks on the path from defining_block to using_block.
  // They must all have unconditional branches (br) only.
  Region *region = defining_block->getParent();
  for (Block &block : region->getBlocks()) {
    // Skips blocks not on the path.
    if (!dom_info.dominates(defining_block, &block) ||
        !dom_info.dominates(&block, using_block)) {
      continue;
    }

    // For blocks on the path, checks if their terminators are unconditional
    // branches only (excluding using_block itself).
    if (&block != using_block) {
      Operation *term_op = block.getTerminator();

      // If the terminator is a conditional branch, this pattern is not
      // satisfied.
      if (isa<neura::CondBr>(term_op)) {
        return false;
      }

      // The terminator must be an unconditional branch (br).
      assert(isa<neura::Br>(term_op) &&
             "The terminator must be an unconditional branch.\n");

      // Ensures no backward edges (loops) exist.
      neura::Br br_op = cast<neura::Br>(term_op);
      Block *dest = br_op.getDest();
      // If the destination block dominates current block, it creates a loop.
      if (dom_info.dominates(dest, &block)) {
        return false;
      }
    }
  }
  return true;
}

DenseMap<Block *, SmallVector<DirectDominatingLiveIn>>
identifyDirectDominatingLiveIns(Region &region, DominanceInfo &dom_info,
                                PostDominanceInfo &post_dom_info) {
  DenseMap<Block *, SmallVector<DirectDominatingLiveIn>>
      using_block_to_dominating_direct_live_ins;
  for (Block &block : region.getBlocks()) {
    // Skips the entry block.
    if (&block == &region.front()) {
      continue;
    }

    // Collects direct live-in values for the block.
    SetVector<Value> live_ins;
    for (Operation &op : block.getOperations()) {
      for (Value operand : op.getOperands()) {
        // If the operand is defined in another block, it is a live-in
        // value.
        if (auto block_arg = dyn_cast<BlockArgument>(operand)) {
          if (block_arg.getOwner() != &block) {
            live_ins.insert(operand);
          }
        } else {
          Operation *def_op = operand.getDefiningOp();
          if (def_op && def_op->getBlock() != &block) {
            live_ins.insert(operand);
          }
        }
      }
    }

    // Checks each live-in value to see if it has direct dominating
    // dependencies.
    // Direct dominating dependency means:
    // 1. The defining block of the live-in value dominates the using block.
    // 2. The using block post-dominates the defining block.
    // 3. We can ensure the live-in in the using block is valid once the
    // defining block is executed.
    //
    // We support two mutually exclusive patterns:
    // - Pattern 1: Single-Source-Single-Sink with only one conditional branch
    // (cond_br).
    // - Pattern 2: Linear path with only unconditional branches (br).
    for (Value live_in : live_ins) {
      Block *defining_block = nullptr;

      if (auto block_arg = dyn_cast<BlockArgument>(live_in)) {
        defining_block = block_arg.getOwner();
      } else {
        Operation *def_op = live_in.getDefiningOp();
        if (def_op) {
          defining_block = def_op->getBlock();
        }
      }

      if (!defining_block) {
        continue;
      }

      // If the using block is a loop header (has a back-edge), we must NOT
      // treat any live-in as a direct dominating live-in. This is because:
      // 1. Live-ins from outer scopes have rate mismatch and need PHI_START
      // 2. Live-ins from inner blocks are loop-carried dependencies that need
      // PHI
      bool using_block_is_loop_header = false;
      for (Block *pred : block.getPredecessors()) {
        if (dom_info.dominates(&block, pred)) {
          using_block_is_loop_header = true;
          break;
        }
      }

      if (using_block_is_loop_header) {
        // Skips direct dominating live-in optimization for loop headers.
        // The value must be promoted to a block argument so that the
        // transform-ctrl-to-data-flow pass creates an inner-rate PHI_START.
        continue;
      }

      // Pattern 1: Single-Source-Single-Sink with one conditional branch.
      if (isSingleSourceSingleSinkPattern(defining_block, &block, dom_info,
                                          post_dom_info)) {
        assert(!isDirectUnconditionalPattern(defining_block, &block, dom_info,
                                             post_dom_info) &&
               "Patterns 1 and 2 are mutually exclusive.");
        DirectDominatingLiveIn direct_dominating_live_in;
        direct_dominating_live_in.value = live_in;
        direct_dominating_live_in.defining_block = defining_block;
        direct_dominating_live_in.using_block = &block;

        using_block_to_dominating_direct_live_ins[&block].push_back(
            direct_dominating_live_in);

        // Pattern 1 matched, skip Pattern 2 check (they are mutually
        // exclusive).
        continue;
      }

      // Pattern 2: Direct Unconditional Path.
      if (isDirectUnconditionalPattern(defining_block, &block, dom_info,
                                       post_dom_info)) {
        assert(!isSingleSourceSingleSinkPattern(defining_block, &block,
                                                dom_info, post_dom_info) &&
               "Patterns 1 and 2 are mutually exclusive.");
        DirectDominatingLiveIn direct_dominating_live_in;
        direct_dominating_live_in.value = live_in;
        direct_dominating_live_in.defining_block = defining_block;
        direct_dominating_live_in.using_block = &block;

        using_block_to_dominating_direct_live_ins[&block].push_back(
            direct_dominating_live_in);

        // Pattern 2 matched.
        continue;
      }

      // TODO: Add more direct dominating live-in patterns based on
      // dominance and post-dominance analysis. Issue:
      // https://github.com/coredac/dataflow/issues/159
    }
  }
  return using_block_to_dominating_direct_live_ins;
}

LogicalResult promoteLiveInValuesToBlockArgs(Region &region,
                                             DominanceInfo &dom_info,
                                             PostDominanceInfo &post_dom_info) {
  if (region.empty()) {
    return success();
  }

  DenseMap<Block *, SmallVector<DirectDominatingLiveIn>>
      direct_dominating_live_ins =
          identifyDirectDominatingLiveIns(region, dom_info, post_dom_info);

  // Maps each block to its dominating direct live-in values.
  DenseMap<Block *, SetVector<Value>> direct_dominating_live_in_values;
  for (auto &[block, dataflow_live_ins] : direct_dominating_live_ins) {
    for (auto &dataflow_live_in : dataflow_live_ins) {
      direct_dominating_live_in_values[block].insert(dataflow_live_in.value);
    }
  }

  // Collects direct live-in values for each block in the region.
  // Without considering the transitive dependencies.
  DenseMap<Block *, SetVector<Value>> direct_live_ins;

  // Initializes the direct live-ins for each block.
  for (Block &block : region.getBlocks()) {
    if (&block == &region.front()) {
      continue;
    }

    SetVector<Value> live_ins;
    for (Operation &op : block.getOperations()) {
      for (Value operand : op.getOperands()) {
        // If the operand is a direct dominating live-in value, skip it.
        if (direct_dominating_live_in_values[&block].contains(operand)) {
          continue;
        }

        // If the operand is defined in another block, it is a live-in
        // value.
        if (auto block_arg = dyn_cast<BlockArgument>(operand)) {
          if (block_arg.getOwner() != &block) {
            live_ins.insert(operand);
          }
        } else {
          Operation *def_op = operand.getDefiningOp();
          if (def_op && def_op->getBlock() != &block) {
            live_ins.insert(operand);
          }
        }
      }
    }

    if (!live_ins.empty()) {
      direct_live_ins[&block] = live_ins;
    }
  }

  // If we update a branch or conditional branch, we may introduce new
  // live-ins for a block. So we need to propagate live-in values until a
  // fixed point is reached.

  // *************************************************************************
  // For example, consider this control flow:
  //
  // Block A:
  //   %0 = constant 1
  //   %1 = constant 2
  //   br B
  //
  // Block B:
  //   br C
  //
  // Block C:
  //   %2 = add %0, %1  // %0 and %1 are live-ins for block C
  //   return
  //
  // Initial direct_live_ins analysis:
  //   - Block C: {%0, %1} (directly used values defined outside C)
  //   - Block B: {} (no direct use of external values)
  //
  // After propagation:
  //   - Block C: {%0, %1}
  //   - Block B: {%0, %1} (needs to pass these values to C)
  //
  // The transformation adds block arguments:
  //   Block A:
  //     %0 = constant 1
  //     %1 = constant 2
  //     br B(%0, %1)
  //
  //   Block B(%b0, %b1):
  //     br C(%b0, %b1)
  //
  //   Block C(%c0, %c1):
  //     %2 = add %c0, %c1
  //     return
  // *************************************************************************
  DenseMap<Block *, SetVector<Value>> all_live_ins = direct_live_ins;
  bool changed = true;

  while (changed) {
    changed = false;

    for (Block &current_block : region.getBlocks()) {
      if (&current_block == &region.front()) {
        continue;
      }

      // Checks if current block has successor blocks and if they have
      // any live-ins.
      for (Block *succ_block : current_block.getSuccessors()) {
        auto succ_live_in_iter = all_live_ins.find(succ_block);
        if (succ_live_in_iter == all_live_ins.end()) {
          continue;
        }

        SetVector<Value> &succ_live_ins = succ_live_in_iter->second;
        SetVector<Value> &current_live_ins = all_live_ins[&current_block];

        // Checks if the live-in value in successor block is defined in the
        // current block.
        for (Value live_in : succ_live_ins) {
          // If it is a direct dominating live-in value for the successor
          // block, we skip it.
          if (direct_dominating_live_in_values[succ_block].contains(live_in)) {
            continue;
          }

          if (direct_dominating_live_in_values[&current_block].contains(
                  live_in)) {
            continue;
          }

          // If it is defined in the current block, that means it is not a
          // live-in value for the current block. We can skip it.
          if (Operation *def_op = live_in.getDefiningOp()) {
            if (def_op->getBlock() == &current_block) {
              continue;
            }
          } else if (auto block_arg = dyn_cast<BlockArgument>(live_in)) {
            if (block_arg.getOwner() == &current_block) {
              continue;
            }
          }

          // If current live-ins do not contain the live-in value,
          // we add it to the current live-ins.
          if (!current_live_ins.contains(live_in)) {
            current_live_ins.insert(live_in);
            changed = true;
          }
        }
      }
    }
  }

  // llvm::errs() << "All live-ins after propagation:\n";
  // for (auto &[block, liveIns] : all_live_ins) {
  //   llvm::errs() << "Block: " << *block << "\nLive-ins: ";
  //   for (Value liveIn : liveIns) {
  //     llvm::errs() << "   " << liveIn << "\n";
  //   }
  //   llvm::errs() << "\n";
  // }

  // Adds all live-in values as block arguments and updates the
  // operations that use these live-in values.
  DenseMap<std::pair<Block *, Value>, Value> block_value_to_arg;
  DenseMap<Block *, unsigned> original_num_args;

  for (auto &[block, live_ins] : all_live_ins) {
    original_num_args[block] = block->getNumArguments();

    // Adds all live-in values as block arguments.
    for (Value value : live_ins) {
      block->addArgument(value.getType(), value.getLoc());
    }

    // Constructs a mapping from live-in values to their corresponding
    // block arguments.
    unsigned index = original_num_args[block];
    for (Value value : live_ins) {
      block_value_to_arg[{block, value}] = block->getArgument(index++);
    }
  }

  // Updates all operations in the region to use the new block arguments
  // instead of the live-in values.
  for (auto &[block, live_ins] : all_live_ins) {
    for (Operation &op : block->getOperations()) {
      for (unsigned i = 0; i < op.getNumOperands(); i++) {
        Value operand = op.getOperand(i);
        auto key = std::make_pair(block, operand);

        if (block_value_to_arg.count(key)) {
          op.setOperand(i, block_value_to_arg[key]);
        }
      }
    }
  }

  // Updates the terminators of predecessor blocks to use the new block
  // arguments instead of the live-in values.
  for (auto &[current_block, live_ins] : all_live_ins) {
    for (Block *pred_block : current_block->getPredecessors()) {
      Operation *term_op = pred_block->getTerminator();

      if (auto br_op = dyn_cast<neura::Br>(term_op)) {
        if (br_op.getDest() == current_block) {
          SmallVector<Value> new_operands(br_op.getOperands().begin(),
                                          br_op.getOperands().end());
          for (Value live_in : live_ins) {
            Operation *def_op = live_in.getDefiningOp();
            BlockArgument block_arg = dyn_cast<BlockArgument>(live_in);
            if (def_op && def_op->getBlock() == pred_block) {
              new_operands.push_back(live_in);
            } else if (block_arg && block_arg.getOwner() == pred_block) {
              new_operands.push_back(block_arg);
            } else if (all_live_ins[pred_block].contains(live_in)) {
              new_operands.push_back(block_value_to_arg[{pred_block, live_in}]);
            } else {
              assert(false && "Unexpected live-in value (br operation)");
            }
          }
          OpBuilder builder(br_op);
          builder.create<neura::Br>(br_op.getLoc(), new_operands,
                                    current_block);
          br_op.erase();
        }
      } else if (auto cond_br_op = dyn_cast<neura::CondBr>(term_op)) {
        bool needs_update = false;
        SmallVector<Value> true_operands(cond_br_op.getTrueArgs().begin(),
                                         cond_br_op.getTrueArgs().end());
        SmallVector<Value> false_operands(cond_br_op.getFalseArgs().begin(),
                                          cond_br_op.getFalseArgs().end());
        // Handles the true branch.
        if (cond_br_op.getTrueDest() == current_block) {
          needs_update = true;
          for (Value live_in : live_ins) {
            Operation *def_op = live_in.getDefiningOp();
            BlockArgument block_arg = dyn_cast<BlockArgument>(live_in);
            if (def_op && def_op->getBlock() == pred_block) {
              true_operands.push_back(live_in);
            } else if (block_arg && block_arg.getOwner() == pred_block) {
              true_operands.push_back(block_arg);
            } else if (direct_dominating_live_in_values[pred_block].contains(
                           live_in)) {
              true_operands.push_back(live_in);
            } else if (all_live_ins[pred_block].contains(live_in)) {
              true_operands.push_back(
                  block_value_to_arg[{pred_block, live_in}]);
            } else {
              assert(false && "Unexpected live-in value (true branch of "
                              "cond_br operation)");
            }
          }
        }

        // Handles the false branch.
        if (cond_br_op.getFalseDest() == current_block) {
          needs_update = true;
          for (Value live_in : live_ins) {
            Operation *def_op = live_in.getDefiningOp();
            BlockArgument block_arg = dyn_cast<BlockArgument>(live_in);
            if (def_op && def_op->getBlock() == pred_block) {
              false_operands.push_back(live_in);
            } else if (block_arg && block_arg.getOwner() == pred_block) {
              false_operands.push_back(block_arg);
            } else if (direct_dominating_live_in_values[pred_block].contains(
                           live_in)) {
              false_operands.push_back(live_in);
            } else if (all_live_ins[pred_block].contains(live_in)) {
              false_operands.push_back(
                  block_value_to_arg[{pred_block, live_in}]);
            } else {
              assert(false && "Unexpected live-in value (false branch of "
                              "cond_br operation)");
            }
          }
        }

        // If an update is needed, create a new conditional branch
        // operation.
        if (needs_update) {
          OpBuilder builder(cond_br_op);
          builder.create<neura::CondBr>(
              cond_br_op.getLoc(), cond_br_op.getCondition(), true_operands,
              false_operands, cond_br_op.getTrueDest(),
              cond_br_op.getFalseDest());
          cond_br_op.erase();
        }
      }
    }
  }
  return success();
}

struct CanonicalizeLiveInPass
    : public PassWrapper<CanonicalizeLiveInPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeLiveInPass)

  StringRef getArgument() const override { return "canonicalize-live-in"; }
  StringRef getDescription() const override {
    return "Canonicalizes live-in values/operations in each basic block.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();

    // Processes functions.
    module_op.walk([&](Operation *op) {
      Region *region = nullptr;
      if (auto func_op = dyn_cast<func::FuncOp>(op)) {
        auto accel_attr =
            func_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
        if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
          return;
        }
        region = &func_op.getBody();
      } else if (auto llvm_func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        auto accel_attr =
            llvm_func->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
        if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
          return;
        }
        region = &llvm_func.getBody();
      } else {
        return;
      }

      if (!region || region->empty()) {
        return;
      }

      DominanceInfo dom_info(op);
      PostDominanceInfo post_dom_info(op);

      if (failed(promoteLiveInValuesToBlockArgs(*region, dom_info,
                                                post_dom_info))) {
        signalPassFailure();
        return;
      }
    });

    // Processes neura.kernel operations.
    module_op.walk([&](neura::KernelOp kernel_op) {
      auto accel_attr =
          kernel_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        return;
      }

      Region &kernel_region = kernel_op.getBody();
      if (kernel_region.empty()) {
        return;
      }

      // Creates dominance info for the kernel region.
      DominanceInfo dom_info(kernel_op);
      PostDominanceInfo post_dom_info(kernel_op);

      if (failed(promoteLiveInValuesToBlockArgs(kernel_region, dom_info,
                                                post_dom_info))) {
        signalPassFailure();
        return;
      }
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createCanonicalizeLiveInPass() {
  return std::make_unique<CanonicalizeLiveInPass>();
}
} // namespace mlir::neura
