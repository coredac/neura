#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;

#define GEN_PASS_DEF_TRANSFORMTOSTEERCONTROL
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
class OperationsToErase {
public:
  void markForErasure(Operation *op) {
    if (op) {
      ops_to_erase.insert(op);
    }
  }

  void eraseMarkedOperations() {
    for (auto it = this->ops_to_erase.rbegin(); it != this->ops_to_erase.rend();
         ++it) {
      if (!(*it)->use_empty()) {
        continue;
      }
      (*it)->erase();
    }
    ops_to_erase.clear();
  }

private:
  llvm::SetVector<Operation *> ops_to_erase;
};

class LoopAnalyzer {
public:
  struct LoopRecurrenceInfo {
    // The reserve operation that starts the loop.
    neura::ReserveOp reserve_op;
    // The phi_start operation that merges values from different iterations.
    neura::PhiStartOp phi_start_op;
    // The initial value before the loop starts.
    Value initial_value;
    // The condition that controls the loop continuation.
    Value condition;
    // The value that is passed back to the next iteration.
    Value backward_value;
    // Whether the loop is invariant (i.e., does not depend on the loop body).
    bool is_invariant;
  };

  LoopAnalyzer(func::FuncOp func) {
    // Map from values to their corresponding reserve operations.
    llvm::DenseMap<Value, neura::ReserveOp> value_to_reserve_map;

    func.walk([&](neura::ReserveOp op) {
      value_to_reserve_map[op.getResult()] = op;
    });

    llvm::DenseMap<Value, llvm::SmallVector<std::pair<Value, neura::CtrlMovOp>>>
        target_to_source_ctrl_mov_map;

    func.walk([&](neura::CtrlMovOp op) {
      target_to_source_ctrl_mov_map[op.getTarget()].push_back(
          {op.getValue(), op});
    });

    // Analyzes phi_start operations and the backward edges.
    func.walk([&](neura::PhiStartOp phi_start_op) {
      for (Value input : phi_start_op->getOperands()) {
        auto reserve_it = value_to_reserve_map.find(input);
        if (reserve_it == value_to_reserve_map.end()) {
          continue;
        }

        neura::ReserveOp reserve_op = reserve_it->second;
        auto ctrl_mov_it =
            target_to_source_ctrl_mov_map.find(reserve_op.getResult());
        assert(ctrl_mov_it != target_to_source_ctrl_mov_map.end() &&
               "Reserve output must be a target of a ctrl_mov operation");

        for (auto &[source_value, ctrl_mov_op] : ctrl_mov_it->second) {
          Value initial_value = nullptr;
          for (Value phi_input : phi_start_op->getOperands()) {
            if (phi_input != reserve_op.getResult()) {
              initial_value = phi_input;
              break;
            }
          }
          assert(initial_value != nullptr && "Phi must have an initial value");

          Value condition = nullptr;
          neura::GrantPredicateOp grant_op = nullptr;
          for (auto phi_user : phi_start_op->getUsers()) {
            if (isa<neura::GrantPredicateOp>(phi_user)) {
              auto def_op = llvm::dyn_cast<neura::GrantPredicateOp>(phi_user);
              if (def_op.getValue() == phi_start_op.getResult() &&
                  !isa<neura::NotOp>(def_op.getPredicate().getDefiningOp())) {
                llvm::errs() << "[ctrl2steer] Found loop condition: "
                             << def_op.getPredicate() << "\n";
                condition = def_op.getPredicate();
                grant_op = def_op;
                break;
              }
            }
          }

          assert(condition && grant_op &&
                 "Phi must have a corresponding grant_predicate operation");

          // Checks if the source_value is a loop invariant.
          bool is_invariant = false;
          if (source_value == grant_op.getResult()) {
            is_invariant = true;
          }

          // Records the loop information.
          this->loop_recurrences.push_back({reserve_op, phi_start_op,
                                            initial_value, condition,
                                            source_value, is_invariant});

          // Maps the phi operation to its loop recurrence index.
          this->phi_to_loop_recurrences[phi_start_op.getResult()] =
              loop_recurrences.size() - 1;

          // Records the reserve operations that are part of loops.
          this->loop_reserves.insert(reserve_op.getResult());
        }
      }
    });
  }

  const llvm::SmallVector<LoopRecurrenceInfo> &getLoopRecurrences() const {
    return loop_recurrences;
  }

  bool isLoopReserve(Value value) const {
    return loop_reserves.contains(value);
  }

  bool isLoopPhi(Value value) const {
    return phi_to_loop_recurrences.contains(value);
  }

  const LoopRecurrenceInfo *getLoopRecurrenceInfo(Value phi_value) const {
    auto it = phi_to_loop_recurrences.find(phi_value);
    if (it != phi_to_loop_recurrences.end()) {
      return &loop_recurrences[it->second];
    }
    return nullptr;
  }

private:
  llvm::SmallVector<LoopRecurrenceInfo> loop_recurrences;
  llvm::DenseMap<Value, unsigned> phi_to_loop_recurrences;
  llvm::DenseSet<Value> loop_reserves;
};

class BackwardValueHandler {
public:
  BackwardValueHandler(PatternRewriter &rewriter) : rewriter(rewriter) {}

  Value createReserveForBackwardValue(Value backward_value,
                                      Operation *insertion_point) {
    auto it = backward_value_reserve_map.find(backward_value);
    if (it != backward_value_reserve_map.end()) {
      return it->second;
    }

    llvm::errs() << "[ctrl2steer] Creating reserve for backward value: "
                 << backward_value << "\n";

    // Creates the reserve operation for the backward value.
    this->rewriter.setInsertionPointToStart(insertion_point->getBlock());
    auto reserve_op = this->rewriter.create<neura::ReserveOp>(
        backward_value.getLoc(), backward_value.getType());
    this->backward_value_reserve_map[backward_value] = reserve_op.getResult();

    llvm::errs() << "[ctrl2steer] Creating ctrl_mov for backward value: "
                 << backward_value << "\n";

    // Creates a ctrl_mov operation to move the backward value into the reserve.
    this->rewriter.setInsertionPointAfter(backward_value.getDefiningOp());
    this->rewriter.create<neura::CtrlMovOp>(
        backward_value.getLoc(), backward_value, reserve_op.getResult());

    return reserve_op.getResult();
  }

private:
  PatternRewriter &rewriter;
  // Map from backward values to their corresponding reserve values.
  llvm::DenseMap<Value, Value> backward_value_reserve_map;
};

class PhiStartToCarryPattern : public OpRewritePattern<neura::PhiStartOp> {
public:
  PhiStartToCarryPattern(MLIRContext *context,
                         const LoopAnalyzer &loop_analyzer,
                         BackwardValueHandler &backward_value_handler,
                         OperationsToErase &ops_to_erase)
      : OpRewritePattern<neura::PhiStartOp>(context),
        loop_analyzer(loop_analyzer),
        backward_value_handler(backward_value_handler),
        ops_to_erase(ops_to_erase) {}

  LogicalResult matchAndRewrite(neura::PhiStartOp phi_start_op,
                                PatternRewriter &rewriter) const override {
    // If the phi operation is not part of a loop, we do not handle it here.
    if (!loop_analyzer.isLoopPhi(phi_start_op.getResult())) {
      return failure();
    }

    const auto *loop_recurrence_info =
        loop_analyzer.getLoopRecurrenceInfo(phi_start_op.getResult());
    assert(loop_recurrence_info && "Loop recurrence info must be available");

    // Creates a reserve operation for the loop condition.
    Value condition = loop_recurrence_info->condition;
    assert(condition && "Loop condition must be available");
    Value condition_reserve =
        this->backward_value_handler.createReserveForBackwardValue(
            condition, phi_start_op);

    // Creates a carry or a invariant operation based on whether the loop
    // recurrence is invariant.
    rewriter.setInsertionPoint(phi_start_op);
    Value result;
    if (loop_recurrence_info->is_invariant) {
      auto invariant_op = rewriter.create<neura::InvariantOp>(
          phi_start_op.getLoc(), phi_start_op.getType(),
          loop_recurrence_info->initial_value, condition_reserve);
      result = invariant_op.getResult();
      // rewriter.replaceOp(phi_op, invariant_op.getResult());
    } else {
      Value backward_reserve =
          this->backward_value_handler.createReserveForBackwardValue(
              loop_recurrence_info->backward_value, phi_start_op);
      auto carry_op = rewriter.create<neura::CarryOp>(
          phi_start_op.getLoc(), phi_start_op.getType(),
          loop_recurrence_info->initial_value, condition_reserve,
          backward_reserve);
      result = carry_op.getResult();
      // rewriter.replaceOp(phi_op, carry_op.getResult());
    }

    llvm::SmallVector<neura::GrantPredicateOp> related_grant_ops;
    for (auto *user : phi_start_op->getUsers()) {
      if (auto grant_op = dyn_cast<neura::GrantPredicateOp>(user)) {
        if (grant_op.getValue() == phi_start_op.getResult() &&
            grant_op.getPredicate() == condition) {
          related_grant_ops.push_back(grant_op);
        }
      }
    }

    // Marks the related operations for erasure.
    for (auto grant_op : related_grant_ops) {
      rewriter.replaceAllOpUsesWith(grant_op, result);
      ops_to_erase.markForErasure(grant_op);
    }

    rewriter.replaceOp(phi_start_op, result);

    this->ops_to_erase.markForErasure(loop_recurrence_info->reserve_op);
    for (auto *user : loop_recurrence_info->reserve_op->getUsers()) {
      if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(user)) {
        ops_to_erase.markForErasure(ctrl_mov_op);
      }
    }
    return success();
  }

private:
  const LoopAnalyzer &loop_analyzer;
  BackwardValueHandler &backward_value_handler;
  OperationsToErase &ops_to_erase;
};

class MergePatternFinder {
public:
  struct MergeCandidate {
    Value condition;
    Value true_value;
    Value false_value;
    neura::GrantPredicateOp true_grant;
    neura::GrantPredicateOp false_grant;
    neura::PhiOp phi_op;
    neura::NotOp not_op;
  };

  MergePatternFinder(func::FuncOp func) {
    // Collects all the not operations.
    llvm::DenseMap<Value, neura::NotOp> not_value_to_op;
    llvm::DenseMap<Value, Value> negated_to_condition;

    func.walk([&](neura::NotOp not_op) {
      not_value_to_op[not_op.getResult()] = not_op;
      negated_to_condition[not_op.getResult()] = not_op.getInput();
    });

    // Collects all the grant_predicate operations based on their conditions.
    llvm::DenseMap<Value, llvm::SmallVector<neura::GrantPredicateOp>>
        condition_to_grants;
    func.walk([&](neura::GrantPredicateOp grant_op) {
      condition_to_grants[grant_op.getPredicate()].push_back(grant_op);
    });

    func.walk([&](neura::PhiOp phi_op) {
      // Phi operation must have two operands.
      if (phi_op.getNumOperands() != 2) {
        return;
      }

      Value input0 = phi_op.getOperand(0);
      Value input1 = phi_op.getOperand(1);

      // Each operand must be produced by a grant_predicate operation.
      auto grant0 = input0.getDefiningOp<neura::GrantPredicateOp>();
      auto grant1 = input1.getDefiningOp<neura::GrantPredicateOp>();
      if (!grant0 || !grant1) {
        return;
      }

      // Checks if the conditions of the two grant_predicate operations are
      // complementary.
      Value cond0 = grant0.getPredicate();
      Value cond1 = grant1.getPredicate();

      // Checks if one condition is the negation of the other.
      neura::NotOp not_op = nullptr;
      Value original_cond;
      bool cond0_is_original = false;

      // case 1: cond0 is the original condition, cond1 is its negation.
      if (auto it = not_value_to_op.find(cond1);
          it != not_value_to_op.end() && it->second.getInput() == cond0) {
        not_op = it->second;
        original_cond = cond0;
        cond0_is_original = true;
      }
      // case 2: cond1 is the original condition, cond0 is its negation.
      else if (auto it = not_value_to_op.find(cond0);
               it != not_value_to_op.end() && it->second.getInput() == cond1) {
        not_op = it->second;
        original_cond = cond1;
        cond0_is_original = false;
      }
      // Conditions are not complementary, not the pattern we are looking for.
      else {
        return;
      }

      // Determines true branch and false branch
      neura::GrantPredicateOp true_grant = cond0_is_original ? grant0 : grant1;
      neura::GrantPredicateOp false_grant = cond0_is_original ? grant1 : grant0;
      Value true_value = true_grant.getValue();
      Value false_value = false_grant.getValue();

      // Records the found merge pattern
      merge_candidates.push_back({original_cond, true_value, false_value,
                                  true_grant, false_grant, phi_op, not_op});

      // Records the operations involved in the merge
      phi_in_merge[phi_op] = merge_candidates.size() - 1;
      grants_in_merge.insert(true_grant);
      grants_in_merge.insert(false_grant);
      if (not_op) {
        nots_in_merge.insert(not_op);
      }
    });
  }

  llvm::SmallVector<MergeCandidate> getMergeCandidates() const {
    return this->merge_candidates;
  }

  bool isPhiInMerge(neura::PhiOp phi_op) const {
    return phi_in_merge.contains(phi_op);
  }

  bool isGrantInMerge(neura::GrantPredicateOp grant_op) const {
    return grants_in_merge.contains(grant_op);
  }

  bool isNotInMerge(neura::NotOp not_op) const {
    return nots_in_merge.contains(not_op);
  }

  const MergeCandidate *getMergeCandidateForPhi(neura::PhiOp phi_op) const {
    auto it = phi_in_merge.find(phi_op);
    if (it != phi_in_merge.end())
      return &merge_candidates[it->second];
    return nullptr;
  }

private:
  llvm::SmallVector<MergeCandidate> merge_candidates;
  llvm::DenseMap<neura::PhiOp, unsigned> phi_in_merge;
  llvm::DenseSet<neura::GrantPredicateOp> grants_in_merge;
  llvm::DenseSet<neura::NotOp> nots_in_merge;
};

class PhiToMergePattern : public OpRewritePattern<neura::PhiOp> {
public:
  PhiToMergePattern(MLIRContext *context,
                    const MergePatternFinder &merge_pattern_finder,
                    OperationsToErase &ops_to_erase)
      : OpRewritePattern<neura::PhiOp>(context),
        merge_pattern_finder(merge_pattern_finder), ops_to_erase(ops_to_erase) {
  }

  LogicalResult matchAndRewrite(neura::PhiOp phi_op,
                                PatternRewriter &rewriter) const override {
    // Checks if the phi operation is part of a merge pattern.
    if (!merge_pattern_finder.isPhiInMerge(phi_op)) {
      return failure();
    }

    const auto *merge_candidate =
        merge_pattern_finder.getMergeCandidateForPhi(phi_op);
    if (!merge_candidate) {
      return failure();
    }

    rewriter.setInsertionPoint(phi_op);
    auto merge_op = rewriter.create<neura::MergeOp>(
        phi_op.getLoc(), phi_op.getType(), merge_candidate->condition,
        merge_candidate->true_value, merge_candidate->false_value);

    rewriter.replaceOp(phi_op, merge_op.getResult());

    // Marks the related operations for erasure.
    ops_to_erase.markForErasure(merge_candidate->true_grant);
    ops_to_erase.markForErasure(merge_candidate->false_grant);
    ops_to_erase.markForErasure(merge_candidate->not_op);

    return success();
  }

private:
  const MergePatternFinder &merge_pattern_finder;
  OperationsToErase &ops_to_erase;
};

class SteerPhiToMergePattern : public OpRewritePattern<neura::PhiOp> {
public:
  SteerPhiToMergePattern(MLIRContext *context, OperationsToErase &ops_to_erase,
                         MergePatternFinder &merge_pattern_finder,
                         LoopAnalyzer &loop_analyzer)
      : OpRewritePattern<neura::PhiOp>(context),
        merge_pattern_finder(merge_pattern_finder),
        loop_analyzer(loop_analyzer) {}

  LogicalResult matchAndRewrite(neura::PhiOp phi_op,
                                PatternRewriter &rewriter) const override {
    // Checks if the phi operation has two operands.
    if (phi_op.getNumOperands() != 2) {
      return failure();
    }

    Value input0 = phi_op.getOperand(0);
    Value input1 = phi_op.getOperand(1);

    // Checks if the phi operation is already part of a merge pattern or a loop
    // recurrence.
    if (merge_pattern_finder.isPhiInMerge(phi_op) ||
        loop_analyzer.isLoopPhi(phi_op.getResult())) {
      return failure();
    }

    Value condition = nullptr;
    Value true_value = nullptr;
    Value false_value = nullptr;

    // Case 1: Direct pattern: true_steer + false_steer
    if (tryMatchDirectSteerPattern(input0, input1, condition, true_value,
                                   false_value) ||
        tryMatchDirectSteerPattern(input1, input0, condition, true_value,
                                   false_value)) {
      createMergeOp(phi_op, rewriter, condition, true_value, false_value);
      return success();
    }

    // Case 2: One input is based on a true_steer, the other is a false_steer.
    if (tryMatchIndirectSteerPattern(input0, input1, condition, true_value,
                                     false_value) ||
        tryMatchIndirectSteerPattern(input1, input0, condition, true_value,
                                     false_value)) {
      createMergeOp(phi_op, rewriter, condition, true_value, false_value);
      return success();
    }

    return failure();
  }

private:
  MergePatternFinder &merge_pattern_finder;
  LoopAnalyzer &loop_analyzer;

  // Checks for direct true_steer + false_steer pattern.
  bool tryMatchDirectSteerPattern(Value input1, Value input2, Value &condition,
                                  Value &true_value, Value &false_value) const {
    auto true_steer = input1.getDefiningOp<neura::TrueSteerOp>();
    if (!true_steer) {
      return false;
    }

    auto false_steer = input2.getDefiningOp<neura::FalseSteerOp>();
    if (!false_steer) {
      return false;
    }

    // Checks if both steer operations share the same condition.
    if (true_steer.getCondition() != false_steer.getCondition()) {
      return false;
    }

    condition = true_steer.getCondition();
    true_value = input1;
    false_value = input2;
    return true;
  }

  // Checks if one input is based on a steer operation.
  bool tryMatchIndirectSteerPattern(Value input1, Value input2,
                                    Value &condition, Value &true_value,
                                    Value &false_value) const {
    // Checks if input2 is a false_steer.
    auto false_steer = input2.getDefiningOp<neura::FalseSteerOp>();
    if (!false_steer) {
      return false;
    }

    condition = false_steer.getCondition();

    // Checks the defining operation of input1.
    Operation *def_op = input1.getDefiningOp();
    if (!def_op) {
      return false;
    }

    // Checks if the defining operation's inputs use true_steer.
    bool found_true_steer = false;
    for (Value operand : def_op->getOperands()) {
      if (auto true_steer = operand.getDefiningOp<neura::TrueSteerOp>()) {
        if (true_steer.getCondition() == condition) {
          found_true_steer = true;
          break;
        }
      }
    }

    if (!found_true_steer) {
      return false;
    }

    true_value = input1;
    false_value = input2;
    return true;
  }

  // Creates a merge operation to replace the phi operation.
  void createMergeOp(neura::PhiOp phi_op, PatternRewriter &rewriter,
                     Value condition, Value true_value,
                     Value false_value) const {
    rewriter.setInsertionPoint(phi_op);
    auto merge_op = rewriter.create<neura::MergeOp>(
        phi_op.getLoc(), phi_op.getType(), condition, true_value, false_value);
    rewriter.replaceOp(phi_op, merge_op.getResult());
  }
};

class GrantPredicateToSteerPattern
    : public OpRewritePattern<neura::GrantPredicateOp> {
public:
  GrantPredicateToSteerPattern(MLIRContext *context,
                               OperationsToErase &ops_to_erase)
      : OpRewritePattern<neura::GrantPredicateOp>(context),
        ops_to_erase(ops_to_erase) {}

  LogicalResult matchAndRewrite(neura::GrantPredicateOp grant_op,
                                PatternRewriter &rewriter) const override {
    Value value = grant_op.getValue();
    Value condition = grant_op.getPredicate();

    if (auto not_op = condition.getDefiningOp<neura::NotOp>()) {
      // If the condition is a Not operation, we can transform it to the
      // false_steer.
      rewriter.setInsertionPoint(grant_op);
      auto false_steer = rewriter.create<neura::FalseSteerOp>(
          grant_op.getLoc(), value.getType(), value, not_op.getInput());
      rewriter.replaceOp(grant_op, false_steer.getResult());
      ops_to_erase.markForErasure(not_op);
    } else {
      // Otherwise, we transform it to the true_steer.
      rewriter.setInsertionPoint(grant_op);
      auto true_steer = rewriter.create<neura::TrueSteerOp>(
          grant_op.getLoc(), value.getType(), value, condition);
      rewriter.replaceOp(grant_op, true_steer.getResult());
    }

    return success();
  }

private:
  OperationsToErase &ops_to_erase;
};

class GrantOnceRemovalPattern : public OpRewritePattern<neura::GrantOnceOp> {
public:
  GrantOnceRemovalPattern(MLIRContext *context)
      : OpRewritePattern<neura::GrantOnceOp>(context) {}

  LogicalResult matchAndRewrite(neura::GrantOnceOp grant_once_op,
                                PatternRewriter &rewriter) const override {
    Value input = grant_once_op.getValue();

    rewriter.replaceOp(grant_once_op, input);
    return success();
  }
};

struct TransformToSteerControlPass
    : public PassWrapper<TransformToSteerControlPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformToSteerControlPass)

  StringRef getArgument() const override {
    return "transform-to-steer-control";
  }

  StringRef getDescription() const override {
    return "Transform control flow into data flow using steer control "
           "operations.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext &context = getContext();
    PatternRewriter rewriter(&context);
    OperationsToErase ops_to_erase;

    RewritePatternSet grant_once_patterns(&context);
    grant_once_patterns.add<GrantOnceRemovalPattern>(&context);
    if (failed(applyPatternsGreedily(func, std::move(grant_once_patterns)))) {
      signalPassFailure();
    }
    MergePatternFinder merge_pattern_finder(func);

    RewritePatternSet merge_patterns(&context);
    merge_patterns.add<PhiToMergePattern>(&context, merge_pattern_finder,
                                          ops_to_erase);
    if (failed(applyPatternsGreedily(func, std::move(merge_patterns)))) {
      signalPassFailure();
    }
    // Erases the marked operations after processing all merge patterns.
    ops_to_erase.eraseMarkedOperations();

    LoopAnalyzer loop_analyzer(func);
    BackwardValueHandler backward_value_handler(rewriter);

    RewritePatternSet phi_patterns(&context);
    phi_patterns.add<PhiStartToCarryPattern>(
        &context, loop_analyzer, backward_value_handler, ops_to_erase);
    if (failed(applyPatternsGreedily(func, std::move(phi_patterns)))) {
      signalPassFailure();
    }
    // Erases the marked operations after processing all phi operations.
    ops_to_erase.eraseMarkedOperations();

    RewritePatternSet steer_patterns(&context);
    steer_patterns.add<GrantPredicateToSteerPattern>(&context, ops_to_erase);

    if (failed(applyPatternsGreedily(func, std::move(steer_patterns)))) {
      signalPassFailure();
    }
    // Erases the marked operations after processing all grant_predicate
    // operations.
    ops_to_erase.eraseMarkedOperations();

    RewritePatternSet steer_phi_patterns(&context);
    steer_phi_patterns.add<SteerPhiToMergePattern>(
        &context, ops_to_erase, merge_pattern_finder, loop_analyzer);
    if (failed(applyPatternsGreedily(func, std::move(steer_phi_patterns)))) {
      signalPassFailure();
    }
    // Erases the marked operations after processing all steer-phi patterns.
    ops_to_erase.eraseMarkedOperations();

    // Cleans up any remaining unused reserve, ctrl_mov, and not operations.
    llvm::SmallVector<Operation *, 16> to_erase;
    func.walk([&](Operation *op) {
      if (isa<neura::ReserveOp>(op) && op->use_empty()) {
        to_erase.push_back(op);
      } else if (isa<neura::NotOp>(op) && op->use_empty()) {
        to_erase.push_back(op);
      } else if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(op)) {
        // Cleans up ctrl_mov operations whose target reserve is unused.
        if (auto target_reserve =
                ctrl_mov_op.getTarget().getDefiningOp<neura::ReserveOp>()) {
          if (target_reserve->use_empty()) {
            // If the target reserve is going to be deleted, this ctrl_mov can
            // also be deleted.
            to_erase.push_back(ctrl_mov_op);
          }
        }
      }
    });

    // Deletes the collected operations in reverse order.
    for (auto it = to_erase.rbegin(); it != to_erase.rend(); ++it) {
      (*it)->erase();
    }

    // Checks if the function is now in predicate mode.
    auto dataflow_mode_attr =
        func->getAttrOfType<StringAttr>(neura::attr::kDataflowMode);
    if (!dataflow_mode_attr ||
        dataflow_mode_attr.getValue() != neura::attr::val::kModePredicate) {
      func.emitError("transform-to-steer-control requires function to be in "
                     "predicate mode");
      signalPassFailure();
      return;
    }
    // Changes the dataflow_mode attribute to "steering".
    func->setAttr(neura::attr::kDataflowMode,
                  StringAttr::get(&context, neura::attr::val::kModeSteering));
    llvm::errs()
        << "[ctrl2steer] Changed dataflow mode from predicate to steering "
           "for function: "
        << func.getName() << "\n";
  }
};
} // namespace

namespace mlir {
namespace neura {

std::unique_ptr<Pass> createTransformToSteerControlPass() {
  return std::make_unique<TransformToSteerControlPass>();
}

} // namespace neura
} // namespace mlir