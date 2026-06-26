//===- ClassifyTaskAndCounterPass.cpp - Classify tasks and counters -------===//
//
// Single pass that annotates every taskflow.task and its contained
// taskflow.counter ops with AMOEBA execution-model attributes:
//
// Per taskflow.counter:
//   counter_hierarchy  – structural role in the loop nest:
//     "root"   : outermost loop (no parent, has children)
//     "relay"  : middle loop   (has parent and children)
//     "leaf"   : innermost loop (no children; also single-loop tasks)
//
//   counter_dynamism   – AMOEBA bound class:
//     "constant_bound" : all bounds are compile-time constants
//     "symbol_bound"   : bounds are known before the task launches
//     "dynamic_bound"  : bounds are produced during task execution
//
//   counter_id         – unique integer index within the task (0-based)
//
// Per taskflow.task:
//   runtime_managable = true when the task can be managed by the AMOEBA
//   runtime: it has at least one symbol-bound counter and its
//   hyperblock bodies do not carry loop-carried dependences.
//   dlp_replicable = true when the task can be replicated for data-level
//   parallelism.
//
// Classification rules for a bound value inside the task body:
//   arith.constant                             → constant_bound
//   affine.apply                               → recurse through operands
//   block arg mapping to an outer value:
//     func.func argument                       → symbol_bound
//     memref.dim / function-scope arithmetic   → symbol_bound
//     arith.constant at call site              → constant_bound
//   anything else                              → dynamic_bound
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Bound kind
//===----------------------------------------------------------------------===//

enum class BoundKind { ConstantBound = 0, SymbolBound = 1, DynamicBound = 2 };

static BoundKind worstCase(BoundKind a, BoundKind b) {
  return static_cast<BoundKind>(
      std::max(static_cast<int>(a), static_cast<int>(b)));
}

static StringRef boundKindToStr(BoundKind k) {
  switch (k) {
  case BoundKind::ConstantBound:
    return "constant_bound";
  case BoundKind::SymbolBound:
    return "symbol_bound";
  case BoundKind::DynamicBound:
    return "dynamic_bound";
  }
  llvm_unreachable("unknown BoundKind");
}

//===----------------------------------------------------------------------===//
// Bound classification helpers
//===----------------------------------------------------------------------===//

// Classifies a value that lives OUTSIDE the task body (a value_input operand
// at the call site) by tracing it to its origin.
static BoundKind classifyOuterValue(Value v) {
  if (auto block_arg = dyn_cast<BlockArgument>(v)) {
    auto *parent_region = block_arg.getParentRegion();
    if (parent_region && isa<func::FuncOp>(parent_region->getParentOp())) {
      return BoundKind::SymbolBound;
    }
    return BoundKind::DynamicBound;
  }
  Operation *def = v.getDefiningOp();
  if (!def) {
    assert(false && "unexpected value with no defining op and not a block arg");
  }
  if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
    return BoundKind::ConstantBound;
  }
  // Any op at the function body scope (memref.dim, preceding taskflow.task
  // results, arith on func args, etc.) is fully determined before this task
  // launches.
  if (def->getParentRegion() &&
      isa<func::FuncOp>(def->getParentRegion()->getParentOp())) {
    return BoundKind::SymbolBound;
  }
  return BoundKind::DynamicBound;
}

// Classifies one lower/upper/step value used by a taskflow.counter.
// task_op is used to map task block arguments back to value_input operands.
static BoundKind classifyCounterBoundValue(Value v, TaskflowTaskOp task_op) {
  if (Operation *def = v.getDefiningOp()) {
    if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
      return BoundKind::ConstantBound;
    }
    // affine.apply is a pure affine computation; its dynamism is determined
    // entirely by its operands (e.g. affine_map<()[s0]->(s0-2)>()[%n]).
    if (isa<affine::AffineApplyOp>(def)) {
      BoundKind k = BoundKind::ConstantBound;
      for (Value operand : def->getOperands()) {
        k = worstCase(k, classifyCounterBoundValue(operand, task_op));
      }
      return k;
    }
    return BoundKind::DynamicBound;
  }

  auto block_arg = dyn_cast<BlockArgument>(v);
  if (!block_arg) {
    return BoundKind::DynamicBound;
  }

  // Block args layout: [dep_read_in…] [dep_write_in…] [value_inputs…]
  unsigned num_dep_read = task_op.getWillReads().size();
  unsigned num_dep_write = task_op.getWillWrites().size();
  unsigned value_input_start = num_dep_read + num_dep_write;
  unsigned arg_idx = block_arg.getArgNumber();

  assert(arg_idx >= value_input_start &&
         "counter bound is a memref dependency block arg — IR is malformed");

  unsigned vi_idx = arg_idx - value_input_start;
  auto value_inputs = task_op.getValueInputs();
  assert(vi_idx < value_inputs.size() &&
         "counter bound block arg index out of range of value inputs");

  return classifyOuterValue(value_inputs[vi_idx]);
}

// Returns the drive-pattern class implied by a counter's lb/ub/step operands.
static BoundKind classifyCounterDynamism(TaskflowCounterOp counter_op,
                                         TaskflowTaskOp task_op) {
  BoundKind k = BoundKind::ConstantBound;
  for (Value bound : {counter_op.getLowerBound(), counter_op.getUpperBound(),
                      counter_op.getStep()}) {
    k = worstCase(k, classifyCounterBoundValue(bound, task_op));
  }
  return k;
}

//===----------------------------------------------------------------------===//
// Hyperblock patterns for runtime management suitability.
//===----------------------------------------------------------------------===//

enum class TaskPattern { LoopCarriedDependence };

static bool taskContainsHyperblock(TaskflowTaskOp task_op) {
  bool saw_hyperblock = false;
  task_op.walk([&](TaskflowHyperblockOp) {
    saw_hyperblock = true;
    return WalkResult::interrupt();
  });
  return saw_hyperblock;
}

static bool
hyperblockMatchesLoopCarriedDependencePattern(TaskflowHyperblockOp hb) {
  if (!hb.getIterArgs().empty()) {
    return true;
  }

  bool matches = false;

  hb.walk([&](TaskflowHyperblockYieldOp yield_op) {
    if (!yield_op.getIterArgsNext().empty()) {
      matches = true;
    }
  });

  hb.walk([&](affine::AffineForOp for_op) {
    if (for_op.getNumIterOperands() > 0 || for_op.getNumResults() > 0) {
      matches = true;
    }
  });

  hb.walk([&](scf::ForOp for_op) {
    if (for_op.getNumRegionIterArgs() > 0 || for_op.getNumResults() > 0) {
      matches = true;
    }
  });

  return matches;
}

static bool taskMatchesLoopCarriedDependencePattern(TaskflowTaskOp task_op) {
  WalkResult result = task_op.walk([&](TaskflowHyperblockOp hb) {
    if (hyperblockMatchesLoopCarriedDependencePattern(hb)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static bool taskMatchesPattern(TaskflowTaskOp task_op, TaskPattern pattern) {
  switch (pattern) {
  case TaskPattern::LoopCarriedDependence:
    return taskMatchesLoopCarriedDependencePattern(task_op);
  }
  llvm_unreachable("unknown TaskPattern");
}

static bool taskHasDlpCapability(TaskflowTaskOp task_op) {
  return taskContainsHyperblock(task_op) &&
         !taskMatchesPattern(task_op, TaskPattern::LoopCarriedDependence);
}

//===----------------------------------------------------------------------===//
// Counter classification
//===----------------------------------------------------------------------===//

static LogicalResult classifyCounters(TaskflowTaskOp task_op) {
  // Pre-flight check: construct-hyperblock-from-task must run first.
  bool has_affine_for = false;
  task_op.walk([&](affine::AffineForOp) -> WalkResult {
    has_affine_for = true;
    return WalkResult::interrupt();
  });
  bool has_counter = false;
  task_op.walk([&](TaskflowCounterOp) -> WalkResult {
    has_counter = true;
    return WalkResult::interrupt();
  });
  if (has_affine_for && !has_counter) {
    return task_op.emitError()
           << "[ClassifyTaskAndCounter]: task '" << task_op.getTaskName()
           << "' contains affine.for loops but no taskflow.counter ops — "
              "run 'construct-hyperblock-from-task' before "
              "'classify-task-and-counter'";
  }

  SmallVector<TaskflowCounterOp> counters;
  task_op.walk(
      [&](TaskflowCounterOp counter_op) { counters.push_back(counter_op); });

  OpBuilder builder(task_op.getContext());

  if (counters.empty()) {
    return success();
  }

  // Build parent-child relationships among counters.
  DenseMap<Value, TaskflowCounterOp> value_to_counter;
  for (TaskflowCounterOp counter_op : counters) {
    value_to_counter[counter_op.getCounterIndex()] = counter_op;
  }

  DenseSet<TaskflowCounterOp> counters_with_children;
  for (TaskflowCounterOp counter_op : counters) {
    if (auto parent_idx = counter_op.getParentIndex()) {
      if (auto parent_counter = value_to_counter.lookup(parent_idx)) {
        counters_with_children.insert(parent_counter);
      }
    }
  }

  int counter_id = 0;

  for (TaskflowCounterOp counter_op : counters) {
    // --- counter_dynamism ---
    BoundKind bound_kind = classifyCounterDynamism(counter_op, task_op);

    // --- counter_hierarchy ---
    bool has_parent = (counter_op.getParentIndex() != nullptr);
    bool has_child = counters_with_children.contains(counter_op);

    StringRef structural_type;
    if (!has_parent && has_child) {
      structural_type = "root";
    } else if (has_parent && has_child) {
      structural_type = "relay";
    } else {
      // No children (or single loop with no parent/child): leaf.
      structural_type = "leaf";
    }

    counter_op.setCounterHierarchyAttr(builder.getStringAttr(structural_type));
    counter_op.setCounterDynamismAttr(
        builder.getStringAttr(boundKindToStr(bound_kind)));
    counter_op.setCounterIdAttr(builder.getI32IntegerAttr(counter_id++));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DLP-capability task tagging
//===----------------------------------------------------------------------===//

static bool taskIsDlpReplicable(TaskflowTaskOp task_op) {
  auto attr = task_op->getAttrOfType<BoolAttr>("dlp_replicable");
  return attr && attr.getValue();
}

static void identifyDlpCapability(TaskflowTaskOp task_op) {
  // Re-running this pass should not preserve task-level tags from an older
  // classification result.
  task_op->removeAttr("dlp_replicable");

  if (taskHasDlpCapability(task_op)) {
    OpBuilder builder(task_op.getContext());
    task_op->setAttr("dlp_replicable", builder.getBoolAttr(true));
  }
}

//===----------------------------------------------------------------------===//
// Runtime-managable task tagging
//===----------------------------------------------------------------------===//

static bool taskHasSymbolBoundCounter(TaskflowTaskOp task_op) {
  WalkResult result = task_op.walk([&](TaskflowCounterOp counter_op) {
    std::optional<StringRef> dynamism = counter_op.getCounterDynamism();
    if (dynamism && *dynamism == "symbol_bound") {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static void identifyRuntimeManagableTask(TaskflowTaskOp task_op) {
  // Re-running this pass should not preserve task-level tags from an older
  // classification result.
  task_op->removeAttr("task_type");
  task_op->removeAttr("runtime_managable");

  if (taskHasSymbolBoundCounter(task_op) && taskIsDlpReplicable(task_op)) {
    OpBuilder builder(task_op.getContext());
    task_op->setAttr("runtime_managable", builder.getBoolAttr(true));
  }
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ClassifyTaskAndCounterPass
    : public PassWrapper<ClassifyTaskAndCounterPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClassifyTaskAndCounterPass)

  StringRef getArgument() const override { return "classify-task-and-counter"; }
  StringRef getDescription() const override {
    return "Classify taskflow counters and mark runtime-managable tasks.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult counter_result =
        module.walk([&](TaskflowTaskOp task_op) -> WalkResult {
          if (failed(classifyCounters(task_op))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (counter_result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    module.walk(
        [&](TaskflowTaskOp task_op) { identifyDlpCapability(task_op); });
    module.walk(
        [&](TaskflowTaskOp task_op) { identifyRuntimeManagableTask(task_op); });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::taskflow::createClassifyTaskAndCounterPass() {
  return std::make_unique<ClassifyTaskAndCounterPass>();
}
