//===- FuseTaskPass.cpp - Task fusion for Taskflow dialect ----------------===//
//
// Fuses taskflow.task ops by merging counter chains and hyperblock bodies.
// Supports producer-consumer fusion (with intermediate store/load elimination)
// and sibling fusion (with shared input deduplication).
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cmath>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Operand helpers
//===----------------------------------------------------------------------===//

// Resolves a value through a task's WAR/RAW dependency chain.
// If v is a dependency_read_out or dependency_write_out of the task,
// returns the corresponding input; otherwise returns v unchanged.
static Value resolveThrough(Value v, TaskflowTaskOp task) {
  for (unsigned i = 0; i < task.getDependencyReadOut().size(); ++i)
    if (v == task.getDependencyReadOut()[i])
      return task.getDependencyReadIn()[i];
  for (unsigned i = 0; i < task.getDependencyWriteOut().size(); ++i)
    if (v == task.getDependencyWriteOut()[i])
      return task.getDependencyWriteIn()[i];
  return v;
}

// Collects unique values from two ranges into a deduped vector with index map.
static void collectUnique(ValueRange a, ValueRange b,
                          SmallVectorImpl<Value> &result,
                          llvm::SmallDenseMap<Value, unsigned> &idx_map) {
  for (Value v : a) {
    idx_map[v] = result.size();
    result.push_back(v);
  }
  for (Value v : b)
    if (!idx_map.count(v)) {
      idx_map[v] = result.size();
      result.push_back(v);
    }
}

//===----------------------------------------------------------------------===//
// Counter chain & hyperblock utilities
//===----------------------------------------------------------------------===//

// Extracts all TaskflowCounterOps from a task body in definition order.
static SmallVector<TaskflowCounterOp> extractCounterChain(TaskflowTaskOp task) {
  SmallVector<TaskflowCounterOp> chain;
  for (Operation &op : task.getBody().front())
    if (auto c = dyn_cast<TaskflowCounterOp>(&op))
      chain.push_back(c);
  return chain;
}

// Returns true if two counter chains have identical bounds and steps.
static bool counterChainsMatch(SmallVectorImpl<TaskflowCounterOp> &a,
                               SmallVectorImpl<TaskflowCounterOp> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  auto get_const_val = [](Value val) -> std::optional<int64_t> {
    if (auto cst = val.getDefiningOp<arith::ConstantIndexOp>()) {
      return cst.value();
    }
    if (auto cst = val.getDefiningOp<arith::ConstantIntOp>()) {
      return cst.value();
    }
    return std::nullopt;
  };
  for (unsigned i = 0; i < a.size(); ++i) {
    auto al = get_const_val(a[i].getLowerBound());
    auto bl = get_const_val(b[i].getLowerBound());
    auto au = get_const_val(a[i].getUpperBound());
    auto bu = get_const_val(b[i].getUpperBound());
    auto as_ = get_const_val(a[i].getStep());
    auto bs_ = get_const_val(b[i].getStep());
    if (!al || !bl || !au || !bu || !as_ || !bs_ || *al != *bl || *au != *bu ||
        *as_ != *bs_) {
      return false;
    }
  }
  return true;
}

// Finds the single TaskflowHyperblockOp in a task body. Returns nullptr
// if none or multiple exist.
static TaskflowHyperblockOp findHyperblock(TaskflowTaskOp task) {
  TaskflowHyperblockOp result = nullptr;
  for (Operation &op : task.getBody().front()) {
    if (auto hb = dyn_cast<TaskflowHyperblockOp>(&op)) {
      if (result)
        return nullptr; // multiple hyperblocks
      result = hb;
    }
  }
  return result;
}

// Finds the single scf::ForOp in a hyperblock body. Returns nullptr
// if none or multiple exist.
static scf::ForOp findInnerForOp(TaskflowHyperblockOp hb) {
  scf::ForOp result = nullptr;
  for (auto &op : hb.getBody().front()) {
    if (auto f = dyn_cast<scf::ForOp>(&op)) {
      if (result)
        return nullptr;
      result = f;
    }
  }
  return result;
}

// Returns true if two scf::ForOp loops have identical constant bounds
// and step.
static bool scfForBoundsMatch(scf::ForOp a, scf::ForOp b) {
  auto get_const = [](Value v) -> std::optional<int64_t> {
    if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
      return cst.value();
    return std::nullopt;
  };
  auto al = get_const(a.getLowerBound()), bl = get_const(b.getLowerBound());
  auto au = get_const(a.getUpperBound()), bu = get_const(b.getUpperBound());
  auto as = get_const(a.getStep()), bs = get_const(b.getStep());
  return al && bl && au && bu && as && bs && *al == *bl && *au == *bu &&
         *as == *bs;
}

//===----------------------------------------------------------------------===//
// Legality checks
//===----------------------------------------------------------------------===//

// Returns true if task1 dominates task2 in the same block.
static bool canFuseTasks(TaskflowTaskOp t1, TaskflowTaskOp t2) {
  return t1 && t2 && t1 != t2 && t1->getBlock() == t2->getBlock() &&
         t1->isBeforeInBlock(t2);
}

// Returns true if any write output of the producer feeds the consumer.
static bool hasProducerConsumerRelation(TaskflowTaskOp producer,
                                        TaskflowTaskOp consumer) {
  for (Value r : producer.getDependencyWriteOut()) {
    for (Value in : consumer.getDependencyReadIn())
      if (r == in)
        return true;
    for (Value in : consumer.getDependencyWriteIn())
      if (r == in)
        return true;
  }
  for (Value r : producer.getValueOutputs())
    for (Value in : consumer.getValueInputs())
      if (r == in)
        return true;
  return false;
}

// Returns true if tasks share at least one input with no data dependency.
// Resolves WAR chains: if t2's input is t1's dependency_read_out, traces
// back to the original memref for comparison.
static bool areSiblingTasks(TaskflowTaskOp t1, TaskflowTaskOp t2) {
  llvm::SmallPtrSet<Value, 8> t1_mem;
  for (Value v : t1.getDependencyReadIn())
    t1_mem.insert(v);
  for (Value v : t1.getDependencyWriteIn())
    t1_mem.insert(v);
  llvm::SmallPtrSet<Value, 8> t1_val(t1.getValueInputs().begin(),
                                     t1.getValueInputs().end());
  bool share = false;
  for (Value v : t2.getDependencyReadIn()) {
    Value resolved = resolveThrough(v, t1);
    if (t1_mem.count(resolved)) {
      share = true;
      break;
    }
  }
  if (!share)
    for (Value v : t2.getDependencyWriteIn()) {
      Value resolved = resolveThrough(v, t1);
      if (t1_mem.count(resolved)) {
        share = true;
        break;
      }
    }
  if (!share)
    for (Value v : t2.getValueInputs())
      if (t1_val.count(v)) {
        share = true;
        break;
      }
  return share && !hasProducerConsumerRelation(t1, t2) &&
         !hasProducerConsumerRelation(t2, t1);
}

// Returns true if any op between producer and consumer uses producer's results.
static bool hasInterveningUses(TaskflowTaskOp producer,
                               TaskflowTaskOp consumer) {
  llvm::SmallPtrSet<Value, 8> results;
  for (Value v : producer.getDependencyReadOut())
    results.insert(v);
  for (Value v : producer.getDependencyWriteOut())
    results.insert(v);
  for (Value v : producer.getValueOutputs())
    results.insert(v);
  bool in_range = false;
  for (Operation &op : *producer->getBlock()) {
    if (&op == producer.getOperation()) {
      in_range = true;
      continue;
    }
    if (&op == consumer.getOperation())
      break;
    if (in_range)
      for (Value v : op.getOperands())
        if (results.count(v))
          return true;
  }
  return false;
}

// Returns true if all outputs of the task have at most one use.
static bool hasOnlySingleUseOutputs(TaskflowTaskOp task) {
  for (Value r : task.getDependencyReadOut())
    if (!r.hasOneUse() && !r.use_empty())
      return false;
  for (Value r : task.getDependencyWriteOut())
    if (!r.hasOneUse() && !r.use_empty())
      return false;
  for (Value r : task.getValueOutputs())
    if (!r.hasOneUse() && !r.use_empty())
      return false;
  return true;
}

// Clones non-structural ops from a task body (skips counters, hyperblocks,
// and yields).
static void cloneTaskBodyMiscOps(Block &body, OpBuilder &builder,
                                 IRMapping &mapping) {
  for (Operation &op : body) {
    if (isa<TaskflowCounterOp, TaskflowHyperblockOp, TaskflowYieldOp>(&op))
      continue;
    builder.clone(op, mapping);
  }
}

// Returns true if two tasks have compatible loop structures for merging.
// Handles both counter-chain and scf.for-inside-hyperblock cases.
static bool haveMatchingLoopStructure(TaskflowTaskOp t1, TaskflowTaskOp t2) {
  auto hb1 = findHyperblock(t1), hb2 = findHyperblock(t2);
  if (!hb1 || !hb2)
    return false;
  if (hb1.getIterArgs().size() > 0 || hb2.getIterArgs().size() > 0)
    return false;

  auto c1 = extractCounterChain(t1), c2 = extractCounterChain(t2);
  if (!c1.empty() && !c2.empty())
    return counterChainsMatch(c1, c2);

  // No counter chains: compares scf.for loops in hyperblock bodies.
  if (c1.empty() && c2.empty()) {
    auto f1 = findInnerForOp(hb1), f2 = findInnerForOp(hb2);
    if (f1 && f2)
      return scfForBoundsMatch(f1, f2);
    // Both have no loops: trivially compatible.
    if (!f1 && !f2)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Producer-consumer fusion
//===----------------------------------------------------------------------===//

// Fuses a producer into its consumer by merging their counter chains and
// hyperblock bodies into a single task with direct SSA value forwarding
// for the intermediate memref.
static TaskflowTaskOp fuseProducerConsumerTasks(TaskflowTaskOp producer,
                                                TaskflowTaskOp consumer,
                                                Value intermediate_memref,
                                                OpBuilder &builder) {
  Location loc = consumer.getLoc();
  auto prod_read = producer.getDependencyReadIn();
  auto prod_write = producer.getDependencyWriteIn();
  auto prod_val = producer.getValueInputs();
  auto cons_read = consumer.getDependencyReadIn();
  auto cons_write = consumer.getDependencyWriteIn();
  auto cons_val = consumer.getValueInputs();

  // Collects fused inputs, keeping intermediate in write_memrefs for
  // block arg availability but excluding it from consumer's read side.
  SmallVector<Value> fused_read, fused_write, fused_val;
  for (Value v : prod_read)
    fused_read.push_back(v);
  for (Value v : cons_read)
    if (v != intermediate_memref)
      fused_read.push_back(resolveThrough(v, producer));
  for (Value v : prod_write)
    fused_write.push_back(v);
  for (Value v : cons_write)
    if (v != intermediate_memref)
      fused_write.push_back(resolveThrough(v, producer));
  for (Value v : prod_val)
    fused_val.push_back(v);
  for (Value v : cons_val)
    fused_val.push_back(resolveThrough(v, producer));

  // Read output types: passthrough of fused read inputs for WAR tracking.
  SmallVector<Type> read_out_types;
  for (Value v : fused_read)
    read_out_types.push_back(v.getType());

  // Write/value output types come from the consumer only.
  SmallVector<Type> write_out_types(
      consumer.getDependencyWriteOut().getTypes());
  SmallVector<Type> val_out_types(consumer.getValueOutputs().getTypes());

  SmallVector<Value> orig_reads, orig_writes;
  for (Value v : producer.getOriginalReadMemrefs())
    orig_reads.push_back(v);
  for (Value v : consumer.getOriginalReadMemrefs())
    orig_reads.push_back(v);
  for (Value v : producer.getOriginalWriteMemrefs())
    orig_writes.push_back(v);
  for (Value v : consumer.getOriginalWriteMemrefs())
    orig_writes.push_back(v);

  auto fused = builder.create<TaskflowTaskOp>(
      loc, read_out_types, write_out_types, val_out_types, fused_read,
      fused_write, fused_val, builder.getStringAttr("fused_pc"), orig_reads,
      orig_writes);

  Block *body = new Block();
  fused.getBody().push_back(body);
  for (Value v : fused_read)
    body->addArgument(v.getType(), loc);
  for (Value v : fused_write)
    body->addArgument(v.getType(), loc);
  for (Value v : fused_val)
    body->addArgument(v.getType(), loc);

  unsigned fused_read_n = fused_read.size();
  unsigned fused_write_n = fused_write.size();

  // --- Maps producer block args to fused block args ---
  IRMapping mapping;
  Block &prod_body = producer.getBody().front();
  for (unsigned i = 0; i < prod_read.size(); ++i)
    mapping.map(prod_body.getArgument(i), body->getArgument(i));
  for (unsigned i = 0; i < prod_write.size(); ++i)
    mapping.map(prod_body.getArgument(prod_read.size() + i),
                body->getArgument(fused_read_n + i));
  for (unsigned i = 0; i < prod_val.size(); ++i)
    mapping.map(prod_body.getArgument(prod_read.size() + prod_write.size() + i),
                body->getArgument(fused_read_n + fused_write_n + i));

  // Identifies the producer's task-body block arg for the intermediate write
  // memref. The intermediate_memref is the producer's write OUTPUT result;
  // finds which write output index it corresponds to, then gets the matching
  // write memref block arg.
  BlockArgument prod_intermediate_arg = nullptr;
  for (unsigned i = 0; i < producer.getDependencyWriteOut().size(); ++i)
    if (producer.getDependencyWriteOut()[i] == intermediate_memref)
      prod_intermediate_arg = prod_body.getArgument(prod_read.size() + i);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  // --- Maps consumer block args to fused block args ---
  Block &cons_body = consumer.getBody().front();
  unsigned cons_read_fused = prod_read.size();
  for (unsigned i = 0; i < cons_read.size(); ++i) {
    if (cons_read[i] == intermediate_memref) {
      if (prod_intermediate_arg)
        mapping.map(cons_body.getArgument(i),
                    mapping.lookupOrDefault(prod_intermediate_arg));
    } else {
      mapping.map(cons_body.getArgument(i),
                  body->getArgument(cons_read_fused++));
    }
  }
  unsigned cons_write_fused = fused_read_n + prod_write.size();
  for (unsigned i = 0; i < cons_write.size(); ++i) {
    if (cons_write[i] == intermediate_memref) {
      if (prod_intermediate_arg)
        mapping.map(cons_body.getArgument(cons_read.size() + i),
                    mapping.lookupOrDefault(prod_intermediate_arg));
    } else {
      mapping.map(cons_body.getArgument(cons_read.size() + i),
                  body->getArgument(cons_write_fused++));
    }
  }
  unsigned cons_val_fused = fused_read_n + fused_write_n + prod_val.size();
  for (unsigned i = 0; i < cons_val.size(); ++i)
    mapping.map(cons_body.getArgument(cons_read.size() + cons_write.size() + i),
                body->getArgument(cons_val_fused + i));

  // Clones non-counter, non-hyperblock, non-yield ops from both task bodies
  // FIRST (e.g. arith.constant for loop bounds) so they are in the mapping
  // before the counter chain is cloned.
  cloneTaskBodyMiscOps(prod_body, builder, mapping);
  cloneTaskBodyMiscOps(cons_body, builder, mapping);

  // Clones counter chain from producer.
  auto prod_counters = extractCounterChain(producer);
  for (auto ctr : prod_counters)
    builder.clone(*ctr, mapping);

  // Locates both hyperblocks.
  auto prod_hb = findHyperblock(producer);
  auto cons_hb = findHyperblock(consumer);
  Block &prod_hb_body = prod_hb.getBody().front();
  Block &cons_hb_body = cons_hb.getBody().front();

  // Creates fused hyperblock.
  SmallVector<Value> triggers;
  for (Value v : prod_hb.getIndices())
    triggers.push_back(mapping.lookupOrDefault(v));
  auto fused_hb = builder.create<TaskflowHyperblockOp>(loc, TypeRange{},
                                                       triggers, ValueRange{});
  Block *hb_body = &fused_hb.getBody().emplaceBlock();
  for (auto arg : prod_hb_body.getArguments())
    hb_body->addArgument(arg.getType(), loc);

  // Maps hyperblock block args.
  for (unsigned i = 0; i < prod_hb_body.getNumArguments(); ++i)
    mapping.map(prod_hb_body.getArgument(i), hb_body->getArgument(i));
  for (unsigned i = 0; i < cons_hb_body.getNumArguments(); ++i)
    mapping.map(cons_hb_body.getArgument(i), hb_body->getArgument(i));

  // Identifies the consumer's intermediate read block arg for load skipping.
  BlockArgument cons_intermediate_arg = nullptr;
  for (unsigned i = 0; i < cons_read.size(); ++i)
    if (cons_read[i] == intermediate_memref)
      cons_intermediate_arg = cons_body.getArgument(i);

  {
    OpBuilder::InsertionGuard hb_guard(builder);
    builder.setInsertionPointToStart(hb_body);

    auto prod_for = findInnerForOp(prod_hb);
    auto cons_for = findInnerForOp(cons_hb);

    if (prod_for && cons_for) {
      // --- SCF loop fusion: merges for-body ops into a single scf.for ---

      // Clones non-for, non-yield ops from both hyperblock bodies.
      for (Operation &op : prod_hb_body) {
        if (isa<TaskflowHyperblockYieldOp, scf::ForOp>(&op))
          continue;
        builder.clone(op, mapping);
      }
      for (Operation &op : cons_hb_body) {
        if (isa<TaskflowHyperblockYieldOp, scf::ForOp>(&op))
          continue;
        builder.clone(op, mapping);
      }

      // Creates merged scf.for with producer's bounds.
      Value lb = mapping.lookupOrDefault(prod_for.getLowerBound());
      Value ub = mapping.lookupOrDefault(prod_for.getUpperBound());
      Value step = mapping.lookupOrDefault(prod_for.getStep());
      auto merged_for = builder.create<scf::ForOp>(loc, lb, ub, step);
      mapping.map(prod_for.getInductionVar(), merged_for.getInductionVar());
      mapping.map(cons_for.getInductionVar(), merged_for.getInductionVar());

      OpBuilder::InsertionGuard for_guard(builder);
      builder.setInsertionPoint(merged_for.getBody()->getTerminator());

      // Clones producer for-body; skips store to intermediate, records
      // the forwarded value.
      Value forwarded = nullptr;
      for (Operation &op : *prod_for.getBody()) {
        if (isa<scf::YieldOp>(&op))
          continue;
        if (auto store = dyn_cast<memref::StoreOp>(&op)) {
          if (prod_intermediate_arg &&
              store.getMemRef() == prod_intermediate_arg) {
            forwarded = mapping.lookupOrDefault(store.getValueToStore());
            continue;
          }
        }
        builder.clone(op, mapping);
      }

      // Clones consumer for-body; replaces loads from intermediate
      // with the forwarded value.
      for (Operation &op : *cons_for.getBody()) {
        if (isa<scf::YieldOp>(&op))
          continue;
        if (auto load = dyn_cast<memref::LoadOp>(&op)) {
          if (cons_intermediate_arg && forwarded &&
              load.getMemRef() == cons_intermediate_arg) {
            mapping.map(load.getResult(), forwarded);
            continue;
          }
        }
        builder.clone(op, mapping);
      }

    } else {
      // --- Counter-chain path: clones hyperblock bodies sequentially ---
      Value forwarded = nullptr;
      for (Operation &op : prod_hb_body) {
        if (isa<TaskflowHyperblockYieldOp>(&op))
          continue;
        if (auto store = dyn_cast<memref::StoreOp>(&op)) {
          if (prod_intermediate_arg &&
              store.getMemRef() == prod_intermediate_arg) {
            forwarded = mapping.lookupOrDefault(store.getValueToStore());
            continue;
          }
        }
        builder.clone(op, mapping);
      }
      for (Operation &op : cons_hb_body) {
        if (isa<TaskflowHyperblockYieldOp>(&op))
          continue;
        if (auto load = dyn_cast<memref::LoadOp>(&op)) {
          if (cons_intermediate_arg && forwarded &&
              load.getMemRef() == cons_intermediate_arg) {
            mapping.map(load.getResult(), forwarded);
            continue;
          }
        }
        builder.clone(op, mapping);
      }
    }

    builder.create<TaskflowHyperblockYieldOp>(loc);
  }

  // Creates the task yield from consumer's yield operands.
  auto cons_yield =
      cast<TaskflowYieldOp>(consumer.getBody().front().getTerminator());

  // Read yield outputs: passthrough fused read block args.
  SmallVector<Value> yield_reads;
  for (unsigned i = 0; i < fused_read.size(); ++i)
    yield_reads.push_back(body->getArgument(i));

  SmallVector<Value> yield_mem, yield_val;
  for (Value v : cons_yield.getMemoryResults())
    yield_mem.push_back(mapping.lookupOrDefault(v));
  for (Value v : cons_yield.getValueResults())
    yield_val.push_back(mapping.lookupOrDefault(v));
  builder.create<TaskflowYieldOp>(loc, yield_reads, yield_mem, yield_val);
  return fused;
}

//===----------------------------------------------------------------------===//
// Sibling fusion
//===----------------------------------------------------------------------===//

// Fuses two sibling tasks by merging their counter chains and hyperblock
// bodies into a single task with deduplicated inputs.
static TaskflowTaskOp fuseSiblingTasks(TaskflowTaskOp t1, TaskflowTaskOp t2,
                                       OpBuilder &builder) {
  Location loc = t1.getLoc();

  // Resolves t2's inputs through t1's WAR chains so that the fused task
  // references original memrefs rather than t1's passthrough results.
  SmallVector<Value> t2_read, t2_write, t2_val;
  for (Value v : t2.getDependencyReadIn())
    t2_read.push_back(resolveThrough(v, t1));
  for (Value v : t2.getDependencyWriteIn())
    t2_write.push_back(resolveThrough(v, t1));
  for (Value v : t2.getValueInputs())
    t2_val.push_back(resolveThrough(v, t1));

  // Deduplicates inputs across both tasks.
  SmallVector<Value> fused_read, fused_write, fused_val;
  llvm::SmallDenseMap<Value, unsigned> read_idx, write_idx, val_idx;
  collectUnique(t1.getDependencyReadIn(), t2_read, fused_read, read_idx);
  collectUnique(t1.getDependencyWriteIn(), t2_write, fused_write, write_idx);
  collectUnique(t1.getValueInputs(), t2_val, fused_val, val_idx);

  // Read output types: passthrough of fused read inputs for WAR tracking.
  SmallVector<Type> read_out_types;
  for (Value v : fused_read)
    read_out_types.push_back(v.getType());

  // Combined write/value output types.
  SmallVector<Type> write_out_types, val_out_types;
  write_out_types.append(t1.getDependencyWriteOut().getTypes().begin(),
                         t1.getDependencyWriteOut().getTypes().end());
  write_out_types.append(t2.getDependencyWriteOut().getTypes().begin(),
                         t2.getDependencyWriteOut().getTypes().end());
  val_out_types.append(t1.getValueOutputs().getTypes().begin(),
                       t1.getValueOutputs().getTypes().end());
  val_out_types.append(t2.getValueOutputs().getTypes().begin(),
                       t2.getValueOutputs().getTypes().end());

  SmallVector<Value> orig_reads, orig_writes;
  for (Value v : t1.getOriginalReadMemrefs())
    orig_reads.push_back(v);
  for (Value v : t2.getOriginalReadMemrefs())
    orig_reads.push_back(v);
  for (Value v : t1.getOriginalWriteMemrefs())
    orig_writes.push_back(v);
  for (Value v : t2.getOriginalWriteMemrefs())
    orig_writes.push_back(v);

  auto fused = builder.create<TaskflowTaskOp>(
      loc, read_out_types, write_out_types, val_out_types, fused_read,
      fused_write, fused_val, builder.getStringAttr("fused_sibling"),
      orig_reads, orig_writes);

  Block *body = new Block();
  fused.getBody().push_back(body);
  for (Value v : fused_read)
    body->addArgument(v.getType(), loc);
  for (Value v : fused_write)
    body->addArgument(v.getType(), loc);
  for (Value v : fused_val)
    body->addArgument(v.getType(), loc);

  unsigned rn = fused_read.size(), wn = fused_write.size();

  // Lambda to map a task's block args to fused block args via index maps.
  // When is_t2 is true, resolves inputs through t1's WAR chains for lookup.
  auto mapBlockArgs = [&](TaskflowTaskOp task, IRMapping &m, bool is_t2) {
    Block &tb = task.getBody().front();
    auto r = task.getDependencyReadIn();
    auto w = task.getDependencyWriteIn();
    auto v = task.getValueInputs();
    for (unsigned i = 0; i < r.size(); ++i) {
      Value key = is_t2 ? resolveThrough(r[i], t1) : r[i];
      m.map(tb.getArgument(i), body->getArgument(read_idx[key]));
    }
    for (unsigned i = 0; i < w.size(); ++i) {
      Value key = is_t2 ? resolveThrough(w[i], t1) : w[i];
      m.map(tb.getArgument(r.size() + i),
            body->getArgument(rn + write_idx[key]));
    }
    for (unsigned i = 0; i < v.size(); ++i) {
      Value key = is_t2 ? resolveThrough(v[i], t1) : v[i];
      m.map(tb.getArgument(r.size() + w.size() + i),
            body->getArgument(rn + wn + val_idx[key]));
    }
  };

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  // Clones non-counter, non-hyperblock, non-yield ops from both task bodies
  // FIRST (e.g. arith.constant for loop bounds) so they are in the mapping
  // before the counter chain is cloned.
  IRMapping mapping;
  mapBlockArgs(t1, mapping, /*is_t2=*/false);
  Block &t1_body = t1.getBody().front();
  Block &t2_body = t2.getBody().front();
  IRMapping mapping2;
  mapBlockArgs(t2, mapping2, /*is_t2=*/true);
  cloneTaskBodyMiscOps(t1_body, builder, mapping);
  cloneTaskBodyMiscOps(t2_body, builder, mapping2);

  // Clones counter chain from t1.
  auto counters = extractCounterChain(t1);
  for (auto ctr : counters)
    builder.clone(*ctr, mapping);

  // Locates both hyperblocks.
  auto hb1 = findHyperblock(t1), hb2 = findHyperblock(t2);
  Block &hb1_body = hb1.getBody().front();
  Block &hb2_body = hb2.getBody().front();

  // Creates fused hyperblock.
  SmallVector<Value> triggers;
  for (Value v : hb1.getIndices())
    triggers.push_back(mapping.lookupOrDefault(v));
  auto fused_hb = builder.create<TaskflowHyperblockOp>(loc, TypeRange{},
                                                       triggers, ValueRange{});
  Block *hb_body = &fused_hb.getBody().emplaceBlock();
  for (auto arg : hb1_body.getArguments())
    hb_body->addArgument(arg.getType(), loc);

  // Maps hyperblock block args.
  for (unsigned i = 0; i < hb1_body.getNumArguments(); ++i)
    mapping.map(hb1_body.getArgument(i), hb_body->getArgument(i));
  for (unsigned i = 0; i < hb2_body.getNumArguments(); ++i)
    mapping2.map(hb2_body.getArgument(i), hb_body->getArgument(i));

  {
    OpBuilder::InsertionGuard hb_guard(builder);
    builder.setInsertionPointToStart(hb_body);

    auto for1 = findInnerForOp(hb1), for2 = findInnerForOp(hb2);

    if (for1 && for2) {
      // --- SCF loop fusion: merges for-body ops into a single scf.for ---

      // Clones non-for, non-yield ops from both hyperblock bodies.
      for (Operation &op : hb1_body) {
        if (isa<TaskflowHyperblockYieldOp, scf::ForOp>(&op))
          continue;
        builder.clone(op, mapping);
      }
      for (Operation &op : hb2_body) {
        if (isa<TaskflowHyperblockYieldOp, scf::ForOp>(&op))
          continue;
        builder.clone(op, mapping2);
      }

      // Creates merged scf.for with t1's bounds.
      Value lb = mapping.lookupOrDefault(for1.getLowerBound());
      Value ub = mapping.lookupOrDefault(for1.getUpperBound());
      Value step = mapping.lookupOrDefault(for1.getStep());
      auto merged_for = builder.create<scf::ForOp>(loc, lb, ub, step);
      mapping.map(for1.getInductionVar(), merged_for.getInductionVar());
      mapping2.map(for2.getInductionVar(), merged_for.getInductionVar());

      OpBuilder::InsertionGuard for_guard(builder);
      builder.setInsertionPoint(merged_for.getBody()->getTerminator());

      // Clones t1 for-body.
      for (Operation &op : *for1.getBody()) {
        if (isa<scf::YieldOp>(&op))
          continue;
        builder.clone(op, mapping);
      }
      // Clones t2 for-body.
      for (Operation &op : *for2.getBody()) {
        if (isa<scf::YieldOp>(&op))
          continue;
        builder.clone(op, mapping2);
      }

    } else {
      // --- Counter-chain path: clones hyperblock bodies sequentially ---
      for (Operation &op : hb1_body) {
        if (isa<TaskflowHyperblockYieldOp>(&op))
          continue;
        builder.clone(op, mapping);
      }
      for (Operation &op : hb2_body) {
        if (isa<TaskflowHyperblockYieldOp>(&op))
          continue;
        builder.clone(op, mapping2);
      }
    }

    builder.create<TaskflowHyperblockYieldOp>(loc);
  }

  // Creates combined task yield.
  auto t1_yield = cast<TaskflowYieldOp>(t1.getBody().front().getTerminator());
  auto t2_yield = cast<TaskflowYieldOp>(t2.getBody().front().getTerminator());

  // Read yield outputs: passthrough fused read block args.
  SmallVector<Value> all_reads;
  for (unsigned i = 0; i < fused_read.size(); ++i)
    all_reads.push_back(body->getArgument(i));

  SmallVector<Value> all_mem, all_val;
  for (Value v : t1_yield.getMemoryResults())
    all_mem.push_back(mapping.lookupOrDefault(v));
  for (Value v : t1_yield.getValueResults())
    all_val.push_back(mapping.lookupOrDefault(v));
  for (Value v : t2_yield.getMemoryResults())
    all_mem.push_back(mapping2.lookupOrDefault(v));
  for (Value v : t2_yield.getValueResults())
    all_val.push_back(mapping2.lookupOrDefault(v));
  builder.create<TaskflowYieldOp>(loc, all_reads, all_mem, all_val);
  return fused;
}

//===----------------------------------------------------------------------===//
// Profitability analysis
//===----------------------------------------------------------------------===//

// Represents metrics for evaluating fusion profitability.
struct FusionMetrics {
  int rec_mii = 1;
  int res_mii = 1;
  int max_fanout = 0;
  int num_ops = 0;
};

// Calculates the maximum fanout across all ops in a region.
static int calculateMaxFanoutInRegion(Region &region) {
  int max_fanout = 0;
  region.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      int fanout = std::distance(result.use_begin(), result.use_end());
      max_fanout = std::max(max_fanout, fanout);
    }
  });
  return max_fanout;
}

// Runs the taskflow-to-neura pipeline on a cloned module and computes
// MII metrics on the resulting "test_fused_kernel" function.
static FusionMetrics
computeRealMetrics(ModuleOp test_module,
                   const neura::Architecture &architecture) {
  FusionMetrics metrics;
  auto cloned = test_module.clone();

  // Phase 1: converts taskflow ops to neura kernels.
  {
    PassManager pm(cloned.getContext());
    pm.addPass(taskflow::createClassifyTaskAndCounterPass());
    pm.addPass(createConvertTaskflowToNeuraPass());
    pm.enableVerifier(false);
    if (failed(pm.run(cloned))) {
      metrics.rec_mii = 100;
      metrics.res_mii = 100;
      cloned.erase();
      return metrics;
    }
  }

  // Converts func.return to neura.return for accelerator-marked functions
  // so that the subsequent neura passes can process them correctly.
  cloned.walk([&](func::ReturnOp ret) {
    auto func_op = ret->getParentOfType<func::FuncOp>();
    if (!func_op || !func_op->getAttrOfType<StringAttr>("accelerator")) {
      return;
    }
    OpBuilder builder(ret);
    auto neura_ret =
        builder.create<neura::ReturnOp>(ret.getLoc(), ret.getOperands());
    if (ret.getNumOperands() > 0) {
      neura_ret->setAttr("return_type", builder.getStringAttr("value"));
    } else {
      neura_ret->setAttr("return_type", builder.getStringAttr("void"));
    }
    ret.erase();
  });

  // Phase 2: runs the neura compilation pipeline.
  {
    PassManager pm(cloned.getContext());
    pm.addPass(neura::createAssignAcceleratorPass());
    pm.addPass(createLowerArithToNeuraPass());
    pm.addPass(createLowerMemRefToNeuraPass());
    pm.addPass(neura::createCanonicalizeReturnPass());
    pm.addPass(neura::createCanonicalizeCastPass());
    pm.addPass(neura::createPromoteInputArgToConstPass());
    pm.addPass(neura::createCanonicalizeLiveInPass());
    pm.addPass(neura::createLeveragePredicatedValuePass());
    pm.addPass(neura::createTransformCtrlToDataFlowPass());
    pm.enableVerifier(false);
    if (failed(pm.run(cloned))) {
      metrics.rec_mii = 100;
      metrics.res_mii = 100;
      cloned.erase();
      return metrics;
    }
  }

  cloned.walk([&](func::FuncOp func_op) {
    if (func_op.getName() != "test_fused_kernel") {
      return;
    }
    metrics.res_mii = neura::calculateResMii(func_op.getBody(), architecture);
    auto cycles = neura::collectRecurrenceCycles(func_op.getBody());
    metrics.rec_mii = 1;
    for (const auto &cycle : cycles) {
      metrics.rec_mii = std::max(metrics.rec_mii, cycle.length);
    }
    int num_ops = 0;
    func_op.walk([&](Operation *op) {
      if (!isa<func::FuncOp>(op) && !op->hasTrait<OpTrait::IsTerminator>()) {
        ++num_ops;
      }
    });
    metrics.num_ops = num_ops;
    if (!func_op.getBody().empty()) {
      metrics.max_fanout = calculateMaxFanoutInRegion(func_op.getBody());
    }
  });

  cloned.erase();
  return metrics;
}

// Creates a test module containing a single task wrapped in a function.
static ModuleOp createTestModuleForTask(TaskflowTaskOp task) {
  MLIRContext *ctx = task->getContext();
  OpBuilder builder(ctx);
  Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Collects unique operand values from the task.
  llvm::SetVector<Value> unique_operands;
  for (Value v : task->getOperands()) {
    unique_operands.insert(v);
  }

  SmallVector<Type> input_types;
  for (Value v : unique_operands) {
    input_types.push_back(v.getType());
  }
  SmallVector<Type> output_types;
  for (Value v : task->getResults()) {
    output_types.push_back(v.getType());
  }
  if (input_types.empty()) {
    input_types.push_back(builder.getI64Type());
  }
  if (output_types.empty()) {
    output_types.push_back(builder.getI64Type());
  }

  auto func_type = builder.getFunctionType(input_types, output_types);
  auto func_op =
      builder.create<func::FuncOp>(loc, "test_fused_kernel", func_type);
  func_op->setAttr("accelerator", builder.getStringAttr("neura"));

  Block *entry = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  IRMapping mapping;
  for (auto [i, v] : llvm::enumerate(unique_operands)) {
    mapping.map(v, entry->getArgument(i));
  }

  auto *cloned = builder.clone(*task.getOperation(), mapping);
  auto cloned_task = cast<TaskflowTaskOp>(cloned);

  SmallVector<Value> ret;
  for (Value v : cloned_task->getResults()) {
    ret.push_back(v);
  }
  if (ret.empty()) {
    ret.push_back(entry->getArgument(0));
  }
  builder.create<func::ReturnOp>(loc, ret);
  return module;
}

// Computes metrics for a single task by running the neura pipeline.
static FusionMetrics
computeSingleTaskMetrics(TaskflowTaskOp task,
                         const neura::Architecture &architecture) {
  auto module = createTestModuleForTask(task);
  FusionMetrics metrics = computeRealMetrics(module, architecture);
  module.erase();
  return metrics;
}

// Creates a test module with two tasks fused together.
static ModuleOp createFusedTestModule(TaskflowTaskOp task1,
                                      TaskflowTaskOp task2,
                                      bool is_producer_consumer,
                                      Value intermediate_memref) {
  MLIRContext *ctx = task1->getContext();
  OpBuilder builder(ctx);
  Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Collects unique external values (excludes task1's results used by task2).
  llvm::SetVector<Value> unique_vals;
  for (Value v : task1->getOperands()) {
    unique_vals.insert(v);
  }
  for (Value v : task2->getOperands()) {
    unique_vals.insert(v);
  }
  for (Value v : task1->getResults()) {
    unique_vals.remove(v);
  }

  SmallVector<Type> input_types;
  for (Value v : unique_vals) {
    input_types.push_back(v.getType());
  }
  if (input_types.empty()) {
    input_types.push_back(builder.getI64Type());
  }

  // Placeholder output type; the fused task's results drive the actual return.
  SmallVector<Type> output_types;
  output_types.push_back(builder.getI64Type());

  auto func_type = builder.getFunctionType(input_types, output_types);
  auto func_op =
      builder.create<func::FuncOp>(loc, "test_fused_kernel", func_type);
  func_op->setAttr("accelerator", builder.getStringAttr("neura"));

  Block *entry = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  IRMapping mapping;
  for (auto [i, v] : llvm::enumerate(unique_vals)) {
    mapping.map(v, entry->getArgument(i));
  }

  // Clones task1.
  auto *ct1_op = builder.clone(*task1.getOperation(), mapping);
  auto ct1 = cast<TaskflowTaskOp>(ct1_op);

  // Maps task1's results so task2's operands resolve correctly.
  for (auto [orig, cloned] :
       llvm::zip(task1->getResults(), ct1->getResults())) {
    mapping.map(orig, cloned);
  }

  // Clones task2.
  auto *ct2_op = builder.clone(*task2.getOperation(), mapping);
  auto ct2 = cast<TaskflowTaskOp>(ct2_op);

  // Resolves the cloned intermediate memref.
  Value cloned_intermediate;
  if (is_producer_consumer && intermediate_memref) {
    cloned_intermediate = mapping.lookupOrDefault(intermediate_memref);
  }

  // Fuses the cloned tasks.
  TaskflowTaskOp fused;
  if (is_producer_consumer) {
    fused = fuseProducerConsumerTasks(ct1, ct2, cloned_intermediate, builder);
  } else {
    fused = fuseSiblingTasks(ct1, ct2, builder);
  }

  // Fixes the function return type and creates a return op.
  SmallVector<Value> ret;
  for (Value v : fused->getResults()) {
    ret.push_back(v);
  }
  if (ret.empty()) {
    ret.push_back(entry->getArgument(0));
  }

  // Removes placeholder return type and rebuilds with actual types.
  SmallVector<Type> real_output_types;
  for (Value v : ret) {
    real_output_types.push_back(v.getType());
  }
  func_op.setFunctionType(
      builder.getFunctionType(input_types, real_output_types));
  builder.create<func::ReturnOp>(loc, ret);

  // Erases un-fused clones (consumer first to drop uses of producer results).
  ct2->erase();
  ct1->erase();
  return module;
}

// Computes metrics for the fused version of two tasks.
static FusionMetrics
computeFusedTaskMetrics(TaskflowTaskOp task1, TaskflowTaskOp task2,
                        bool is_producer_consumer, Value intermediate_memref,
                        const neura::Architecture &architecture) {
  auto module = createFusedTestModule(task1, task2, is_producer_consumer,
                                      intermediate_memref);
  FusionMetrics metrics = computeRealMetrics(module, architecture);
  module.erase();
  return metrics;
}

// Estimates the effective MII considering utilization and fanout penalties.
static int estimateMII(const FusionMetrics &metrics, int total_tiles) {
  const float alpha = 0.5f;
  const float beta = 0.5f;
  int mii = std::max(metrics.rec_mii, metrics.res_mii);
  float utilization_factor =
      1.0f + alpha * (metrics.num_ops / static_cast<float>(total_tiles));
  float fanout_factor = 1.0f + beta * std::max(metrics.max_fanout - 4, 0);
  return static_cast<int>(std::ceil(utilization_factor * fanout_factor * mii));
}

// Checks if fusion is profitable by comparing estimated MII of the fused
// task against the individual tasks. For producer-consumer fusion, the
// unfused MII is the sum (sequential execution); for sibling fusion, it
// is the max (independent execution).
static bool isFusionProfitable(TaskflowTaskOp task1, TaskflowTaskOp task2,
                               bool is_producer_consumer,
                               Value intermediate = nullptr) {
  neura::Architecture architecture(1, 1);
  int total_tiles = architecture.getNumTiles();

  FusionMetrics m1 = computeSingleTaskMetrics(task1, architecture);
  FusionMetrics m2 = computeSingleTaskMetrics(task2, architecture);
  FusionMetrics fused = computeFusedTaskMetrics(
      task1, task2, is_producer_consumer, intermediate, architecture);

  int mii_1 = estimateMII(m1, total_tiles);
  int mii_2 = estimateMII(m2, total_tiles);
  int mii_fused = estimateMII(fused, total_tiles);

  // Uses raw MII (max of recurrence and resource MII) for the core
  // comparison, since the estimateMII utilization penalty is additive
  // and already reflected in res_mii.
  int raw_1 = std::max(m1.rec_mii, m1.res_mii);
  int raw_2 = std::max(m2.rec_mii, m2.res_mii);
  int raw_fused = std::max(fused.rec_mii, fused.res_mii);
  int unfused_mii =
      is_producer_consumer ? (raw_1 + raw_2) : std::max(raw_1, raw_2);
  bool profitable = raw_fused <= unfused_mii;

  llvm::errs() << "[fuse-task] Profitability:"
               << " m1(rec=" << m1.rec_mii << " res=" << m1.res_mii
               << " ops=" << m1.num_ops << " fan=" << m1.max_fanout
               << " mii=" << mii_1 << ")"
               << " m2(rec=" << m2.rec_mii << " res=" << m2.res_mii
               << " ops=" << m2.num_ops << " fan=" << m2.max_fanout
               << " mii=" << mii_2 << ")"
               << " fused(rec=" << fused.rec_mii << " res=" << fused.res_mii
               << " ops=" << fused.num_ops << " fan=" << fused.max_fanout
               << " mii=" << mii_fused << ")"
               << " -> " << (profitable ? "PROFITABLE" : "REJECTED") << "\n";
  return profitable;
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

// Fuses a producer task into its consumer when the producer's output feeds
// directly into the consumer and the loop structures match.
struct ProducerConsumerTaskFusion : public OpRewritePattern<TaskflowTaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskflowTaskOp consumer,
                                PatternRewriter &rewriter) const override {
    TaskflowTaskOp producer = nullptr;
    Value intermediate;

    // Searches consumer's read and write memrefs for a fusible producer.
    auto tryFindProducer = [&](ValueRange inputs) -> bool {
      for (Value in : inputs) {
        auto def = in.getDefiningOp<TaskflowTaskOp>();
        // Only write outputs represent true producer-consumer (RAW) links.
        // Read outputs are WAR dependency chains and must be skipped.
        if (!def || !llvm::is_contained(def.getDependencyWriteOut(), in))
          continue;
        if (!canFuseTasks(def, consumer) || !hasOnlySingleUseOutputs(def) ||
            hasInterveningUses(def, consumer) ||
            !haveMatchingLoopStructure(def, consumer) ||
            !isFusionProfitable(def, consumer, true, in))
          continue;
        producer = def;
        intermediate = in;
        return true;
      }
      return false;
    };
    if (!tryFindProducer(consumer.getDependencyReadIn()) &&
        !tryFindProducer(consumer.getDependencyWriteIn()))
      return failure();

    auto fused =
        fuseProducerConsumerTasks(producer, consumer, intermediate, rewriter);

    // Replaces producer's dependency_read_out results.
    for (unsigned i = 0; i < producer.getDependencyReadOut().size(); ++i) {
      Value orig_read = producer.getDependencyReadIn()[i];
      for (unsigned j = 0; j < fused.getDependencyReadIn().size(); ++j) {
        if (fused.getDependencyReadIn()[j] == orig_read) {
          producer.getDependencyReadOut()[i].replaceAllUsesWith(
              fused.getDependencyReadOut()[j]);
          break;
        }
      }
    }
    // Replaces consumer's dependency_read_out results.
    for (unsigned i = 0; i < consumer.getDependencyReadOut().size(); ++i) {
      Value orig_read = consumer.getDependencyReadIn()[i];
      if (orig_read == intermediate)
        continue;
      Value resolved = resolveThrough(orig_read, producer);
      for (unsigned j = 0; j < fused.getDependencyReadIn().size(); ++j) {
        if (fused.getDependencyReadIn()[j] == resolved) {
          consumer.getDependencyReadOut()[i].replaceAllUsesWith(
              fused.getDependencyReadOut()[j]);
          break;
        }
      }
    }
    // Replaces consumer's write and value outputs.
    for (auto [o, n] : llvm::zip(consumer.getDependencyWriteOut(),
                                 fused.getDependencyWriteOut()))
      o.replaceAllUsesWith(n);
    for (auto [o, n] :
         llvm::zip(consumer.getValueOutputs(), fused.getValueOutputs()))
      o.replaceAllUsesWith(n);
    rewriter.eraseOp(consumer);
    rewriter.eraseOp(producer);
    return success();
  }
};

// Fuses sibling tasks that share inputs and have matching loop structures.
struct SiblingTaskFusion : public OpRewritePattern<TaskflowTaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskflowTaskOp t1,
                                PatternRewriter &rewriter) const override {
    TaskflowTaskOp t2 = nullptr;
    for (Operation *op = t1->getNextNode(); op; op = op->getNextNode()) {
      auto next = dyn_cast<TaskflowTaskOp>(op);
      if (!next)
        continue;
      if (areSiblingTasks(t1, next) && canFuseTasks(t1, next) &&
          haveMatchingLoopStructure(t1, next) &&
          isFusionProfitable(t1, next, false)) {
        t2 = next;
        break;
      }
    }
    if (!t2)
      return failure();

    auto fused = fuseSiblingTasks(t1, t2, rewriter);

    // Replaces dependency_read_out for both tasks.
    for (unsigned i = 0; i < t1.getDependencyReadOut().size(); ++i) {
      Value orig_read = t1.getDependencyReadIn()[i];
      for (unsigned j = 0; j < fused.getDependencyReadIn().size(); ++j) {
        if (fused.getDependencyReadIn()[j] == orig_read) {
          t1.getDependencyReadOut()[i].replaceAllUsesWith(
              fused.getDependencyReadOut()[j]);
          break;
        }
      }
    }
    for (unsigned i = 0; i < t2.getDependencyReadOut().size(); ++i) {
      Value orig_read = t2.getDependencyReadIn()[i];
      Value resolved = resolveThrough(orig_read, t1);
      for (unsigned j = 0; j < fused.getDependencyReadIn().size(); ++j) {
        if (fused.getDependencyReadIn()[j] == resolved) {
          t2.getDependencyReadOut()[i].replaceAllUsesWith(
              fused.getDependencyReadOut()[j]);
          break;
        }
      }
    }

    // Replaces dependency_write_out and value_outputs.
    unsigned t1_wo = t1.getDependencyWriteOut().size();
    unsigned t1_vo = t1.getValueOutputs().size();
    for (unsigned i = 0; i < t1_wo; ++i)
      t1.getDependencyWriteOut()[i].replaceAllUsesWith(
          fused.getDependencyWriteOut()[i]);
    for (unsigned i = 0; i < t1_vo; ++i)
      t1.getValueOutputs()[i].replaceAllUsesWith(fused.getValueOutputs()[i]);
    for (unsigned i = 0; i < t2.getDependencyWriteOut().size(); ++i)
      t2.getDependencyWriteOut()[i].replaceAllUsesWith(
          fused.getDependencyWriteOut()[t1_wo + i]);
    for (unsigned i = 0; i < t2.getValueOutputs().size(); ++i)
      t2.getValueOutputs()[i].replaceAllUsesWith(
          fused.getValueOutputs()[t1_vo + i]);
    rewriter.eraseOp(t1);
    rewriter.eraseOp(t2);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

// Applies producer-consumer and sibling task fusion using MII-based
// profitability analysis and loop body merging.
struct FuseTaskPass
    : public PassWrapper<FuseTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseTaskPass)

  StringRef getArgument() const override { return "fuse-task"; }
  StringRef getDescription() const override {
    return "Fuses taskflow.task ops via counter chain and hyperblock merging.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, func::FuncDialect, arith::ArithDialect,
                memref::MemRefDialect, scf::SCFDialect, cf::ControlFlowDialect,
                math::MathDialect, affine::AffineDialect, neura::NeuraDialect,
                taskflow::TaskflowDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ProducerConsumerTaskFusion>(&getContext(), /*benefit=*/10);
    patterns.add<SiblingTaskFusion>(&getContext(), /*benefit=*/5);
    if (failed(applyPatternsGreedily(
            getOperation(), FrozenRewritePatternSet(std::move(patterns)))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::taskflow {
std::unique_ptr<Pass> createFuseTaskPass() {
  return std::make_unique<FuseTaskPass>();
}
} // namespace mlir::taskflow
