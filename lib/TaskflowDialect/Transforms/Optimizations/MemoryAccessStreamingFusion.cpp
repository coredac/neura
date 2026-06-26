//===- MemoryAccessStreamingFusion.cpp - Fuses tasks by memory deps -------===//
//
// This pass identifies and fuses taskflow.task operations that are connected
// by memory access dependencies (one task writes a memref, another reads it).
// It eliminates intermediate memref allocations and converts memory access
// dependencies into direct SSA value dependencies.
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Task Information and Dependency Analysis.
//===----------------------------------------------------------------------===//

// Stores information about a single task and its memory dependencies.
struct TaskInfo {
  taskflow::TaskflowTaskOp task_op;

  // Memrefs accessed by this task.
  SmallVector<Value> read_memrefs;
  SmallVector<Value> write_memrefs;

  // Memory dependency graph edges.
  // Upstream producers: tasks whose write_memrefs this task reads from.
  // e.g., if Task A writes %alloc and this task reads %alloc,
  //        then Task A is in this->memory_writers.
  SmallVector<TaskInfo *> memory_writers;

  // Downstream consumers: tasks that read from this task's write_memrefs.
  // e.g., if this task writes %alloc and Task B reads %alloc,
  //        then Task B is in this->memory_readers.
  SmallVector<TaskInfo *> memory_readers;

  // SSA value inputs (true producer-consumer dependencies).
  SmallVector<Value> value_inputs;

  TaskInfo() : task_op(nullptr) {}
  TaskInfo(taskflow::TaskflowTaskOp op) : task_op(op) {}
};

// Represents a candidate pair of tasks that can be fused.
struct FusionCandidate {
  TaskInfo *memory_writer;   // Task that writes the intermediate memref.
  TaskInfo *memory_reader;   // Task that reads the intermediate memref.
  Value intermediate_memref; // The memref to be eliminated.
  int fusion_benefit;        // Fusion benefit score for greedy selection.

  FusionCandidate(TaskInfo *writer, TaskInfo *reader, Value memref, int benefit)
      : memory_writer(writer), memory_reader(reader),
        intermediate_memref(memref), fusion_benefit(benefit) {}
};

//===----------------------------------------------------------------------===//
// Memory Dependency Graph Builder
//===----------------------------------------------------------------------===//

// Builds the memory dependency graph for all tasks in the function.
class MemoryDependencyAnalysis {
public:
  MemoryDependencyAnalysis(func::FuncOp func_op) : func_op(func_op) {}

  // Analyzes all tasks and builds the memory dependency graph.
  void analyze(DenseMap<Operation *, TaskInfo> &task_map) {
    // Collects all task operations.
    SmallVector<taskflow::TaskflowTaskOp> tasks;
    this->func_op.walk([&](taskflow::TaskflowTaskOp task_op) {
      tasks.push_back(task_op);
      task_map[task_op.getOperation()] = TaskInfo(task_op);
    });

    // Extracts memref accesses for each task.
    for (auto task_op : tasks) {
      auto &task_info = task_map[task_op.getOperation()];
      extractMemrefAccesses(task_op, task_info);
    }

    // Builds memory dependency edges.
    buildMemoryDependencies(tasks, task_map);
  }

private:
  // Extracts read and write memrefs from a task operation.
  void extractMemrefAccesses(taskflow::TaskflowTaskOp task_op,
                             TaskInfo &task_info) {
    // Extracts read memrefs from the task operands.
    for (Value memref : task_op.getWillReads()) {
      task_info.read_memrefs.push_back(memref);
    }

    // Extracts write memrefs from the task operands.
    for (Value memref : task_op.getWillWrites()) {
      task_info.write_memrefs.push_back(memref);
    }

    // Extracts value inputs (SSA producer-consumer dependencies).
    for (Value input : task_op.getValueInputs()) {
      task_info.value_inputs.push_back(input);
    }
  }

  // Builds memory dependency edges between tasks by tracing SSA value flow.
  //
  // The two sets of read/write memrefs serve distinct purposes:
  //   - read/write_memrefs: carry SSA values (write_outputs results from
  //     upstream tasks). These encode memory ordering (RAW, WAW, WAR deps).
  //   - original_read/write_memrefs: track the physical %alloc buffers.
  //     These are used later to identify intermediate memrefs to eliminate.
  //
  // This function builds the dependency graph using the SSA value flow:
  //   If Task B's read_memrefs or write_memrefs contains a value produced
  //   by Task A's write_outputs, then Task A → Task B is a dependency.
  //
  // Example:
  //   %alloc = memref.alloc()
  //   %wo_A = taskflow.task @A write_memrefs(%alloc) ...
  //   %wo_B = taskflow.task @B read_memrefs(%wo_A) write_memrefs(%wo_A) ...
  //   Here %wo_A in B's read_memrefs → RAW dep (A→B),
  //        %wo_A in B's write_memrefs → WAW dep (A→B).
  void buildMemoryDependencies(ArrayRef<taskflow::TaskflowTaskOp> tasks,
                               DenseMap<Operation *, TaskInfo> &task_map) {
    // Maps each write_outputs SSA value to the task that produces it.
    DenseMap<Value, TaskInfo *> write_output_to_producer;

    for (auto task_op : tasks) {
      auto &task_info = task_map[task_op.getOperation()];
      // Map read_outputs for WAR dependency tracking.
      for (Value ro : task_op.getDoneReads()) {
        write_output_to_producer[ro] = &task_info;
      }
      for (Value wo : task_op.getDoneWrites()) {
        write_output_to_producer[wo] = &task_info;
      }
    }

    // For each task, check if its read/write_memrefs consume another
    // task's write_outputs. If so, establish a dependency edge.
    for (auto task_op : tasks) {
      auto &task_info = task_map[task_op.getOperation()];
      DenseSet<TaskInfo *> seen_writers;

      auto addDependencyIfProduced = [&](Value operand) {
        auto it = write_output_to_producer.find(operand);
        if (it == write_output_to_producer.end()) {
          return;
        }
        TaskInfo *writer = it->second;
        if (writer == &task_info) {
          return; // No self-dependency.
        }
        if (!seen_writers.insert(writer).second) {
          return; // Deduplicate (same value may appear in both read and
                  // write).
        }
        task_info.memory_writers.push_back(writer);
        writer->memory_readers.push_back(&task_info);
      };

      // RAW: read_memrefs consuming a write_outputs value.
      for (Value operand : task_op.getWillReads()) {
        addDependencyIfProduced(operand);
      }

      // WAW/WAR: write_memrefs consuming a write_outputs value.
      for (Value operand : task_op.getWillWrites()) {
        addDependencyIfProduced(operand);
      }
    }
  }

  func::FuncOp func_op;
};

//===----------------------------------------------------------------------===//
// Fusion Candidate Identification
//===----------------------------------------------------------------------===//

// Identifies viable fusion candidates using greedy strategy.
class FusionCandidateIdentifier {
public:
  FusionCandidateIdentifier(DenseMap<Operation *, TaskInfo> &map)
      : task_map(map) {}

  // Identifies all valid fusion candidates and sorts them by benefit.
  SmallVector<FusionCandidate> identify() {
    SmallVector<FusionCandidate> candidates;

    // Iterates through all tasks to find fusion opportunities.
    for (auto &entry : task_map) {
      TaskInfo &task_info = entry.second;

      // Checks if this task has exactly one memory reader (fusion condition).
      if (task_info.memory_readers.size() == 1) {
        TaskInfo *reader = task_info.memory_readers[0];

        // Checks if fusion is valid.
        if (canFuse(&task_info, reader)) {
          // Finds the intermediate memref to eliminate.
          Value intermediate = findIntermediateMemref(&task_info, reader);
          if (intermediate) {
            int benefit = calculateFusionBenefit(&task_info, reader);
            candidates.emplace_back(&task_info, reader, intermediate, benefit);
          }
        }
      }
    }

    // Sorts candidates by fusion benefit (highest first) for greedy
    // selection.
    llvm::sort(candidates,
               [](const FusionCandidate &a, const FusionCandidate &b) {
                 return a.fusion_benefit > b.fusion_benefit;
               });

    return candidates;
  }

private:
  // Checks if two tasks can be fused.
  bool canFuse(TaskInfo *writer, TaskInfo *reader) {
    auto writer_op = writer->task_op;
    auto reader_op = reader->task_op;

    // 0. Writers with value_outputs are not fusable.
    //    The fused task only produces the reader's outputs, so the
    //    writer's value_outputs would be lost. A future enhancement
    //    could propagate them through the fused task.
    if (!writer_op.getValueOutputs().empty()) {
      llvm::errs() << "  canFuse: writer has value_outputs\n";
      return false;
    }

    // 1. Extracts loop nests from both tasks and checks bounds compatibility.
    auto writer_loops = extractOutermostLoopNest(writer_op);
    auto reader_loops = extractOutermostLoopNest(reader_op);

    if (writer_loops.empty() || reader_loops.empty()) {
      llvm::errs() << "  canFuse: empty loop nest\n";
      return false;
    }

    if (!areLoopBoundsCompatible(writer_loops, reader_loops)) {
      llvm::errs() << "  canFuse: incompatible loop bounds\n";
      return false;
    }

    // 2. Checks that the intermediate memref is not used outside
    //    writer/reader (e.g., not returned by the function).
    Value intermediate = findIntermediateMemref(writer, reader);
    if (!intermediate) {
      return false;
    }

    for (Operation *user : intermediate.getUsers()) {
      if (user == writer_op.getOperation() ||
          user == reader_op.getOperation()) {
        continue;
      }
      // Allow memref.alloc which defines it.
      if (isa<memref::AllocOp>(user)) {
        continue;
      }
      llvm::errs() << "  canFuse: intermediate memref has external use\n";
      return false;
    }

    // 3. No cyclic dependency: writer must not read any original memref
    //    that reader writes (excluding the intermediate itself).
    for (Value w_read : writer_op.getOriginalReadMemrefs()) {
      if (w_read == intermediate) {
        continue;
      }
      for (Value r_write : reader_op.getOriginalWriteMemrefs()) {
        if (r_write == intermediate) {
          continue;
        }
        if (w_read == r_write) {
          llvm::errs() << "  canFuse: cyclic dependency\n";
          return false;
        }
      }
    }

    return true;
  }

  // Extracts the outermost chain of perfectly nested affine.for loops
  // from a task body.
  SmallVector<affine::AffineForOp>
  extractOutermostLoopNest(taskflow::TaskflowTaskOp task_op) {
    SmallVector<affine::AffineForOp> loops;
    Block &body = task_op.getBody().front();

    // Finds the single outermost affine.for in the task body.
    affine::AffineForOp outermost = nullptr;
    for (Operation &op : body) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
        if (outermost) {
          return {}; // Multiple top-level loops: bail.
        }
        outermost = for_op;
      }
    }
    if (!outermost) {
      return {};
    }

    // Walks the perfectly nested chain.
    auto current = outermost;
    while (current) {
      loops.push_back(current);
      // Checks for a single nested affine.for.
      affine::AffineForOp nested = nullptr;
      for (Operation &op : current.getBody()->getOperations()) {
        if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
          if (nested) {
            nested = nullptr;
            break;
          } // Non-perfect: stop.
          nested = for_op;
        }
      }
      current = nested;
    }
    return loops;
  }

  // Checks if two loop nests have compatible bounds.
  bool areLoopBoundsCompatible(SmallVector<affine::AffineForOp> &writer_loops,
                               SmallVector<affine::AffineForOp> &reader_loops) {
    if (writer_loops.size() != reader_loops.size()) {
      return false;
    }
    for (size_t i = 0; i < writer_loops.size(); ++i) {
      if (!writer_loops[i].hasConstantBounds() ||
          !reader_loops[i].hasConstantBounds()) {
        return false;
      }
      if (writer_loops[i].getConstantLowerBound() !=
              reader_loops[i].getConstantLowerBound() ||
          writer_loops[i].getConstantUpperBound() !=
              reader_loops[i].getConstantUpperBound() ||
          writer_loops[i].getStepAsInt() != reader_loops[i].getStepAsInt()) {
        return false;
      }
    }
    return true;
  }

  // Finds the intermediate memref (original alloc) between writer and
  // reader. Uses original_write/read_memrefs since the IR passes SSA results
  // (not raw allocs) as read_memrefs.
  Value findIntermediateMemref(TaskInfo *writer, TaskInfo *reader) {
    auto writer_op = writer->task_op;
    auto reader_op = reader->task_op;
    for (Value orig_write : writer_op.getOriginalWriteMemrefs()) {
      for (Value orig_read : reader_op.getOriginalReadMemrefs()) {
        if (orig_write == orig_read) {
          return orig_write;
        }
      }
    }
    return nullptr;
  }

  // Calculates the fusion benefit score for greedy selection.
  int calculateFusionBenefit(TaskInfo *writer, TaskInfo *reader) {
    int benefit = 0;

    // Base benefit: eliminates one memref allocation.
    benefit += 100;

    // Bonus for element-wise operations (same loop bounds, same memref
    // shape).
    if (writer->write_memrefs.size() == 1 && reader->read_memrefs.size() == 1) {
      benefit += 50;
    }

    return benefit;
  }

  DenseMap<Operation *, TaskInfo> &task_map;
};

//===----------------------------------------------------------------------===//
// Task Fuser - Performs the actual fusion transformation
//===----------------------------------------------------------------------===//

// Fuses two tasks connected by a memory dependency into a single task.
class TaskFuser {
public:
  TaskFuser(func::FuncOp func) : function(func) {}

  // Performs fusion of a candidate pair. Returns true if fusion succeeded.
  bool performFusion(FusionCandidate &candidate) {
    auto writer_op = candidate.memory_writer->task_op;
    auto reader_op = candidate.memory_reader->task_op;
    Value intermediate = candidate.intermediate_memref;

    llvm::errs() << "  Fusing writer: " << writer_op.getTaskName()
                 << " + reader: " << reader_op.getTaskName() << "\n";

    // Step 1: Builds merged operand lists.
    SmallVector<Value> fused_read_memrefs;
    SmallVector<Value> fused_write_memrefs;
    SmallVector<Value> fused_value_inputs;
    SmallVector<Value> fused_original_read_memrefs;
    SmallVector<Value> fused_original_write_memrefs;

    buildMergedOperands(writer_op, reader_op, intermediate, fused_read_memrefs,
                        fused_write_memrefs, fused_value_inputs,
                        fused_original_read_memrefs,
                        fused_original_write_memrefs);

    // Step 2: Builds the result types (same as reader's outputs).
    // Writer's value_outputs are not included because canFuse rejects
    // writers with value_outputs.
    SmallVector<Type> read_output_types;
    SmallVector<Type> write_output_types;
    SmallVector<Type> value_output_types;
    // read_outputs corresponds to fused_reads (passthrough for WAR tracking).
    for (Value v : fused_read_memrefs) {
      read_output_types.push_back(v.getType());
    }
    for (Value v : reader_op.getDoneWrites()) {
      write_output_types.push_back(v.getType());
    }
    for (Value v : reader_op.getValueOutputs()) {
      value_output_types.push_back(v.getType());
    }

    // Step 3: Creates the fused task.
    OpBuilder builder(reader_op);
    std::string fused_name =
        (writer_op.getTaskName() + "_" + reader_op.getTaskName() + "_fused")
            .str();

    auto fused_task = builder.create<taskflow::TaskflowTaskOp>(
        writer_op.getLoc(), read_output_types, write_output_types,
        value_output_types, fused_read_memrefs, fused_write_memrefs,
        fused_value_inputs, fused_name, fused_original_read_memrefs,
        fused_original_write_memrefs);

    // Step 4: Builds fused task body by merging loop nests.
    if (!buildFusedBody(fused_task, writer_op, reader_op, intermediate,
                        fused_read_memrefs, fused_write_memrefs,
                        fused_value_inputs)) {
      fused_task.erase();
      return false;
    }

    // Step 5: Replaces uses and cleans up.
    replaceUsesAndCleanup(writer_op, reader_op, fused_task, intermediate);

    llvm::errs() << "  Fusion succeeded: " << fused_name << "\n";
    return true;
  }

private:
  // Builds merged operand lists for the fused task.
  // The intermediate is the original alloc value. We must use
  // original_read/write_memrefs to identify which operands to exclude
  // from the reader (since reader's read_memrefs contain SSA results).
  void buildMergedOperands(taskflow::TaskflowTaskOp writer_op,
                           taskflow::TaskflowTaskOp reader_op,
                           Value intermediate, SmallVector<Value> &fused_reads,
                           SmallVector<Value> &fused_writes,
                           SmallVector<Value> &fused_values,
                           SmallVector<Value> &fused_orig_reads,
                           SmallVector<Value> &fused_orig_writes) {

    // read_memrefs = writer.reads ∪ reader.reads - intermediate
    DenseSet<Value> seen;
    auto writer_reads = writer_op.getWillReads();
    auto writer_orig_reads = writer_op.getOriginalReadMemrefs();
    for (unsigned i = 0; i < writer_reads.size(); ++i) {
      Value orig = (i < writer_orig_reads.size()) ? writer_orig_reads[i]
                                                  : writer_reads[i];
      if (orig != intermediate && seen.insert(writer_reads[i]).second) {
        fused_reads.push_back(writer_reads[i]);
      }
    }

    auto reader_reads = reader_op.getWillReads();
    auto reader_orig_reads = reader_op.getOriginalReadMemrefs();
    for (unsigned i = 0; i < reader_reads.size(); ++i) {
      Value orig = (i < reader_orig_reads.size()) ? reader_orig_reads[i]
                                                  : reader_reads[i];
      if (orig != intermediate && seen.insert(reader_reads[i]).second) {
        fused_reads.push_back(reader_reads[i]);
      }
    }

    // write_memrefs = reader.writes ∪ (writer.writes - intermediate)
    seen.clear();
    auto reader_writes = reader_op.getWillWrites();
    auto reader_orig_writes = reader_op.getOriginalWriteMemrefs();
    for (unsigned i = 0; i < reader_writes.size(); ++i) {
      Value orig = (i < reader_orig_writes.size()) ? reader_orig_writes[i]
                                                   : reader_writes[i];
      if (orig != intermediate && seen.insert(reader_writes[i]).second) {
        fused_writes.push_back(reader_writes[i]);
      }
    }

    auto writer_writes = writer_op.getWillWrites();
    auto writer_orig_writes = writer_op.getOriginalWriteMemrefs();
    for (unsigned i = 0; i < writer_writes.size(); ++i) {
      Value orig = (i < writer_orig_writes.size()) ? writer_orig_writes[i]
                                                   : writer_writes[i];
      if (orig != intermediate && seen.insert(writer_writes[i]).second) {
        fused_writes.push_back(writer_writes[i]);
      }
    }

    // value_inputs = writer.values ∪ reader.values
    for (Value v : writer_op.getValueInputs()) {
      fused_values.push_back(v);
    }
    for (Value v : reader_op.getValueInputs()) {
      fused_values.push_back(v);
    }

    // original_read/write_memrefs: same merge rules (using originals
    // directly).
    seen.clear();
    for (Value v : writer_op.getOriginalReadMemrefs()) {
      if (v != intermediate && seen.insert(v).second) {
        fused_orig_reads.push_back(v);
      }
    }
    for (Value v : reader_op.getOriginalReadMemrefs()) {
      if (v != intermediate && seen.insert(v).second) {
        fused_orig_reads.push_back(v);
      }
    }

    seen.clear();
    for (Value v : reader_op.getOriginalWriteMemrefs()) {
      if (v != intermediate && seen.insert(v).second) {
        fused_orig_writes.push_back(v);
      }
    }
    for (Value v : writer_op.getOriginalWriteMemrefs()) {
      if (v != intermediate && seen.insert(v).second) {
        fused_orig_writes.push_back(v);
      }
    }
  }

  // Builds the fused task body by merging writer and reader loop nests.
  // Returns false if fusion fails (e.g., unexpected IR structure).
  bool buildFusedBody(taskflow::TaskflowTaskOp fused_task,
                      taskflow::TaskflowTaskOp writer_op,
                      taskflow::TaskflowTaskOp reader_op, Value intermediate,
                      ArrayRef<Value> fused_reads, ArrayRef<Value> fused_writes,
                      ArrayRef<Value> fused_values) {

    // Creates the entry block with all operands as block arguments.
    Block *fused_block = new Block();
    fused_task.getBody().push_back(fused_block);

    // Block args: read_memrefs, write_memrefs, value_inputs.
    for (Value v : fused_reads) {
      fused_block->addArgument(v.getType(), v.getLoc());
    }
    for (Value v : fused_writes) {
      fused_block->addArgument(v.getType(), v.getLoc());
    }
    for (Value v : fused_values) {
      fused_block->addArgument(v.getType(), v.getLoc());
    }

    // Builds a mapping from writer/reader block args to fused block args.
    IRMapping writer_mapping;
    IRMapping reader_mapping;

    Block &writer_body = writer_op.getBody().front();
    Block &reader_body = reader_op.getBody().front();

    // Maps writer's block args to fused block args.
    unsigned fused_arg_idx = 0;
    mapBlockArgs(writer_op, writer_body, fused_block, fused_arg_idx,
                 writer_mapping, intermediate, fused_reads, fused_writes,
                 fused_values);

    // Maps reader's block args to fused block args.
    mapBlockArgs(reader_op, reader_body, fused_block, fused_arg_idx,
                 reader_mapping, intermediate, fused_reads, fused_writes,
                 fused_values);

    // Clones the writer's loop nest into the fused body.
    OpBuilder body_builder(fused_block, fused_block->end());

    // Finds the writer's outermost affine.for.
    affine::AffineForOp writer_outer_loop = nullptr;
    for (Operation &op : writer_body) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
        writer_outer_loop = for_op;
        break;
      }
    }
    if (!writer_outer_loop) {
      return false;
    }

    // Finds the reader's outermost affine.for.
    affine::AffineForOp reader_outer_loop = nullptr;
    for (Operation &op : reader_body) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
        reader_outer_loop = for_op;
        break;
      }
    }
    if (!reader_outer_loop) {
      return false;
    }

    // Clones the writer's entire loop nest.
    Operation *cloned_writer =
        body_builder.clone(*writer_outer_loop, writer_mapping);
    auto cloned_writer_loop = cast<affine::AffineForOp>(cloned_writer);

    // Finds the innermost loop body in the cloned writer nest.
    affine::AffineForOp innermost_writer = cloned_writer_loop;
    while (true) {
      affine::AffineForOp nested = nullptr;
      for (Operation &op : innermost_writer.getBody()->getOperations()) {
        if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
          nested = for_op;
          break;
        }
      }
      if (!nested) {
        break;
      }
      innermost_writer = nested;
    }

    // Maps reader's loop induction variables to writer's (cloned) loop IVs.
    // This is valid because we verified bounds compatibility in canFuse.
    {
      auto writer_loops_chain = getLoopChain(cloned_writer_loop);
      auto reader_loops_chain = getLoopChain(reader_outer_loop);
      for (size_t i = 0;
           i < writer_loops_chain.size() && i < reader_loops_chain.size();
           ++i) {
        reader_mapping.map(reader_loops_chain[i].getInductionVar(),
                           writer_loops_chain[i].getInductionVar());
      }
    }

    // In the innermost writer loop body, finds affine.store to intermediate.
    // Maps: for each store to intermediate, the stored value becomes the
    // replacement for the corresponding load in the reader.
    // For now, handle the common case: single store to intermediate.
    Value store_value = nullptr;
    Operation *store_to_intermediate = nullptr;
    for (Operation &op : innermost_writer.getBody()->getOperations()) {
      if (auto store_op = dyn_cast<affine::AffineStoreOp>(op)) {
        // Checks if this store writes to the intermediate memref's
        // block arg (mapped from the writer's original arg).
        // The stored-to memref is the writer's block arg for the
        // intermediate.
        store_value = store_op.getValueToStore();
        store_to_intermediate = &op;
      }
    }

    if (!store_value) {
      llvm::errs() << "  No store to intermediate found in writer\n";
      return false;
    }

    // Now clones reader's innermost loop body ops into the writer's
    // innermost loop. For affine.load from intermediate, replaces with
    // the store_value (SSA direct connection).
    affine::AffineForOp reader_innermost = reader_outer_loop;
    while (true) {
      affine::AffineForOp nested = nullptr;
      for (Operation &op : reader_innermost.getBody()->getOperations()) {
        if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
          nested = for_op;
          break;
        }
      }
      if (!nested) {
        break;
      }
      reader_innermost = nested;
    }

    // Inserts reader's ops before the yield (or end) of the writer's
    // innermost loop body.
    OpBuilder inner_builder(innermost_writer.getBody(),
                            innermost_writer.getBody()->end());
    // Positions before the affine.yield terminator if it exists.
    if (!innermost_writer.getBody()->empty()) {
      Operation *terminator = innermost_writer.getBody()->getTerminator();
      if (terminator) {
        inner_builder.setInsertionPoint(terminator);
      }
    }

    for (Operation &op : reader_innermost.getBody()->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      if (auto load_op = dyn_cast<affine::AffineLoadOp>(op)) {
        // Checks if this load reads from the intermediate memref.
        // The reader's block arg corresponding to the intermediate.
        Value load_memref = load_op.getMemRef();
        bool is_intermediate_load = false;

        // Checks if this memref is a block arg that maps to intermediate.
        // Uses original_read/write_memrefs since reader's read_memrefs
        // contain SSA results, not the raw alloc.
        if (auto block_arg = dyn_cast<BlockArgument>(load_memref)) {
          unsigned arg_num = block_arg.getArgNumber();
          unsigned total_reads = reader_op.getWillReads().size();
          unsigned total_writes = reader_op.getWillWrites().size();

          if (arg_num < total_reads) {
            // Use original_read_memrefs to check against intermediate.
            Value orig_memref = reader_op.getOriginalReadMemrefs()[arg_num];
            if (orig_memref == intermediate)
              is_intermediate_load = true;
          } else if (arg_num < total_reads + total_writes) {
            Value orig_memref =
                reader_op.getOriginalWriteMemrefs()[arg_num - total_reads];
            if (orig_memref == intermediate)
              is_intermediate_load = true;
          }
        }

        if (is_intermediate_load) {
          // Replaces this load with the store_value (SSA streaming).
          reader_mapping.map(load_op.getResult(), store_value);
          continue; // Don't clone the load.
        }
      }

      // Clones the op with the reader mapping.
      inner_builder.clone(op, reader_mapping);
    }

    // Removes the store to intermediate (no longer needed).
    if (store_to_intermediate) {
      store_to_intermediate->erase();
    }

    // Step 5: Creates the yield for the fused task.
    // Yields the reader's output memrefs.
    // Remove any existing terminator (if the block already has one).
    if (fused_block->mightHaveTerminator()) {
      if (auto *yield_point = fused_block->getTerminator()) {
        yield_point->erase();
      }
    }

    OpBuilder yield_builder(fused_block, fused_block->end());
    SmallVector<Value> yield_reads;
    SmallVector<Value> yield_writes;
    SmallVector<Value> yield_values;

    // Finds the yield in the reader's original body for reference.
    auto reader_yield =
        cast<taskflow::TaskflowYieldOp>(reader_body.getTerminator());

    // Read yield outputs: passthrough the fused block's read_memref args.
    // These correspond to fused_reads (writer reads + reader reads -
    // intermediate).
    for (unsigned i = 0; i < fused_reads.size(); ++i) {
      yield_reads.push_back(fused_block->getArgument(i));
    }

    // Maps reader yield's memory results to fused block args.
    for (Value v : reader_yield.getDoneWrites()) {
      if (reader_mapping.contains(v)) {
        yield_writes.push_back(reader_mapping.lookup(v));
      } else {
        yield_writes.push_back(v);
      }
    }
    for (Value v : reader_yield.getValueResults()) {
      if (reader_mapping.contains(v)) {
        yield_values.push_back(reader_mapping.lookup(v));
      } else {
        yield_values.push_back(v);
      }
    }

    yield_builder.create<taskflow::TaskflowYieldOp>(
        reader_op.getLoc(), yield_reads, yield_writes, yield_values);

    return true;
  }

  // Maps a task's block args to the corresponding fused block args.
  void mapBlockArgs(taskflow::TaskflowTaskOp task_op, Block &original_body,
                    Block *fused_block, unsigned &fused_arg_idx,
                    IRMapping &mapping, Value intermediate,
                    ArrayRef<Value> fused_reads, ArrayRef<Value> fused_writes,
                    ArrayRef<Value> fused_values) {

    unsigned orig_arg_idx = 0;
    unsigned num_reads = task_op.getWillReads().size();
    unsigned num_writes = task_op.getWillWrites().size();
    unsigned num_values = task_op.getValueInputs().size();

    // Maps read_memrefs block args.
    // Uses original_read_memrefs to identify intermediate (since reader's
    // read_memrefs contain SSA results, not raw allocs).
    auto orig_reads = task_op.getOriginalReadMemrefs();
    for (unsigned i = 0; i < num_reads; ++i) {
      Value orig_memref = (i < orig_reads.size())
                              ? orig_reads[i]
                              : task_op.getWillReads()[i];
      if (orig_memref == intermediate) {
        // Intermediate memref — no corresponding fused arg. Skip.
        orig_arg_idx++;
        continue;
      }
      // Finds the fused block arg for this outer memref.
      Value outer_memref = task_op.getWillReads()[i];
      int fused_idx = findInFusedArgs(outer_memref, fused_reads, fused_writes,
                                      fused_values);
      if (fused_idx >= 0) {
        mapping.map(original_body.getArgument(orig_arg_idx),
                    fused_block->getArgument(fused_idx));
      }
      orig_arg_idx++;
    }

    // Maps write_memrefs block args.
    auto orig_writes = task_op.getOriginalWriteMemrefs();
    for (unsigned i = 0; i < num_writes; ++i) {
      Value orig_memref = (i < orig_writes.size())
                              ? orig_writes[i]
                              : task_op.getWillWrites()[i];
      if (orig_memref == intermediate) {
        orig_arg_idx++;
        continue;
      }
      Value outer_memref = task_op.getWillWrites()[i];
      int fused_idx = findInFusedArgs(outer_memref, fused_reads, fused_writes,
                                      fused_values);
      if (fused_idx >= 0) {
        mapping.map(original_body.getArgument(orig_arg_idx),
                    fused_block->getArgument(fused_idx));
      }
      orig_arg_idx++;
    }

    // Maps value_inputs block args.
    for (unsigned i = 0; i < num_values; ++i) {
      Value outer_value = task_op.getValueInputs()[i];
      int fused_idx =
          findInFusedArgs(outer_value, fused_reads, fused_writes, fused_values);
      if (fused_idx >= 0) {
        mapping.map(original_body.getArgument(orig_arg_idx),
                    fused_block->getArgument(fused_idx));
      }
      orig_arg_idx++;
    }
  }

  // Finds the index of an outer value in the fused block's argument list.
  // Returns -1 if not found.
  int findInFusedArgs(Value outer_val, ArrayRef<Value> fused_reads,
                      ArrayRef<Value> fused_writes,
                      ArrayRef<Value> fused_values) {
    unsigned idx = 0;
    for (Value v : fused_reads) {
      if (v == outer_val) {
        return idx;
      }
      idx++;
    }
    for (Value v : fused_writes) {
      if (v == outer_val) {
        return idx;
      }
      idx++;
    }
    for (Value v : fused_values) {
      if (v == outer_val) {
        return idx;
      }
      idx++;
    }
    return -1;
  }

  // Gets the chain of nested affine.for ops starting from the outermost.
  SmallVector<affine::AffineForOp> getLoopChain(affine::AffineForOp outermost) {
    SmallVector<affine::AffineForOp> chain;
    auto current = outermost;
    while (current) {
      chain.push_back(current);
      affine::AffineForOp nested = nullptr;
      for (Operation &op : current.getBody()->getOperations()) {
        if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
          nested = for_op;
          break;
        }
      }
      current = nested;
    }
    return chain;
  }

  // Replaces uses of original tasks' results with fused task results
  // and erases original ops.
  void replaceUsesAndCleanup(taskflow::TaskflowTaskOp writer_op,
                             taskflow::TaskflowTaskOp reader_op,
                             taskflow::TaskflowTaskOp fused_task,
                             Value intermediate) {

    // Helper: finds the index of an outer memref in fused_reads.
    auto findInFusedReads = [&](Value outer_memref) -> int {
      for (unsigned i = 0; i < fused_task.getWillReads().size(); ++i) {
        if (fused_task.getWillReads()[i] == outer_memref)
          return i;
      }
      return -1;
    };

    // Replaces writer's read_outputs: map each to the fused task's
    // corresponding read_output by finding the writer's read_memref
    // in the fused task's read_memrefs.
    for (unsigned i = 0; i < writer_op.getDoneReads().size(); ++i) {
      Value writer_read_input = writer_op.getWillReads()[i];
      int fused_idx = findInFusedReads(writer_read_input);
      if (fused_idx >= 0) {
        writer_op.getDoneReads()[i].replaceAllUsesWith(
            fused_task.getDoneReads()[fused_idx]);
      }
    }

    // Replaces reader's read_outputs: skip intermediate, map others.
    for (unsigned i = 0; i < reader_op.getDoneReads().size(); ++i) {
      Value orig = (i < reader_op.getOriginalReadMemrefs().size())
                       ? reader_op.getOriginalReadMemrefs()[i]
                       : reader_op.getWillReads()[i];
      if (orig == intermediate)
        continue; // Intermediate read_output is dead after fusion.
      Value reader_read_input = reader_op.getWillReads()[i];
      int fused_idx = findInFusedReads(reader_read_input);
      if (fused_idx >= 0) {
        reader_op.getDoneReads()[i].replaceAllUsesWith(
            fused_task.getDoneReads()[fused_idx]);
      }
    }

    // Replaces reader's write_outputs with fused task's write_outputs.
    for (unsigned i = 0; i < reader_op.getDoneWrites().size(); ++i) {
      reader_op.getDoneWrites()[i].replaceAllUsesWith(
          fused_task.getDoneWrites()[i]);
    }

    // Replaces reader's value_outputs with fused task's value_outputs.
    for (unsigned i = 0; i < reader_op.getValueOutputs().size(); ++i) {
      reader_op.getValueOutputs()[i].replaceAllUsesWith(
          fused_task.getValueOutputs()[i]);
    }

    // Erases original tasks (reader first since writer might be used by it
    // through the intermediate, but we've already replaced all uses).
    reader_op.erase();
    writer_op.erase();

    // Erases the intermediate memref allocation if it's now dead.
    if (auto alloc_op = intermediate.getDefiningOp<memref::AllocOp>()) {
      if (alloc_op.getResult().use_empty()) {
        alloc_op.erase();
      }
    }
  }

  func::FuncOp function;
};

//===----------------------------------------------------------------------===//
// Memory Access Streaming Fusion Pass
//===----------------------------------------------------------------------===//

struct MemoryAccessStreamingFusionPass
    : public PassWrapper<MemoryAccessStreamingFusionPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryAccessStreamingFusionPass)

  StringRef getArgument() const final {
    return "memory-access-streaming-fusion";
  }

  StringRef getDescription() const final {
    return "Memory Access Streaming Fusion";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::errs() << "[MemoryAccessStreamingFusion] Running "
                    "MemoryAccessStreamingFusion on function: "
                 << func.getName() << "\n";

    // Iterative fusion: re-analyze after each round to catch chains.
    // e.g., Task A → Task B → Task C: first round fuses A+B, second round
    // fuses (A+B)+C.
    unsigned total_fusions = 0;
    constexpr unsigned kMaxIterations = 100;

    for (unsigned iter = 0; iter < kMaxIterations; ++iter) {
      // Re-builds memory dependency graph from current IR state.
      DenseMap<Operation *, TaskInfo> task_map;
      MemoryDependencyAnalysis analysis(func);
      analysis.analyze(task_map);

      llvm::errs() << "[MemoryAccessStreamingFusion] Iteration " << iter
                   << ": Found " << task_map.size() << " tasks\n";

      // Identifies fusion candidates.
      FusionCandidateIdentifier identifier(task_map);
      auto candidates = identifier.identify();

      llvm::errs() << "[MemoryAccessStreamingFusion] Found "
                   << candidates.size() << " fusion candidates\n";

      if (candidates.empty()) {
        break;
      }

      // Performs greedy fusion for this round.
      DenseSet<Operation *> fused_tasks;
      TaskFuser fuser(func);
      unsigned round_fusions = 0;

      for (auto &candidate : candidates) {
        Operation *writer_op = candidate.memory_writer->task_op.getOperation();
        Operation *reader_op = candidate.memory_reader->task_op.getOperation();

        // Skips if either task was already consumed by a previous fusion
        // in this round.
        if (fused_tasks.count(writer_op) || fused_tasks.count(reader_op)) {
          continue;
        }

        llvm::errs() << "[MemoryAccessStreamingFusion] Attempting to fuse "
                        "tasks (benefit: "
                     << candidate.fusion_benefit << ")\n";

        if (fuser.performFusion(candidate)) {
          fused_tasks.insert(writer_op);
          fused_tasks.insert(reader_op);
          ++round_fusions;
        }
      }

      llvm::errs() << "[MemoryAccessStreamingFusion] Round " << iter
                   << ": fused " << round_fusions << " task pairs\n";

      total_fusions += round_fusions;

      // If no fusions happened this round, we've converged.
      if (round_fusions == 0) {
        break;
      }
    }

    llvm::errs() << "[MemoryAccessStreamingFusion] Total fusions: "
                 << total_fusions << "\n";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::taskflow::createMemoryAccessStreamingFusionPass() {
  return std::make_unique<MemoryAccessStreamingFusionPass>();
}
