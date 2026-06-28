#ifndef TASKFLOW_OPS_TD
#define TASKFLOW_OPS_TD

include "TaskflowDialect/TaskflowDialect.td"
include "TaskflowDialect/TaskflowTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

//----------------------------------------------------------------------
// Base Class for all Taskflow operations.
//----------------------------------------------------------------------
class TaskflowOpBase<string mnemonic, list<Trait> traits = []>
    : Op<TaskflowDialect, mnemonic, traits>;

//----------------------------------------------------------------------
// Task Level Operations.
//----------------------------------------------------------------------

// Defines a uniform computation task operation within a Taskflow graph.
def TaskflowTaskOp
    : TaskflowOpBase<
          "task", [IsolatedFromAbove, AutomaticAllocationScope,
                   AttrSizedOperandSegments, AttrSizedResultSegments,
                   SingleBlockImplicitTerminator<"TaskflowYieldOp">,
                   NativeOpTrait<"AffineScope">]> {
  let summary = "Computation task operation within a Taskflow graph.";

  let description = [{
    Represents a computational task that consumes and produces explicit
    dependence-state SSA values. Tasks are isolated from their surrounding scope
    and can only communicate through these explicit dependencies.

    Tasks has two types of inputs/outputs:
    1. Memory dependence states: memref SSA versions that the task reads or
       writes. The done_reads and done_writes results still refer to the same
       underlying memrefs, but represent the states after this task has observed
       or modified them.
    2. Value dependencies: SSA values from producer tasks

    The `original_read_memrefs` and `original_write_memrefs` operands record
    the actual original memrefs that this task accesses. These operands keep
    the memory identity separate from the dependence-state SSA value, enabling
    dependence analysis and data placement for multi-CGRA mapping.

    Example:
      // Memory state inputs: %mem_state, %out_state. Value input: %val.
      $out_mem, %out_val = taskflow.task "Task_0"
        will_reads(%mem_state : memref<4xi32>)
        will_writes(%out_state : memref<?xi32>)
        value_inputs(%val : i32)
        [original_read_memrefs(%arg0 : memref<?x8x6xi32>),
         original_write_memrefs(%arg5 : memref<?xi32>)] {
      ^bb0(%a0: memref<4xi32>, %a1: memref<?xi32>, %a2: i32):
        affine.for %i = 0 to 4 {
          %v = affine.load %a0[%i] : memref<4xi32>
          %sum = arith.addi %v, %a2 : i32
          affine.store %sum, %a1[%i] : memref<?xi32>
        }
        taskflow.yield done_reads(%a0 : memref<4xi32>) done_writes(%a1 : memref<?xi32>) values(%a2 : i32)
      } : (memref<4xi32>, memref<?xi32>, i32) -> (memref<4xi32>, memref<?xi32>, i32)
  }];

  let arguments = (ins Variadic<AnyMemRef>:$will_reads,
      Variadic<AnyMemRef>:$will_writes, Variadic<AnyType>:$value_inputs,
      StrAttr:$task_name, Variadic<AnyMemRef>:$original_read_memrefs,
      Variadic<AnyMemRef>:$original_write_memrefs);

  let results = (outs Variadic<AnyMemRef>:$done_reads,
      Variadic<AnyMemRef>:$done_writes,
      Variadic<AnyType>:$value_outputs);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

// Defines the yield operation to terminate a Taskflow task.
def TaskflowYieldOp
    : TaskflowOpBase<"yield", [Terminator, Pure, ReturnLike,
                               AttrSizedOperandSegments,
                               ParentOneOf<["TaskflowTaskOp"]>]> {
  let summary = "Yield operation for Taskflow task";
  let description = [{
    Yields dependence-state and value results from a task body. Read results
    are sparse: a task only needs to yield a read state when a later writer must
    depend on that read for WAR ordering. The number and types of operands must
    match the result types of the parent taskflow.task operation.
    
    Example:
      taskflow.task "Task_0" will_reads(%arg0 : memref<4xi32>) value_inputs(%arg1 : i32) {
        ...
        taskflow.yield done_reads(%a0 : memref<4xi32>)
      } : (memref<4xi32>, i32) -> memref<4xi32>
  }];

  let arguments = (ins Variadic<AnyMemRef>:$done_reads,
      Variadic<AnyMemRef>:$done_writes, Variadic<AnyType>:$value_results);

  let hasCustomAssemblyFormat = 1;

  let builders = [
      // Default builder for empty yield.
      OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{}, ValueRange{}, ValueRange{});
    }]>];
}

// Defines the data dependency edge operation that carries data dependencies
// between tasks in a Taskflow graph.
def TaskflowChannelOp
    : TaskflowOpBase<"channel", [Pure, SameOperandsAndResultType]> {
  let summary = "Data dependency edge that carries data dependencies between "
                "tasks in a Taskflow graph";
  let description = [{
    Represents a data dependency edge between tasks in the taskflow graph.
    A channel connects a producer task's output to a consumer task's input.

    Channels enforce explicit data dependencies and can be used for:
    - Producer-consumer relationships
    - Read-after-write (RAW) dependencies
    - Write-after-read (WAR) dependencies
    - Write-after-write (WAW) dependencies

    Example:
      %0 = taskflow.task "producer_task" (...) { ... } : (...) -> memref<4xi32>
      %1 = taskflow.channel %0 : memref<4xi32>
      %2 = taskflow.task "consumer_task" (%1, ...) { ... } : (memref<4xi32>, ...) -> ...
  }];

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$target);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($target)
  }];
}

//----------------------------------------------------------------------
// Intra-Task Operations.
//----------------------------------------------------------------------
// Counter operation representing loop iteration control within a Taskflow task.
def TaskflowCounterOp : TaskflowOpBase<"counter", []> {
  let summary = "Loop counter operation with hardware counter semantics";

  let description = [{
    Represents a hardware loop counter. Bounds and step are all SSA values,
    supporting both static and dynamic trip counts uniformly.

    Counter classification:
    - "root": Top-level counter with no parent (drives entire loop nest)
    - "relay": Intermediate counter with both parent and child counters
    - "leaf": Innermost counter with no child counters (maps to CGRA tile array)

    Example:
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c16 step %c1 : index
      %j = taskflow.counter parent(%i : index) from %c0 to %N step %c1 : index
  }];

  let arguments = (ins Optional<AnyType>:$parent_index,
      Index:$lower_bound,
      Index:$upper_bound,
      Index:$step,
      OptionalAttr<StrAttr>:$counter_hierarchy,
      OptionalAttr<StrAttr>:$counter_dynamism,
      OptionalAttr<I32Attr>:$counter_id);

  let results = (outs AnyType:$counter_index);

  let assemblyFormat = [{
    (`parent` `(` $parent_index^ `:` type($parent_index) `)`)?
    `from` $lower_bound `to` $upper_bound `step` $step
    attr-dict-with-keyword
    `:` type($counter_index)
  }];
}

def TaskflowHyperblockOp
    : TaskflowOpBase<"hyperblock", [AutomaticAllocationScope,
                                    AttrSizedOperandSegments,
                                    SingleBlockImplicitTerminator<
                                        "TaskflowHyperblockYieldOp">]> {
  let summary = "Hyperblock operation containing loop body computation";

  let description = [{
    Represents the loop body computation as a hyperblock controlled by taskflow.counter operation.
    The hyperblock takes the counter indices as input to trigger its execution.

    If the hyperblock has a return value, it must return the final value produced by the hyperblock (i.e., from the last iteration).

    Example:
      %result = taskflow.hyperblock indices(%i : index), iter_args(%init_val : i32) {
        ^bb0(%idx: index, %arg: i32):
          // Loop body computation using %idx and %arg
          ...
          taskflow.hyperblock.yield %output : i32
      } -> i32
  }];

  let arguments = (ins Variadic<AnyType>:$indices,
      Variadic<AnyType>:$iter_args);

  let results = (outs Variadic<AnyType>:$outputs);

  let regions = (region SizedRegion<1>:$body);

  // let assemblyFormat = [{
  //   (`indices` `(` $indices^ `:` type($indices) `)`)?
  //   attr-dict-with-keyword
  //   $body
  //   `->` `(` type($outputs) `)`
  // }];
}

def TaskflowHyperblockYieldOp
    : TaskflowOpBase<
          "hyperblock.yield", [Terminator, Pure, ReturnLike,
                               AttrSizedOperandSegments,
                               ParentOneOf<["TaskflowHyperblockOp"]>]> {
  let summary = "Yield operation for Taskflow hyperblock";

  let description = [{
    Terminates the hyperblock body.
  }];

  let arguments = (ins Variadic<AnyType>:$iter_args_next,
      Variadic<AnyType>:$results);

  let assemblyFormat = [{
    (`iter_args_next` `(` $iter_args_next^ `:` type($iter_args_next) `)`)?
    (`results` `(` $results^ `:` type($results) `)`)?
    attr-dict
  }];

  let builders = [OpBuilder<
      (ins), [{build($_builder, $_state, ValueRange{}, ValueRange{});}]>];
}

#endif // TASKFLOW_OPS_TD
