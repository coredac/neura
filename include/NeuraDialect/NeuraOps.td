// NeuraOps.td - Custom operation definitions.

include "NeuraDialect/NeuraDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// ----------------------------------------------------
// Defines neura kernel related operations.
// ----------------------------------------------------

def Neura_KernelOp : Op<NeuraDialect, "kernel",
                        [IsolatedFromAbove, RecursiveMemoryEffects,
                         AttrSizedOperandSegments]> {
  let summary = "Marks a region for CGRA execution.";
  let description = [{
    Defines a computation region that should be offloaded to CGRA.
    The region contains operations that will be lowered to neura dataflow IR.
    Operations outside this region remain on CPU.

    The kernel captures values from the surrounding context through explicit
    operands via `ins`. Results are returned via `neura.yield`.

    Example with return value (reduction pattern):
      %result = neura.kernel ins(%input, %coeff : memref<?xf32>, memref<?xf32>) {
        %sum = scf.for ... iter_args(%acc = %init) -> (f32) {
          ...
          scf.yield %new_acc : f32
        }
        neura.yield %sum : f32
      } : f32
      memref.store %result, %output[%idx] : memref<?xf32>

    Example without return value:
      neura.kernel ins(%input, %output : memref<?xf32>, memref<?xf32>) {
        affine.for %i = 0 to 128 {
          %v = affine.load %input[%i] : memref<?xf32>
          affine.store %v, %output[%i] : memref<?xf32>
        }
      }
  }];

  let arguments = (ins Variadic<AnyType>:$inputs, // Input operands from
                                                  // surrounding context.
      Variadic<AnyType>:$iter_args_init,  // Initial values for loop carried
                                          // variables.
      OptionalAttr<I32Attr>:$cgra_id,     // Target CGRA ID (for multi-CGRA
                                          // systems).
      OptionalAttr<StrAttr>:$kernel_name, // Name of the kernel (for
                                          // identification).
      OptionalAttr<StrAttr>:$accelerator  // Target accelerator name.
  );

  let results = (outs Variadic<AnyType>:$outputs);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`inputs` `(` $inputs^ `:` type($inputs) `)` )?
    (`iter_args_init` `(` $iter_args_init^ `:` type($iter_args_init) `)` )?
    attr-dict-with-keyword
    $body
    (`:` type($outputs)^)?
  }];
}

// Yield operation for fused_op and kernel regions.
def Neura_YieldOp
    : Op<NeuraDialect,
         "yield", [Terminator, Pure, ReturnLike, AttrSizedOperandSegments]> {
  let summary = "Yield values from a neura.kernel or neura.fused_op region.";
  let description = [{
    Returns values from a neura.kernel or neura.fused_op region to the parent operation.
    This operation must be the terminator of a region when the parent operation is a neura.kernel or neura.fused_op.

    Example:
      %result = neura.kernel ins(...) {
        %v = ... // computations
        neura.yield %v : f32
      } : f32
  }];

  let arguments = (ins Variadic<AnyType>:$iter_args_next,
      Variadic<AnyType>:$results);

  let builders = [OpBuilder<
      (ins), [{ build($_builder, $_state, ValueRange{}, ValueRange{}); }]>];

  let assemblyFormat = [{
  (`iter_args_next` `(` $iter_args_next^ `:` type($iter_args_next) `)` )?
  (`results` `(` $results^ `:` type($results) `)` )? attr-dict}];

  let hasVerifier = 1;
}

// ----------------------------------------------------
// Defines basic scalar operations.
// ----------------------------------------------------

def Neura_ConstantOp : Op<NeuraDialect, "constant"> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "attr-dict `:` type($result)";
}

// Defines an addition operation.
def Neura_AddOp : Op<NeuraDialect, "add"> {
  let summary = "Integer addition operation";
  let opName = "add";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_SubOp : Op<NeuraDialect, "sub"> {
  let summary = "Integer substraction operation";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_MulOp : Op<NeuraDialect, "mul"> {
  let summary = "Integer multiplication operation";
  let opName = "mul";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_DivOp : Op<NeuraDialect, "div"> {
  let summary = "Integer division operation";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_RemOp : Op<NeuraDialect, "rem"> {
  let summary = "Integer remainder operation";
  let description = [{
    Performs an integer remainder operation, computing the result of
    a % b, where % is the remainder operator.

    Example:
      %result = neura.rem %a, %b : i32
  }];
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point addition operation.
def Neura_FAddOp : Op<NeuraDialect, "fadd"> {
  let summary = "Floating addition operation";
  let opName = "fadd";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:`
  // type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point substraction operation.
def Neura_FSubOp : Op<NeuraDialect, "fsub"> {
  let summary = "Floating substraction operation";
  let opName = "fsub";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point negation operation.
def Neura_FNegOp : Op<NeuraDialect, "fneg"> {
  let summary = "Floating negation operation";
  let opName = "fneg";
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point multiplication operation.
def Neura_FMulOp : Op<NeuraDialect, "fmul"> {
  let summary = "Floating multiplication operation";
  let opName = "fmul";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  // let traits = [SameOperandsAndResultElementType];
}

def Neura_FDivOp : Op<NeuraDialect, "fdiv"> {
  let summary = "Floating division operation";
  let description = [{
    Performs a floating-point division operation, computing the result of
    a / b, where / is the floating-point division operator.

    Example:
      %result = neura.fdiv %a, %b : f32
  }];
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point maximum operation.
def Neura_FMaxOp : Op<NeuraDialect, "fmax"> {
  let summary = "Floating-point maximum operation";
  let description = [{
    Returns the maximum of two floating-point values.
    Supports two NaN propagation semantics:
    - "maxnum": Returns non-NaN value when one operand is NaN (llvm.maxnum)
    - "maximum": Propagates NaN when any operand is NaN (llvm.maximum)
    
    Example:
      %result = neura.fmax<"maxnum">(%a, %b : f32) : f32 -> f32
      %result = neura.fmax<"maximum">(%a, %b : f32) : f32 -> f32
  }];
  let opName = "fmax";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs,
      DefaultValuedAttr<StrAttr, "\"maxnum\"">:$nan_semantic);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
  let assemblyFormat =
      "`<` $nan_semantic `>` `(` $lhs (`,` $rhs^ `:` type($rhs))? `)` "
      "attr-dict `:` type($lhs) `->` type($result)";
}

// Defines a floating-point minimum operation.
def Neura_FMinOp : Op<NeuraDialect, "fmin"> {
  let summary = "Floating-point minimum operation";
  let description = [{
    Returns the minimum of two floating-point values.
    Supports two NaN propagation semantics:
    - "minnum": Returns non-NaN value when one operand is NaN (llvm.minnum)
    - "minimum": Propagates NaN when any operand is NaN (llvm.minimum)
    
    Example:
      %result = neura.fmin<"minnum">(%a, %b : f32) : f32 -> f32
      %result = neura.fmin<"minimum">(%a, %b : f32) : f32 -> f32
  }];
  let opName = "fmin";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs,
      DefaultValuedAttr<StrAttr, "\"minnum\"">:$nan_semantic);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
  let assemblyFormat =
      "`<` $nan_semantic `>` `(` $lhs (`,` $rhs^ `:` type($rhs))? `)` "
      "attr-dict `:` type($lhs) `->` type($result)";
}

// Defines a bitwise OR operation.
def Neura_AndOp : Op<NeuraDialect, "and"> {
  let summary = "Bitwise AND operation";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_OrOp : Op<NeuraDialect, "or"> {
  let summary = "Bitwise OR operation";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines an integer compare operation.
def Neura_ICmpOp : Op<NeuraDialect, "icmp"> {
  let summary = "Integer compare operation";
  let opName = "icmp";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs, StrAttr:$cmpType);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $cmpTypeAttr `,` $cmp_type
  // attr-dict `:` type($result)"; let traits =
  // [SameOperandsAndResultElementType];
}

// Defines a floating-point compare operation.
def Neura_FCmpOp : Op<NeuraDialect, "fcmp"> {
  let summary = "Floating-point compare operation";
  let opName = "fcmp";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs, StrAttr:$cmpType);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $cmpType attr-dict `:`
  // type($result)"; let traits = [SameOperandsAndResultElementType];
}

// Defines a load operation.
def Neura_LoadOp : Op<NeuraDialect, "load"> {
  let arguments = (ins AnyType:$addr);
  let results = (outs AnyType:$value);
  // let assemblyFormat = "$addr `,` attr-dict `:` type($value)";
}

// Defines a store operation.
def Neura_StoreOp : Op<NeuraDialect, "store"> {
  let arguments = (ins AnyType:$value, Optional<AnyType>:$addr);
  let results = (outs);
  // let assemblyFormat = "$value `,` $addr `,` attr-dict";
}

// Defines a memset operation for setting a block of memory.
def Neura_MemsetOp : Op<NeuraDialect, "memset"> {
  let summary = "Memset operation that sets a block of memory to a value";
  let description = [{
    Sets len bytes of memory starting at dest to the given value.
    This is a semantic operation that RTL can lower to appropriate hardware implementation.
    Example:
      neura.memset %dest, %value, %len {is_volatile = false} : !neura.data<ptr, i1>, i8, i64
  }];
  let arguments = (ins AnyType:$dest, AnyType:$value, AnyType:$len,
      BoolAttr:$is_volatile);
  let results = (outs);
  // let assemblyFormat = "$dest `,` $value `,` $len attr-dict `:` type($dest)
  // `,` type($value) `,` type($len)";
}

// Defines a load operation with integrated address calculation.
def Neura_LoadIndexedOp
    : Op<NeuraDialect, "load_indexed", [AttrSizedOperandSegments]> {
  let summary =
      "Load with integrated address calculation for multi-dimensional arrays";
  let description = [{
    Calculates the address using the base address and indices.
    Load the value at the calculated address.
    Example:
      %value = neura.load_indexed %base [%arg1, %arg2] : f32
  }];
  let arguments = (ins Optional<AnyType>:$base, Variadic<AnyType>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base `[` $indices `:` type($indices) `]` type($base) "
                       "attr-dict `:` type($result)";
}

// Defines a store operation with integrated address calculation.
def Neura_StoreIndexedOp
    : Op<NeuraDialect, "store_indexed", [AttrSizedOperandSegments]> {
  let summary =
      "Store with integrated address calculation for multi-dimensional arrays";
  let description = [{
    Calculates the address using the base address and indices.
    Store the value at the calculated address.
    Example:
      neura.store_indexed %value, %base [%arg1, %arg2] : f32
  }];
  let arguments = (ins AnyType:$value, Optional<AnyType>:$base,
      Variadic<AnyType>:$indices);
  let results = (outs);
  let assemblyFormat = "$value `to` $base `[` $indices `:` type($indices) `]` "
                       "type($base) attr-dict `:` type($value)";
}

// Defines a pointer computation operation.
def Neura_GEP : Op<NeuraDialect, "gep", [AttrSizedOperandSegments]> {
  let summary = "Pointer computation using offset indices";
  let arguments = (ins Optional<AnyType>:$base, Variadic<AnyType>:$indices);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$base `[` $indices `]` `,` $predicate attr-dict";
}

// Defines a conditional branch operation.
def Neura_CondBr
    : Op<NeuraDialect, "cond_br", [Terminator, AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$condition, Variadic<AnyType>:$trueArgs,
      Variadic<AnyType>:$falseArgs);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$condition `:` type($condition) `then` ($trueArgs^ `:` "
                       "type($trueArgs))? `to` $trueDest `else` ($falseArgs^ "
                       "`:` type($falseArgs))? `to` $falseDest attr-dict";
}

// Defines an unconditional branch operation.
def Neura_Br : Op<NeuraDialect, "br", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "($args^ `:` type($args))? `to` $dest attr-dict";
}

def Neura_SelOp : Op<NeuraDialect, "sel"> {
  let arguments = (ins AnyType:$cond, AnyType:$ifTrue, AnyType:$ifFalse);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$cond `,` $ifTrue `,` $ifFalse attr-dict `:`
  // type($result)";
}

def Neura_NotOp : Op<NeuraDialect, "not"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  // let assemblyFormat = "$input attr-dict `:` type($output)";
}

// Defines a return operation.
def Neura_ReturnOp : Op<NeuraDialect, "return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  // let assemblyFormat = "($values^)? `,` attr-dict";
}

// Defines a return operation for void functions in dataflow mode.
def Neura_ReturnVoidOp : Op<NeuraDialect, "return_void", []> {
  let summary = "Return operation for void functions in dataflow mode.";
  let description = [{
    Terminates a void function. The trigger operand provides the execution
    condition in dataflow mode - the return executes when the trigger becomes valid.
    
    Without a trigger (empty operands), this represents a static return that
    always executes. After canonicalization, void returns should have triggers.

    Example:
      neura.return_void %cond : !neura.data<i1, i1>  // In dataflow mode
      ...
      neura.yield                       // Function ends here.

    This is NOT a terminator - the block ends with neura.yield instead.
  }];
  let arguments = (ins Optional<AnyType>:$trigger);
  let assemblyFormat = "($trigger^ `:` type($trigger))? attr-dict";
}

// Defines a return operation for non-void fnctions in dataflow mode.
def Neura_ReturnValueOp : Op<NeuraDialect, "return_value", []> {
  let summary = "Return operation for non-void functions in dataflow mode.";
  let description = [{
    Represents a return point for non-void functions in dataflow mode.
    The value operand provides the return value and the trigger operand.

    This is NOT a terminator - the block ends with neura.yield instead.

    Example:
      neura.return_value %ret_val : !neura.data<i32, i1> // In dataflow mode
      ...
      neura.yield                       // Function ends here.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

// Defines a cast operation for type conversion.
def Neura_CastOp : Op<NeuraDialect, "cast"> {
  let summary = "Generic type conversion operation";
  let arguments = (ins AnyType:$input, StrAttr:$cast_type);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$input type($input) `->` type($output) `,`
  // attr-dict";
}

// Defines an alloca operation for memory allocation.
def Neura_AllocaOp : Op<NeuraDialect, "alloca"> {
  let summary = "Memory allocation operation";
  let description = [{
    Allocates memory on the stack, similar to llvm.alloca.
    Takes a predicated size value and returns a pointer to the allocated memory.

    Example:
      %ptr = neura.alloca %size : !neura.data<i32, i1> -> !llvm.ptr
  }];

  let arguments = (ins Optional<AnyType>:$size);
  let results = (outs AnyType:$result);
  let assemblyFormat = "($size^ `:` type($size))? attr-dict `->` type($result)";
}

// Defines a sign extension operation.
def Neura_SExtOp : Op<NeuraDialect, "sext"> {
  let summary = "Sign extension operation";
  let description = [{
    Sign extends a value from a smaller integer type to a larger integer type.
    Similar to llvm.sext, but works with predicated values.

    Example:
      %extended = neura.sext %value : !neura.data<i8, i1> -> !neura.data<i32, i1>
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

// Defines a zero extension operation.
def Neura_ZExtOp : Op<NeuraDialect, "zext"> {
  let summary = "Zero extension operation";
  let description = [{
    Zero extends a value from a smaller integer type to a larger integer type.
    Similar to llvm.zext, but works with predicated values.

    Example:
      %extended = neura.zext %value : !neura.data<i8, i1> -> !neura.data<i32, i1>
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

// Defines a logical shift left operation.
def Neura_ShlOp : Op<NeuraDialect, "shl"> {
  let summary = "Logical shift left operation";
  let description = [{
    Performs a logical left shift on an integer value.
    Similar to llvm.shl, but works with predicated values.

    Example:
      %shifted = neura.shl %value, %shiftAmount : !neura.data<i32, i1> -> !neura.data<i32, i1>
  }];
  let arguments = (ins AnyType:$value, Optional<AnyType>:$shiftAmount);
  let results = (outs AnyType:$result);
}

// ----------------------------------------------------
// Defines vector operations.
// ----------------------------------------------------

// Defines a vector multiplication operation.
def Neura_VFMulOp : Op<NeuraDialect, "vfmul"> {
  let summary = "Vector floating multiplication operation";
  let opName = "vfmul";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a vector integer multiplication operation.
def Neura_VMulOp : Op<NeuraDialect, "vmul"> {
  let summary = "Vector integer multiplication operation";
  let opName = "vmul";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
}

// Defines a vector integer addition operation.
def Neura_VAddOp : Op<NeuraDialect, "vadd"> {
  let summary = "Vector integer addition operation";
  let opName = "vadd";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
}

// Defines a vector floating addition operation.
def Neura_VFAddOp : Op<NeuraDialect, "vfadd"> {
  let summary = "Vector floating addition operation";
  let opName = "vfadd";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
}

// Defines a vector reduction add operation (reduces vector to scalar).
def Neura_VectorReduceAddOp : Op<NeuraDialect, "vector.reduce.add"> {
  let summary = "Vector reduction add operation - reduces vector to scalar by "
                "summing elements";
  let description = [{
    Reduces a vector to a scalar by summing all elements.
    Similar to llvm.intr.vector.reduce.add.
    
    Example:
      %sum = neura.vector.reduce.add %vec : vector<4xi32> -> i32
  }];
  let opName = "vector.reduce.add";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

// ----------------------------------------------------
// Defines fused operations.
// ----------------------------------------------------

def Neura_FAddFAddOp : Op<NeuraDialect, "fadd_fadd"> {
  let summary = "Fused fadd(fadd(a, b), c)";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_FMulFAddOp : Op<NeuraDialect, "fmul_fadd"> {
  let summary = "Fused fadd(fmul(a, b), c)";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_MulAddOp : Op<NeuraDialect, "mul_add"> {
  let summary = "Fused add(mul(a, b), c)";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c,
      Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` $predicate attr-dict `:`
  // type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// ----------------------------------------------------
// Defines move operations.
def Neura_DataMovOp : Op<NeuraDialect, "data_mov"> {
  let summary = "Data movement operation";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  // let assemblyFormat = "$input attr-dict `:` type($input) `->`
  // type($output)";
}

// ----------------------------------------------------
// Defines ctrl-related operations.
// ----------------------------------------------------

// Defines phi operation for merging values in dataflow form.
def Neura_PhiOp : Op<NeuraDialect, "phi"> {
  let summary = "Phi node in dataflow form.";
  let description = [{
    Merges values from different control paths in dataflow form.
    Used with reserve and ctrl_mov to represent control flow.

    Example:
      %v = neura.reserve : f32           // Create placeholder
      %result = neura.phi %init, %v      // Merge initial and loop-carried values
      neura.ctrl_mov %next to %v         // Connect next iteration
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);

  // Explicitly specifies types for operands in the assembly format.
  // let assemblyFormat = "$init_val `:` type($init_val) `,` $loop_val `:`
  // type($loop_val) attr-dict `,` type($result)";
}

// Defines phi_start operation specifically for phi operations with
// neura.reserve as an operand.
def Neura_PhiStartOp : Op<NeuraDialect, "phi_start"> {
  let summary = "Phi start operation for neura.reserve operand.";
  let description = [{
    Special phi operation for loop initialization where the first operand
    must be initial values and subsequent operands
    are the reserved or other values from different control paths.
    
    This operation distinguishes loop initialization patterns from general
    phi merges, making it easier to identify and optimize recurrence cycles.
    
    The first operand (reserve) represents the placeholder for the loop-carried
    value, while other operands provide the initial values for the first iteration.
    
    Example:
      %reserved = neura.reserve : f32
      %init = neura.grant_once ... : f32
      %result = neura.phi_start %init, %reserved : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
      // ... loop body uses %result ...
      neura.ctrl_mov %next_val -> %reserved
  }];

  let arguments = (ins AnyType:$init_value, AnyType:$reserved);
  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$init_value `,` $reserved attr-dict `:` type($init_value) `,` "
      "type($reserved) `->` type($result)";

  let hasVerifier = 1;
}

// Defines control movement extending base move but with different signature.
def Neura_CtrlMovOp : Op<NeuraDialect, "ctrl_mov"> {
  let summary = "Control movement operation";
  let description = [{
    Connects a value to a reserved placeholder in the dataflow.
    Used to establish control flow dependencies.

    Example:
      ctrl_mov %value to %placeholder : f32  // Connect value to placeholder
  }];

  // Adds type constraints for both operands.
  let arguments = (ins AnyType:$value, AnyType:$target);
  let results = (outs);

  // Corrects assembly format - types must be space-separated.
  let assemblyFormat =
      "$value `->` $target attr-dict `:` type($value) type($target)";
}

// Defines reserve operation for control flow values.
def Neura_ReserveOp : Op<NeuraDialect, "reserve"> {
  let summary = "Creates a placeholder for control flow values";
  let description = [{
    Creates a placeholder value that will be connected via ctrl_mov.
    Used to represent control flow dependencies in dataflow form.

    Example:
      %v = neura.reserve : f32      // Create placeholder
      %result = neura.phi %init, %v // Use in phi node
      neura.ctrl_mov %next to %v    // Connect value
  }];

  let arguments = (ins);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Neura_GrantPredicateOp : Op<NeuraDialect, "grant_predicate"> {
  let summary = "Grants a new predicate to a predicated value.";
  let description = [{
    Takes a predicated value and a predicate (i1), producing a new predicated
    value whose predicate bit is set to the given condition.

    Example:
      %g = neura.grant_predicate %val, %pred : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  }];

  let arguments = (ins AnyType:$value, AnyType:$predicate);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value `,` $predicate attr-dict `:` type($value) `,` "
                       "type($predicate) `->` type($result)";
}

def Neura_GrantOnceOp : Op<NeuraDialect, "grant_once"> {
  let summary = "Marks a value as valid once.";
  let description = [{
    Either grants a value a one-time predicate, or creates a constant value with a one-time predicate. The resulting value is considered valid
    only during its first activation. Used to initialize recurrence cycles.

    Example:
      %v = neura.grant_once %init : !neura.data<f32, i1> -> !neura.data<f32, i1>
      %c = neura.grant_once <{value = 42 : i64}> : !neura.data<i64, i1>
  }];

  let arguments = (ins Optional<AnyType>:$value,
      OptionalAttr<AnyAttr>:$constant_value);
  let results = (outs AnyType:$result);

  // let assemblyFormat = "$value attr-dict `:` type($value) `->`
  // type($result)";
}

def Neura_GrantAlwaysOp : Op<NeuraDialect, "grant_always"> {
  let summary = "Marks a value as valid always.";
  let description = [{
    Grants a value always-valid predicate: the resulting value is considered valid
    during the entire application lifetime.

    Example:
      %v = neura.grant_always %init : !neura.data<f32, i1> -> !neura.data<f32, i1>
  }];

  let arguments = (ins Optional<AnyType>:$value,
      OptionalAttr<AnyAttr>:$constant_value);
  let results = (outs AnyType:$result);

  // let assemblyFormat = "$value attr-dict `:` type($value) `->`
  // type($result)";
}

// ----------------------------------------------------
// Defines fused control flow operations.
// ----------------------------------------------------

def Neura_LoopControlOp : Op<NeuraDialect, "loop_control"> {
  let summary = "Generates loop indicies and valid predicates.";
  let description = [{
    Controls loop execution with minimal recurrence cycle length (1).
    Takes the current index and produces the next index and validity predicate.

    This operation combines comparison and increment logic into a single operation,
    enabling efficient loop execution on dataflow architectures with modulo scheduling.

    The first iteration automatically uses the start value because the reserve operation's
    predicate bit is initially 0, while the start value's predicate bit is 1.

    Example:
      // Shows loop control that calculates next index and validity in one step.
      %next_idx, %loop_valid = neura.loop_control(
        parent_valid = %parent_valid,
      ) <{
           start_value = 0 : i64, 
           end_value = 100 : i64, 
           step_value = 1 : i64,
           iteration_type = "increment"
         }> : !neura.data<i1, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
  }];

  let arguments =
      (ins AnyType:$parentValid,  // Valid predicate from the parent loop.
          AnyAttr:$iterationType, // "increment" or "decrement".
          AnyAttr:$start,         // Start index value.
          AnyAttr:$end,           // End index value.
          AnyAttr:$step           // Step value (increment or decrement based on
                                  // iteration_type).
      );

  let results = (outs AnyType:$nextindex, // Current loop index
      AnyType:$valid // Valid predicate for the current index
  );

  // let assemblyFormat =
  //   " `(``parent_valid` `=` $parentValid `,` `start` `=` $start `,` `end` `=`
  //   $end `,` `step` `=` $step`)` attr-dict `:` type($parentValid) `,`
  //   type($start) `,` type($end) `,` type($step) `->` type($nextindex) `,`
  //   type($valid)";
}

// Defines an operation for hardware loop counters.
def Neura_CounterOp : Op<NeuraDialect, "counter", [Pure, AttrSizedOperandSegments]> {
  let summary = "Hardware loop counter for CGRA execution.";
  let description = [{
    Represents a hardware loop counter unit that generates loop indices.
    This maps directly to a counter FU on the CGRA.

    Bounds and step are SSA Values (Index type), supporting both static
    and dynamic trip counts uniformly. Static bounds are typically
    arith.constant ops; dynamic bounds come from kernel inputs.

    The counter produces:
    - current_index: the current loop index value.

    Example (static):
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %idx = neura.counter from %c0 to %c32 step %c1
               {counter_hierarchy = "leaf", counter_dynamism = "constant_bound", counter_id = 0 : i32} : index

    Example (dynamic upper bound from kernel input):
      %idx = neura.counter from %c0 to %N step %c1
               {counter_hierarchy = "leaf", counter_dynamism = "symbol_dynamic", counter_id = 0 : i32} : index
  }];
  let arguments = (ins Optional<AnyType>:$lower_bound, Optional<AnyType>:$upper_bound,
      Optional<AnyType>:$step, OptionalAttr<StrAttr>:$counter_hierarchy,
      OptionalAttr<StrAttr>:$counter_dynamism, I32Attr:$counter_id);

  let results = (outs AnyType:$current_index);
  let assemblyFormat = [{
    (`from` $lower_bound^ `:` type($lower_bound))?
    (`to` $upper_bound^ `:` type($upper_bound))?
    (`step` $step^ `:` type($step))?
    attr-dict-with-keyword
    `->` type($current_index)
  }];
}

// Defines an operation to extract the predicate bit from a predicated value.
def Neura_ExtractPredicateOp : Op<NeuraDialect, "extract_predicate"> {
  let summary = "Extracts the predicate bit from a predicated value.";
  let description = [{
    Extracts the predicate bit from a predicated value,
    producing a boolean predicated value: !neura.predicated<i1, i1>.

    Example:
      %counter = neura.counter {bound = 16} : !neura.predicated<index, i1>
      %is_valid = neura.extract_predicate %counter : !neura.predicated<index, i1> -> !neura.predicated<i1, i1>
      
      // Use for gating final results:
      %gated = neura.grant_predicate %result, %is_valid
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$predicate);
  let assemblyFormat =
      "$input attr-dict `:` type($input) `->` type($predicate)";
}

// ----------------------------------------------------
// Defines operations for steering-control based DFG execution.
// ----------------------------------------------------

// Defines the true_steer operation.
def Neura_TrueSteerOp : Op<NeuraDialect, "true_steer"> {
  let summary = "Conditionally pass a value when condition is true.";
  let description =
      [{When the condition is true, the input value is passed to the output; otherwise, the output is empty.
    Example:
      %out = neura.true_steer %in, %cond : i32, i1 -> f32
  }];

  let arguments = (ins AnyType:$input, AnyType:$condition);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$input `,` $condition attr-dict `:` type($input) `,` "
                       "type($condition) `->` type($output)";
}

// Defines the false_steer operation.
def Neura_FalseSteerOp : Op<NeuraDialect, "false_steer"> {
  let summary = "Conditionally pass a value when condition is false.";
  let description =
      [{When the condition is false, the input value is passed to the output; otherwise, the output is empty.
    Example:
      %out = neura.false_steer %in, %cond : i32, i1 -> f32
  }];

  let arguments = (ins AnyType:$input, AnyType:$condition);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$input `,` $condition attr-dict `:` type($input) `,` "
                       "type($condition) `->` type($output)";
}

// Defines the carry operation.
def Neura_CarryOp : Op<NeuraDialect, "carry"> {
  let summary = "Carry state across iterations.";
  let description = [{
  Three inputs for carry operation:
    - initial value: used in the first execution.
    - condition: determines whether to use the carried value.
    - carried value: used when condition is true.
  The output is the initial value when it is executed for the first time, otherwise it is the carried value when the condition is true.
  Example:
    %out = neura.carry %init, %cond, %carry_val : i32, i1, i32 -> i32
  }];

  let arguments = (ins AnyType:$initial, AnyType:$condition, AnyType:$carried);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$initial `,` $condition `,` $carried attr-dict `:` type($initial) `,` "
      "type($condition) `,` type($carried) `->` type($result)";
}

// Defines the merge operation.
def Neura_MergeOp : Op<NeuraDialect, "merge"> {
  let summary = "Merge multiple inputs into one output.";
  let description = [{
  Merges multiple input values into a single output value based on the condition.
  Example:
      %out = neura.merge %cond, %in1, %in2 : i1, i32, i32 -> i32
  }];

  let arguments = (ins AnyType:$condition, AnyType:$true_value,
      AnyType:$false_value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict "
                       "`:` type($condition) `,` type($true_value) `,` "
                       "type($false_value) `->` type($result)";
}

// Defines the invariant operation.
def Neura_InvariantOp : Op<NeuraDialect, "invariant"> {
  let summary = "Invariant value across DFG execution.";
  let description = [{
  Invariant operation is a subset of carry operation where the output is always the initial value.
  Example:
      %out = neura.invariant %init %cond : i32, i1 -> i32
  }];
  let arguments = (ins AnyType:$initial, AnyType:$condition);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$initial `,` $condition attr-dict `:` type($initial) "
                       "`,` type($condition) `->` type($result)";
}

// ----------------------------------------------------
// Defines fused operations identified by graph mining.
// ----------------------------------------------------

// Fused Operation - wraps a frequently occurring pattern
def Neura_FusedOp : Op<NeuraDialect, "fused_op", [NoTerminator]> {
  let summary = "Represents a fused operation identified by graph mining";
  let description = [{
    This operation wraps a frequently occurring subgraph pattern identified
    by graph mining passes. The pattern body is contained in a region
    that preserves the original operations and their data flow edges.
    
    Attributes:
    - pattern_id: unique identifier for this pattern type
    - pattern_name: string description of the pattern
    - frequency: how many times this pattern occurs in the original graph
    
    Example:
        %result = "neura.fused_op"(%arg0, %arg1) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "mul->add"}> ({
        ^bb0(%arg0: !neura.data<!llvm.ptr, i1>, %arg1: !neura.data<!llvm.ptr, i1>):
          %0 = "neura.mul"(%arg0, %arg1) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
          %1 = "neura.add"(%0, %arg1) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
          "neura.yield"(%1) : (!neura.data<!llvm.ptr, i1>) -> ()
        })
  }];

  let arguments = (ins Variadic<AnyType>:$inputs, I64Attr:$pattern_id,
      StrAttr:$pattern_name, I64Attr:$frequency);

  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 0;
}
