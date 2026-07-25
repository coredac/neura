// ConversionPasses.td - Passes for dialect conversion

#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Math Emulation Pass.
//=========================================================//
def EmulateMathWithArith : Pass<"emulate-math-with-arith", "ModuleOp"> {
  let summary = "Emulate math.fpowi and math.tanh with arith/math.exp";
  let description = [{
    Emulates math.fpowi(constant-exponent) with chains of arith.mulf and
    math.tanh with math.exp + arith ops. This must run before affine loop
    conversion so that constant operands are not promoted to block arguments.
  }];
  let constructor = "mlir::createEmulateMathWithArithPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect"];
}

//=========================================================//
// Neura Conversion Passes.
//=========================================================//
def LowerArithToNeura : Pass<"lower-arith-to-neura", "FuncOp"> {
  let summary = "Lower arith to Neura dialect";
  let description =
      [{Lower arith dialect operations to Neura dialect operations.}];
  let constructor = "mlir::createLowerArithToNeuraPass()";
}

def LowerLlvmToNeura : Pass<"lower-llvm-to-neura", "ModuleOp"> {
  let summary = "Lower LLVM to Neura dialect";
  let description = [{Lower LLVM operations to Neura dialect operations.}];
  let constructor = "mlir::createLowerLlvmToNeuraPass()";
}

def LowerMemRefToNeura : Pass<"lower-memref-to-neura", "ModuleOp"> {
  let summary = "Lower MemRef to Neura dialect";
  let description = [{Lower MemRef operations to Neura dialect operations.}];
  let constructor = "mlir::createLowerMemRefToNeuraPass()";
}

def LowerBuiltinToNeura : Pass<"lower-builtin-to-neura", "ModuleOp"> {
  let summary = "Lower Builtin to Neura dialect";
  let description = [{Lower Builtin operations to Neura dialect operations.}];
  let constructor = "mlir::createLowerBuiltinToNeuraPass()";
}

def LowerTorchToNeura : Pass<"lower-torch-to-neura", "ModuleOp"> {
  let summary = "Lower torch frontend marker ops to Neura dialect";
  let description = [{
    Rewrites the opaque frontend marker operation "torch.neura.gather"
    (produced by the PyTorch custom op and stripped of torch types by the
    Python bridge) into the native neura.gather operation. This is a dedicated
    lowering route that bypasses the linalg pipeline used by generic ops.
  }];
  let constructor = "mlir::createLowerTorchToNeuraPass()";
  let dependentDialects = ["mlir::neura::NeuraDialect"];
}

def LowerAffineToNeura : Pass<"lower-affine-to-neura", "func::FuncOp"> {
  let summary =
      "Lower Affine perfect nested loops to Neura loop_control operations";
  let description = [{
    Converts perfectly nested affine.for loops directly to Neura dialect using 
    loop_control operations, avoiding the need to flatten to LLVM IR first.
    This preserves loop structure information for better optimization on 
    dataflow architectures.
  }];
  let constructor = "mlir::createLowerAffineToNeuraPass()";
  let dependentDialects = ["mlir::neura::NeuraDialect",
                           "mlir::affine::AffineDialect"];
}

//=========================================================//
// TaskFlow Conversion Passes.
//=========================================================//

def ConvertAffineToTaskflow : Pass<"convert-affine-to-taskflow", "ModuleOp"> {
  let summary = "Convert top-level affine.for operations to Taskflow dialect";
  let description = [{
    This pass converts top-level affine.for loops in a function into
    taskflow.task operations within a taskflow.graph. Each top-level loop
    becomes a separate task, and data dependencies between tasks are made
    explicit through taskflow.channel operations.
    
    The pass:
    1. Identifies all top-level affine.for operations
    2. Analyzes data dependencies (RAW, WAR, WAW) between loops
    3. Creates a taskflow.graph containing the loops
    4. Converts each loop to a taskflow.task
    5. Inserts taskflow.channel operations for data dependencies
  }];

  let constructor = "mlir::createConvertAffineToTaskflowPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect"];
}

def ConvertTaskflowToNeura : Pass<"convert-taskflow-to-neura", "ModuleOp"> {
  let summary = "Convert taskflow operations to neura.kernel";
  let description = [{
    Converts taskflow.hyperblock operations into neura.kernel operations suitable for
    CGRA tile array mapping.

    Converts the entire taskflow.hyperblock operations and corresponding counters ops
    to neura.kernel, based on the input architecture spec file.

    mlir-neura-opt --convert-taskflow-to-neura input.mlir --architecture-spec=$PATH/architecture.yaml -o output.mlir
  }];
  let constructor = "mlir::createConvertTaskflowToNeuraPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect"
                           "mlir::affine::AffineDialect"];
}

def FoldSubView : Pass<"fold-subview", "func::FuncOp"> {
  let summary = "Folds memref.subview into affine load/store operations";
  let description = [{
    Eliminates memref.subview operations by folding them into their users
    (affine.load and affine.store). Adjusts indices to account for subview
    offsets and strides, enabling direct access to the source memref.
  }];
  let constructor = "mlir::createFoldSubViewPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect"];
}

def ConvertCopyToAffineLoops
    : Pass<"convert-copy-to-affine-loops", "func::FuncOp"> {
  let summary = "Converts memref.copy to explicit affine loop nests";
  let description = [{
    Converts memref.copy operations into explicit affine.for loop nests
    with affine.load and affine.store operations. This makes data movement
    explicit for CGRA compilation with global addressing.
  }];
  let constructor = "mlir::createConvertCopyToAffineLoopsPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::func::FuncDialect"];
}

#endif // CONVERSION_PASSES_TD