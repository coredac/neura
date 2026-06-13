#include "TaskflowDialect/TaskflowPasses.h"
#include "Conversion/ConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef//Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

// This pass pipeline can convert affine dialect into taskflow dialect with
// neura.kernel op.
void mlir::taskflow::registerTaskflowConversionPassPipeline() {
  PassPipelineRegistration<>(
      "taskflow-conversion",
      "Converts affine dialects to taskflow dialect with neura.kernel ops.",
      [](OpPassManager &pm) {
        pm.addPass(mlir::createConvertAffineToTaskflowPass());
        pm.addPass(mlir::taskflow::createConstructHyperblockFromTaskPass());
        pm.addPass(mlir::taskflow::createClassifyTaskAndCounterPass());
        pm.addPass(mlir::createConvertTaskflowToNeuraPass());
      });
}

// This pass pipeline converts TOSA dialect to Affine dialect with cleanup.
void mlir::taskflow::registerTosaToAffineConversionPassPipeline() {
  PassPipelineRegistration<>(
      "tosa-to-affine-conversion",
      "Complete pipeline: TOSA to Linalg to Affine with subview/copy cleanup",
      [](OpPassManager &pm) {
        // Step 1-3: TOSA to Linalg (function-level passes).
        pm.nest<func::FuncOp>().addPass(
            mlir::tosa::createTosaInferShapesPass());
        pm.nest<func::FuncOp>().addPass(
            mlir::tosa::createTosaMakeBroadcastablePass());
        pm.nest<func::FuncOp>().addPass(mlir::tosa::createTosaToLinalgNamed());
        pm.nest<func::FuncOp>().addPass(mlir::tosa::createTosaToLinalg());
        pm.nest<func::FuncOp>().addPass(mlir::tosa::createTosaToArith());
        pm.nest<func::FuncOp>().addPass(mlir::tosa::createTosaToTensor());

        // Step 4: Linalg Generalization (function-level).
        pm.nest<func::FuncOp>().addPass(createLinalgGeneralizeNamedOpsPass());

        // Step 5: Canonicalization (module-level).
        pm.addPass(createCanonicalizerPass());

        // Step 6: Bufferization with proper options (module-level).
        OneShotBufferizationOptions bufferization_options;
        bufferization_options.bufferizeFunctionBoundaries = true;
        bufferization_options.setFunctionBoundaryTypeConversion(
            LayoutMapOption::IdentityLayoutMap);
        pm.addPass(createOneShotBufferizePass(bufferization_options));

        // Step 7: Linalg to Affine Loops (function-level).
        pm.nest<func::FuncOp>().addPass(createConvertLinalgToAffineLoopsPass());

        // Step 8: Cleanup subview and copy operations (function-level).
        pm.nest<func::FuncOp>().addPass(createFoldSubViewPass());
        pm.nest<func::FuncOp>().addPass(createConvertCopyToAffineLoopsPass());

        // Step 9: Final Affine cleanup (function-level).
        pm.nest<func::FuncOp>().addPass(
            mlir::affine::createAffineLoopNormalizePass());
        pm.nest<func::FuncOp>().addPass(
            mlir::affine::createSimplifyAffineStructuresPass());
        pm.addPass(createCanonicalizerPass());
      });
}

// This pass pipeline converts Linag-on-Tensors (torch-mlir LINALG_ON_TENSORS
// output) to Affine dialect.
void mlir::taskflow::registerLinalgToAffineConversionPassPipeline() {
  PassPipelineRegistration<>(
      "linalg-to-affine-conversion",
      "BUfferizes linalg-on-tensors IR then lowers to affine loops.",
      [](OpPassManager &pm) {
        // Step 1: Generalizes named linalg ops (matmul, transpose, conv, etc.)
        // to their generic form.
        pm.nest<func::FuncOp>().addPass(createLinalgGeneralizeNamedOpsPass());

        pm.nest<func::FuncOp>().addPass(createEmptyTensorEliminationPass());

        // Step 2: Canonicalizes before bufferization.
        pm.addPass(createCanonicalizerPass());

        // Step 3: One-Shot Bufferization - tensor -> memref.
        // allowReturnAllocsFromLoops=true is needed for dynamic shapes.
        OneShotBufferizationOptions bufferization_options;
        bufferization_options.bufferizeFunctionBoundaries = true;
        bufferization_options.allowReturnAllocsFromLoops = true;
        bufferization_options.setFunctionBoundaryTypeConversion(
            LayoutMapOption::IdentityLayoutMap);
        pm.addPass(createOneShotBufferizePass(bufferization_options));

        // Step 4: Converts Linalg to Affine loops.
        pm.nest<func::FuncOp>().addPass(createConvertLinalgToAffineLoopsPass());

        pm.nest<func::FuncOp>().addPass(
            mlir::memref::createFoldMemRefAliasOpsPass());

        // Step 5: Cleans up subview and copy operations introduced by
        // bufferization.
        pm.nest<func::FuncOp>().addPass(createFoldSubViewPass());
        pm.nest<func::FuncOp>().addPass(createConvertCopyToAffineLoopsPass());

        // Step 6: Affine cleanup for downstream passes.
        pm.nest<func::FuncOp>().addPass(
            mlir::affine::createAffineLoopNormalizePass());
        pm.nest<func::FuncOp>().addPass(
            mlir::affine::createSimplifyAffineStructuresPass());
        pm.addPass(createCanonicalizerPass());

        // Step 7: Expand math.fpowi / math.tanh into arith/math.exp
        // primitives before taskflow conversion, while constants (e.g.
        // exponent 3 for GELU's x^3) are still visible as arith.constant
        // inside affine loops.
        pm.addPass(mlir::createExpandMathToArithPass());
      });
}