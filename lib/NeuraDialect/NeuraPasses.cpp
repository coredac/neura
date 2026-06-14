#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/ViewOpGraph.h"

std::string filename = "opgraph.dot";
std::error_code EC;
llvm::raw_fd_ostream os(filename, EC, llvm::sys::fs::OF_Text);

// This pass pipeline can convert all the other dialects into the Neura dialect
void mlir::neura::registerNeuraConversionPassPipeline() {
  PassPipelineRegistration<>(
      "neura-conversion", "Convert all dialects to Neura dialect",
      [](OpPassManager &pm) {
        // Strip taskflow.task wrappers first (if present from upstream
        // taskflow-conversion pipeline), promoting neura.kernel to func level.
        pm.addPass(mlir::createStripTaskflowTaskPass());

        pm.addPass(mlir::neura::createAssignAcceleratorPass());

        pm.addPass(mlir::createLowerAffineToNeuraPass());
        pm.addPass(mlir::createLowerArithToNeuraPass());
        pm.addPass(mlir::createLowerMemRefToNeuraPass());
        pm.addPass(mlir::createLowerBuiltinToNeuraPass());
        pm.addPass(mlir::createLowerLlvmToNeuraPass());
        pm.addPass(mlir::createPrintOpGraphPass(os));

        pm.addPass(mlir::neura::createCanonicalizeReturnPass());
        pm.addPass(mlir::neura::createCanonicalizeCastPass());
        pm.addPass(mlir::neura::createPromoteInputArgToConstPass());
        pm.addPass(mlir::neura::createFoldConstantPass());
        pm.addPass(mlir::neura::createCanonicalizeLiveInPass());
        pm.addPass(mlir::neura::createLeveragePredicatedValuePass());
        pm.addPass(mlir::createPrintOpGraphPass(os));

        pm.addPass(mlir::neura::createTransformCtrlToDataFlowPass());
        pm.addPass(mlir::neura::createFoldConstantPass());
        pm.addPass(mlir::neura::createFusePatternPass());
        pm.addPass(mlir::neura::createInsertDataMovPass());
        pm.addPass(mlir::createPrintOpGraphPass(os));

        pm.addPass(mlir::neura::createMapToAcceleratorPass());
        pm.addPass(mlir::neura::createGenerateCodePass());
      });
}

// This pass pipeline provides the full Python-frontend lowering path:
//   Linalg-on-Tensors (from torch-mlir)
//     → Bufferization → Affine Loops
//     → Taskflow (multi-CGRA decomposition)
//     → Neura (CGRA dataflow dialect)
//     → Code generation (JSON instructions)
//
// This mirrors the Python neura_pipeline.py script but can be invoked
// directly from mlir-neura-opt / neura-compiler.
void mlir::neura::registerPythonToNeuraPassPipeline() {
  PassPipelineRegistration<>(
      "python-to-neura",
      "Python-frontend lowering: Linalg-on-Tensors → Affine → Taskflow → Neura → Codegen",
      [](OpPassManager &pm) {
        // ---- Phase 1: Linalg-on-Tensors → Affine ----
        // Generalize named linalg ops to generic form
        pm.nest<mlir::func::FuncOp>().addPass(
            mlir::createLinalgGeneralizeNamedOpsPass());
        pm.nest<mlir::func::FuncOp>().addPass(
            mlir::bufferization::createEmptyTensorEliminationPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // One-shot bufferization: tensor → memref
        mlir::bufferization::OneShotBufferizationOptions bufOpts;
        bufOpts.bufferizeFunctionBoundaries = true;
        bufOpts.allowReturnAllocsFromLoops = true;
        bufOpts.setFunctionBoundaryTypeConversion(
            mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
        pm.addPass(
            mlir::bufferization::createOneShotBufferizePass(bufOpts));

        // Linalg → Affine loops
        pm.nest<mlir::func::FuncOp>().addPass(
            mlir::createConvertLinalgToAffineLoopsPass());
        pm.nest<mlir::func::FuncOp>().addPass(
            mlir::memref::createFoldMemRefAliasOpsPass());

        // Cleanup subview / copy
        pm.nest<mlir::func::FuncOp>().addPass(mlir::createFoldSubViewPass());
        pm.nest<mlir::func::FuncOp>().addPass(
            mlir::createConvertCopyToAffineLoopsPass());

        // Affine cleanup
        pm.nest<mlir::func::FuncOp>().addPass(
            mlir::affine::createAffineLoopNormalizePass());
        pm.nest<mlir::func::FuncOp>().addPass(
            mlir::affine::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // ---- Phase 2: Affine → Taskflow (multi-CGRA) ----
        pm.addPass(mlir::createConvertAffineToTaskflowPass());
        pm.addPass(
            mlir::taskflow::createConstructHyperblockFromTaskPass());
        pm.addPass(
            mlir::taskflow::createClassifyTaskAndCounterPass());
        pm.addPass(mlir::createConvertTaskflowToNeuraPass());
        pm.addPass(mlir::createStripTaskflowTaskPass());

        // ---- Phase 3: Neura lowering + mapping + codegen ----
        pm.addPass(mlir::neura::createAssignAcceleratorPass());
        pm.addPass(mlir::createLowerAffineToNeuraPass());
        pm.addPass(mlir::createLowerArithToNeuraPass());
        pm.addPass(mlir::createLowerMemRefToNeuraPass());
        pm.addPass(mlir::createLowerBuiltinToNeuraPass());
        pm.addPass(mlir::createLowerLlvmToNeuraPass());
        pm.addPass(mlir::createPrintOpGraphPass(os));

        pm.addPass(mlir::neura::createCanonicalizeReturnPass());
        pm.addPass(mlir::neura::createCanonicalizeCastPass());
        pm.addPass(mlir::neura::createPromoteInputArgToConstPass());
        pm.addPass(mlir::neura::createFoldConstantPass());
        pm.addPass(mlir::neura::createCanonicalizeLiveInPass());
        pm.addPass(mlir::neura::createLeveragePredicatedValuePass());
        pm.addPass(mlir::createPrintOpGraphPass(os));

        pm.addPass(mlir::neura::createTransformCtrlToDataFlowPass());
        pm.addPass(mlir::neura::createFoldConstantPass());
        pm.addPass(mlir::neura::createFusePatternPass());
        pm.addPass(mlir::neura::createInsertDataMovPass());
        pm.addPass(mlir::createPrintOpGraphPass(os));

        pm.addPass(mlir::neura::createMapToAcceleratorPass());
        pm.addPass(mlir::neura::createGenerateCodePass());
      });
}
