// ConversionPasses.h - Header file for conversion passes

#ifndef CONVERSION_PASSES_H
#define CONVERSION_PASSES_H
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {

// Passes defined in GraphPasses.td.
#define GEN_PASS_DECL
#include "Conversion/ConversionPasses.h.inc"

// Neura Conversion Passes.
std::unique_ptr<mlir::Pass> createLowerArithToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerLlvmToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerMemRefToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerBuiltinToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerAffineToNeuraPass();

// Math Expansion Pass.
std::unique_ptr<mlir::Pass> createExpandMathToArithPass();

// TaskFlow Conversion Passes.
std::unique_ptr<mlir::Pass> createConvertAffineToTaskflowPass();
std::unique_ptr<mlir::Pass> createConvertTaskflowToNeuraPass();
std::unique_ptr<mlir::Pass> createStripTaskflowTaskPass();

// Memref SubView and Copy Conversion Passes.
std::unique_ptr<mlir::Pass> createFoldSubViewPass();
std::unique_ptr<mlir::Pass> createConvertCopyToAffineLoopsPass();
#define GEN_PASS_REGISTRATION
#include "Conversion/ConversionPasses.h.inc"

} // namespace mlir

#endif // CONVERSION_PASSES_H