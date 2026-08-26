// NeuraConversionPasses.h - Header file for Neura conversion passes

#ifndef NEURA_CONVERSION_PASSES_H
#define NEURA_CONVERSION_PASSES_H
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {

// Passes defined in NeuraConversionPasses.td.
#define GEN_PASS_DECL
#include "Conversion/NeuraConversionPasses.h.inc"

// Neura conversion passes.
std::unique_ptr<mlir::Pass> createLowerArithToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerLlvmToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerMemRefToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerBuiltinToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerTorchToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerAffineToNeuraPass();
std::unique_ptr<mlir::Pass> createExpandMemrefCopyPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/NeuraConversionPasses.h.inc"

} // namespace mlir

#endif // NEURA_CONVERSION_PASSES_H
