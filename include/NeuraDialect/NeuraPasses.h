// NeuraPasses.h - Header file for Neura passes

#ifndef NEURA_PASSES_H
#define NEURA_PASSES_H

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
namespace neura {

void registerNeuraConversionPassPipeline();

// Passes defined in NeuraPasses.td
#define GEN_PASS_DECL
#include "NeuraDialect/NeuraPasses.h.inc"
std::unique_ptr<mlir::Pass> createInsertDataMovPass();
std::unique_ptr<mlir::Pass> createInsertCtrlMovPass();
std::unique_ptr<mlir::Pass> createAssignAcceleratorPass();
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass();
std::unique_ptr<mlir::Pass> createLeveragePredicatedValuePass();
// Creates the MapToAccelerator pass. Tile dimensions default to 0 (use
// architecture singleton) when not specified via options.
std::unique_ptr<mlir::Pass> createMapToAcceleratorPass(
    const MapToAcceleratorOptions &options = MapToAcceleratorOptions{});
std::unique_ptr<mlir::Pass> createGenerateCodePass();
std::unique_ptr<mlir::Pass> createCanonicalizeReturnPass();
std::unique_ptr<mlir::Pass> createCanonicalizeLiveInPass();
std::unique_ptr<mlir::Pass> createPromoteInputArgToConstPass();
std::unique_ptr<mlir::Pass> createTransformToSteerControlPass();
std::unique_ptr<mlir::Pass> createRemovePredicatedTypePass();

// ====================================
// Optimization Passes
// ====================================
// Hardware specific optimization passes
std::unique_ptr<mlir::Pass> createFuseLoopControlPass();
std::unique_ptr<mlir::Pass> createFusePatternPass();

// Hardware agnostic optimization passes
std::unique_ptr<mlir::Pass> createFoldConstantPass();
std::unique_ptr<mlir::Pass> createCanonicalizeCastPass();

// Graph mining passes
std::unique_ptr<mlir::Pass> createIterMergePatternPass();
std::unique_ptr<mlir::Pass> createInitPatternPass();

// Hardware optimization passes
std::unique_ptr<mlir::Pass> createHardwareMergePass();
std::unique_ptr<mlir::Pass> createInitExecLatencyPass();

#define GEN_PASS_REGISTRATION
#include "NeuraDialect/NeuraPasses.h.inc"

} // namespace neura
} // namespace mlir

#endif // NEURA_PASSES_H