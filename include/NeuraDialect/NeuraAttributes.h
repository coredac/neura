#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace neura {

namespace attr {

// Attribute Keys
constexpr llvm::StringLiteral kKernelMetadata = "kernel_metadata";
constexpr llvm::StringLiteral kKind = "kind";

// Specifies the dataflow representation mode, as opposed to control-flow.
constexpr llvm::StringLiteral kDataflowMode = "dataflow_mode";

// Specifies the mapping strategy mode, can be either 'spatial-only' or
// 'spatial-temporal'.
constexpr llvm::StringLiteral kMappingMode = "mapping_mode";

constexpr llvm::StringLiteral kMappingStrategy = "mapping_strategy";
constexpr llvm::StringLiteral kBacktrackConfig = "backtrack_config";

// Keys for the template mapping input constraints.
constexpr llvm::StringLiteral kPlacement = "placement";
constexpr llvm::StringLiteral kX = "x";
constexpr llvm::StringLiteral kY = "y";

// Identification & Results.
constexpr llvm::StringLiteral kDfgId = "dfg_id";
constexpr llvm::StringLiteral kMappingInfo = "mapping_info";
constexpr llvm::StringLiteral kXTiles = "x_tiles";
constexpr llvm::StringLiteral kYTiles = "y_tiles";
constexpr llvm::StringLiteral kCompiledII = "compiled_ii";
constexpr llvm::StringLiteral kRecMII = "rec_mii";
constexpr llvm::StringLiteral kResMII = "res_mii";

// Values & Constants Keys.
constexpr llvm::StringLiteral kValue = "value";
constexpr llvm::StringLiteral kConstantValue = "constant_value";
constexpr llvm::StringLiteral kRhsValue = "rhs_value";
constexpr llvm::StringLiteral kLhsValue = "lhs_value";

// Attribute Values & Constants
namespace val {
// Strategy & Mode
constexpr llvm::StringLiteral kSpatialOnly = "spatial-only";
constexpr llvm::StringLiteral kSpatialTemporal = "spatial-temporal";
constexpr llvm::StringLiteral kTemplate = "template";
constexpr llvm::StringLiteral kHeuristic = "heuristic";
constexpr llvm::StringLiteral kCustomized = "customized";
constexpr llvm::StringLiteral kSimple = "simple";
constexpr llvm::StringLiteral kGreedy = "greedy";
constexpr llvm::StringLiteral kExhaustive = "exhaustive";

// Identifiers
constexpr llvm::StringLiteral kModeSteering = "steering";
constexpr llvm::StringLiteral kModePredicate = "predicate";

// Operation Logic
constexpr llvm::StringLiteral kOpFused = "fused_op";
constexpr llvm::StringLiteral kNeuraFusedOp = "neura.fused_op";

} // namespace val

} // namespace attr

// Returns whether an operation carries template kernel metadata.
inline bool isTemplateKernel(Operation *operation) {
  auto metadata =
      operation->getAttrOfType<DictionaryAttr>(attr::kKernelMetadata);
  auto kind =
      metadata ? metadata.getAs<StringAttr>(attr::kKind) : StringAttr{};
  return kind && kind.getValue() == attr::val::kTemplate;
}

} // namespace neura
} // namespace mlir
