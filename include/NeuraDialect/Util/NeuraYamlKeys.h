#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace neura {
namespace yamlkeys {

constexpr llvm::StringLiteral kArchitecture = "architecture";
constexpr llvm::StringLiteral kMultiCgraDefaults = "multi_cgra_defaults";
constexpr llvm::StringLiteral kPerCgraDefaults = "per_cgra_defaults";
constexpr llvm::StringLiteral kTileDefaults = "tile_defaults";
constexpr llvm::StringLiteral kTileOverrides = "tile_overrides";
constexpr llvm::StringLiteral kLinkDefaults = "link_defaults";
constexpr llvm::StringLiteral kLinkOverrides = "link_overrides";
constexpr llvm::StringLiteral kFuTypes = "fu_types";
constexpr llvm::StringLiteral kNumRegisters = "num_registers";
constexpr llvm::StringLiteral kLatency = "latency";
constexpr llvm::StringLiteral kBandwidth = "bandwidth";
constexpr llvm::StringLiteral kExistence = "existence";
constexpr llvm::StringLiteral kRows = "rows";
constexpr llvm::StringLiteral kColumns = "columns";
constexpr llvm::StringLiteral kBaseTopology = "base_topology";
constexpr llvm::StringLiteral kCtrlMemItems = "ctrl_mem_items";
constexpr llvm::StringLiteral kGlobalReduceEntries = "global_reduce_entries";
constexpr llvm::StringLiteral kTileX = "tile_x";
constexpr llvm::StringLiteral kTileY = "tile_y";
constexpr llvm::StringLiteral kCgraX = "cgra_x";
constexpr llvm::StringLiteral kCgraY = "cgra_y";
constexpr llvm::StringLiteral kSrcTileX = "src_tile_x";
constexpr llvm::StringLiteral kSrcTileY = "src_tile_y";
constexpr llvm::StringLiteral kDstTileX = "dst_tile_x";
constexpr llvm::StringLiteral kDstTileY = "dst_tile_y";
constexpr llvm::StringLiteral kMesh = "mesh";
constexpr llvm::StringLiteral kKingMesh = "king_mesh";
constexpr llvm::StringLiteral kKingMeshAlt = "king mesh";
constexpr llvm::StringLiteral kRing = "ring";

} // namespace yamlkeys
} // namespace neura
} // namespace mlir
