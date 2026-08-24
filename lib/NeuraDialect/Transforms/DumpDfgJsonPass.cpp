//===- DumpDfgJsonPass.cpp - Emit pre-map DFG + arch as JSON -------------===//
//
// Emits the lowered Neura DFG (placed ops, dependence edges with a
// loop-carried iteration distance omega) together with the target CGRA
// architecture (tiles, per-FU-class tile support, mesh links, registers,
// ctrl_mem_items) as a JSON document. This feeds the exact modulo-scheduling
// oracle (test/cost-model/exact_oracle*.py) so the oracle solves the exact same
// problem instance the mapper faces (same op-kind / recurrence primitives).
//
//===----------------------------------------------------------------------===//

#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_DUMPDFGJSON
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// occupiesFU (placement set) and fuClassOf (an op's FU class) come from
// mapping_util.h -- the single source of truth shared with the mapper, so the
// JSON this pass emits buckets ops exactly as the mapper does.

// Safe placed-producer unwrap (does not assert like
// getMaterializedProducer).
Operation *placedProducer(Value value) {
  Operation *producer = value.getDefiningOp();
  if (!producer) {
    return nullptr;
  }
  if (isa<neura::ReserveOp>(producer)) {
    return nullptr; // loop-carried placeholder; handled via ctrl_mov edges.
  }
  if (auto move = dyn_cast<neura::DataMovOp>(producer)) {
    Operation *inner_producer = move.getOperand().getDefiningOp();
    if (inner_producer && !isa<neura::ReserveOp>(inner_producer)) {
      return inner_producer;
    }
    return nullptr;
  }
  return producer;
}

struct DumpDfgJsonPass
    : public PassWrapper<DumpDfgJsonPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpDfgJsonPass)
  DumpDfgJsonPass() = default;
  DumpDfgJsonPass(const DumpDfgJsonPass &pass)
      : PassWrapper<DumpDfgJsonPass, OperationPass<ModuleOp>>(pass) {}
  StringRef getArgument() const override { return "dump-dfg-json"; }
  StringRef getDescription() const override {
    return "Emit the pre-map DFG + architecture as JSON for the exact oracle.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<neura::NeuraDialect>();
  }

  // Tile-array shape controls, mirroring --cost-model-analytical /
  // --map-to-accelerator, so the DFG+arch can be dumped for ANY CGRA shape
  // (incl. multi-CGRA rectangles and irregular L/T blocks). 0 = the global
  // single-CGRA architecture.
  Option<int> x_tiles{*this, "x-tiles",
                      llvm::cl::desc("Total tiles in X (0 = arch singleton)."),
                      llvm::cl::init(0)};
  Option<int> y_tiles{*this, "y-tiles",
                      llvm::cl::desc("Total tiles in Y (0 = arch singleton)."),
                      llvm::cl::init(0)};
  Option<std::string> valid_tiles{
      *this, "valid-tiles",
      llvm::cl::desc("Comma-separated tile coords (x_y) for non-rect shapes."),
      llvm::cl::init("")};

  std::unique_ptr<Architecture>
  buildCustomArch(const Architecture &global_arch) {
    if (x_tiles.getValue() <= 0 || y_tiles.getValue() <= 0) {
      return nullptr;
    }
    std::vector<TileOverride> overrides;
    if (!valid_tiles.getValue().empty()) {
      // applyTileOverrides can only REMOVE tiles (existence=false); it cannot
      // re-add a removed tile. So emit a removal override for every tile NOT in
      // the valid set, and leave the valid tiles untouched.
      std::set<std::pair<int, int>> valid_coords;
      llvm::SmallVector<llvm::StringRef, 4> coords;
      llvm::StringRef(valid_tiles.getValue()).split(coords, ',');
      for (llvm::StringRef coord : coords) {
        coord = coord.trim(); // tolerate "0_0, 1_1" with spaces after commas.
        if (coord.empty()) {
          continue;
        }
        auto parts = coord.split('_');
        int x, y;
        if (!parts.first.trim().getAsInteger(10, x) &&
            !parts.second.trim().getAsInteger(10, y)) {
          // Ignore out-of-grid coords rather than silently removing real tiles.
          if (x >= 0 && x < x_tiles.getValue() && y >= 0 &&
              y < y_tiles.getValue()) {
            valid_coords.insert({x, y});
          } else {
            llvm::errs() << "[dump-dfg-json] valid-tiles coord " << x << "_" << y
                         << " is outside the " << x_tiles.getValue() << "x"
                         << y_tiles.getValue() << " grid; ignored\n";
          }
        }
      }
      if (valid_coords.empty()) {
        llvm::errs() << "[dump-dfg-json] valid-tiles selected no tiles in the "
                        "grid; using the full "
                     << x_tiles.getValue() << "x" << y_tiles.getValue()
                     << " rectangle\n";
      } else {
        for (int y = 0; y < y_tiles.getValue(); ++y) {
          for (int x = 0; x < x_tiles.getValue(); ++x) {
            if (!valid_coords.count({x, y})) {
              TileOverride tile_override;
              tile_override.tile_x = x;
              tile_override.tile_y = y;
              tile_override.existence = false;
              overrides.push_back(tile_override);
            }
          }
        }
      }
    }
    return global_arch.cloneWithNewDimensions(y_tiles.getValue(),
                                              x_tiles.getValue(), overrides);
  }

  void emitRegion(Region &region, const Architecture &arch,
                  llvm::raw_ostream &os) {
    // Assign dense ids to placed ops.
    llvm::DenseMap<Operation *, int> op_id;
    std::vector<Operation *> placed_ops;
    region.walk([&](Operation *op) {
      if (occupiesFU(op)) {
        op_id[op] = (int)placed_ops.size();
        placed_ops.push_back(op);
      }
    });

    struct Edge {
      int src, dst, omega;
    };
    std::vector<Edge> edges;
    // Forward (intra-iteration) edges, omega=0. Dedup by (producer, consumer):
    // an op that uses the same value twice (e.g. x + x) has two data_movs to the
    // same producer, but that is ONE dependence net — emitting it twice would
    // double-book the shared link/route on import.
    std::set<std::pair<int, int>> seen_forward;
    for (Operation *consumer : placed_ops) {
      for (Value operand : consumer->getOperands()) {
        Operation *producer = placedProducer(operand);
        if (producer && op_id.count(producer)) {
          if (seen_forward.insert({op_id[producer], op_id[consumer]}).second) {
            edges.push_back({op_id[producer], op_id[consumer], 0});
          }
        }
      }
    }
    // Loop-carried edges, omega=1: producer of a ctrl_mov's value -> users of
    // the reserve it targets. Represents value[i] feeding the placeholder for
    // iteration i+1.
    region.walk([&](neura::CtrlMovOp ctrl_mov) {
      Operation *producer = placedProducer(ctrl_mov.getValue());
      auto reserve = ctrl_mov.getTarget().getDefiningOp<neura::ReserveOp>();
      if (!producer || !op_id.count(producer) || !reserve) {
        return;
      }
      for (Operation *user : reserve.getResult().getUsers()) {
        Operation *placed_user = user;
        if (is_non_materialized(user)) {
          for (Operation *router_user : user->getUsers()) {
            if (op_id.count(router_user)) {
              placed_user = router_user;
            }
          }
        }
        if (op_id.count(placed_user)) {
          edges.push_back({op_id[producer], op_id[placed_user], 1});
        }
      }
    });

    // Architecture: per-FU-class tile support + mesh links + registers.
    os << "{\n  \"arch\": {\n";
    os << "    \"num_tiles\": " << arch.getNumTiles()
       << ", \"ctrl_mem_items\": " << arch.getMaxCtrlMemItems() << ",\n";
    os << "    \"tiles\": [";
    auto tiles = arch.getAllTiles();
    for (size_t i = 0; i < tiles.size(); ++i) {
      Tile *tile = tiles[i];
      // regs = total registers on the tile; regfiles = number of register files
      // (each with one read + one write port) — both read straight from the arch
      // so the oracle's port model doesn't hardcode the regs-per-file constant.
      os << (i ? ", " : "") << "{\"id\": " << tile->getId()
         << ", \"x\": " << tile->getX() << ", \"y\": " << tile->getY()
         << ", \"regs\": " << (int)tile->getRegisters().size()
         << ", \"regfiles\": " << (int)tile->getRegisterFiles().size() << "}";
    }
    os << "],\n    \"fu_class_tiles\": {";
    bool first_class = true;
    for (const auto &[fu_class_name, kinds] : kFuTypesToOperations) {
      if (kinds.empty()) {
        continue;
      }
      OperationKind probe_kind = kinds.front();
      std::string tile_id_list;
      for (Tile *tile : tiles) {
        if (tile->canSupportOperation(probe_kind)) {
          tile_id_list += (tile_id_list.empty() ? "" : ", ") +
                          std::to_string(tile->getId());
        }
      }
      os << (first_class ? "" : ", ") << "\"" << fu_class_name << "\": ["
         << tile_id_list << "]";
      first_class = false;
    }
    os << "},\n    \"links\": [";
    auto links = arch.getAllLinks();
    for (size_t i = 0; i < links.size(); ++i) {
      Link *link = links[i];
      os << (i ? ", " : "") << "[" << link->getSrcTile()->getId() << ", "
         << link->getDstTile()->getId() << ", " << link->getLatency() << "]";
    }
    os << "]\n  },\n";

    // Ops.
    os << "  \"ops\": [";
    for (size_t i = 0; i < placed_ops.size(); ++i) {
      os << (i ? ", " : "") << "{\"id\": " << i << ", \"class\": \""
         << fuClassOf(placed_ops[i]) << "\", \"latency\": "
         << std::max(1, getOpLatency(placed_ops[i])) << "}";
    }
    os << "],\n  \"edges\": [";
    for (size_t i = 0; i < edges.size(); ++i) {
      os << (i ? ", " : "") << "{\"s\": " << edges[i].src
         << ", \"d\": " << edges[i].dst << ", \"w\": " << edges[i].omega << "}";
    }
    os << "]\n}\n";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const Architecture &global_arch = mlir::neura::getArchitecture();
    std::unique_ptr<Architecture> custom_arch = buildCustomArch(global_arch);
    const Architecture &arch = custom_arch ? *custom_arch : global_arch;
    bool emitted = false;
    auto tryEmit = [&](Operation *op, Region &region) {
      auto accel_attr = op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (accel_attr && accel_attr.getValue() == accel::kNeuraTarget &&
          !region.empty()) {
        emitRegion(region, arch, llvm::outs());
        emitted = true;
      }
    };
    module.walk(
        [&](neura::KernelOp kernel) { tryEmit(kernel, kernel.getBody()); });
    module.walk([&](func::FuncOp func) { tryEmit(func, func.getBody()); });
    // Fallback: nothing tagged for the accelerator -> emit the first non-empty
    // func so a hand-written kernel still works standalone.
    if (!emitted) {
      module.walk([&](func::FuncOp func) {
        if (!emitted && !func.getBody().empty()) {
          emitRegion(func.getBody(), arch, llvm::outs());
          emitted = true;
        }
      });
    }
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createDumpDfgJsonPass() {
  return std::make_unique<DumpDfgJsonPass>();
}
} // namespace mlir::neura
