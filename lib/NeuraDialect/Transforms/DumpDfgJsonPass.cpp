//===- DumpDfgJsonPass.cpp - Emit pre-map DFG + arch as JSON -------------===//
//
// Emits the lowered Neura DFG (materialized ops, dependence edges with a
// loop-carried iteration distance omega) together with the target CGRA
// architecture (tiles, per-FU-class tile support, mesh links, registers,
// ctrl_mem_items) as a JSON document. This feeds the exact modulo-scheduling
// oracle (test/cost-model/exact_oracle.py). It reuses the same primitives the
// mapper uses (getOperationKindFromMlirOp, getMaterializedProducer semantics,
// collectRecurrenceCycles' reserve/ctrl_mov structure) so the oracle solves the
// exact same problem instance the mapper faces.
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
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_DUMPDFGJSON
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

bool isMaterial(Operation *op) {
  if (isa<func::FuncOp, ModuleOp, neura::KernelOp>(op))
    return false;
  if (is_non_materialized(op)) // reserve / data_mov / ctrl_mov / yield
    return false;
  if (Operation *p = op->getParentOp())
    if (isa<neura::FusedOp>(p))
      return false;
  return true;
}

const std::map<OperationKind, std::string> &kindToClass() {
  static const std::map<OperationKind, std::string> t = [] {
    std::map<OperationKind, std::string> m;
    for (const auto &[name, ops] : kFuTypesToOperations)
      for (OperationKind k : ops)
        m[k] = name;
    return m;
  }();
  return t;
}

std::string classOf(Operation *op) {
  if (isa<neura::FusedOp>(op))
    if (auto n = op->getAttrOfType<StringAttr>("pattern_name"))
      return n.getValue().str();
  auto it = kindToClass().find(getOperationKindFromMlirOp(op));
  return it == kindToClass().end() ? "other" : it->second;
}

// Safe materialized-producer unwrap (does not assert like getMaterializedProducer).
Operation *matProducer(Value v) {
  Operation *p = v.getDefiningOp();
  if (!p)
    return nullptr;
  if (isa<neura::ReserveOp>(p))
    return nullptr; // loop-carried placeholder; handled via ctrl_mov edges.
  if (auto mov = dyn_cast<neura::DataMovOp>(p)) {
    Operation *inner = mov.getOperand().getDefiningOp();
    if (inner && !isa<neura::ReserveOp>(inner))
      return inner;
    return nullptr;
  }
  return p;
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
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<neura::NeuraDialect>();
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

  std::unique_ptr<Architecture> buildCustomArch(const Architecture &global) {
    if (x_tiles.getValue() <= 0 || y_tiles.getValue() <= 0)
      return nullptr;
    std::vector<TileOverride> overrides;
    if (!valid_tiles.getValue().empty()) {
      llvm::SmallVector<llvm::StringRef, 4> coords;
      llvm::StringRef(valid_tiles.getValue()).split(coords, ',');
      for (int y = 0; y < y_tiles.getValue(); ++y)
        for (int x = 0; x < x_tiles.getValue(); ++x) {
          TileOverride to; to.tile_x = x; to.tile_y = y; to.existence = false;
          overrides.push_back(to);
        }
      for (llvm::StringRef c : coords) {
        auto pr = c.split('_');
        int x, y;
        if (!pr.first.getAsInteger(10, x) && !pr.second.getAsInteger(10, y)) {
          TileOverride to; to.tile_x = x; to.tile_y = y; to.existence = true;
          overrides.push_back(to);
        }
      }
    }
    return global.cloneWithNewDimensions(y_tiles.getValue(), x_tiles.getValue(),
                                         overrides);
  }

  void emitRegion(Region &region, const Architecture &arch, llvm::raw_ostream &os) {
    // Assign dense ids to materialized ops.
    llvm::DenseMap<Operation *, int> id;
    std::vector<Operation *> ops;
    region.walk([&](Operation *op) {
      if (isMaterial(op)) {
        id[op] = (int)ops.size();
        ops.push_back(op);
      }
    });

    // Forward edges (omega=0).
    struct Edge { int s, d, w; };
    std::vector<Edge> edges;
    for (Operation *v : ops) {
      for (Value operand : v->getOperands()) {
        Operation *u = matProducer(operand);
        if (u && id.count(u))
          edges.push_back({id[u], id[v], 0});
      }
    }
    // Loop-carried edges (omega=1): producer of ctrl_mov value -> users of the
    // reserve it targets. Represents value[i] feeding the placeholder for i+1.
    region.walk([&](neura::CtrlMovOp cm) {
      Operation *P = matProducer(cm.getValue());
      auto reserve = cm.getTarget().getDefiningOp<neura::ReserveOp>();
      if (!P || !id.count(P) || !reserve)
        return;
      for (Operation *user : reserve.getResult().getUsers()) {
        Operation *mu = user;
        if (is_non_materialized(user)) {
          for (Operation *uu : user->getUsers())
            if (id.count(uu)) mu = uu;
        }
        if (id.count(mu))
          edges.push_back({id[P], id[mu], 1});
      }
    });

    // Arch: per-FU-class tile support + mesh links + registers.
    os << "{\n  \"arch\": {\n";
    os << "    \"num_tiles\": " << arch.getNumTiles()
       << ", \"ctrl_mem_items\": " << arch.getMaxCtrlMemItems() << ",\n";
    os << "    \"tiles\": [";
    auto tiles = arch.getAllTiles();
    for (size_t i = 0; i < tiles.size(); ++i) {
      Tile *t = tiles[i];
      os << (i ? ", " : "") << "{\"id\": " << t->getId() << ", \"x\": " << t->getX()
         << ", \"y\": " << t->getY()
         << ", \"regs\": " << (int)t->getRegisters().size() << "}";
    }
    os << "],\n    \"fu_class_tiles\": {";
    bool first = true;
    for (const auto &[name, kinds] : kFuTypesToOperations) {
      if (kinds.empty()) continue;
      OperationKind probe = kinds.front();
      std::string list;
      for (Tile *t : tiles)
        if (t->canSupportOperation(probe))
          list += (list.empty() ? "" : ", ") + std::to_string(t->getId());
      os << (first ? "" : ", ") << "\"" << name << "\": [" << list << "]";
      first = false;
    }
    os << "},\n    \"links\": [";
    auto links = arch.getAllLinks();
    for (size_t i = 0; i < links.size(); ++i) {
      Link *l = links[i];
      os << (i ? ", " : "") << "[" << l->getSrcTile()->getId() << ", "
         << l->getDstTile()->getId() << ", " << l->getLatency() << "]";
    }
    os << "]\n  },\n";

    // Ops.
    os << "  \"ops\": [";
    for (size_t i = 0; i < ops.size(); ++i)
      os << (i ? ", " : "") << "{\"id\": " << i << ", \"class\": \""
         << classOf(ops[i]) << "\", \"latency\": "
         << std::max(1, getOpLatency(ops[i])) << "}";
    os << "],\n  \"edges\": [";
    for (size_t i = 0; i < edges.size(); ++i)
      os << (i ? ", " : "") << "{\"s\": " << edges[i].s << ", \"d\": "
         << edges[i].d << ", \"w\": " << edges[i].w << "}";
    os << "]\n}\n";
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    const Architecture &global = mlir::neura::getArchitecture();
    std::unique_ptr<Architecture> custom = buildCustomArch(global);
    const Architecture &arch = custom ? *custom : global;
    bool done = false;
    auto tryEmit = [&](Operation *op, Region &r) {
      auto a = op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (a && a.getValue() == accel::kNeuraTarget && !r.empty()) {
        emitRegion(r, arch, llvm::outs());
        done = true;
      }
    };
    m.walk([&](neura::KernelOp k) { tryEmit(k, k.getBody()); });
    m.walk([&](func::FuncOp f) { tryEmit(f, f.getBody()); });
    if (!done)
      m.walk([&](func::FuncOp f) {
        if (!done && !f.getBody().empty()) { emitRegion(f.getBody(), arch, llvm::outs()); done = true; }
      });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createDumpDfgJsonPass() {
  return std::make_unique<DumpDfgJsonPass>();
}
} // namespace mlir::neura
