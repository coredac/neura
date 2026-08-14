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
#include "mlir/IR/SymbolTable.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
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

// MLIR symbol names are near-arbitrary strings -- @"a\"b" parses -- so a name
// pasted raw into the dump could itself produce the unparseable file this pass
// exists to avoid. Escapes the characters JSON forbids inside a string.
std::string escapeJsonString(StringRef s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[7];
        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
        out += buf;
      } else {
        out += c;
      }
    }
  }
  return out;
}

// A stable, human-recognizable label for one dumped region. func.func carries a
// symbol; neura.kernel does not, so a kernel is named after the function that
// holds it plus its walk index, which keeps two kernels in one function
// distinguishable in the dump.
std::string regionLabel(Operation *op, unsigned index) {
  if (auto sym =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
    return sym.getValue().str();
  }
  std::string label = op->getName().getStringRef().str();
  if (Operation *parent = op->getParentOp()) {
    if (auto parent_sym =
            parent->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      label = parent_sym.getValue().str() + "/" + label;
  }
  return label + "#" + std::to_string(index);
}

// collectPlacedOps (the placement set, in the order --import-mapping replays),
// buildDfgEdges (the dependence graph), fuClassOf (an op's FU class) and
// tilesProvidingFuClass (which tiles run a class) all come from mapping_util.h
// -- the single source of truth shared with the analytical cost model and the
// mapper, so the JSON this pass emits describes the same instance the cost
// model reasons about.

struct DumpDfgJsonPass
    : public PassWrapper<DumpDfgJsonPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpDfgJsonPass)
  DumpDfgJsonPass() = default;
  DumpDfgJsonPass(const DumpDfgJsonOptions &options) : DumpDfgJsonPass() {
    x_tiles = options.x_tiles;
    y_tiles = options.y_tiles;
    valid_tiles = options.valid_tiles;
    output_file = options.output_file;
  }
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
  // Writing to stdout is fine for a human reading one kernel, but the exact
  // mapper is a separate process that takes a FILE, and a caller that wants to
  // solve a kernel mid-compilation cannot pick stdout back out of its own
  // output. With this set the dump goes to the named file instead.
  Option<std::string> output_file{
      *this, "output-file",
      llvm::cl::desc("Write the JSON here instead of stdout."),
      llvm::cl::init("")};

  // The instance handed to the exact CP-SAT mapper must be the SAME
  // architecture --map-to-accelerator and --cost-model-analytical price, or the
  // oracle answers a different question than the one asked; they therefore all
  // share buildShapedArchitecture.
  std::unique_ptr<Architecture>
  buildCustomArch(const Architecture &global_arch) {
    return buildShapedArchitecture(global_arch, x_tiles.getValue(),
                                   y_tiles.getValue(), valid_tiles.getValue(),
                                   "[dump-dfg-json]");
  }

  // Emits ONE region as a self-contained JSON object, with no trailing newline
  // and no separator, so the caller decides whether this object stands alone
  // (the single-region dump every existing consumer reads) or is one element of
  // an array. `name` is emitted only in the multi-region case, so the
  // single-region bytes are exactly what they were before.
  void emitRegion(Region &region, const Architecture &arch,
                  llvm::raw_ostream &os, StringRef name = StringRef()) {
    // Placed ops in the order --import-mapping replays onto, and the dependence
    // edges over them (forward operand edges with omega=0, ctrl_mov/reserve
    // back edges with omega=1, de-duplicated on (src, dst, omega)).
    std::vector<Operation *> placed_ops = collectPlacedOps(region);
    std::vector<DependenceEdge> edges = buildDfgEdges(region, placed_ops);

    // Architecture: per-FU-class tile support + mesh links + registers.
    os << "{\n";
    if (!name.empty()) {
      os << "  \"name\": \"" << escapeJsonString(name) << "\",\n";
    }
    os << "  \"arch\": {\n";
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
      // A class the table describes with no OperationKind constrains nothing,
      // so no key is emitted for it and the mapper's
      // `fu_class_tiles.get(c, all_tiles)` falls back to every tile -- the same
      // answer tilesProvidingFuClass gives for an undescribed class. An EMPTY
      // list below means the opposite: the class is real and no tile provides
      // it, so the op cannot be placed and the kernel is infeasible here.
      if (kinds.empty()) {
        continue;
      }
      std::string tile_id_list;
      for (Tile *tile : tilesProvidingFuClass(arch, fu_class_name)) {
        tile_id_list +=
            (tile_id_list.empty() ? "" : ", ") + std::to_string(tile->getId());
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
    os << "]\n}";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const Architecture &global_arch = mlir::neura::getArchitecture();
    std::unique_ptr<Architecture> custom_arch = buildCustomArch(global_arch);
    const Architecture &arch = custom_arch ? *custom_arch : global_arch;
    // Every region is collected BEFORE anything is written, because the shape
    // of a valid document depends on how many there are: one region is a bare
    // object (the format the exact mapper and the cost model already read), and
    // writing a second bare object after it produces two concatenated
    // top-level values, which is not JSON at all -- `json.load` stops at the
    // first and raises "Extra data". A stream that has already emitted the
    // first object cannot take that back, so the count has to be known first.
    SmallVector<std::pair<std::string, Region *>, 4> regions;
    unsigned walk_index = 0;
    auto tryCollect = [&](Operation *op, Region &region) {
      auto accel_attr = op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (accel_attr && accel_attr.getValue() == accel::kNeuraTarget &&
          !region.empty()) {
        regions.emplace_back(regionLabel(op, walk_index++), &region);
      }
    };
    module.walk(
        [&](neura::KernelOp kernel) { tryCollect(kernel, kernel.getBody()); });
    module.walk([&](func::FuncOp func) { tryCollect(func, func.getBody()); });
    // Fallback: nothing tagged for the accelerator -> emit the first non-empty
    // func so a hand-written kernel still works standalone.
    if (regions.empty()) {
      module.walk([&](func::FuncOp func) {
        if (regions.empty() && !func.getBody().empty())
          regions.emplace_back(regionLabel(func, 0), &func.getBody());
      });
    }
    std::unique_ptr<llvm::raw_fd_ostream> file_stream;
    if (!output_file.getValue().empty()) {
      std::error_code error;
      file_stream = std::make_unique<llvm::raw_fd_ostream>(
          output_file.getValue(), error, llvm::sys::fs::OF_Text);
      if (error) {
        llvm::errs() << "[dump-dfg-json] cannot write "
                     << output_file.getValue() << ": " << error.message()
                     << "\n";
        signalPassFailure();
        return;
      }
    }
    llvm::raw_ostream &os = file_stream ? *file_stream : llvm::outs();

    // Nothing to dump. The output file is still created (empty), exactly as
    // before, so a caller that opened it does not see a missing file.
    if (regions.empty()) {
      return;
    }

    if (regions.size() == 1) {
      // The historical, and by far the common, case: one instance, one object.
      // Byte-for-byte what this pass emitted before, so exact_mapper_cpsat.py
      // and every committed dump keep working unchanged.
      emitRegion(*regions.front().second, arch, os);
      os << "\n";
      return;
    }

    // More than one instance. An array is the only valid JSON document that can
    // carry them all, and each element is labelled so the reader can tell which
    // kernel it is looking at. Announced on stderr because the exact mapper
    // solves ONE instance and indexes the document as an object: it must not be
    // handed an array by accident and told nothing about it.
    llvm::errs() << "[dump-dfg-json] module has " << regions.size()
                 << " accelerator regions (";
    for (size_t i = 0; i < regions.size(); ++i) {
      llvm::errs() << (i ? ", " : "") << regions[i].first;
    }
    llvm::errs() << "); emitting a JSON array. Single-instance consumers such "
                    "as test/cost-model/exact_mapper_cpsat.py expect one "
                    "object -- select a single region before dumping.\n";

    os << "[\n";
    for (size_t i = 0; i < regions.size(); ++i) {
      emitRegion(*regions[i].second, arch, os, regions[i].first);
      os << (i + 1 < regions.size() ? ",\n" : "\n");
    }
    os << "]\n";
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createDumpDfgJsonPass(const DumpDfgJsonOptions &options) {
  return std::make_unique<DumpDfgJsonPass>(options);
}
} // namespace mlir::neura
