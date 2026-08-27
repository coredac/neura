//===- DumpDfgJsonPass.cpp - Emit pre-map DFG + arch as JSON -------------===//
//
// Emits the lowered Neura DFG (placed ops, dependence edges with a
// loop-carried iteration distance omega) together with the target CGRA
// architecture (tiles, per-FU-class tile support, mesh links, registers,
// ctrl_mem_items) as a JSON document. This feeds the exact modulo-scheduling
// oracle (test/cost-model/exact_mapper_cpsat.py) so the oracle solves the exact same
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
#include <string>

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_DUMPDFGJSON
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

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

  // Writes one lowered Neura region in the JSON contract consumed by the
  // external exact CP-SAT mapper.
  void emitRegion(Region &region, const Architecture &arch,
                  llvm::raw_ostream &os) {
    emitExactMapperJson(region, arch, os);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const Architecture &global_arch = mlir::neura::getArchitecture();
    std::unique_ptr<Architecture> custom_arch = buildArchitectureForShape(
        global_arch, x_tiles.getValue(), y_tiles.getValue(), valid_tiles);
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
    assert(emitted && "dump-dfg-json requires a non-empty neura accelerator region");
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createDumpDfgJsonPass() {
  return std::make_unique<DumpDfgJsonPass>();
}
} // namespace mlir::neura
