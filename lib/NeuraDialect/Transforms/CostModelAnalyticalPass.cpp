//===- CostModelAnalyticalPass.cpp - Parametric single-task II predictor -===//
//
// Runs the analytical cost model (analytical_cost_model.h) on every Neura
// kernel/func targeting the accelerator and reports the predicted II and each
// resource bound. It NEVER invokes the mapper; --map-to-accelerator is the
// offline validation oracle. Results are recorded under a dedicated
// `analytical_cost_model` attribute so they never clash with the mapper's
// `mapping_info`.
//
//===----------------------------------------------------------------------===//

#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
#include "NeuraDialect/Mapping/analytical_cost_model.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <set>
#include <utility>
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_COSTMODELANALYTICAL
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

constexpr llvm::StringLiteral kAnalyticalAttr = "analytical_cost_model";

struct CostModelAnalyticalPass
    : public PassWrapper<CostModelAnalyticalPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CostModelAnalyticalPass)

  StringRef getArgument() const override { return "cost-model-analytical"; }
  StringRef getDescription() const override {
    return "Analytical (parametric) single-task II predictor (no mapper).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  CostModelAnalyticalPass() = default;
  CostModelAnalyticalPass(const CostModelAnalyticalOptions &options)
      : CostModelAnalyticalPass() {
    this->x_tiles = options.x_tiles;
    this->y_tiles = options.y_tiles;
    this->valid_tiles = options.valid_tiles;
    this->write_attr = options.write_attr;
  }
  CostModelAnalyticalPass(const CostModelAnalyticalPass &pass)
      : PassWrapper<CostModelAnalyticalPass, OperationPass<ModuleOp>>(pass) {}

  Option<int> x_tiles{
      *this, "x-tiles",
      llvm::cl::desc("Total tiles in X (0 = architecture singleton)."),
      llvm::cl::init(0)};
  Option<int> y_tiles{
      *this, "y-tiles",
      llvm::cl::desc("Total tiles in Y (0 = architecture singleton)."),
      llvm::cl::init(0)};
  Option<std::string> valid_tiles{
      *this, "valid-tiles",
      llvm::cl::desc("Comma-separated tile coords (x_y) for non-rect shapes."),
      llvm::cl::init("")};
  Option<bool> write_attr{
      *this, "write-attr",
      llvm::cl::desc("Record analytical_ii and bounds in an attribute."),
      llvm::cl::init(true)};

  // Builds the target architecture, honouring x/y-tiles + valid-tiles exactly
  // like --map-to-accelerator so predictions can be compared per CGRA shape.
  std::unique_ptr<Architecture>
  buildCustomArch(const Architecture &global_arch) {
    if (x_tiles.getValue() <= 0 || y_tiles.getValue() <= 0)
      return nullptr;
    std::vector<TileOverride> overrides;
    if (!valid_tiles.getValue().empty()) {
      // applyTileOverrides can only REMOVE tiles (existence=false), not re-add
      // them; so remove every tile NOT in the valid set, leaving valid ones.
      std::set<std::pair<int, int>> keep;
      llvm::SmallVector<llvm::StringRef, 4> coords;
      llvm::StringRef(valid_tiles.getValue()).split(coords, ',');
      for (llvm::StringRef coord : coords) {
        auto pr = coord.split('_');
        int x, y;
        if (!pr.first.getAsInteger(10, x) && !pr.second.getAsInteger(10, y))
          keep.insert({x, y});
      }
      for (int y = 0; y < y_tiles.getValue(); ++y)
        for (int x = 0; x < x_tiles.getValue(); ++x)
          if (!keep.count({x, y})) {
            TileOverride to;
            to.tile_x = x;
            to.tile_y = y;
            to.existence = false;
            overrides.push_back(to);
          }
    }
    return global_arch.cloneWithNewDimensions(y_tiles.getValue(),
                                              x_tiles.getValue(), overrides);
  }

  void writeAttr(Operation *op, const AnalyticalIIBreakdown &bd) {
    MLIRContext *ctx = op->getContext();
    auto i32 = [&](int v) {
      return IntegerAttr::get(IntegerType::get(ctx, 32), v);
    };
    SmallVector<NamedAttribute, 10> attrs;
    auto add = [&](StringRef k, Attribute v) {
      attrs.push_back(NamedAttribute(StringAttr::get(ctx, k), v));
    };
    add("analytical_ii", i32(bd.final_ii));
    add("res_mii", i32(bd.res.value));
    add("rec_mii", i32(bd.rec.value));
    add("mem_mii", i32(bd.mem.value));
    add("route_mii", i32(bd.route.value));
    add("reg_mii", i32(bd.reg.value));
    add("issue_mii", i32(bd.issue.value));
    add("max_ii", i32(bd.max_ii));
    add("dominant", StringAttr::get(ctx, bd.dominant));
    op->setAttr(kAnalyticalAttr, DictionaryAttr::get(ctx, attrs));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const Architecture &global_arch = mlir::neura::getArchitecture();
    std::unique_ptr<Architecture> custom = buildCustomArch(global_arch);
    const Architecture &arch = custom ? *custom : global_arch;

    int processed = 0;
    auto process = [&](Operation *op, Region &region, StringRef name) {
      if (region.empty())
        return;
      AnalyticalIIBreakdown bd = computeAnalyticalII(region, arch);
      llvm::errs() << "[cost-model-analytical] region=" << name
                   << " tiles=" << arch.getNumTiles() << "\n";
      bd.print(llvm::errs());
      if (write_attr.getValue())
        writeAttr(op, bd);
      ++processed;
    };

    bool any_accel = false;
    module.walk([&](neura::KernelOp k) {
      auto a = k->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (a && a.getValue() == accel::kNeuraTarget) {
        any_accel = true;
        process(k, k.getBody(), "kernel");
      }
    });
    module.walk([&](func::FuncOp f) {
      auto a = f->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (a && a.getValue() == accel::kNeuraTarget) {
        any_accel = true;
        process(f, f.getBody(), f.getName());
      }
    });

    // Fallback: no op is explicitly tagged for the accelerator (e.g. a
    // hand-written regression kernel). Process every non-empty func so the
    // model is still usable standalone.
    if (!any_accel) {
      module.walk([&](func::FuncOp f) { process(f, f.getBody(), f.getName()); });
    }

    if (processed == 0)
      llvm::errs() << "[cost-model-analytical] no regions processed\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass>
createCostModelAnalyticalPass(const CostModelAnalyticalOptions &options) {
  return std::make_unique<CostModelAnalyticalPass>(options);
}
} // namespace mlir::neura
