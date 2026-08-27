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
#include "NeuraDialect/Mapping/analytical_cost_model.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
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

  // Stores only the stable, printable summary; detailed diagnostics remain on
  // stderr so this pass does not change the mapper's mapping_info contract.
  void writeAttr(Operation *op, const AnalyticalIIBreakdown &breakdown) {
    MLIRContext *ctx = op->getContext();
    auto i32 = [&](int value) {
      return IntegerAttr::get(IntegerType::get(ctx, 32), value);
    };
    SmallVector<NamedAttribute, 10> attrs;
    auto add = [&](StringRef key, Attribute value) {
      attrs.push_back(NamedAttribute(StringAttr::get(ctx, key), value));
    };
    add("analytical_ii", i32(breakdown.final_ii));
    add("compute_mii", i32(breakdown.compute.value));
    add("rec_mii", i32(breakdown.rec.value));
    add("mem_mii", i32(breakdown.mem.value));
    add("route_mii", i32(breakdown.route.value));
    add("reg_mii", i32(breakdown.reg.value));
    add("res_mii", i32(breakdown.resource.value));
    add("max_ii", i32(breakdown.max_ii));
    add("dominant", StringAttr::get(ctx, breakdown.dominant));
    op->setAttr(kAnalyticalAttr, DictionaryAttr::get(ctx, attrs));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const Architecture &global_arch = mlir::neura::getArchitecture();
    std::unique_ptr<Architecture> custom = buildArchitectureForShape(
        global_arch, x_tiles.getValue(), y_tiles.getValue(), valid_tiles);
    const Architecture &arch = custom ? *custom : global_arch;

    int num_processed = 0;
    auto process = [&](Operation *op, Region &region, StringRef name) {
      if (region.empty()) {
        return;
      }
      AnalyticalIIBreakdown breakdown = computeAnalyticalII(region, arch);
      llvm::errs() << "[cost-model-analytical] region=" << name
                   << " tiles=" << arch.getNumTiles() << "\n";
      breakdown.print(llvm::errs());
      if (write_attr.getValue()) {
        writeAttr(op, breakdown);
      }
      ++num_processed;
    };

    bool any_accel = false;
    module.walk([&](neura::KernelOp kernel) {
      auto accel_attr =
          kernel->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (accel_attr && accel_attr.getValue() == accel::kNeuraTarget) {
        any_accel = true;
        process(kernel, kernel.getBody(), "kernel");
      }
    });
    module.walk([&](func::FuncOp func) {
      auto accel_attr =
          func->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (accel_attr && accel_attr.getValue() == accel::kNeuraTarget) {
        any_accel = true;
        process(func, func.getBody(), func.getName());
      }
    });

    assert(any_accel &&
           "cost-model-analytical requires a neura accelerator region");
    assert(num_processed > 0 && "no non-empty accelerator region processed");
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass>
createCostModelAnalyticalPass(const CostModelAnalyticalOptions &options) {
  return std::make_unique<CostModelAnalyticalPass>(options);
}
} // namespace mlir::neura
