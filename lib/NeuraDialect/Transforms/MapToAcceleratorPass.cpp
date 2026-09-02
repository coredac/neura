#include <deque>
#include <fstream>
#include <memory>

#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/TemplateMapping/TemplateMapping.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "NeuraDialect/Util/NeuraYamlKeys.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;
using namespace mlir::neura::yamlkeys;

#define GEN_PASS_DEF_MAPTOACCELERATOR
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct MapToAcceleratorPass
    : public PassWrapper<MapToAcceleratorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapToAcceleratorPass)

  StringRef getArgument() const override { return "map-to-accelerator"; }
  StringRef getDescription() const override {
    return "Maps IR to the target accelerator.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  MapToAcceleratorPass() = default;
  MapToAcceleratorPass(const MapToAcceleratorOptions &options)
      : MapToAcceleratorPass() {
    this->x_tiles = options.x_tiles;
    this->y_tiles = options.y_tiles;
    this->valid_tiles = options.valid_tiles;
  }
  MapToAcceleratorPass(const MapToAcceleratorPass &pass)
      : PassWrapper<MapToAcceleratorPass, OperationPass<ModuleOp>>(pass) {}
  Option<std::string> mappingStrategy{
      *this, "mapping-strategy",
      llvm::cl::desc("Mapping strategy to use for mapping operations to the "
                     "accelerator. Options: heuristic (default) and template."),
      llvm::cl::init(attr::val::kHeuristic.str())};
  Option<std::string> mappingMode{
      *this, "mapping-mode",
      llvm::cl::desc(
          "Mapping mode to use for mapping operations to the "
          "accelerator. Options: spatial-only, spatial-temporal (default)."),
      llvm::cl::init(attr::val::kSpatialTemporal.str())};
  Option<std::string> backtrackConfig{
      *this, "backtrack-config",
      llvm::cl::desc(
          "Backtrack configuration used for mapping operations to the "
          "accelerator. Options: simple, greedy, exhaustive, "
          "customized=max_loc,max_depth (default "
          "max_loc=5, max_depth=3)"),
      llvm::cl::init(attr::val::kCustomized.str())};
  Option<bool> dumpMappingTable{
      *this, "dump-mapping-table",
      llvm::cl::desc(
          "Dump the resource allocation table after mapping (default: true)"),
      llvm::cl::init(true)};
  Option<int> x_tiles{
      *this, "x-tiles",
      llvm::cl::desc("Override number of tiles in X dimension (0 = default)."),
      llvm::cl::init(0)};
  Option<int> y_tiles{
      *this, "y-tiles",
      llvm::cl::desc("Override number of tiles in Y dimension (0 = default)."),
      llvm::cl::init(0)};
  Option<std::string> valid_tiles{
      *this, "valid-tiles",
      llvm::cl::desc("Comma separated list of valid tile coords x_y,x_y to "
                     "support non-rectangular shapes."),
      llvm::cl::init("")};

  // Configures mapping strategy and mode based on command-line options.
  bool configureMappingStrategy(StringRef mapping_strategy_opt,
                                StringRef backtrack_config_opt,
                                StringRef mapping_mode_opt,
                                std::unique_ptr<Mapping> &mapping_strategy,
                                std::string &resolved_mapping_mode,
                                std::string &resolved_mapping_strategy,
                                bool &is_spatial_only) {
    StringRef mapping_mode_str = mapping_mode_opt;
    if (mapping_mode_str.empty()) {
      mapping_mode_str = attr::val::kSpatialTemporal;
    }
    if (mapping_mode_str == attr::val::kSpatialOnly ||
        mapping_mode_str == attr::val::kSpatialTemporal) {
      llvm::errs() << "[MapToAcceleratorPass] Using Mapping Mode: "
                   << mapping_mode_str << "\n";
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping mode: "
                   << mapping_mode_str << "\n";
      return false;
    }
    resolved_mapping_mode = mapping_mode_str.str();
    is_spatial_only = (mapping_mode_str == attr::val::kSpatialOnly);

    StringRef mapping_strategy_str = mapping_strategy_opt;
    if (mapping_strategy_str.empty()) {
      mapping_strategy_str = attr::val::kHeuristic;
    }

    // The template strategy uses the placement written by a spatial template.
    // It currently supports spatial-only mapping because every materialized
    // operation is assigned to a dedicated hardware tile.
    if (mapping_strategy_str == attr::val::kTemplate) {
      if (!is_spatial_only) {
        llvm::errs() << "[MapToAcceleratorPass] Template mapping requires "
                        "spatial-only mapping mode.\n";
        return false;
      }

      mapping_strategy = std::make_unique<TemplateMapping>();
      resolved_mapping_strategy = attr::val::kTemplate.str();
      return true;
    }

    StringRef backtrack_str = backtrack_config_opt;
    if (mapping_strategy_str.empty() ||
        mapping_strategy_str == attr::val::kHeuristic) {
      if (backtrack_str.empty()) {
        backtrack_str = attr::val::kHeuristic;
      }
      if (backtrack_str == attr::val::kSimple) {
        mapping_strategy = std::make_unique<HeuristicMapping>(1, 1);
      } else if (backtrack_str == attr::val::kGreedy) {
        mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, 1);
      } else if (backtrack_str == attr::val::kExhaustive) {
        mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, INT_MAX);
      } else if (backtrack_str == attr::val::kCustomized) {
        mapping_strategy = std::make_unique<HeuristicMapping>(5, 3);
      } else if (backtrack_str.starts_with("customized=")) {
        StringRef params = backtrack_str.substr(strlen("customized="));
        size_t comma_pos = params.find(',');
        if (comma_pos != StringRef::npos) {
          StringRef max_loc_str = params.substr(0, comma_pos);
          StringRef max_depth_str = params.substr(comma_pos + 1);
          int max_loc = 0, max_depth = 0;
          if (!max_loc_str.getAsInteger(10, max_loc) &&
              !max_depth_str.getAsInteger(10, max_depth)) {
            mapping_strategy =
                std::make_unique<HeuristicMapping>(max_loc, max_depth);
            llvm::errs()
                << "[MapToAcceleratorPass] Use custom backtrack parameters: "
                << "max_location_to_try=" << max_loc
                << ", max_backtrack_depth=" << max_depth << "\n";
          } else {
            llvm::errs() << "[MapToAcceleratorPass] Illegal customized "
                            "parameters format: "
                         << backtrack_str << "\n";
            return false;
          }
        } else {
          llvm::errs() << "[MapToAcceleratorPass] Illegal customized "
                          "parameters format: "
                       << backtrack_str << "\n";
          return false;
        }
      } else {
        llvm::errs() << "[MapToAcceleratorPass] Unsupported backtrack config: "
                     << backtrack_str << "\n";
        return false;
      }
      resolved_mapping_strategy = mapping_strategy_str.str();
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping strategy: "
                   << mapping_strategy_str << "\n";
      return false;
    }
    return true;
  }

  // Assigns unique dfg_id to all operations in SSA topological order.
  void assignDfgIdsInRegion(Region &region, int &next_id) {
    // Uses existing topological sort to get all operations in order.
    std::vector<Operation *> sorted_ops = getTopologicallySortedOps(region);

    auto ctx = region.getContext();

    // Assigns ID to each operation in topological order.
    for (Operation *op : sorted_ops) {
      op->setAttr(attr::kDfgId,
                  IntegerAttr::get(IntegerType::get(ctx, 32), next_id));
      llvm::errs() << "[MapToAcceleratorPass] Assigned dfg_id=" << next_id
                   << " to " << *op << "\n";
      next_id++;
    }

    llvm::errs() << "[MapToAcceleratorPass] Assigned " << next_id
                 << " dfg_id(s) in total\n";
  }

  // Generic mapping function works for both function and kernel mapping.
  template <typename OpType>
  bool mapRegion(OpType op, Region &region, const Architecture &architecture,
                 Mapping *mapping_strategy, bool is_spatial_only,
                 const std::string &resolved_mapping_mode,
                 const std::string &resolved_mapping_strategy) {
    // Checks steering mode compatibility with architecture.
    auto dataflow_mode_attr =
        op->template getAttrOfType<StringAttr>(attr::kDataflowMode);
    bool is_steering_mode =
        (dataflow_mode_attr &&
         dataflow_mode_attr.getValue() == attr::val::kModeSteering);
    if (is_steering_mode) {
      if (!is_spatial_only) {
        op.emitError()
            << "Steering mode mapping only supports spatial-only mapping mode.";
        return false;
      }
    }

    // Collects and reports recurrence cycles found in the function.
    auto recurrence_cycles = collectRecurrenceCycles(region);
    std::set<Operation *> critical_ops;
    RecurrenceCycle *longest = nullptr;
    for (auto &cycle : recurrence_cycles) {
      llvm::outs() << "[DEBUG] Recurrence cycle (length " << cycle.length
                   << "):\n";
      for (Operation *op : cycle.operations) {
        critical_ops.insert(op);
        llvm::outs() << "  " << *op << "\n";
      }
      if (!longest || cycle.length > longest->length) {
        longest = &cycle;
      }
    }

    if (longest) {
      llvm::outs() << "[MapToAcceleratorPass] Longest recurrence cycle (length "
                   << longest->length << "):\n";
      for (Operation *op : longest->operations) {
        op->print(llvm::outs()), llvm::outs() << "\n";
      }
    }

    int rec_mii = calculateRecMii(recurrence_cycles);

    llvm::errs() << "[MapToAcceleratorPass] Calculated Recurrence MII: "
                 << rec_mii << "\n";

    int res_mii = calculateResMii(region, architecture);

    const int possible_min_ii = std::max(rec_mii, res_mii);
    const int max_ii = architecture.getMaxCtrlMemItems();

    std::vector<Operation *> topologically_sorted_ops =
        getTopologicallySortedOps(region);
    if (topologically_sorted_ops.empty()) {
      assert(false && "Mapping aborted due to empty op list.");
    }

    // Filters out operations inside fused_op regions.
    // Only maps the fused_op itself, not the operations within its region.
    std::vector<Operation *> filtered_ops;
    int skipped_count = 0;
    for (Operation *op : topologically_sorted_ops) {
      Operation *parent_op = op->getParentOp();
      // Checks if the parent is a fused_op by inspecting the operation name.
      if (parent_op &&
          parent_op->getName().getStringRef().contains(attr::val::kOpFused)) {
        // Skips operations inside a fused_op region.
        llvm::outs() << "[MapToAcceleratorPass] Skipping op inside fused_op: "
                     << *op << "\n";
        skipped_count++;
        continue;
      }
      filtered_ops.push_back(op);
    }
    topologically_sorted_ops = std::move(filtered_ops);

    if (skipped_count > 0) {
      llvm::errs() << "[MapToAcceleratorPass] Filtered out " << skipped_count
                   << " operations inside fused_op regions\n";
    }

    for (Operation *op : topologically_sorted_ops) {
      llvm::outs() << "[MapToAcceleratorPass] Topologically sorted op: " << *op
                   << "\n";
    }
    std::vector<std::vector<Operation *>> level_buckets =
        getOpsInAlapLevels(topologically_sorted_ops, critical_ops);
    for (int level = 0; level < static_cast<int>(level_buckets.size());
         ++level) {
      llvm::outs() << "[MapToAcceleratorPass] ALAP Bucket Level " << level
                   << ": " << level_buckets[level].size() << " ops\n";
      for (Operation *op : level_buckets[level]) {
        llvm::outs() << "  " << *op << "\n";
      }
    }
    std::vector<std::pair<Operation *, int>> sorted_ops_with_alap_levels =
        flatten_level_buckets(level_buckets, critical_ops);
    for (const auto &[op, level] : sorted_ops_with_alap_levels) {
      llvm::outs() << "[MapToAcceleratorPass] ALAP sorted op: " << *op
                   << " (ALAP level: " << level << ")\n";
    }
    // assert(false);
    for (int ii = possible_min_ii; ii <= max_ii; ++ii) {
      llvm::errs() << "[MapToAcceleratorPass] Start mapping with target II of "
                   << ii << "\n";
      // Creates a mapping state for the current II.
      MappingState mapping_state(architecture, ii, is_spatial_only);
      if (mapping_strategy->map(sorted_ops_with_alap_levels, critical_ops,
                                architecture, mapping_state)) {
        // Success.
        if (dumpMappingTable) {
          // Logs to stderr.
          mapping_state.dumpOpToLocs();
        }
        mapping_state.encodeMappingState();

        // Assigns unique dfg_id to all operations in SSA topological order.
        int next_id = 0;
        assignDfgIdsInRegion(region, next_id);

        // Sets the mapping_info attribute on the function.
        auto ctx = op->getContext();
        SmallVector<NamedAttribute, 8> mapping_attrs;
        mapping_attrs.push_back(
            NamedAttribute(StringAttr::get(ctx, attr::kXTiles),
                           IntegerAttr::get(IntegerType::get(ctx, 32),
                                            architecture.getPerCgraColumns())));
        mapping_attrs.push_back(
            NamedAttribute(StringAttr::get(ctx, attr::kYTiles),
                           IntegerAttr::get(IntegerType::get(ctx, 32),
                                            architecture.getPerCgraRows())));
        mapping_attrs.push_back(
            NamedAttribute(StringAttr::get(ctx, attr::kMappingStrategy),
                           StringAttr::get(ctx, resolved_mapping_strategy)));
        mapping_attrs.push_back(
            NamedAttribute(StringAttr::get(ctx, attr::kMappingMode),
                           StringAttr::get(ctx, resolved_mapping_mode)));
        mapping_attrs.push_back(
            NamedAttribute(StringAttr::get(ctx, attr::kCompiledII),
                           IntegerAttr::get(IntegerType::get(ctx, 32), ii)));
        mapping_attrs.push_back(NamedAttribute(
            StringAttr::get(ctx, attr::kRecMII),
            IntegerAttr::get(IntegerType::get(ctx, 32), rec_mii)));
        mapping_attrs.push_back(NamedAttribute(
            StringAttr::get(ctx, attr::kResMII),
            IntegerAttr::get(IntegerType::get(ctx, 32), res_mii)));
        DictionaryAttr mapping_info = DictionaryAttr::get(ctx, mapping_attrs);

        op->setAttr(attr::kMappingInfo, mapping_info);
        return true;
      }
      llvm::errs() << "[MapToAcceleratorPass] Mapping failed for target II of "
                   << ii << "\n";
      mapping_state.dumpOpToLocs();
    }
    llvm::errs()
        << "[MapToAcceleratorPass] Mapping failed for all target II values.\n";
    return false;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::errs() << "[MapToAcceleratorPass] Starting mapping pass...\n";
    std::unique_ptr<Mapping> mapping_strategy;
    std::string resolved_mapping_mode;
    std::string resolved_mapping_strategy;
    bool is_spatial_only = false;
    if (!configureMappingStrategy(
            mappingStrategy.getValue(), backtrackConfig.getValue(),
            mappingMode.getValue(), mapping_strategy, resolved_mapping_mode,
            resolved_mapping_strategy, is_spatial_only)) {
      return;
    }

    const Architecture &global_arch = mlir::neura::getArchitecture();
    std::unique_ptr<Architecture> custom_arch;
    const Architecture *target_arch = &global_arch;

    if (x_tiles.getValue() > 0 && y_tiles.getValue() > 0) {
      std::vector<TileOverride> additional_overrides;
      if (!valid_tiles.getValue().empty()) {
        llvm::SmallVector<llvm::StringRef, 4> coords;
        llvm::StringRef(valid_tiles.getValue()).split(coords, ',');

        // Default: mark all tiles as non-existent first if valid_tiles
        // provided.
        for (int y = 0; y < y_tiles.getValue(); ++y) {
          for (int x = 0; x < x_tiles.getValue(); ++x) {
            TileOverride to;
            to.tile_x = x;
            to.tile_y = y;
            to.existence = false;
            additional_overrides.push_back(to);
          }
        }

        // Then mark the valid ones as existent.
        for (llvm::StringRef coord : coords) {
          auto pair = coord.split('_');
          int x, y;
          if (!pair.first.getAsInteger(10, x) &&
              !pair.second.getAsInteger(10, y)) {
            TileOverride to;
            to.tile_x = x;
            to.tile_y = y;
            to.existence = true;
            additional_overrides.push_back(to);
          }
        }
      }

      // Builds a custom architecture with the requested tile dimensions.
      // For non-rectangular shapes, tiles marked existence=false are removed
      // before inter-tile links are created, so no boundary links connect to
      // absent tiles.
      custom_arch = global_arch.cloneWithNewDimensions(
          y_tiles.getValue(), x_tiles.getValue(), additional_overrides);
      target_arch = custom_arch.get();
      llvm::errs()
          << "[MapToAcceleratorPass] Overriding architecture dimensions to "
          << y_tiles.getValue() << "x" << x_tiles.getValue() << " tiles.\n";
    }

    const Architecture &architecture = *target_arch;

    // Maps kernels.
    module.walk([&](neura::KernelOp kernel_op) {
      auto accel_attr =
          kernel_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        return;
      }

      Region &kernel_region = kernel_op.getBody();
      if (!mapRegion(kernel_op, kernel_region, architecture,
                     mapping_strategy.get(), is_spatial_only,
                     resolved_mapping_mode, resolved_mapping_strategy)) {
        llvm::errs() << "[MapToAcceleratorPass] Mapping failed for kernel.\n";
        signalPassFailure();
      }
    });

    // Maps functions.
    module.walk([&](func::FuncOp func_op) {
      auto accel_attr =
          func_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        return;
      }

      Region &func_region = func_op.getBody();

      if (!mapRegion(func_op, func_region, architecture, mapping_strategy.get(),
                     is_spatial_only, resolved_mapping_mode,
                     resolved_mapping_strategy)) {
        llvm::errs() << "[MapToAcceleratorPass] Failed to map function.\n";
        signalPassFailure();
      }
    });
  }
};

} // namespace

namespace mlir::neura {

std::unique_ptr<Pass>
createMapToAcceleratorPass(const MapToAcceleratorOptions &options) {
  return std::make_unique<MapToAcceleratorPass>(options);
}

} // namespace mlir::neura
