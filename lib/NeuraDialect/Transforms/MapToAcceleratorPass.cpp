#include <deque>
#include <fstream>
#include <memory>

#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
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
                     "accelerator. Options: heuristic (default)."),
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
  Option<std::string> importMapping{
      *this, "import-mapping",
      llvm::cl::desc(
          "Path to an exact-mapper placement JSON (op index -> tile "
          "+ time_step, produced by exact_mapper_cpsat.py --emit). "
          "When set, the search is skipped: each materialized op is "
          "placed at its given tile/time and the existing router "
          "wires the operands, reproducing the exact optimal II. The "
          "op order must match --dump-dfg-json (same lowered IR)."),
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

  // A single op placement read from an exact-mapper emit: which tile the op
  // sits on and at which (absolute) time step.
  struct ImportedPlace {
    int tile_id;
    int time_step;
  };

  // One hop of a value's concrete route: the tile it occupies and the cycle.
  struct RouteHop {
    int tile;
    int cycle;
  };
  // Concrete routes keyed by (producer op index, consumer op index). Each is
  // the ordered tile/cycle path the value travels (producer tile first).
  // Present only when the emit carries exact routes; then the importer replays
  // them instead of greedily re-routing.
  using RouteMap = std::map<std::pair<int, int>, std::vector<RouteHop>>;

  // Materialized ops in the SAME order --dump-dfg-json emits them, so the i-th
  // op here corresponds to placements[i] of the exact-mapper JSON. Uses the
  // shared neura::occupiesFU (the exact predicate DumpDfgJsonPass emits
  // with) so the two indexings cannot drift apart.
  static std::vector<Operation *>
  collectMaterializedInDumpOrder(Region &region) {
    std::vector<Operation *> materialized;
    region.walk([&](Operation *walked) {
      if (occupiesFU(walked)) {
        materialized.push_back(walked);
      }
    });
    return materialized;
  }

  // Parses an exact-mapper emit (compiled_ii + placements[{id,tile,time_step}]
  // and optional routes[{s,d,path}]).
  static bool parseImportedMapping(StringRef path, int &imported_ii,
                                   std::vector<ImportedPlace> &places,
                                   RouteMap &routes, std::string &err) {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
      err = "cannot open " + path.str();
      return false;
    }
    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
      err = "invalid JSON: " + llvm::toString(parsed.takeError());
      return false;
    }
    llvm::json::Object *root = parsed->getAsObject();
    if (!root) {
      err = "top-level JSON is not an object";
      return false;
    }
    std::optional<int64_t> ii = root->getInteger("compiled_ii");
    llvm::json::Array *placements = root->getArray("placements");
    if (!ii || !placements) {
      err = "missing compiled_ii or placements";
      return false;
    }
    imported_ii = static_cast<int>(*ii);
    // placements are indexed by their "id"; store them in id order so index i
    // lines up with the i-th materialized op.
    places.assign(placements->size(), ImportedPlace{-1, -1});
    for (llvm::json::Value &entry : *placements) {
      llvm::json::Object *record = entry.getAsObject();
      if (!record) {
        err = "placement entry is not an object";
        return false;
      }
      std::optional<int64_t> id = record->getInteger("id");
      std::optional<int64_t> tile = record->getInteger("tile");
      std::optional<int64_t> time_step = record->getInteger("time_step");
      if (!id || !tile || !time_step) {
        err = "placement entry missing id/tile/time_step";
        return false;
      }
      if (*id < 0 || *id >= static_cast<int64_t>(places.size())) {
        err = "placement id out of range";
        return false;
      }
      places[*id] =
          ImportedPlace{static_cast<int>(*tile), static_cast<int>(*time_step)};
    }
    // Optional exact routes.
    if (llvm::json::Array *route_arr = root->getArray("routes")) {
      for (llvm::json::Value &entry : *route_arr) {
        llvm::json::Object *record = entry.getAsObject();
        if (!record) {
          continue;
        }
        std::optional<int64_t> src_idx = record->getInteger("s");
        std::optional<int64_t> dst_idx = record->getInteger("d");
        llvm::json::Array *path = record->getArray("path");
        if (!src_idx || !dst_idx || !path) {
          continue;
        }
        std::vector<RouteHop> hops;
        for (llvm::json::Value &node : *path) {
          llvm::json::Array *hop_pair = node.getAsArray();
          if (!hop_pair || hop_pair->size() != 2) {
            continue;
          }
          std::optional<int64_t> hop_tile = (*hop_pair)[0].getAsInteger();
          std::optional<int64_t> hop_cycle = (*hop_pair)[1].getAsInteger();
          if (hop_tile && hop_cycle) {
            hops.push_back(RouteHop{static_cast<int>(*hop_tile),
                                    static_cast<int>(*hop_cycle)});
          }
        }
        routes[{static_cast<int>(*src_idx), static_cast<int>(*dst_idx)}] =
            std::move(hops);
      }
    }
    return true;
  }

  // Placement-only fallback: bind every op at its imported tile/time and let
  // the existing greedy router (placeAndRoute) wire the operands.
  //
  // NOTE: currently unexercised. It runs only when the imported JSON has NO
  // "routes" array, but the exact mapper always emits routes with --emit (and
  // its II+1 fallback guarantees a routable solution is found), so the normal
  // flow always takes placeImportedExact instead. Kept as a safety net for a
  // hypothetical placement-only emit -- and because greedy re-routing can fail
  // on large kernels where the joint solver would not, which is exactly why the
  // exact-route path exists.
  //
  // The IR must be the same lowered form --dump-dfg-json consumed, so op i <->
  // placements[i]. Returns false (leaving diagnostics) if the op count
  // disagrees or any op fails to place/route.
  bool placeImportedSolution(Region &region, const Architecture &architecture,
                             const std::vector<ImportedPlace> &places,
                             MappingState &mapping_state) {
    std::vector<Operation *> materialized =
        collectMaterializedInDumpOrder(region);
    if (materialized.size() != places.size()) {
      llvm::errs()
          << "[MapToAcceleratorPass] import-mapping op-count mismatch: "
          << materialized.size() << " materialized ops vs " << places.size()
          << " placements. The JSON must come from "
          << "--dump-dfg-json on this exact lowered IR.\n";
      return false;
    }
    // tile id -> tile resource, so an imported tile id becomes a MappingLoc.
    llvm::DenseMap<int, Tile *> tile_by_id;
    for (Tile *tile : architecture.getAllTiles()) {
      tile_by_id[tile->getId()] = tile;
    }

    for (size_t i = 0; i < materialized.size(); ++i) {
      Operation *op = materialized[i];
      auto found = tile_by_id.find(places[i].tile_id);
      if (found == tile_by_id.end()) {
        llvm::errs() << "[MapToAcceleratorPass] imported tile id "
                     << places[i].tile_id << " not in architecture\n";
        return false;
      }
      MappingLoc loc{found->second, places[i].time_step};
      if (!placeAndRoute(op, loc, mapping_state)) {
        llvm::errs() << "[MapToAcceleratorPass] imported placement failed to "
                        "route op #"
                     << i << " (" << op->getName() << ") at tile "
                     << places[i].tile_id << " t=" << places[i].time_step
                     << "\n";
        return false;
      }
    }
    return true;
  }

  // Turns one emitted route (ordered tile/cycle hops) into a MappingLoc path of
  // links and registers, the form MappingState::reserveRoute expects. Same-tile
  // runs become a register hold (one register for the whole run); a tile change
  // becomes the link between the two tiles. Returns false if a link/register
  // the route needs is unavailable (should not happen -- the solver already
  // proved the routing feasible under the same resource limits).
  bool buildRoutePath(const std::vector<RouteHop> &hops,
                      llvm::DenseMap<int, Tile *> &tile_by_id,
                      MappingState &mapping_state, neura::DataMovOp mov_op,
                      std::vector<MappingLoc> &out_path) {
    size_t hop_idx = 0;
    while (hop_idx + 1 < hops.size()) {
      if (hops[hop_idx].tile == hops[hop_idx + 1].tile) {
        // Maximal same-tile run -> a single register holds the value across it.
        size_t run_end = hop_idx;
        while (run_end + 1 < hops.size() &&
               hops[run_end + 1].tile == hops[hop_idx].tile) {
          ++run_end;
        }
        Tile *tile = tile_by_id[hops[hop_idx].tile];
        int cycle_begin = hops[hop_idx].cycle,
            cycle_end = hops[run_end].cycle; // exclusive
        Register *reg = getAvailableRegister(mapping_state, tile, cycle_begin,
                                             cycle_end, mov_op);
        if (!reg) {
          llvm::errs() << "[MapToAcceleratorPass] no free register on tile "
                       << hops[hop_idx].tile << " for [" << cycle_begin << ","
                       << cycle_end << ")\n";
          return false;
        }
        for (int cycle = cycle_begin; cycle < cycle_end; ++cycle) {
          out_path.push_back(MappingLoc{reg, cycle});
        }
        hop_idx = run_end;
      } else {
        // Tile change -> the link between the two tiles at this cycle.
        Tile *src = tile_by_id[hops[hop_idx].tile];
        Link *link = nullptr;
        for (Link *candidate : src->getOutLinks()) {
          if (candidate->getDstTile()->getId() == hops[hop_idx + 1].tile) {
            link = candidate;
            break;
          }
        }
        if (!link) {
          llvm::errs() << "[MapToAcceleratorPass] no link " << hops[hop_idx].tile
                       << "->" << hops[hop_idx + 1].tile << "\n";
          return false;
        }
        out_path.push_back(MappingLoc{link, hops[hop_idx].cycle});
        ++hop_idx;
      }
    }
    return true;
  }

  // Exact-route import: bind every op at its tile/time, then reserve the
  // SOLVER'S route for every value move (instead of greedily re-routing).
  // Because the routes are the joint solution, this reproduces the optimal
  // mapping even on large kernels where greedy per-net routing gets stuck.
  // Binding is done in a first pass so every producer location exists before
  // routes are reserved.
  bool placeImportedExact(Region &region, const Architecture &architecture,
                          const std::vector<ImportedPlace> &places,
                          const RouteMap &routes, MappingState &mapping_state) {
    std::vector<Operation *> materialized =
        collectMaterializedInDumpOrder(region);
    if (materialized.size() != places.size()) {
      llvm::errs()
          << "[MapToAcceleratorPass] import-mapping op-count mismatch: "
          << materialized.size() << " vs " << places.size() << "\n";
      return false;
    }
    llvm::DenseMap<int, Tile *> tile_by_id;
    for (Tile *tile : architecture.getAllTiles()) {
      tile_by_id[tile->getId()] = tile;
    }
    llvm::DenseMap<Operation *, int> op_index;
    for (size_t i = 0; i < materialized.size(); ++i) {
      op_index[materialized[i]] = (int)i;
    }

    // Pass 1: bind every op at its imported tile/time.
    for (size_t i = 0; i < materialized.size(); ++i) {
      Operation *op = materialized[i];
      auto found = tile_by_id.find(places[i].tile_id);
      if (found == tile_by_id.end()) {
        llvm::errs() << "[MapToAcceleratorPass] imported tile id "
                     << places[i].tile_id << " not in architecture\n";
        return false;
      }
      MappingLoc loc{found->second, places[i].time_step};
      int latency = getOpLatency(op);
      bool ok = latency > 1 ? mapping_state.bindMultiCycleOp(
                                  loc.resource, loc.time_step, latency, op)
                            : mapping_state.bindOp(loc, op);
      if (!ok) {
        llvm::errs() << "[MapToAcceleratorPass] failed to bind op #" << i
                     << " at tile " << places[i].tile_id << "\n";
        return false;
      }
    }

    // Pass 2: reserve the exact route of every value move.
    auto reserve = [&](Operation *mov, int producer_idx, int consumer_idx) {
      auto route_it = routes.find({producer_idx, consumer_idx});
      if (route_it == routes.end()) {
        llvm::errs() << "[MapToAcceleratorPass] missing route " << producer_idx
                     << "->" << consumer_idx << "\n";
        return false;
      }
      std::vector<MappingLoc> path;
      if (!buildRoutePath(route_it->second, tile_by_id, mapping_state,
                          dyn_cast<neura::DataMovOp>(mov), path)) {
        return false;
      }
      if (!path.empty()) {
        mapping_state.reserveRoute(mov, path);
      }
      return true;
    };
    for (size_t i = 0; i < materialized.size(); ++i) {
      Operation *op = materialized[i];
      // Forward operand moves.
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def || isa<neura::ReserveOp>(def)) {
          continue; // loop-carried placeholder handled via ctrl_mov below.
        }
        if (!isa<neura::DataMovOp>(def)) {
          continue;
        }
        Operation *producer = getMaterializedProducer(operand);
        if (!producer || !op_index.count(producer)) {
          continue;
        }
        if (!reserve(def, op_index[producer], (int)i)) {
          return false;
        }
      }
      // Backward (loop-carried) moves: op -> phi/reserve via ctrl_mov.
      for (Operation *user : getCtrlMovUsers(op)) {
        auto ctrl_mov = dyn_cast<neura::CtrlMovOp>(user);
        if (!ctrl_mov) {
          continue;
        }
        Operation *backward = getMaterializedBackwardUser(ctrl_mov);
        if (!backward || !op_index.count(backward)) {
          continue;
        }
        if (!reserve(ctrl_mov, (int)i, op_index[backward])) {
          return false;
        }
      }
    }
    return true;
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
    int rec_mii = 1;
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
      rec_mii = longest->length;
    } else if (!longest) {
      rec_mii = 1; // No recurrence cycles found, set MII to 1.
    }

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
    // Records a successful mapping (encode + dfg ids + mapping_info) for a
    // given II. Shared by the heuristic search and the imported-solution path.
    auto finalizeMapping = [&](MappingState &mapping_state, int ii,
                               StringRef strategy_label) {
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
                         StringAttr::get(ctx, strategy_label)));
      mapping_attrs.push_back(
          NamedAttribute(StringAttr::get(ctx, attr::kMappingMode),
                         StringAttr::get(ctx, resolved_mapping_mode)));
      mapping_attrs.push_back(
          NamedAttribute(StringAttr::get(ctx, attr::kCompiledII),
                         IntegerAttr::get(IntegerType::get(ctx, 32), ii)));
      mapping_attrs.push_back(
          NamedAttribute(StringAttr::get(ctx, attr::kRecMII),
                         IntegerAttr::get(IntegerType::get(ctx, 32), rec_mii)));
      mapping_attrs.push_back(
          NamedAttribute(StringAttr::get(ctx, attr::kResMII),
                         IntegerAttr::get(IntegerType::get(ctx, 32), res_mii)));
      op->setAttr(attr::kMappingInfo, DictionaryAttr::get(ctx, mapping_attrs));
    };

    // Imported exact solution: skip the search, place each materialized op at
    // its given tile/time, and let the existing router wire the operands. This
    // reproduces the exact-optimal II that a search-based mapper could not
    // reach (e.g. the CP-SAT mapper's provably optimal placement).
    if (!importMapping.getValue().empty()) {
      int imported_ii = 0;
      std::vector<ImportedPlace> places;
      RouteMap routes;
      std::string parse_err;
      if (!parseImportedMapping(importMapping.getValue(), imported_ii, places,
                                routes, parse_err)) {
        llvm::errs() << "[MapToAcceleratorPass] import-mapping error: "
                     << parse_err << "\n";
        return false;
      }
      llvm::errs() << "[MapToAcceleratorPass] Importing exact mapping ("
                   << places.size() << " ops, " << routes.size()
                   << " routes, II=" << imported_ii << ") from "
                   << importMapping.getValue() << "\n";
      MappingState mapping_state(architecture, imported_ii, is_spatial_only);
      // With exact routes, replay them (reproduces the solver's joint routing).
      // The routes.empty() branch is the placement-only fallback and is not hit
      // by the current emit, which always carries routes (see
      // placeImportedSolution).
      bool placed = routes.empty()
                        ? placeImportedSolution(region, architecture, places,
                                                mapping_state)
                        : placeImportedExact(region, architecture, places,
                                             routes, mapping_state);
      if (placed) {
        finalizeMapping(mapping_state, imported_ii, "exact-cpsat");
        return true;
      }
      llvm::errs() << "[MapToAcceleratorPass] Imported solution did not "
                      "place/route; no mapping produced.\n";
      return false;
    }

    for (int ii = possible_min_ii; ii <= max_ii; ++ii) {
      llvm::errs() << "[MapToAcceleratorPass] Start mapping with target II of "
                   << ii << "\n";
      // Creates a mapping state for the current II.
      MappingState mapping_state(architecture, ii, is_spatial_only);
      if (mapping_strategy->map(sorted_ops_with_alap_levels, critical_ops,
                                architecture, mapping_state)) {
        finalizeMapping(mapping_state, ii, resolved_mapping_strategy);
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
            TileOverride tile_override;
            tile_override.tile_x = x;
            tile_override.tile_y = y;
            tile_override.existence = false;
            additional_overrides.push_back(tile_override);
          }
        }

        // Then mark the valid ones as existent.
        for (llvm::StringRef coord : coords) {
          auto coord_pair = coord.split('_');
          int x, y;
          if (!coord_pair.first.getAsInteger(10, x) &&
              !coord_pair.second.getAsInteger(10, y)) {
            TileOverride tile_override;
            tile_override.tile_x = x;
            tile_override.tile_y = y;
            tile_override.existence = true;
            additional_overrides.push_back(tile_override);
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
