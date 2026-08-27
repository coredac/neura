#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <utility>

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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
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
          "Path to a CP-SAT mapping witness (op index -> tile "
          "+ time_step, produced by exact_mapper_cpsat.py --emit). "
          "When set, the search is skipped: each materialized op is "
          "placed at its given tile/time and the emitted routes are replayed. "
          "The op order must match --dump-dfg-json (same lowered IR)."),
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
    std::string fu_class;
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

  // Rejects an imported placement that binds an op to a tile whose functional
  // units cannot run it.
  //
  // The internal mapper never even considers such a location: its candidate
  // filter (mapping_util) drops every tile that does not provide the op's FU.
  // The import path used to skip that filter entirely and bind whatever the
  // JSON said, so an external solver that models FU classes differently -- or a
  // JSON that is simply stale with respect to the architecture spec -- produced
  // a mapping the hardware cannot execute while the compiler still reported
  // success (mapping_strategy = "exact-cpsat", a compiled_ii, generated code).
  // Since --import-mapping exists to replay an external solver's witness so
  // its II can be measured, that silent acceptance turns straight into a
  // fabricated result; failing is the only safe answer.
  //
  // Legality is asked through tilesProvidingFuClass(fuClassOf(op)) -- the same
  // single source of truth --dump-dfg-json uses to TELL the external solver
  // which tiles each FU class has ("fu_class_tiles"). So this verifies exactly
  // the constraint the solver was handed, including its two distinct cases: a
  // class the FU table does not describe is UNCONSTRAINED (every tile is legal,
  // matching the solver's `fu_class_tiles.get(c, all_tiles)` fallback), while a
  // described class that no tile provides is EMPTY (no legal tile at all).
  //
  // Reports every violation before returning, so a systematically wrong emit
  // does not have to be debugged one op at a time.
  static bool
  validateImportedFuLegality(const std::vector<Operation *> &materialized,
                             const Architecture &architecture,
                             const std::vector<ImportedPlace> &places) {
    llvm::StringMap<llvm::DenseSet<int>> legal_tiles_by_fu_class;
    bool all_legal = true;
    for (size_t i = 0; i < materialized.size(); ++i) {
      Operation *op = materialized[i];
      std::string fu_class = fuClassOf(op);
      if (places[i].fu_class != fu_class) {
        llvm::errs() << "[MapToAcceleratorPass] import-mapping placement for "
                        "op #"
                     << i << " (" << op->getName() << ") declares FU class \""
                     << places[i].fu_class << "\", but this IR maps it to \""
                     << fu_class << "\".\n";
        all_legal = false;
        continue;
      }
      auto found = legal_tiles_by_fu_class.find(fu_class);
      if (found == legal_tiles_by_fu_class.end()) {
        llvm::DenseSet<int> tile_ids;
        for (Tile *tile : tilesProvidingFuClass(architecture, fu_class)) {
          tile_ids.insert(tile->getId());
        }
        found = legal_tiles_by_fu_class.insert({fu_class, std::move(tile_ids)})
                    .first;
      }
      if (!found->second.contains(places[i].tile_id)) {
        llvm::errs() << "[MapToAcceleratorPass] import-mapping illegal "
                        "placement: op #"
                     << i << " (" << op->getName() << ", fu class \""
                     << fu_class << "\") placed on tile " << places[i].tile_id
                     << ", whose functional units cannot run it. Legal tiles: ";
        if (found->second.empty()) {
          llvm::errs() << "(none on this architecture)";
        } else {
          llvm::SmallVector<int> sorted_ids(found->second.begin(),
                                            found->second.end());
          llvm::sort(sorted_ids);
          llvm::interleaveComma(sorted_ids, llvm::errs());
        }
        llvm::errs() << ".\n";
        all_legal = false;
      }
    }
    if (!all_legal) {
      llvm::errs() << "[MapToAcceleratorPass] import-mapping rejected: the "
                      "imported solution is not executable on this "
                      "architecture. Re-run the external mapper against the "
                      "architecture spec in use (--dump-dfg-json emits its FU "
                      "class -> tile table as \"fu_class_tiles\").\n";
    }
    return all_legal;
  }

  // Checks that an imported route is the route for an actual DFG edge and that
  // every hop agrees with the architecture's directed links and latencies.
  // The JSON records a tile/cycle at each node: a same-tile step is one
  // register hold, while a tile change is one link traversal whose cycle delta
  // is that link's latency. Endpoints are checked against the producer's ready
  // cycle and the consumer's omega-shifted deadline before any state changes.
  static bool validateImportedRoutes(
      Region &region, const std::vector<Operation *> &materialized,
      const std::vector<ImportedPlace> &places, const RouteMap &routes, int ii,
      const llvm::DenseMap<int, Tile *> &tile_by_id, std::string &err) {
    std::map<std::pair<int, int>, int> expected;
    for (const DependenceEdge &edge : buildDfgEdges(region, materialized)) {
      auto key = std::make_pair(edge.src, edge.dst);
      auto found = expected.find(key);
      if (found != expected.end() && found->second != edge.omega) {
        err = "DFG contains both forward and loop-carried routes for " +
              std::to_string(edge.src) + "->" + std::to_string(edge.dst) +
              "; the imported route schema cannot disambiguate them";
        return false;
      }
      expected[key] = edge.omega;
    }

    for (const auto &[key, omega] : expected) {
      if (routes.find(key) == routes.end()) {
        err = "missing route " + std::to_string(key.first) + "->" +
              std::to_string(key.second);
        return false;
      }
      (void)omega;
    }
    for (const auto &[key, hops] : routes) {
      if (expected.find(key) == expected.end()) {
        err = "route " + std::to_string(key.first) + "->" +
              std::to_string(key.second) + " is not a DFG edge";
        return false;
      }
      if (hops.empty()) {
        err = "route " + std::to_string(key.first) + "->" +
              std::to_string(key.second) + " has an empty path";
        return false;
      }
      if (hops.size() == 1 &&
          hops.front().cycle == std::numeric_limits<int>::max()) {
        err = "route " + std::to_string(key.first) + "->" +
              std::to_string(key.second) +
              " has a one-node path at the maximum cycle value";
        return false;
      }

      const int src_idx = key.first;
      const int dst_idx = key.second;
      const int omega = expected.find(key)->second;
      const int64_t ready_cycle =
          static_cast<int64_t>(places[src_idx].time_step) +
          std::max(1, getOpLatency(materialized[src_idx]));
      const int64_t deadline = static_cast<int64_t>(places[dst_idx].time_step) +
                               static_cast<int64_t>(omega) * ii;
      if (hops.front().tile != places[src_idx].tile_id ||
          hops.front().cycle != ready_cycle) {
        err = "route " + std::to_string(src_idx) + "->" +
              std::to_string(dst_idx) + " starts at tile/cycle " +
              std::to_string(hops.front().tile) + "/" +
              std::to_string(hops.front().cycle) + ", expected producer " +
              std::to_string(places[src_idx].tile_id) + "/" +
              std::to_string(ready_cycle);
        return false;
      }
      if (hops.back().tile != places[dst_idx].tile_id ||
          hops.back().cycle != deadline) {
        err = "route " + std::to_string(src_idx) + "->" +
              std::to_string(dst_idx) + " ends at tile/cycle " +
              std::to_string(hops.back().tile) + "/" +
              std::to_string(hops.back().cycle) + ", expected consumer " +
              std::to_string(places[dst_idx].tile_id) + "/" +
              std::to_string(deadline);
        return false;
      }

      for (size_t hop_idx = 0; hop_idx < hops.size(); ++hop_idx) {
        if (tile_by_id.find(hops[hop_idx].tile) == tile_by_id.end()) {
          err = "route " + std::to_string(src_idx) + "->" +
                std::to_string(dst_idx) + " references unknown tile " +
                std::to_string(hops[hop_idx].tile);
          return false;
        }
        if (hop_idx == 0)
          continue;

        const RouteHop &previous = hops[hop_idx - 1];
        const RouteHop &current = hops[hop_idx];
        if (current.tile == previous.tile) {
          if (static_cast<int64_t>(current.cycle) !=
              static_cast<int64_t>(previous.cycle) + 1) {
            err = "route " + std::to_string(src_idx) + "->" +
                  std::to_string(dst_idx) +
                  " has a same-tile hop that is not one cycle after its "
                  "predecessor";
            return false;
          }
          continue;
        }

        Tile *src_tile = tile_by_id.lookup(previous.tile);
        Tile *dst_tile = tile_by_id.lookup(current.tile);
        Link *link = nullptr;
        for (Link *candidate : src_tile->getOutLinks()) {
          if (candidate->getDstTile() == dst_tile) {
            link = candidate;
            break;
          }
        }
        if (!link) {
          err = "route " + std::to_string(src_idx) + "->" +
                std::to_string(dst_idx) + " has no link " +
                std::to_string(previous.tile) + "->" +
                std::to_string(current.tile);
          return false;
        }
        if (link->getLatency() <= 0) {
          err = "route " + std::to_string(src_idx) + "->" +
                std::to_string(dst_idx) + " uses link " +
                std::to_string(previous.tile) + "->" +
                std::to_string(current.tile) + " with non-positive latency";
          return false;
        }
        const int64_t cycle_delta =
            static_cast<int64_t>(current.cycle) - previous.cycle;
        if (cycle_delta != link->getLatency()) {
          err = "route " + std::to_string(src_idx) + "->" +
                std::to_string(dst_idx) + " uses link " +
                std::to_string(previous.tile) + "->" +
                std::to_string(current.tile) + " with cycle delta " +
                std::to_string(cycle_delta) + ", expected link latency " +
                std::to_string(link->getLatency());
          return false;
        }
      }
    }
    return true;
  }

  // Parses an imported mapping witness (compiled_ii + placements[{id,class,
  // tile,time_step}] and optional routes[{s,d,path}]). Parsing is deliberately
  // strict: malformed records must not disappear and leave a partial mapping
  // that appears valid to a later pass.
  static bool parseImportedMapping(StringRef path, int &imported_ii,
                                   std::vector<ImportedPlace> &places,
                                   RouteMap &routes, bool &has_routes,
                                   std::string &err) {
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
    if (placements->empty()) {
      err = "placements must not be empty";
      return false;
    }
    if (*ii <= 0 || *ii > std::numeric_limits<int>::max()) {
      err = "compiled_ii must be a positive 32-bit integer";
      return false;
    }
    imported_ii = static_cast<int>(*ii);
    // placements are indexed by their "id"; store them in id order so index i
    // lines up with the i-th materialized op.
    places.assign(placements->size(), ImportedPlace{-1, -1, ""});
    std::vector<bool> seen(placements->size(), false);
    for (llvm::json::Value &entry : *placements) {
      llvm::json::Object *record = entry.getAsObject();
      if (!record) {
        err = "placement entry is not an object";
        return false;
      }
      std::optional<int64_t> id = record->getInteger("id");
      std::optional<llvm::StringRef> fu_class = record->getString("class");
      std::optional<int64_t> tile = record->getInteger("tile");
      std::optional<int64_t> time_step = record->getInteger("time_step");
      if (!id || !fu_class || !tile || !time_step) {
        err = "placement entry missing id/class/tile/time_step";
        return false;
      }
      if (*id < 0 || *id >= static_cast<int64_t>(places.size())) {
        err = "placement id out of range";
        return false;
      }
      if (*tile < 0 || *tile > std::numeric_limits<int>::max() ||
          *time_step < 0 || *time_step > std::numeric_limits<int>::max()) {
        err = "placement tile and time_step must be non-negative 32-bit "
              "integers";
        return false;
      }
      if (fu_class->empty()) {
        err = "placement class must not be empty";
        return false;
      }
      if (seen[*id]) {
        err = "duplicate placement id " + std::to_string(*id);
        return false;
      }
      seen[*id] = true;
      places[*id] =
          ImportedPlace{static_cast<int>(*tile), static_cast<int>(*time_step),
                        fu_class->str()};
    }
    for (size_t id = 0; id < seen.size(); ++id) {
      if (!seen[id]) {
        err = "missing placement id " + std::to_string(id);
        return false;
      }
    }
    // Optional route witness. If the key is present, it must be an array and
    // every record/node in it must have the documented shape.
    has_routes = root->get("routes") != nullptr;
    if (has_routes) {
      llvm::json::Array *route_arr = root->getArray("routes");
      if (!route_arr) {
        err = "routes must be an array";
        return false;
      }
      for (llvm::json::Value &entry : *route_arr) {
        llvm::json::Object *record = entry.getAsObject();
        if (!record) {
          err = "route entry is not an object";
          return false;
        }
        std::optional<int64_t> src_idx = record->getInteger("s");
        std::optional<int64_t> dst_idx = record->getInteger("d");
        llvm::json::Array *path = record->getArray("path");
        if (!src_idx || !dst_idx || !path) {
          err = "route entry missing s/d/path";
          return false;
        }
        if (*src_idx < 0 || *dst_idx < 0 ||
            *src_idx >= static_cast<int64_t>(places.size()) ||
            *dst_idx >= static_cast<int64_t>(places.size())) {
          err = "route endpoint index out of range";
          return false;
        }
        if (path->empty()) {
          err = "route path must contain at least its source hop";
          return false;
        }
        std::vector<RouteHop> hops;
        for (llvm::json::Value &node : *path) {
          llvm::json::Array *hop_pair = node.getAsArray();
          if (!hop_pair || hop_pair->size() != 2) {
            err = "route path hop must be a [tile, cycle] pair";
            return false;
          }
          std::optional<int64_t> hop_tile = (*hop_pair)[0].getAsInteger();
          std::optional<int64_t> hop_cycle = (*hop_pair)[1].getAsInteger();
          if (!hop_tile || !hop_cycle || *hop_tile < 0 || *hop_cycle < 0 ||
              *hop_tile > std::numeric_limits<int>::max() ||
              *hop_cycle > std::numeric_limits<int>::max()) {
            err = "route path tile and cycle must be non-negative 32-bit "
                  "integers";
            return false;
          }
          hops.push_back(RouteHop{static_cast<int>(*hop_tile),
                                  static_cast<int>(*hop_cycle)});
        }
        auto key = std::make_pair(static_cast<int>(*src_idx),
                                  static_cast<int>(*dst_idx));
        if (!routes.emplace(key, std::move(hops)).second) {
          err = "duplicate route " + std::to_string(key.first) + "->" +
                std::to_string(key.second);
          return false;
        }
      }
    }
    return true;
  }

  // Placement-only fallback: bind every op at its imported tile/time and let
  // the existing greedy router (placeAndRoute) wire the operands.
  //
  // NOTE: currently unexercised. It runs only when the imported JSON has NO
  // "routes" array. The CP-SAT emitter normally includes routes with --emit;
  // this remains a safety net for a placement-only witness.
  //
  // The IR must be the same lowered form --dump-dfg-json consumed, so op i <->
  // placements[i]. Returns false (leaving diagnostics) if the op count
  // disagrees or any op fails to place/route.
  bool placeImportedSolution(Region &region, const Architecture &architecture,
                             const std::vector<ImportedPlace> &places,
                             MappingState &mapping_state) {
    std::vector<Operation *> materialized = collectPlacedOps(region);
    if (materialized.size() != places.size()) {
      llvm::errs()
          << "[MapToAcceleratorPass] import-mapping op-count mismatch: "
          << materialized.size() << " materialized ops vs " << places.size()
          << " placements. The JSON must come from "
          << "--dump-dfg-json on this exact lowered IR.\n";
      return false;
    }
    // An op may only be bound to a tile that can actually run it -- the same
    // FU constraint the internal placer enforces (see
    // validateImportedFuLegality).
    if (!validateImportedFuLegality(materialized, architecture, places)) {
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
    if (hops.empty()) {
      llvm::errs() << "[MapToAcceleratorPass] cannot reserve an empty route\n";
      return false;
    }

    // A one-node route is an immediate same-tile value transfer. It still
    // needs a local register mapping: GenerateCodePass uses that register to
    // wire the consumer, and treating the route as an empty path leaves an
    // UNRESOLVED operand. Model the slot as a one-cycle local register deposit
    // and read (the register file has independent read/write ports).
    if (hops.size() == 1) {
      auto tile_it = tile_by_id.find(hops.front().tile);
      if (tile_it == tile_by_id.end()) {
        llvm::errs() << "[MapToAcceleratorPass] route references tile "
                     << hops.front().tile << " not in architecture\n";
        return false;
      }
      Tile *tile = tile_it->second;
      Register *reg =
          getAvailableRegister(mapping_state, tile, hops.front().cycle,
                               hops.front().cycle + 1, mov_op);
      if (!reg) {
        llvm::errs() << "[MapToAcceleratorPass] no free register on tile "
                     << hops.front().tile << " for immediate same-tile route "
                     << "at t=" << hops.front().cycle << "\n";
        return false;
      }
      out_path.push_back(MappingLoc{reg, hops.front().cycle});
      return true;
    }

    size_t hop_idx = 0;
    while (hop_idx + 1 < hops.size()) {
      if (hops[hop_idx].tile == hops[hop_idx + 1].tile) {
        // Maximal same-tile run -> a single register holds the value across it.
        size_t run_end = hop_idx;
        while (run_end + 1 < hops.size() &&
               hops[run_end + 1].tile == hops[hop_idx].tile) {
          ++run_end;
        }
        auto tile_it = tile_by_id.find(hops[hop_idx].tile);
        if (tile_it == tile_by_id.end()) {
          llvm::errs() << "[MapToAcceleratorPass] route references tile "
                       << hops[hop_idx].tile << " not in architecture\n";
          return false;
        }
        Tile *tile = tile_it->second;
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
        auto src_it = tile_by_id.find(hops[hop_idx].tile);
        if (src_it == tile_by_id.end()) {
          llvm::errs() << "[MapToAcceleratorPass] route references tile "
                       << hops[hop_idx].tile << " not in architecture\n";
          return false;
        }
        Tile *src = src_it->second;
        Link *link = nullptr;
        for (Link *candidate : src->getOutLinks()) {
          if (candidate->getDstTile()->getId() == hops[hop_idx + 1].tile) {
            link = candidate;
            break;
          }
        }
        if (!link) {
          llvm::errs() << "[MapToAcceleratorPass] no link "
                       << hops[hop_idx].tile << "->" << hops[hop_idx + 1].tile
                       << "\n";
          return false;
        }
        MappingLoc link_loc{link, hops[hop_idx].cycle};
        if (!mapping_state.isAvailableAcrossTime(link_loc, mov_op)) {
          llvm::errs() << "[MapToAcceleratorPass] link " << hops[hop_idx].tile
                       << "->" << hops[hop_idx + 1].tile
                       << " is unavailable at t=" << hops[hop_idx].cycle
                       << " for imported route\n";
          return false;
        }
        out_path.push_back(link_loc);
        ++hop_idx;
      }
    }
    return true;
  }

  // Route-witness import: bind every op at its tile/time, then reserve the
  // emitted route for every value move instead of greedily re-routing.
  // Binding is done in a first pass so every producer location exists before
  // routes are reserved.
  bool placeImportedExact(Region &region, const Architecture &architecture,
                          const std::vector<ImportedPlace> &places,
                          const RouteMap &routes, MappingState &mapping_state) {
    std::vector<Operation *> materialized = collectPlacedOps(region);
    if (materialized.size() != places.size()) {
      llvm::errs()
          << "[MapToAcceleratorPass] import-mapping op-count mismatch: "
          << materialized.size() << " vs " << places.size() << "\n";
      return false;
    }
    // An op may only be bound to a tile that can actually run it -- the same
    // FU constraint the internal placer enforces (see
    // validateImportedFuLegality).
    if (!validateImportedFuLegality(materialized, architecture, places)) {
      return false;
    }
    llvm::DenseMap<int, Tile *> tile_by_id;
    for (Tile *tile : architecture.getAllTiles()) {
      tile_by_id[tile->getId()] = tile;
    }
    std::string route_error;
    if (!validateImportedRoutes(region, materialized, places, routes,
                                mapping_state.getII(), tile_by_id,
                                route_error)) {
      llvm::errs() << "[MapToAcceleratorPass] import-mapping route error: "
                   << route_error << "\n";
      return false;
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

    // Pass 2: reserve the emitted route of every value move.
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
      if (path.empty()) {
        llvm::errs() << "[MapToAcceleratorPass] imported route " << producer_idx
                     << "->" << consumer_idx
                     << " produced no reservable resources\n";
        return false;
      }
      mapping_state.reserveRoute(mov, path);
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

    std::set<Operation *> critical_ops;
    int rec_mii = calculateRecMii(region, &critical_ops);

    llvm::errs() << "[MapToAcceleratorPass] Calculated Recurrence MII: "
                 << rec_mii << "\n";

    int res_mii = calculateResourceMii(region, architecture);

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

    // Imported witness: skip the search and replay the supplied placements and
    // routes. The II is selected by the external solver's budget; this path
    // does not claim that the witness is the first feasible II.
    if (!importMapping.getValue().empty()) {
      int imported_ii = 0;
      std::vector<ImportedPlace> places;
      RouteMap routes;
      bool has_routes = false;
      std::string parse_err;
      if (!parseImportedMapping(importMapping.getValue(), imported_ii, places,
                                routes, has_routes, parse_err)) {
        llvm::errs() << "[MapToAcceleratorPass] import-mapping error: "
                     << parse_err << "\n";
        return false;
      }
      llvm::errs() << "[MapToAcceleratorPass] Importing mapping witness ("
                   << places.size() << " ops, " << routes.size()
                   << " routes, II=" << imported_ii << ") from "
                   << importMapping.getValue() << "\n";
      MappingState mapping_state(architecture, imported_ii, is_spatial_only);
      // A missing route field denotes a placement-only witness. When the field
      // is present, even an empty array is intentional and must be checked
      // against the complete DFG rather than silently falling back to greedy
      // routing.
      bool placed = !has_routes
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

    // Builds a custom architecture with the requested tile dimensions, minus
    // every tile absent from `valid_tiles`. The parsing (and the reason valid
    // tiles are LEFT ALONE rather than re-marked existent) lives in
    // buildShapedArchitecture, shared with --cost-model-analytical,
    // --dump-dfg-json and the task allocator so that every one of them prices a
    // shape on the same tiles. Its inline predecessor here skipped the
    // coordinate trim, the grid-bounds check and the empty-set fallback, so a
    // mistyped list could hand the mapper the zero-tile architecture that same
    // fallback exists to prevent.
    custom_arch = buildShapedArchitecture(
        global_arch, x_tiles.getValue(), y_tiles.getValue(),
        valid_tiles.getValue(), "[MapToAcceleratorPass]");
    if (custom_arch) {
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
