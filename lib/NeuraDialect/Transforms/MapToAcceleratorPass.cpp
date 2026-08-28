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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;
using namespace mlir::neura::yamlkeys;

#define GEN_PASS_DEF_MAPTOACCELERATOR
#include "NeuraDialect/NeuraPasses.h.inc"

#ifndef NEURA_EXACT_MAPPER_CPSAT_SCRIPT
#error "CMake must define the packaged CP-SAT script path"
#endif
#ifndef NEURA_EXACT_MAPPER_CPSAT_INSTALL_SCRIPT
#error "CMake must define the installed CP-SAT script path"
#endif

namespace {
void canonicalizeDataLayoutSpec(ModuleOp module) {
  auto spec = module->getAttrOfType<DataLayoutSpecAttr>("dlti.dl_spec");
  if (!spec) {
    return;
  }
  SmallVector<DataLayoutEntryInterface> entries(spec.getEntries().begin(),
                                                spec.getEntries().end());
  auto printed = [](DataLayoutEntryInterface entry) {
    std::string text;
    llvm::raw_string_ostream os(text);
    entry.print(os);
    return os.str();
  };
  llvm::sort(entries, [&](DataLayoutEntryInterface lhs,
                          DataLayoutEntryInterface rhs) {
    return printed(lhs) < printed(rhs);
  });
  module->setAttr("dlti.dl_spec",
                  DataLayoutSpecAttr::get(module.getContext(), entries));
}

// Generic FusedOp is graph-mining IR, not an architecture resource. Inline it
// before physical mapping so every DataMov source is an operation that can
// receive a real FU/tile binding. A composite-FU contract would be a separate
// representation; never infer one from a fused pattern name here.
bool inlineFusedOpsForPhysicalMapping(Region &region) {
  SmallVector<neura::FusedOp> fused_ops;
  region.walk([&](neura::FusedOp fused_op) { fused_ops.push_back(fused_op); });

  // Lower nested patterns into their parent patterns before lowering the
  // parent into the mapped region.
  (void)llvm::reverse(fused_ops);
  for (neura::FusedOp fused_op : fused_ops) {
    Region &body = fused_op.getBody();
    assert(llvm::hasSingleElement(body) &&
           "FusedOp must have exactly one body block before mapping");
    Block &body_block = body.front();
    auto yield = dyn_cast<neura::YieldOp>(body_block.getTerminator());
    assert(yield &&
           "FusedOp body must terminate with neura.yield before mapping");
    assert(fused_op->getNumOperands() == body_block.getNumArguments() &&
           "FusedOp body arguments must correspond to its inputs");
    assert(fused_op->getNumResults() == yield.getNumOperands() &&
           "FusedOp results must correspond to its yielded values");

    for (unsigned i = 0; i < body_block.getNumArguments(); ++i) {
      body_block.getArgument(i).replaceAllUsesWith(fused_op->getOperand(i));
    }

    SmallVector<Value> yielded_values(yield.getOperands().begin(),
                                      yield.getOperands().end());
    SmallVector<Operation *> body_operations;
    for (Operation &body_op : body_block) {
      if (&body_op != yield.getOperation()) {
        body_operations.push_back(&body_op);
      }
    }
    yield.erase();
    for (Operation *body_op : body_operations) {
      body_op->moveBefore(fused_op);
    }
    fused_op->replaceAllUsesWith(yielded_values);
    fused_op->erase();
  }
  return !fused_ops.empty();
}

// InsertDataMovPass intentionally leaves fused bodies untouched. Once those
// bodies are inlined above, make their direct inputs explicit transport edges
// before the mapper asks every non-reserve operand for a routed producer.
void wrapInlinedFusedOperandsForPhysicalMapping(Region &region) {
  SmallVector<Operation *> operations;
  region.walk([&](Operation *operation) {
    if (occupiesFU(operation)) {
      operations.push_back(operation);
    }
  });

  for (Operation *operation : operations) {
    OpBuilder builder(operation);
    for (unsigned i = 0; i < operation->getNumOperands(); ++i) {
      Value operand = operation->getOperand(i);
      Operation *producer = operand.getDefiningOp();
      if (isa_and_nonnull<neura::DataMovOp, neura::ReserveOp>(producer)) {
        continue;
      }
      auto move = builder.create<neura::DataMovOp>(
          operation->getLoc(), operand.getType(), operand);
      operation->setOperand(i, move);
    }
  }
}

// Routing reserves locations per DataMovOp. A fused block argument can fan out
// after inlining even though its incoming DataMovOp was formerly consumed once
// by the FusedOp container, so give each physical consumer its own transport
// edge rather than reusing one routed edge.
void splitSharedDataMovUsesForPhysicalMapping(Region &region) {
  SmallVector<neura::DataMovOp> moves;
  region.walk([&](neura::DataMovOp move) { moves.push_back(move); });
  for (neura::DataMovOp move : moves) {
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : move.getResult().getUses()) {
      uses.push_back(&use);
    }
    for (auto *use : llvm::drop_begin(uses)) {
      OpBuilder builder(use->getOwner());
      auto copy = builder.create<neura::DataMovOp>(
          move->getLoc(), move.getResult().getType(), move.getOperand());
      use->set(copy);
    }
  }
}

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
                     "accelerator. Options: heuristic (default), analytical "
                     "(run exact_mapper_cpsat.py and replay its joint result)."),
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
          "Path to an exact-mapper witness JSON (op index -> tile "
          "+ time_step, produced by exact_mapper_cpsat.py --emit). "
          "When set, the search is skipped: each materialized op is "
          "placed at its given tile/time and every emitted route is "
          "replayed, reproducing the exact mapping witness. The "
          "op order must match --dump-dfg-json (same lowered IR)."),
      llvm::cl::init("")};
  Option<std::string> cpSatScript{
      *this, "cpsat-script",
      llvm::cl::desc("Path to exact_mapper_cpsat.py for mapping-strategy=analytical. "
                     "Defaults to the CMake-packaged build-tree resource."),
      llvm::cl::init(NEURA_EXACT_MAPPER_CPSAT_SCRIPT)};
  Option<std::string> pythonExecutable{
      *this, "python-executable",
      llvm::cl::desc("Python executable used by mapping-strategy=analytical."),
      llvm::cl::init("python3")};
  Option<int> cpSatMaxRouteNodes{
      *this, "cpsat-max-route-nodes",
      llvm::cl::desc("Maximum time-expanded routing vertices constructed by "
                     "analytical mapping (default: 50000)."),
      llvm::cl::init(50000)};

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
    } else if (mapping_strategy_str == "analytical") {
      resolved_mapping_strategy = "exact-cpsat";
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

  // Parses an exact-mapper witness. A witness is useful only with its complete
  // route set: replaying placement alone would silently replace CP-SAT's
  // routing proof with the heuristic router.
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
    llvm::json::Array *route_arr = root->getArray("routes");
    if (!ii || !placements || !route_arr) {
      err = "missing compiled_ii, placements, or mandatory routes";
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
    for (llvm::json::Value &entry : *route_arr) {
      llvm::json::Object *record = entry.getAsObject();
      if (!record) {
        err = "route entry is not an object";
        return false;
      }
      std::optional<int64_t> src_idx = record->getInteger("s");
      std::optional<int64_t> dst_idx = record->getInteger("d");
      llvm::json::Array *path = record->getArray("path");
      if (!src_idx || !dst_idx || !path || path->empty()) {
        err = "route entry missing s/d/non-empty path";
        return false;
      }
      std::vector<RouteHop> hops;
      for (llvm::json::Value &node : *path) {
        llvm::json::Array *hop_pair = node.getAsArray();
        if (!hop_pair || hop_pair->size() != 2) {
          err = "route hop must be [tile, cycle]";
          return false;
        }
        std::optional<int64_t> hop_tile = (*hop_pair)[0].getAsInteger();
        std::optional<int64_t> hop_cycle = (*hop_pair)[1].getAsInteger();
        if (!hop_tile || !hop_cycle) {
          err = "route hop tile/cycle must be integers";
          return false;
        }
        hops.push_back(RouteHop{static_cast<int>(*hop_tile),
                                static_cast<int>(*hop_cycle)});
      }
      auto inserted = routes.emplace(
          std::make_pair(static_cast<int>(*src_idx), static_cast<int>(*dst_idx)),
          std::move(hops));
      if (!inserted.second) {
        err = "duplicate route for one producer/consumer pair";
        return false;
      }
    }
    return true;
  }

  // Runs the exact mapper for this exact Region/Architecture pair. The JSON is
  // produced in-process with emitExactMapperJson, then the emitted witness is
  // replayed below; no hand-maintained dump/import ordering is involved.
  bool createAnalyticalWitness(Region &region, const Architecture &architecture,
                               int min_ii, int max_ii,
                               std::string &witness_path) {
    llvm::SmallString<128> input_path;
    int input_fd = -1;
    std::error_code ec = llvm::sys::fs::createTemporaryFile(
        "neura-cpsat-input", "json", input_fd, input_path);
    if (ec) {
      llvm::errs() << "[MapToAcceleratorPass] cannot create CP-SAT input: "
                   << ec.message() << "\n";
      return false;
    }
    {
      llvm::raw_fd_ostream input(input_fd, /*shouldClose=*/true);
      emitExactMapperJson(region, architecture, input);
    }

    llvm::SmallString<128> output_path;
    int output_fd = -1;
    ec = llvm::sys::fs::createTemporaryFile("neura-cpsat-witness", "json",
                                             output_fd, output_path);
    if (ec) {
      llvm::sys::fs::remove(input_path);
      llvm::errs() << "[MapToAcceleratorPass] cannot create CP-SAT output: "
                   << ec.message() << "\n";
      return false;
    }
    { llvm::raw_fd_ostream output(output_fd, /*shouldClose=*/true); }

    std::string min_ii_arg = std::to_string(min_ii);
    std::string max_ii_arg = std::to_string(max_ii);
    std::string max_route_nodes_arg = std::to_string(cpSatMaxRouteNodes);
    std::string script = cpSatScript.getValue();
    // The default names the build-tree copy; an installed binary resolves the
    // corresponding installed resource if that copy is absent.  An explicitly
    // requested path never falls through to another script.
    if (cpSatScript.getNumOccurrences() == 0 &&
        !llvm::sys::fs::exists(script)) {
      script = NEURA_EXACT_MAPPER_CPSAT_INSTALL_SCRIPT;
    }
    if (!llvm::sys::fs::exists(script)) {
      llvm::sys::fs::remove(input_path);
      llvm::sys::fs::remove(output_path);
      llvm::errs() << "[MapToAcceleratorPass] cannot find CP-SAT script: "
                   << script << "\n";
      return false;
    }
    auto python_path = llvm::sys::findProgramByName(pythonExecutable.getValue());
    if (!python_path) {
      llvm::sys::fs::remove(input_path);
      llvm::sys::fs::remove(output_path);
      llvm::errs() << "[MapToAcceleratorPass] cannot find Python executable "
                   << pythonExecutable.getValue() << "\n";
      return false;
    }
    std::string program = python_path.get();
    llvm::SmallVector<llvm::StringRef, 14> args = {
        program, script, input_path,
        "--min-ii", min_ii_arg, "--max-ii", max_ii_arg,
        "--max-route-nodes", max_route_nodes_arg,
        "--emit", output_path};
    std::string execution_error;
    int exit_code = llvm::sys::ExecuteAndWait(program, args,
                                              std::nullopt, {}, 0, 0,
                                              &execution_error);
    llvm::sys::fs::remove(input_path);
    if (exit_code != 0) {
      llvm::sys::fs::remove(output_path);
      llvm::errs() << "[MapToAcceleratorPass] CP-SAT failed (exit "
                   << exit_code << "): " << execution_error << "\n";
      return false;
    }
    witness_path = output_path.str().str();
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
    // wire the consumer, and an empty resource path leaves it unresolved.
    if (hops.size() == 1) {
      auto tile_it = tile_by_id.find(hops.front().tile);
      if (tile_it == tile_by_id.end()) {
        llvm::errs() << "[MapToAcceleratorPass] route references tile "
                     << hops.front().tile << " not in architecture\n";
        return false;
      }
      Register *reg = getAvailableRegister(mapping_state, tile_it->second,
                                           hops.front().cycle,
                                           hops.front().cycle + 1, mov_op);
      if (!reg) {
        llvm::errs() << "[MapToAcceleratorPass] no free register on tile "
                     << hops.front().tile
                     << " for immediate same-tile route at t="
                     << hops.front().cycle << "\n";
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
    std::vector<Operation *> materialized = collectPlacedOps(region);
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
    // Do this before all MII, exact-witness, and routing work so every mapper
    // path sees the same physically materializable DFG.
    if (inlineFusedOpsForPhysicalMapping(region)) {
      splitSharedDataMovUsesForPhysicalMapping(region);
      wrapInlinedFusedOperandsForPhysicalMapping(region);
    }

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

    const int rec_mii = calculateRecMii(region);

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

    // Imported exact solution: skip the search and replay every materialized
    // op placement plus every solver-proven route. No heuristic router is
    // consulted, so the imported witness remains the CP-SAT result.
    const bool analytical_mode = mappingStrategy.getValue() == "analytical";
    if (analytical_mode || !importMapping.getValue().empty()) {
      if (analytical_mode && !importMapping.getValue().empty()) {
        llvm::errs() << "[MapToAcceleratorPass] analytical mode and "
                        "import-mapping are mutually exclusive\n";
        return false;
      }
      std::string generated_witness;
      StringRef mapping_path = importMapping.getValue();
      if (analytical_mode) {
        if (!createAnalyticalWitness(region, architecture, possible_min_ii,
                                     max_ii, generated_witness)) {
          return false;
        }
        mapping_path = generated_witness;
      }
      int imported_ii = 0;
      std::vector<ImportedPlace> places;
      RouteMap routes;
      std::string parse_err;
      if (!parseImportedMapping(mapping_path, imported_ii, places,
                                routes, parse_err)) {
        if (analytical_mode)
          llvm::sys::fs::remove(mapping_path);
        llvm::errs() << "[MapToAcceleratorPass] import-mapping error: "
                     << parse_err << "\n";
        return false;
      }
      llvm::errs() << "[MapToAcceleratorPass] Importing exact mapping ("
                   << places.size() << " ops, " << routes.size()
                   << " routes, II=" << imported_ii << ") from "
                   << mapping_path << "\n";
      MappingState mapping_state(architecture, imported_ii, is_spatial_only);
      bool placed = placeImportedExact(region, architecture, places, routes,
                                       mapping_state);
      if (analytical_mode)
        llvm::sys::fs::remove(mapping_path);
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
    // Shape parsing is centralized with the analytical and JSON passes: an
    // invalid coordinate is rejected instead of silently changing hardware.
    std::unique_ptr<Architecture> custom_arch = buildArchitectureForShape(
        global_arch, x_tiles.getValue(), y_tiles.getValue(), valid_tiles);
    const Architecture &architecture = custom_arch ? *custom_arch : global_arch;

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
    canonicalizeDataLayoutSpec(module);
  }
};

} // namespace

namespace mlir::neura {

std::unique_ptr<Pass>
createMapToAcceleratorPass(const MapToAcceleratorOptions &options) {
  return std::make_unique<MapToAcceleratorPass>(options);
}

} // namespace mlir::neura
