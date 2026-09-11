#include "Common/AcceleratorAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/IR/AffineMap.h"
#include <set>

using namespace mlir;
using namespace neura;

namespace {

struct Operand {
  std::string operand;
  std::string color;
  // Optional runtime binding; offsets count elements, not bytes.
  std::string access;
  std::vector<int64_t> offsets;
  Operand(const std::string &op, const std::string &c = "RED")
      : operand(op), color(c) {}
};

struct Instruction {
  std::string opcode;
  std::vector<Operand> src_operands;
  std::vector<Operand> dst_operands;
  int id = -1;           // Unique instruction ID for debug/DFG.
  int time_step = -1;    // original scheduling timestep.
  int index_per_ii = -1; // grouping key (typically time_step % compiled_ii).
  int invalid_iterations = 0; // prologue length before the op becomes valid.
  Instruction(const std::string &op) : opcode(op) {}
};

// A single-entry model per tile for now. Can be extended later.
struct Entry {
  std::string entry_id;
  std::string type;
  std::vector<Instruction> instructions;
  Entry(const std::string &id, const std::string &t = "loop")
      : entry_id(id), type(t) {}
};

struct Tile {
  int col_idx, row_idx;
  int core_id;
  Entry entry; // single entry per tile.
  Tile(int c, int r, int id)
      : col_idx(c), row_idx(r), core_id(id), entry("entry0", "loop") {}
};

struct ArrayConfig {
  int columns;
  int rows;
  int compiled_ii = -1;
  std::vector<Tile> cores;
};

struct TileLocation {
  int col_idx = -1, row_idx = -1, time_step = -1;
  int index_per_ii = -1;
  int invalid_iterations = 0;
  bool has_tile = false;
};

// ---- Operation kind helpers ----.
static bool isDataMov(Operation *op) {
  return dyn_cast<DataMovOp>(op) != nullptr;
}
static bool isCtrlMov(Operation *op) {
  return dyn_cast<CtrlMovOp>(op) != nullptr;
}
static bool isPhiStart(Operation *op) {
  return dyn_cast<PhiStartOp>(op) != nullptr;
}
static bool isPhi(Operation *op) { return dyn_cast<PhiOp>(op) != nullptr; }
static bool isReserve(Operation *op) {
  return dyn_cast<ReserveOp>(op) != nullptr;
}
static bool isConstant(Operation *op) {
  return dyn_cast<ConstantOp>(op) != nullptr;
}
static bool isFusedOp(Operation *op) {
  return dyn_cast<FusedOp>(op) != nullptr;
}
// ---- Constant for phi_start operation ----.
static constexpr unsigned kReserveOpIndex = 1;

// Returns the reserve operand for phi_start/phi.
// - phi_start: reserve is operand #1 by definition.
// - phi: scans inputs and returns the operand defined by neura.reserve.
static Value getReserveOperand(Operation *op) {
  if (auto phi_start = dyn_cast<PhiStartOp>(op)) {
    assert(op->getNumOperands() > kReserveOpIndex &&
           "phi_start must have a reserve at operand #1");
    Value candidate = phi_start->getOperand(kReserveOpIndex);
    assert((!candidate || isa<ReserveOp>(candidate.getDefiningOp())) &&
           "phi_start operand #1 must be a ReserveOp");
    return candidate;
  }

  if (auto phi = dyn_cast<PhiOp>(op)) {
    for (Value operand : phi.getInputs()) {
      if (auto def_op = operand.getDefiningOp()) {
        if (isa<ReserveOp>(def_op))
          return operand;
      }
    }
  }

  return Value();
}

namespace mapping_utils {
// ----- placement helpers -----.
static TileLocation getTileLocation(Operation *op) {
  TileLocation tile_location;
  if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d)
        continue;
      auto resource = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (auto timestep = dyn_cast_or_null<IntegerAttr>(d.get("time_step")))
        tile_location.time_step = timestep.getInt();
      if (auto index_attr =
              dyn_cast_or_null<IntegerAttr>(d.get("index_per_ii")))
        tile_location.index_per_ii = index_attr.getInt();
      if (auto invalid_attr =
              dyn_cast_or_null<IntegerAttr>(d.get("invalid_iterations")))
        tile_location.invalid_iterations = invalid_attr.getInt();
      if (resource && resource.getValue() == "tile") {
        if (auto x_coord = dyn_cast_or_null<IntegerAttr>(d.get("x")))
          tile_location.col_idx = x_coord.getInt();
        if (auto y_coord = dyn_cast_or_null<IntegerAttr>(d.get("y")))
          tile_location.row_idx = y_coord.getInt();
        tile_location.has_tile = true;
      }
    }
  }
  // If tile mappings exist, x/y must be valid.
  assert(!tile_location.has_tile ||
         (tile_location.col_idx >= 0 && tile_location.row_idx >= 0));
  return tile_location;
}

// Helper to get mapping_locations arrays.
static ArrayAttr getMappingLocations(Operation *op) {
  return op->getAttrOfType<ArrayAttr>("mapping_locs");
}

static std::optional<int> getMappedRegId(Operation *op) {
  if (auto mapping_locations = getMappingLocations(op)) {
    for (Attribute location_attr : mapping_locations) {
      auto location_dict = dyn_cast<DictionaryAttr>(location_attr);
      if (!location_dict)
        continue;
      auto resource_attr =
          dyn_cast_or_null<StringAttr>(location_dict.get("resource"));
      if (!resource_attr)
        continue;
      if (resource_attr.getValue() == "register" ||
          resource_attr.getValue() == "reg") {
        if (auto per_tile_register_id = dyn_cast_or_null<IntegerAttr>(
                location_dict.get("per_tile_register_id"))) {
          return per_tile_register_id.getInt();
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace mapping_utils

static std::string getOpcode(Operation *op) {
  if (auto mac = dyn_cast<MacOp>(op)) {
    return mac.getInput1() ? "MUL_ADD" : "MUL";
  }

  std::string opcode = op->getName().getStringRef().str();
  if (opcode.rfind("neura.", 0) == 0) {
    opcode = opcode.substr(6);
  }
  if (isConstant(op)) {
    return "CONSTANT";
  }
  std::transform(opcode.begin(), opcode.end(), opcode.begin(), ::toupper);

  // For comparison operations, appends the comparison type to the opcode.
  if (auto icmp_op = dyn_cast<ICmpOp>(op)) {
    std::string cmp_type = icmp_op.getCmpType().str();
    std::transform(cmp_type.begin(), cmp_type.end(), cmp_type.begin(),
                   ::toupper);
    return opcode + "_" + cmp_type;
  }
  if (auto fcmp_op = dyn_cast<FCmpOp>(op)) {
    std::string cmp_type = fcmp_op.getCmpType().str();
    std::transform(cmp_type.begin(), cmp_type.end(), cmp_type.begin(),
                   ::toupper);
    return opcode + "_" + cmp_type;
  }

  // For cast operations, appends the cast type to the opcode.
  if (auto cast_op = dyn_cast<CastOp>(op)) {
    std::string cast_type = cast_op.getCastType().str();
    std::transform(cast_type.begin(), cast_type.end(), cast_type.begin(),
                   ::toupper);
    return opcode + "_" + cast_type;
  }

  return opcode;
}

// Extracts constant literal from an attribute.
// Returns formatted string like "#10" or "#3.0", or "arg0" for "%arg0", or
// empty string if not found.
static std::string extractConstantLiteralFromAttr(Attribute attr) {
  if (!attr)
    return "";

  if (auto integer_attr = dyn_cast<IntegerAttr>(attr))
    return "#" + std::to_string(integer_attr.getInt());
  if (auto float_attr = dyn_cast<FloatAttr>(attr))
    return "#" + std::to_string(float_attr.getValueAsDouble());

  // Handles string attributes like "%arg0" -> "arg0".
  if (auto string_attr = dyn_cast<StringAttr>(attr)) {
    std::string value = string_attr.getValue().str();
    // Checks if the string starts with "%arg" followed by digits.
    if (value.size() > 4 && value.substr(0, 4) == "%arg") {
      return value.substr(1);
    }
  }

  return "";
}

// Literals for CONSTANT operations, e.g. "#10" / "#0" / "#3.0".
static std::string getConstantLiteral(Operation *op) {
  if (isConstant(op)) {
    if (auto value_attr = op->getAttr(attr::kValue)) {
      std::string result = extractConstantLiteralFromAttr(value_attr);
      if (!result.empty())
        return result;
    }
    return "#0";
  }

  // Checks for constant_value attribute in non-CONSTANT operations.
  if (auto constant_value_attr = op->getAttr(attr::kConstantValue)) {
    std::string result = extractConstantLiteralFromAttr(constant_value_attr);
    if (!result.empty())
      return result;
  }

  // Checks for rhs_value attribute (for binary operations with constant RHS).
  if (auto rhs_value_attr = op->getAttr(attr::kRhsValue)) {
    std::string result = extractConstantLiteralFromAttr(rhs_value_attr);
    if (!result.empty())
      return result;
  }

  return "";
}

namespace mapping_utils {
// ----- Topology from Architecture -----.
struct Topology {
  DenseMap<int, std::pair<int, int>>
      link_ends; // link_id -> (srcTileId, dstTileId).
  DenseMap<int, std::pair<int, int>> tile_location; // tileId -> (x,y).
  DenseMap<std::pair<int, int>, int> coord_to_tile; // (x,y) -> tileId.

  StringRef getDirBetween(int src_tile_id, int dst_tile_id) const {
    auto [src_x, src_y] = tile_location.lookup(src_tile_id);
    auto [dst_x, dst_y] = tile_location.lookup(dst_tile_id);
    int dc = dst_x - src_x, dr = dst_y - src_y;
    if (dc == 1 && dr == 0)
      return "EAST";
    if (dc == -1 && dr == 0)
      return "WEST";
    if (dc == 0 && dr == 1)
      return "NORTH";
    if (dc == 0 && dr == -1)
      return "SOUTH";
    if (dc == 1 && dr == 1)
      return "NORTHEAST";
    if (dc == -1 && dr == 1)
      return "NORTHWEST";
    if (dc == 1 && dr == -1)
      return "SOUTHEAST";
    if (dc == -1 && dr == -1)
      return "SOUTHWEST";
    return "LOCAL";
  }
  StringRef dirFromLink(int link_id) const {
    auto it = link_ends.find(link_id);
    if (it == link_ends.end())
      return "LOCAL";
    return getDirBetween(it->second.first, it->second.second);
  }
  StringRef invertDir(StringRef d) const {
    if (d == "EAST")
      return "WEST";
    if (d == "WEST")
      return "EAST";
    if (d == "NORTH")
      return "SOUTH";
    if (d == "SOUTH")
      return "NORTH";
    if (d == "NORTHEAST")
      return "SOUTHWEST";
    if (d == "NORTHWEST")
      return "SOUTHEAST";
    if (d == "SOUTHEAST")
      return "NORTHWEST";
    if (d == "SOUTHWEST")
      return "NORTHEAST";
    return "LOCAL";
  }
  int srcTileOfLink(int link_id) const {
    return link_ends.lookup(link_id).first;
  }
  int dstTileOfLink(int link_id) const {
    return link_ends.lookup(link_id).second;
  }
  int tileIdAt(int x, int y) const {
    auto it = coord_to_tile.find({x, y});
    return (it == coord_to_tile.end()) ? -1 : it->second;
  }
};

static Topology getTopologyFromArchitecture(int per_cgra_rows,
                                            int per_cgra_columns) {
  Topology topo;
  const Architecture &architecture = mlir::neura::getArchitecture();

  for (auto *tile : architecture.getAllTiles()) {
    topo.tile_location[tile->getId()] = {tile->getX(), tile->getY()};
    topo.coord_to_tile[{tile->getX(), tile->getY()}] = tile->getId();
  }
  for (auto *link : architecture.getAllLinks()) {
    auto *src_tile = link->getSrcTile();
    auto *dst_tile = link->getDstTile();
    topo.link_ends[link->getId()] = {src_tile->getId(), dst_tile->getId()};
  }
  return topo;
}

// ----- Extract mapping steps (sorted by time) -----.
struct LinkStep {
  int link_id;
  int time_step;
};
struct RegStep {
  int regId;
  int time_step;
};

static SmallVector<LinkStep, 8> collectLinkSteps(Operation *op) {
  SmallVector<LinkStep, 8> steps;
  if (auto mapping_locations = getMappingLocations(op)) {
    for (Attribute location_attr : mapping_locations) {
      auto location_dict = dyn_cast<DictionaryAttr>(location_attr);
      if (!location_dict)
        continue;
      auto resource_attr =
          dyn_cast_or_null<StringAttr>(location_dict.get("resource"));
      if (!resource_attr || resource_attr.getValue() != "link")
        continue;
      auto link_id = dyn_cast_or_null<IntegerAttr>(location_dict.get("id"));
      auto time_step =
          dyn_cast_or_null<IntegerAttr>(location_dict.get("time_step"));
      if (!link_id || !time_step)
        continue;
      steps.push_back({(int)link_id.getInt(), (int)time_step.getInt()});
    }
  }
  llvm::sort(steps, [](const LinkStep &a, const LinkStep &b) {
    return a.time_step < b.time_step;
  });
  return steps;
}

static SmallVector<RegStep, 4> collectRegSteps(Operation *op) {
  SmallVector<RegStep, 4> steps;
  if (auto mapping_locations = getMappingLocations(op)) {
    for (Attribute location_attr : mapping_locations) {
      auto location_dict = dyn_cast<DictionaryAttr>(location_attr);
      if (!location_dict)
        continue;
      auto resource_attr =
          dyn_cast_or_null<StringAttr>(location_dict.get("resource"));
      if (!resource_attr)
        continue;
      if (resource_attr.getValue() == "register" ||
          resource_attr.getValue() == "reg") {
        auto per_tile_register_id = dyn_cast_or_null<IntegerAttr>(
            location_dict.get("per_tile_register_id"));
        auto time_step =
            dyn_cast_or_null<IntegerAttr>(location_dict.get("time_step"));
        if (!per_tile_register_id || !time_step)
          continue;
        steps.push_back(
            {(int)per_tile_register_id.getInt(), (int)time_step.getInt()});
      }
    }
  }
  llvm::sort(steps, [](const RegStep &a, const RegStep &b) {
    return a.time_step < b.time_step;
  });
  return steps;
}
} // namespace mapping_utils

// Keep existing call sites stable (byte-identical behavior) by re-exporting the
// names.
using mapping_utils::collectLinkSteps;
using mapping_utils::collectRegSteps;
using mapping_utils::getMappedRegId;
using mapping_utils::getMappingLocations;
using mapping_utils::getTileLocation;
using mapping_utils::getTopologyFromArchitecture;
using mapping_utils::LinkStep;
using mapping_utils::RegStep;
using mapping_utils::Topology;

// ----- Pass -----.
struct InstructionReference {
  int col_idx, row_idx, t, idx;
};

struct GenerateCodePass
    : public PassWrapper<GenerateCodePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateCodePass)

  StringRef getArgument() const override { return "generate-code"; }
  StringRef getDescription() const override {
    return "CGRA YAML/ASM gen (multi-hop routers + endpoint register deposit + "
           "timing-aware rewiring, with CTRL_MOV kept).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  DenseMap<Operation *, TileLocation> operation_placements;

  // Maps of tile coordinates (x,y) -> time_step -> vector of Instructions.
  std::map<std::pair<int, int>, std::map<int, std::vector<Instruction>>>
      tile_time_instructions;

  // Back references from IR operations to emitted instructions.
  DenseMap<Operation *, InstructionReference>
      operation_to_instruction_reference;
  DenseMap<Operation *, InstructionReference> forwarding_instruction_reference;
  DenseMap<Operation *, SmallVector<Value>> operation_to_operands;
  // Map dfg_id -> op for later adjustments.
  DenseMap<int, Operation *> dfg_id_to_op;
  // Map (col,row,index_per_ii,local_idx_in_bucket) -> global instruction id.
  std::map<std::tuple<int, int, int, int>, int> instruction_id_map;
  int next_instruction_id = 0;
  int current_compiled_ii = -1;
  bool timing_field_error = false;
  KernelOp current_kernel;
  int stationary_input_index = -1;
  AffineMapAttr stationary_map;
  bool static_dataflow = false;

  // De-dup sets.
  std::unordered_set<uint64_t>
      hop_signatures; // (midTileId, time_step, link_id).
  std::unordered_set<uint64_t>
      deposit_signatures; // (dstTileId, time_step, regId).
  std::unordered_set<uint64_t>
      egress_signatures; // (srcTileId, time_step, regId, out_dir).

  // ---------- helpers to place materialized instructions ----------.
  void placeRouterHop(const Topology &topology, int tile_id, int time_step,
                      StringRef input_direction, StringRef output_direction,
                      bool asCtrlMov = false, int assigned_id = -1) {
    auto [tile_x, tile_y] = topology.tile_location.lookup(tile_id);
    Instruction instruction(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    instruction.id = assigned_id;
    instruction.time_step = time_step;
    instruction.index_per_ii = indexPerIiFromTimeStep(time_step);
    instruction.invalid_iterations = syntheticInvalidIterations(time_step);
    instruction.src_operands.emplace_back(input_direction.str(), "RED");
    instruction.dst_operands.emplace_back(output_direction.str(), "RED");
    tile_time_instructions[{tile_x, tile_y}][instruction.index_per_ii]
        .push_back(std::move(instruction));
  }

  // ---------- initialization helpers ----------.
  void clearState() {
    operation_placements.clear();
    tile_time_instructions.clear();
    operation_to_instruction_reference.clear();
    forwarding_instruction_reference.clear();
    operation_to_operands.clear();
    dfg_id_to_op.clear();
    hop_signatures.clear();
    deposit_signatures.clear();
    egress_signatures.clear();
    instruction_id_map.clear();
    next_instruction_id = 0;
    timing_field_error = false;
    current_kernel = nullptr;
    stationary_input_index = -1;
    stationary_map = nullptr;
    static_dataflow = false;
  }

  std::pair<int, int> getArrayDimensions(Operation *scope) {
    const Architecture &architecture = mlir::neura::getArchitecture();
    int columns = architecture.getPerCgraColumns();
    int rows = architecture.getPerCgraRows();
    if (auto mapping_info =
            scope->getAttrOfType<DictionaryAttr>(attr::kMappingInfo)) {
      if (auto x_tiles =
              dyn_cast_or_null<IntegerAttr>(mapping_info.get(attr::kXTiles)))
        columns = x_tiles.getInt();
      if (auto y_tiles =
              dyn_cast_or_null<IntegerAttr>(mapping_info.get(attr::kYTiles)))
        rows = y_tiles.getInt();
    }
    return {columns, rows};
  }

  int getCompiledII(Operation *scope) {
    if (auto mapping_info =
            scope->getAttrOfType<DictionaryAttr>(attr::kMappingInfo)) {
      if (auto compiled_ii = dyn_cast_or_null<IntegerAttr>(
              mapping_info.get(attr::kCompiledII))) {
        return compiled_ii.getInt();
      }
    }
    return -1;
  }

  // Derives index_per_ii from an absolute time_step; falls back to time_step
  // when II is unknown.
  int indexPerIiFromTimeStep(int time_step) const {
    return current_compiled_ii > 0 ? time_step % current_compiled_ii
                                   : time_step;
  }

  int syntheticInvalidIterations(int time_step) const {
    if (static_dataflow) {
      return 0;
    }
    return current_compiled_ii > 0 ? time_step / current_compiled_ii : 0;
  }

  // Extracts index_per_ii and invalid_iterations from mapping_locs.
  bool getIndexAndInvalid(Operation *op, int &index_per_ii,
                          int &invalid_iterations) {
    index_per_ii = -1;
    invalid_iterations = 0;
    bool has_index = false, has_invalid = false;
    if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
      for (Attribute a : arr) {
        auto dict = dyn_cast<DictionaryAttr>(a);
        if (!dict)
          continue;
        if (auto idx_attr =
                dyn_cast_or_null<IntegerAttr>(dict.get("index_per_ii"))) {
          index_per_ii = idx_attr.getInt();
          has_index = true;
        }
        if (auto inv_attr =
                dyn_cast_or_null<IntegerAttr>(dict.get("invalid_iterations"))) {
          invalid_iterations = inv_attr.getInt();
          has_invalid = true;
        }
      }
    }
    if (!has_index || !has_invalid) {
      std::string loc_str;
      llvm::raw_string_ostream rso(loc_str);
      rso << op->getLoc();
      rso.flush();

      std::string op_name = op->getName().getStringRef().str();
      std::stringstream errMsg;
      errMsg << "Operation '" << op_name << "' at " << loc_str
             << " missing index_per_ii or invalid_iterations in mapping_locs";
      op->emitError(errMsg.str());
      timing_field_error = true;
      return false;
    }
    return true;
  }

  // ---------- Single-walk indexing ----------.
  // Do everything that needs walks in a single pass:.
  //   - record operation_placements.
  //   - materialize compute/phi/const instructions.
  //   - collect DATA_MOV and CTRL_MOV ops.
  //   - collect reserve_to_phi_maps (PHI's operand#0 is the reserve).
  void indexIR(Operation *scope, SmallVector<Operation *> &data_movs,
               SmallVector<Operation *> &ctrl_movs,
               DenseMap<Value, Operation *> &reserve_to_phi_map) {
    scope->walk([&](Operation *op) {
      // Skips operations inside fused_op regions.
      if (op->getParentOp() && isFusedOp(op->getParentOp())) {
        return;
      }

      // Records Records placement for every op (even for mov/reserve).
      operation_placements[op] = getTileLocation(op);

      // Builds reserve -> phi mapping for loop-carried dependencies.
      // Supports both phi_start and phi forms.
      if (isPhiStart(op) || isPhi(op)) {
        if (Value reserve = getReserveOperand(op)) {
          reserve_to_phi_map[reserve] = op;
        }
      }

      // Collects forwarders for later expansion.
      if (isDataMov(op)) {
        data_movs.push_back(op);
        return;
      }
      if (isCtrlMov(op)) {
        ctrl_movs.push_back(op);
        return;
      }

      // Skips Reserve from materialization.
      if (isReserve(op))
        return;

      // Materializes all other ops placed on tiles
      // (compute/phi/const/fused_op/etc.).
      TileLocation placement = operation_placements[op];
      if (!placement.has_tile)
        return;

      std::string opcode = getOpcode(op);
      Instruction inst(opcode);
      inst.id = getDfgId(op);
      inst.time_step = placement.time_step;

      if (isConstant(op)) {
        inst.src_operands.emplace_back(getConstantLiteral(op), "RED");
      } else if (op->getAttr(attr::kConstantValue)) {
        // Checks if operation has constant_value attribute (for non-CONSTANT
        // operations).
        inst.src_operands.emplace_back(getConstantLiteral(op), "RED");
      } else {
        // Handles normal operands and folded constants (lhs_value/rhs_value).
        SmallVector<Value> operands;
        operands.reserve(op->getNumOperands());

        auto appendLiteralSlot = [&](Attribute attr) -> bool {
          // If there is no attribute, there is no folded constant.
          if (!attr)
            return false;
          std::string literal = extractConstantLiteralFromAttr(attr);
          if (literal.empty()) {
            // Attribute exists but cannot be serialized to current literal
            // form. Keep slot alignment and surface a clear diagnostic.
            op->emitError(
                "unsupported constant attribute in lhs_value/rhs_value")
                << ": " << attr;
            literal = "UNSUPPORTED_CONSTANT";
          }
          inst.src_operands.emplace_back(literal, "RED");
          // Keeps index alignment with operation_to_operands for rewiring.
          operands.push_back(Value());
          return true;
        };
        auto appendValueSlot = [&](Value v) {
          operands.push_back(v);
          inst.src_operands.emplace_back("UNRESOLVED", "RED");
        };

        // Configured operands extend the normal per-operation encoder.
        // Other operation kinds continue through the generic path below.
        if (!encodeConfiguredOperands(op, inst, operands)) {
          if (auto store_indexed_op = dyn_cast<StoreIndexedOp>(op)) {
            bool lhs_folded = appendLiteralSlot(op->getAttr(attr::kLhsValue));
            if (!lhs_folded) {
              appendValueSlot(store_indexed_op.getValue());
            }

            bool rhs_folded = appendLiteralSlot(op->getAttr(attr::kRhsValue));
            if (!rhs_folded) {
              Value base = store_indexed_op.getBase();
              if (base) {
                appendValueSlot(base);
              }
            }

            for (Value index : store_indexed_op.getIndices()) {
              appendValueSlot(index);
            }
          } else {
            // Generic handling:
            // - lhs_value is the leading source slot.
            // - remaining Value operands keep original order.
            // - rhs_value is the trailing source slot.
            appendLiteralSlot(op->getAttr(attr::kLhsValue));
            for (Value value : op->getOperands()) {
              appendValueSlot(value);
            }
            appendLiteralSlot(op->getAttr(attr::kRhsValue));
          }
        }

        operation_to_operands[op] = std::move(operands);
      }

      if (auto mapped_register_id = getMappedRegId(op))
        inst.dst_operands.emplace_back(
            "$" + std::to_string(*mapped_register_id), "RED");

      int index_per_ii = -1, invalid_iterations = 0;
      if (!getIndexAndInvalid(op, index_per_ii, invalid_iterations))
        return;

      inst.index_per_ii = index_per_ii;
      inst.invalid_iterations = static_dataflow ? 0 : invalid_iterations;

      auto &bucket = getInstructionBucket(placement.col_idx, placement.row_idx,
                                          index_per_ii);
      bucket.push_back(std::move(inst));
      operation_to_instruction_reference[op] =
          InstructionReference{placement.col_idx, placement.row_idx,
                               index_per_ii, (int)bucket.size() - 1};
    });
  }

  // ---------- unified forwarder expansion helpers ----------.
  static SmallVector<LinkStep, 8> getLinkChain(Operation *forwarder) {
    return collectLinkSteps(forwarder);
  }
  static SmallVector<RegStep, 4> getRegisterSteps(Operation *forwarder) {
    return collectRegSteps(forwarder);
  }

  // Validates forwarder op arities: DATA_MOV: at least 1 in/1 out; CTRL_MOV: at
  // least 2 inputs (src,reserve).
  template <bool IsCtrl> bool validateForwarderShape(Operation *forwarder) {
    if constexpr (!IsCtrl) {
      return forwarder->getNumOperands() >= 1 &&
             forwarder->getNumResults() >= 1;
    } else {
      return forwarder->getNumOperands() >= 2;
    }
  }

  // Computes producer first-hop directions and consumer last-hop directions (or
  // LOCAL if link-less).
  std::pair<StringRef, StringRef>
  computeDirections(const SmallVector<LinkStep, 8> &links,
                    const Topology &topo) {
    StringRef producer_direction("LOCAL");
    StringRef consumer_direction("LOCAL");
    if (!links.empty()) {
      producer_direction = topo.dirFromLink(links.front().link_id);
      consumer_direction =
          topo.invertDir(topo.dirFromLink(links.back().link_id));
    }
    return {producer_direction, consumer_direction};
  }

  // Adds producer endpoints:
  // - If pre-link registers exist: producer writes ONLY to the earliest
  // pre-link register.
  //   Later source-side register transfers and egress are emitted as synthetic
  //   instructions.
  // - Else: producer writes to producer_direction if link-based, or $reg for
  // reg-only paths.
  void setProducerDestination(Value source, StringRef producer_direction,
                              const SmallVector<RegStep, 4> &regs,
                              ArrayRef<RegStep> src_reg_steps) {
    Operation *producer = source.getDefiningOp();
    Instruction *instruction = nullptr;
    if (auto result = dyn_cast<OpResult>(source);
        result && isa<MacOp>(producer) && result.getResultNumber() == 1) {
      instruction = getForwardingInstruction(result);
    } else {
      instruction = getInstructionPointer(producer);
    }

    if (instruction) {
      if (!src_reg_steps.empty()) {
        // This mov uses source-side register transfers + egress to send on
        // link.time_step. We still must NOT delete existing directional
        // outputs, because the same producer may fan-out to other consumers via
        // other mov paths.
        setUniqueDestination(instruction,
                             "$" + std::to_string(src_reg_steps.front().regId));
        return;
      }

      if (!producer_direction.empty() && producer_direction != "LOCAL") {
        setUniqueDestination(instruction, producer_direction.str());
      } else if (!regs.empty()) {
        setUniqueDestination(instruction,
                             "$" + std::to_string(regs.back().regId));
      }
    }
  }

  // Egress: on source tile at first_link_ts, move [$src_reg] -> [out_dir].
  void placeSrcEgress(const Topology &topology, int src_tile_id, int time_step,
                      StringRef out_dir, int reg_id, bool asCtrlMov = false,
                      int assigned_id = -1) {
    // Signature must be stable and support arbitrary direction strings
    // (arch-spec dependent).
    uint64_t signature = static_cast<uint64_t>(
        llvm::hash_combine(src_tile_id, time_step, reg_id, out_dir));
    if (!egress_signatures.insert(signature).second)
      return;

    auto [tile_x, tile_y] = topology.tile_location.lookup(src_tile_id);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.id = assigned_id;
    inst.time_step = time_step;
    inst.index_per_ii = indexPerIiFromTimeStep(time_step);
    inst.invalid_iterations = syntheticInvalidIterations(time_step);
    inst.src_operands.emplace_back("$" + std::to_string(reg_id), "RED");
    inst.dst_operands.emplace_back(out_dir.str(), "RED");
    tile_time_instructions[{tile_x, tile_y}][inst.index_per_ii].push_back(
        std::move(inst));
  }

  struct MovRegSplit {
    SmallVector<RegStep, 4>
        src_reg_steps; // all regs with time_step < first_link_time_step
    SmallVector<RegStep, 4>
        dst_reg_steps; // all regs with time_step > last_link_time_step
  };

  SmallVector<std::pair<RegStep, RegStep>, 4>
  collectRegisterTransfers(ArrayRef<RegStep> reg_steps) const {
    SmallVector<std::pair<RegStep, RegStep>, 4> transfers;
    if (reg_steps.size() < 2)
      return transfers;

    for (size_t i = 1; i < reg_steps.size(); ++i) {
      const RegStep &prev = reg_steps[i - 1];
      const RegStep &cur = reg_steps[i];
      // Consecutive uses of the same physical register are a hold, not a move.
      if (prev.regId == cur.regId)
        continue;
      transfers.emplace_back(prev, cur);
    }
    return transfers;
  }

  MovRegSplit splitMovRegs(const SmallVector<RegStep, 4> &regs,
                           const SmallVector<LinkStep, 8> &links) const {
    MovRegSplit out;
    if (regs.empty() || links.empty())
      return out;
    int first_link_time_step = links.front().time_step;
    int last_link_time_step = links.back().time_step;
    for (const RegStep &r : regs) {
      if (r.time_step < first_link_time_step) {
        out.src_reg_steps.push_back(r);
        continue;
      }
      if (r.time_step > last_link_time_step)
        out.dst_reg_steps.push_back(r);
    }
    return out;
  }

  SmallVector<RegStep, 4>
  collectRegsBetweenLinks(const SmallVector<RegStep, 4> &regs,
                          int start_time_step, int end_time_step) const {
    SmallVector<RegStep, 4> segment;
    for (const RegStep &reg_step : regs) {
      if (reg_step.time_step <= start_time_step)
        continue;
      if (reg_step.time_step >= end_time_step)
        break;
      segment.push_back(reg_step);
    }
    return segment;
  }

  template <bool IsCtrl>
  void placeRegisterTransfer(const Topology &topology, int tile_id,
                             int time_step, int src_reg_id, int dst_reg_id,
                             int assigned_id = -1) {
    auto [tile_x, tile_y] = topology.tile_location.lookup(tile_id);
    Instruction inst(IsCtrl ? "CTRL_MOV" : "DATA_MOV");
    inst.id = assigned_id;
    inst.time_step = time_step;
    inst.index_per_ii = indexPerIiFromTimeStep(time_step);
    inst.invalid_iterations = syntheticInvalidIterations(time_step);
    inst.src_operands.emplace_back("$" + std::to_string(src_reg_id), "RED");
    inst.dst_operands.emplace_back("$" + std::to_string(dst_reg_id), "RED");
    tile_time_instructions[{tile_x, tile_y}][inst.index_per_ii].push_back(
        std::move(inst));
  }

  template <bool IsCtrl>
  size_t emitSourceSyntheticChain(const MovRegSplit &reg_split,
                                  const SmallVector<LinkStep, 8> &links,
                                  const Topology &topo, int base_mov_id) {
    if (links.empty() || reg_split.src_reg_steps.empty())
      return 0;

    int src_tile_id = topo.srcTileOfLink(links.front().link_id);
    int next_synthetic_offset = 0;
    SmallVector<std::pair<RegStep, RegStep>, 4> transfers =
        collectRegisterTransfers(reg_split.src_reg_steps);

    for (const auto &[prev, cur] : transfers) {
      int synthetic_id =
          base_mov_id >= 0 ? base_mov_id * 10000 + next_synthetic_offset : -1;
      ++next_synthetic_offset;
      placeRegisterTransfer<IsCtrl>(topo, src_tile_id, cur.time_step,
                                    prev.regId, cur.regId, synthetic_id);
    }

    int egress_instruction_id =
        base_mov_id >= 0 ? base_mov_id * 10000 + next_synthetic_offset : -1;
    placeSrcEgress(topo, src_tile_id, links.front().time_step,
                   topo.dirFromLink(links.front().link_id),
                   reg_split.src_reg_steps.back().regId,
                   /*asCtrlMov=*/IsCtrl, egress_instruction_id);
    ++next_synthetic_offset;
    return next_synthetic_offset;
  }

  template <bool IsCtrl>
  size_t emitDestinationSyntheticChain(const MovRegSplit &reg_split,
                                       const SmallVector<LinkStep, 8> &links,
                                       const Topology &topo, int base_mov_id,
                                       size_t next_synthetic_offset) {
    if (links.empty() || reg_split.dst_reg_steps.empty())
      return next_synthetic_offset;

    int dst_tile_id = topo.dstTileOfLink(links.back().link_id);
    StringRef incoming_dir =
        topo.invertDir(topo.dirFromLink(links.back().link_id));

    // The IR-backed mov_id materializes the deposit onto the first post-link
    // register.
    placeDstDeposit(topo, dst_tile_id,
                    reg_split.dst_reg_steps.front().time_step, incoming_dir,
                    reg_split.dst_reg_steps.front().regId,
                    /*asCtrlMov=*/IsCtrl, base_mov_id);

    SmallVector<std::pair<RegStep, RegStep>, 4> transfers =
        collectRegisterTransfers(reg_split.dst_reg_steps);
    for (const auto &[prev, cur] : transfers) {
      int synthetic_id =
          base_mov_id >= 0
              ? base_mov_id * 10000 + static_cast<int>(next_synthetic_offset)
              : -1;
      ++next_synthetic_offset;
      placeRegisterTransfer<IsCtrl>(topo, dst_tile_id, cur.time_step,
                                    prev.regId, cur.regId, synthetic_id);
    }
    return next_synthetic_offset;
  }

  std::optional<int>
  selectRegisterForConsumer(ArrayRef<RegStep> reg_steps,
                            const TileLocation &consumer_placement,
                            bool allow_prearrival_register) const {
    if (reg_steps.empty())
      return std::nullopt;
    if (!consumer_placement.has_tile || consumer_placement.time_step < 0) {
      if (allow_prearrival_register)
        return reg_steps.front().regId;
      return std::nullopt;
    }

    std::optional<int> selected_reg;
    for (const RegStep &step : reg_steps) {
      if (step.time_step >= consumer_placement.time_step)
        break;
      selected_reg = step.regId;
    }
    if (selected_reg)
      return selected_reg;
    if (allow_prearrival_register)
      return reg_steps.front().regId;
    return std::nullopt;
  }

  // Emits router hops for multi-hop paths (from the second hop onwards).
  // CTRL_MOV emits CTRL_MOV hops.
  template <bool IsCtrl>
  void generateIntermediateHops(const SmallVector<LinkStep, 8> &links,
                                const SmallVector<RegStep, 4> &regs,
                                const Topology &topo, int base_mov_id,
                                size_t &hop_counter) {
    for (size_t i = 1; i < links.size(); ++i) {
      int prev_link = links[i - 1].link_id;
      int cur_link = links[i].link_id;
      int mid_tile = topo.srcTileOfLink(cur_link);
      StringRef incoming_dir = topo.invertDir(topo.dirFromLink(prev_link));
      StringRef outgoing_dir = topo.dirFromLink(cur_link);

      SmallVector<RegStep, 4> inter_link_regs = collectRegsBetweenLinks(
          regs, links[i - 1].time_step, links[i].time_step);
      if (inter_link_regs.empty()) {
        int time_step = links[i].time_step;
        uint64_t sig = static_cast<uint64_t>(
            llvm::hash_combine(mid_tile, time_step, cur_link));
        int hop_id = base_mov_id >= 0
                         ? base_mov_id * 10000 + static_cast<int>(hop_counter)
                         : -1;
        ++hop_counter;
        if (hop_signatures.insert(sig).second) {
          placeRouterHop(topo, mid_tile, time_step, incoming_dir, outgoing_dir,
                         /*asCtrlMov=*/IsCtrl, hop_id);
        }
        continue;
      }

      int deposit_id = base_mov_id >= 0
                           ? base_mov_id * 10000 + static_cast<int>(hop_counter)
                           : -1;
      ++hop_counter;
      placeDstDeposit(topo, mid_tile, inter_link_regs.front().time_step,
                      incoming_dir, inter_link_regs.front().regId,
                      /*asCtrlMov=*/IsCtrl, deposit_id);

      SmallVector<std::pair<RegStep, RegStep>, 4> transfers =
          collectRegisterTransfers(inter_link_regs);
      for (const auto &[prev, cur] : transfers) {
        int transfer_id = base_mov_id >= 0 ? base_mov_id * 10000 +
                                                 static_cast<int>(hop_counter)
                                           : -1;
        ++hop_counter;
        placeRegisterTransfer<IsCtrl>(topo, mid_tile, cur.time_step, prev.regId,
                                      cur.regId, transfer_id);
      }

      int egress_id = base_mov_id >= 0
                          ? base_mov_id * 10000 + static_cast<int>(hop_counter)
                          : -1;
      ++hop_counter;
      placeSrcEgress(topo, mid_tile, links[i].time_step, outgoing_dir,
                     inter_link_regs.back().regId, /*asCtrlMov=*/IsCtrl,
                     egress_id);
    }
  }

  // Consumers for DATA_MOV: all users of forwarder results(0).
  SmallVector<std::pair<Operation *, Value>, 2>
  collectDataMovConsumers(Operation *forwarder) {
    SmallVector<std::pair<Operation *, Value>, 2> consumers;
    Value out = forwarder->getResult(0);
    for (OpOperand &use : out.getUses())
      consumers.push_back({use.getOwner(), use.get()});
    return consumers;
  }

  // Consumers for CTRL_MOV: find PHI via reserve->phi maps; wire the PHI's
  // *data* inputs (sources).
  SmallVector<std::pair<Operation *, Value>, 2>
  collectCtrlMovConsumers(Operation *forwarder,
                          const DenseMap<Value, Operation *> &reserve2phi) {
    SmallVector<std::pair<Operation *, Value>, 2> consumers;
    Value reserve = forwarder->getOperand(1);
    if (Operation *phi = reserve2phi.lookup(reserve))
      consumers.push_back({phi, reserve});
    else
      forwarder->emitWarning(
          "ctrl_mov dest is not consumed by a PHI operand#0; skipping.");
    return consumers;
  }

  // Try register-based rewiring.
  // For cross-tile movs, destination deposits/register transfers have already
  // been materialized during routing instruction generation; this helper only
  // decides which local register a consumer should read, if any.
  template <bool IsCtrl>
  bool handleRegisterRewiring(Operation *consumer_operation,
                              Value value_at_consumer,
                              const SmallVector<RegStep, 4> &regs,
                              ArrayRef<RegStep> dst_reg_steps,
                              const SmallVector<LinkStep, 8> &links,
                              const Topology &topo, int mov_dfg_id) {
    (void)mov_dfg_id;
    if (!links.empty()) {
      if (dst_reg_steps.empty())
        return false;

      int dst_tile = topo.dstTileOfLink(links.back().link_id);
      TileLocation consumer_placement =
          operation_placements.lookup(consumer_operation);
      if (consumer_placement.has_tile) {
        int consumer_tile_id = topo.tileIdAt(consumer_placement.col_idx,
                                             consumer_placement.row_idx);
        if (consumer_tile_id != dst_tile)
          return false;

        if (std::optional<int> register_id = selectRegisterForConsumer(
                dst_reg_steps, consumer_placement,
                /*allow_prearrival_register=*/IsCtrl)) {
          setConsumerSourceExact(consumer_operation, value_at_consumer,
                                 "$" + std::to_string(*register_id));
          return true;
        }
      }
    } else {
      // Same-tile: must go via register.
      if (regs.empty())
        return false;
      TileLocation consumer_placement =
          operation_placements.lookup(consumer_operation);
      std::optional<int> register_id =
          selectRegisterForConsumer(regs, consumer_placement,
                                    /*allow_prearrival_register=*/IsCtrl);
      if (!register_id)
        register_id = regs.back().regId;
      setConsumerSourceExact(consumer_operation, value_at_consumer,
                             "$" + std::to_string(*register_id));
      return true;
    }
    return false;
  }

  template <bool IsCtrl>
  void handleDirectionRewiring(Operation *consumer_operation,
                               Value value_at_consumer,
                               StringRef consumer_direction,
                               const SmallVector<LinkStep, 8> &links,
                               const Topology &topo, Operation *forwarder) {
    if (!links.empty()) {
      // Computes the direction from the link destination tile to the consumer
      // tile.
      TileLocation consumer_placement =
          operation_placements.lookup(consumer_operation);
      if (consumer_placement.has_tile) {
        int dst_tile_id = topo.dstTileOfLink(links.back().link_id);
        int consumer_tile_id = topo.tileIdAt(consumer_placement.col_idx,
                                             consumer_placement.row_idx);

        // If consumer is on the link destination tile, use the incoming
        // direction.
        if (consumer_tile_id == dst_tile_id) {
          setConsumerSourceExact(consumer_operation, value_at_consumer,
                                 consumer_direction.str());
        } else {
          // Computes direction from link destination tile to consumer tile.
          StringRef actual_dir =
              topo.invertDir(topo.getDirBetween(dst_tile_id, consumer_tile_id));
          setConsumerSourceExact(consumer_operation, value_at_consumer,
                                 actual_dir.str());
        }
      } else {
        // Falls back to consumer_direction if consumer placement is unknown.
        setConsumerSourceExact(consumer_operation, value_at_consumer,
                               consumer_direction.str());
      }
    } else {
      forwarder->emitError(
          IsCtrl ? "same-tile ctrl_mov without register mapping is illegal. "
                   "Provide a register in mapping_locs."
                 : "same-tile data_mov without register mapping is illegal. "
                   "Provide a register in mapping_locs.");
      assert(false && "same-tile mov without register mapping");
    }
  }

  template <bool IsCtrl> struct MovBasics {
    int mov_dfg_id = -1;
    Value source;
    Operation *producer = nullptr;
    SmallVector<LinkStep, 8> links;
    SmallVector<RegStep, 4> regs;
    MovRegSplit reg_split;
    StringRef producer_direction;
    StringRef consumer_direction;
  };

  template <bool IsCtrl>
  MovBasics<IsCtrl> buildMovBasics(Operation *forwarder, const Topology &topo) {
    MovBasics<IsCtrl> basics;
    basics.mov_dfg_id = getDfgId(forwarder);

    // Basic info from forwarders.
    Value source = forwarder->getOperand(0);
    basics.source = source;
    basics.producer = source.getDefiningOp();
    basics.links = getLinkChain(forwarder);
    basics.regs = getRegisterSteps(forwarder);
    basics.reg_split = splitMovRegs(basics.regs, basics.links);
    std::pair<StringRef, StringRef> directions =
        computeDirections(basics.links, topo);
    basics.producer_direction = directions.first;
    basics.consumer_direction = directions.second;
    return basics;
  }

  template <bool IsCtrl>
  void emitMovRoutingInstructions(Operation *forwarder,
                                  const MovBasics<IsCtrl> &basics,
                                  const Topology &topo) {
    (void)forwarder;
    // Producer endpoints & intermediate hops.
    setProducerDestination(basics.source, basics.producer_direction,
                           basics.regs, basics.reg_split.src_reg_steps);
    size_t next_synthetic_offset = emitSourceSyntheticChain<IsCtrl>(
        basics.reg_split, basics.links, topo, basics.mov_dfg_id);
    generateIntermediateHops<IsCtrl>(basics.links, basics.regs, topo,
                                     basics.mov_dfg_id, next_synthetic_offset);
    emitDestinationSyntheticChain<IsCtrl>(basics.reg_split, basics.links, topo,
                                          basics.mov_dfg_id,
                                          next_synthetic_offset);
  }

  template <bool IsCtrl>
  SmallVector<std::pair<Operation *, Value>, 2>
  collectMovConsumers(Operation *forwarder,
                      const DenseMap<Value, Operation *> &reserve2phi) {
    if constexpr (IsCtrl) {
      return collectCtrlMovConsumers(forwarder, reserve2phi);
    } else {
      return collectDataMovConsumers(forwarder);
    }
  }

  template <bool IsCtrl>
  void rewriteMovConsumers(
      Operation *forwarder, const MovBasics<IsCtrl> &basics,
      const SmallVector<std::pair<Operation *, Value>, 2> &consumers,
      const Topology &topo) {
    // Wires each consumer: prefer register rewiring; fallback to direction
    // rewiring.
    for (const std::pair<Operation *, Value> &consumer_pair : consumers) {
      Operation *consumer_operation = consumer_pair.first;
      Value value_at_consumer = consumer_pair.second;
      if (!handleRegisterRewiring<IsCtrl>(
              consumer_operation, value_at_consumer, basics.regs,
              basics.reg_split.dst_reg_steps, basics.links, topo,
              basics.mov_dfg_id))
        handleDirectionRewiring<IsCtrl>(consumer_operation, value_at_consumer,
                                        basics.consumer_direction, basics.links,
                                        topo, forwarder);
    }
  }

  template <bool IsCtrl>
  void expandMovImpl(Operation *forwarder, const Topology &topo,
                     const DenseMap<Value, Operation *> &reserve2phi) {
    if (!validateForwarderShape<IsCtrl>(forwarder))
      return;

    // Checks if this data_mov/ctrl_mov has mapping_locs assigned by
    // MapToAcceleratorPass.
    auto mapping_locs = getMappingLocations(forwarder);
    if (!mapping_locs || mapping_locs.empty()) {
      // Skips this mov operation - it will be handled by its consumer or does
      // not need routing. This is expected for data_mov that only feeds into
      // ctrl_mov.
      if constexpr (!IsCtrl) {
        // For data_mov without mapping, verifies if it is only used by
        // ctrl_mov.
        bool only_ctrl_mov_users = true;
        for (OpOperand &use : forwarder->getResult(0).getUses()) {
          if (!isa<CtrlMovOp>(use.getOwner())) {
            only_ctrl_mov_users = false;
            break;
          }
        }
        if (only_ctrl_mov_users) {
          // This is expected - ctrl_mov handles this data transfer implicitly.
          return;
        } else {
          // This data_mov has non-ctrl_mov users but no mapping - this is an
          // error.
          forwarder->emitWarning(
              "data_mov without mapping_locs has non-ctrl_mov users");
        }
      }
      return;
    }

    MovBasics<IsCtrl> basics = buildMovBasics<IsCtrl>(forwarder, topo);

    emitMovRoutingInstructions<IsCtrl>(forwarder, basics, topo);

    SmallVector<std::pair<Operation *, Value>, 2> consumers =
        collectMovConsumers<IsCtrl>(forwarder, reserve2phi);
    if constexpr (IsCtrl) {
      if (consumers.empty())
        return;
    }

    rewriteMovConsumers<IsCtrl>(forwarder, basics, consumers, topo);
  }

  // ---------- output generation ----------.
  void logUnresolvedOperands() {
    unsigned unsrc = 0, undst = 0;
    for (auto &tile_entry : tile_time_instructions) {
      std::pair<int, int> tile_key = tile_entry.first;
      int column = tile_key.first, row = tile_key.second;
      for (auto &timestep_entry : tile_entry.second) {
        int time_step = timestep_entry.first;
        std::vector<Instruction> &vec = timestep_entry.second;
        for (size_t i = 0; i < vec.size(); ++i) {
          Instruction &inst = vec[i];
          for (size_t si = 0; si < inst.src_operands.size(); ++si) {
            Operand &s = inst.src_operands[si];
            if (s.operand == "UNRESOLVED") {
              s.color = "ERROR";
              ++unsrc;
              llvm::errs() << "[UNRESOLVED SRC] tile(" << column << "," << row
                           << ") t=" << time_step << " inst#" << i
                           << " op=" << inst.opcode << " src_idx=" << si
                           << "\n";
            }
          }
          inst.dst_operands.erase(
              std::remove_if(inst.dst_operands.begin(), inst.dst_operands.end(),
                             [](const Operand &o) {
                               return o.operand.empty() ||
                                      o.operand == "UNKNOWN";
                             }),
              inst.dst_operands.end());
          for (size_t di = 0; di < inst.dst_operands.size(); ++di) {
            Operand &d = inst.dst_operands[di];
            if (d.operand == "UNRESOLVED") {
              d.color = "ERROR";
              ++undst;
              llvm::errs() << "[UNRESOLVED DST] tile(" << column << "," << row
                           << ") t=" << time_step << " inst#" << i
                           << " op=" << inst.opcode << " dst_idx=" << di
                           << "\n";
            }
          }
        }
      }
    }
    if (unsrc + undst) {
      ModuleOp module = getOperation();
      auto diag = module.emitWarning(
          "GenerateCodePass: UNRESOLVED operands kept for debugging");
      diag << " (src=" << unsrc << ", dst=" << undst
           << "); they are highlighted with color=ERROR in YAML.";
    }
  }

  // Assigns unique IDs to all materialized instructions (including data/ctrl
  // mov hops).
  void assignInstructionIds(std::unordered_set<int> &materialized_ids) {
    instruction_id_map.clear();
    int max_assigned = -1;
    for (auto &[tile_key, timestep_map] : tile_time_instructions) {
      for (auto &[time_step, inst_vec] : timestep_map) {
        for (Instruction &inst : inst_vec) {
          if (inst.id >= 0)
            max_assigned = std::max(max_assigned, inst.id);
        }
      }
    }
    next_instruction_id = max_assigned + 1;
    for (auto &[tile_key, timestep_map] : tile_time_instructions) {
      int col = tile_key.first;
      int row = tile_key.second;
      for (auto &[time_step, inst_vec] : timestep_map) {
        for (size_t idx = 0; idx < inst_vec.size(); ++idx) {
          Instruction &inst = inst_vec[idx];
          if (inst.id < 0)
            inst.id = next_instruction_id++;
          instruction_id_map[{col, row, time_step, (int)idx}] = inst.id;
          materialized_ids.insert(inst.id);
        }
      }
    }
  }

  // Looks up instruction ID by InstructionReference (col,row,time_step,idx in
  // bucket).
  int lookupInstructionId(const InstructionReference &ref) const {
    auto it =
        instruction_id_map.find({ref.col_idx, ref.row_idx, ref.t, ref.idx});
    if (it == instruction_id_map.end())
      return -1;
    return it->second;
  }

  // Helper to escape strings for DOT/JSON.
  static std::string escape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      if (c == '"')
        out += "\\\"";
      else if (c == '\\')
        out += "\\\\";
      else if (c == '\n')
        out += "\\n";
      else
        out += c;
    }
    return out;
  }

  // Helper to extract dfg_id from operation.
  static int getDfgId(Operation *op) {
    if (auto id_attr = op->getAttrOfType<IntegerAttr>(attr::kDfgId)) {
      return id_attr.getInt();
    }
    return -1;
  }

  // Helper to extract tile coordinates and time_step from mapping_locs.
  struct LocationInfo {
    int tile_x = -1;
    int tile_y = -1;
    int time_step = -1;
    int index_per_ii = -1;
    int invalid_iterations = 0;
    bool has_tile = false;
  };

  static LocationInfo getLocationInfo(Operation *op) {
    LocationInfo info;
    if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
      for (Attribute a : arr) {
        auto d = dyn_cast<DictionaryAttr>(a);
        if (!d)
          continue;
        auto resource = dyn_cast_or_null<StringAttr>(d.get("resource"));

        // Extracts time_step from any resource type.
        if (auto ts_attr = dyn_cast_or_null<IntegerAttr>(d.get("time_step"))) {
          info.time_step = ts_attr.getInt();
        }
        if (auto index_attr =
                dyn_cast_or_null<IntegerAttr>(d.get("index_per_ii"))) {
          info.index_per_ii = index_attr.getInt();
        }
        if (auto invalid_attr =
                dyn_cast_or_null<IntegerAttr>(d.get("invalid_iterations"))) {
          info.invalid_iterations = invalid_attr.getInt();
        }

        // Only tile resources have x/y coordinates.
        if (resource && resource.getValue() == "tile") {
          if (auto x_attr = dyn_cast_or_null<IntegerAttr>(d.get("x"))) {
            info.tile_x = x_attr.getInt();
          }
          if (auto y_attr = dyn_cast_or_null<IntegerAttr>(d.get("y"))) {
            info.tile_y = y_attr.getInt();
          }
          info.has_tile = true;
        }
      }
    }
    return info;
  }

  struct DfgNodeInfo {
    std::string opcode;
    int tile_x = -1;
    int tile_y = -1;
    int time_step = -1;
  };
  using DfgNodeMap = std::map<int, DfgNodeInfo>;
  using DfgEdgeList = std::vector<std::pair<int, int>>;

  struct HopRewriteInfo {
    int mov_id = -1;
    int producer_id = -1;
    SmallVector<int, 4> consumer_ids;
    SmallVector<DfgNodeInfo, 8> pre_mov_nodes;
    SmallVector<DfgNodeInfo, 8> post_mov_nodes;
  };

  void collectCtrlMovReserves(func::FuncOp func,
                              DenseMap<Value, SmallVector<Operation *, 2>>
                                  &reserve_to_ctrl_movs) const {
    func.walk([&](Operation *operation) {
      if (auto ctrl_mov_op = dyn_cast<CtrlMovOp>(operation)) {
        if (ctrl_mov_op->getNumOperands() >= 2) {
          Value reserve_value = ctrl_mov_op->getOperand(1);
          reserve_to_ctrl_movs[reserve_value].push_back(operation);
        }
      }
    });
  }

  void addEdge(DfgEdgeList &edges, int src, int dst) const {
    if (src < 0 || dst < 0)
      return;
    if (src == dst)
      return; // Avoids self-loop.
    edges.emplace_back(src, dst);
  }

  void buildSsaNodesAndEdges(
      func::FuncOp func,
      const DenseMap<Value, SmallVector<Operation *, 2>> &reserve_to_ctrl_movs,
      DfgNodeMap &nodes, DfgEdgeList &edges) {
    func.walk([&](Operation *operation) {
      if (operation == func.getOperation())
        return; // Skips function itself.
      if (isReserve(operation))
        return; // Skips reserve nodes entirely (bypass later).
      if (isa<YieldOp>(operation))
        return; // Skips yield nodes entirely (bypass later).
      // Skips operations inside fused_op regions - they are handled by hardware
      if (operation->getParentOp() && isFusedOp(operation->getParentOp())) {
        return;
      }

      int dfg_id = getDfgId(operation);
      if (dfg_id < 0) {
        llvm::errs() << "[WARN] Operation without dfg_id: " << *operation
                     << "\n";
        return;
      }
      dfg_id_to_op[dfg_id] = operation;

      std::string opcode = getOpcode(operation);
      LocationInfo location_info = getLocationInfo(operation);

      if ((isDataMov(operation) || isCtrlMov(operation)) &&
          !location_info.has_tile) {
        nodes[dfg_id] = DfgNodeInfo{opcode, -1, -1, location_info.time_step};
      } else {
        nodes[dfg_id] =
            DfgNodeInfo{opcode, location_info.tile_x, location_info.tile_y,
                        location_info.time_step};
      }

      for (Value operand : operation->getOperands()) {
        Operation *producer_op = operand.getDefiningOp();
        if (producer_op && isReserve(producer_op)) {
          auto it = reserve_to_ctrl_movs.find(operand);
          if (it != reserve_to_ctrl_movs.end()) {
            for (Operation *ctrl_mov_op : it->second) {
              if (ctrl_mov_op == operation)
                continue;
              int producer_id = getDfgId(ctrl_mov_op);
              if (producer_id >= 0)
                addEdge(edges, producer_id, dfg_id);
            }
          }
          continue;
        }
        if (producer_op) {
          int producer_id = getDfgId(producer_op);
          addEdge(edges, producer_id, dfg_id);
        }
      }
    });
  }

  void
  collectHopRewrites(func::FuncOp func, const Topology &topology,
                     DfgNodeMap &nodes, const DfgEdgeList &original_edges,
                     DfgEdgeList &edges,
                     llvm::SmallDenseSet<std::pair<int, int>, 32> &skip_edges,
                     std::vector<HopRewriteInfo> &rewrites) {
    func.walk([&](Operation *operation) {
      if (!isDataMov(operation) && !isCtrlMov(operation))
        return;
      int mov_dfg_id = getDfgId(operation);
      if (mov_dfg_id < 0)
        return;

      int producer_id = -1;
      if (operation->getNumOperands() >= 1) {
        if (Operation *producer = operation->getOperand(0).getDefiningOp())
          producer_id = getDfgId(producer);
      }

      SmallVector<int, 4> consumer_ids;
      if (operation->getNumResults() >= 1) {
        for (OpOperand &use : operation->getResult(0).getUses()) {
          Operation *user = use.getOwner();
          while (user && isReserve(user)) {
            if (user->getNumResults() == 0) {
              user = nullptr;
              break;
            }
            bool forwarded = false;
            for (OpOperand &reserve_use : user->getResult(0).getUses()) {
              Operation *forward_user = reserve_use.getOwner();
              if (!forward_user)
                continue;
              int consumer_id = getDfgId(forward_user);
              if (consumer_id >= 0)
                consumer_ids.push_back(consumer_id);
              forwarded = true;
            }
            user = nullptr;
            if (forwarded)
              break;
          }
          if (user) {
            int consumer_id = getDfgId(user);
            if (consumer_id >= 0)
              consumer_ids.push_back(consumer_id);
          }
        }
      }

      SmallVector<LinkStep, 8> link_steps = collectLinkSteps(operation);
      SmallVector<DfgNodeInfo, 8> pre_mov_nodes;
      SmallVector<DfgNodeInfo, 8> post_mov_nodes;
      if (!link_steps.empty()) {
        SmallVector<RegStep, 4> regs = collectRegSteps(operation);
        MovRegSplit reg_split = splitMovRegs(regs, link_steps);
        std::string opcode = isCtrlMov(operation) ? "CTRL_MOV" : "DATA_MOV";

        if (!reg_split.src_reg_steps.empty()) {
          int src_tile_id = topology.srcTileOfLink(link_steps.front().link_id);
          auto coord = topology.tile_location.lookup(src_tile_id);
          SmallVector<std::pair<RegStep, RegStep>, 4> transfers =
              collectRegisterTransfers(reg_split.src_reg_steps);

          for (const auto &[prev, cur] : transfers) {
            (void)prev;
            pre_mov_nodes.push_back(
                DfgNodeInfo{opcode, coord.first, coord.second, cur.time_step});
          }
          pre_mov_nodes.push_back(DfgNodeInfo{opcode, coord.first, coord.second,
                                              link_steps.front().time_step});
        }

        // Build intermediate routing nodes so the synthetic DFG matches the
        // materialized instruction stream:
        // - plain link-to-link segments emit one hop node
        // - link-to-reg-to-link segments emit deposit / reg-transfer / egress
        // nodes
        if (link_steps.size() > 1) {
          for (size_t i = 1; i < link_steps.size(); ++i) {
            int middle_tile_id = topology.srcTileOfLink(link_steps[i].link_id);
            auto coord = topology.tile_location.lookup(middle_tile_id);
            SmallVector<RegStep, 4> inter_link_regs = collectRegsBetweenLinks(
                regs, link_steps[i - 1].time_step, link_steps[i].time_step);
            if (inter_link_regs.empty()) {
              pre_mov_nodes.push_back(DfgNodeInfo{
                  opcode, coord.first, coord.second, link_steps[i].time_step});
              continue;
            }

            pre_mov_nodes.push_back(
                DfgNodeInfo{opcode, coord.first, coord.second,
                            inter_link_regs.front().time_step});

            SmallVector<std::pair<RegStep, RegStep>, 4> transfers =
                collectRegisterTransfers(inter_link_regs);
            for (const auto &[prev, cur] : transfers) {
              (void)prev;
              pre_mov_nodes.push_back(DfgNodeInfo{opcode, coord.first,
                                                  coord.second, cur.time_step});
            }

            pre_mov_nodes.push_back(DfgNodeInfo{
                opcode, coord.first, coord.second, link_steps[i].time_step});
          }
        }

        if (reg_split.dst_reg_steps.size() > 1) {
          int dst_tile_id = topology.dstTileOfLink(link_steps.back().link_id);
          auto coord = topology.tile_location.lookup(dst_tile_id);
          SmallVector<std::pair<RegStep, RegStep>, 4> transfers =
              collectRegisterTransfers(reg_split.dst_reg_steps);

          for (const auto &[prev, cur] : transfers) {
            (void)prev;
            post_mov_nodes.push_back(
                DfgNodeInfo{opcode, coord.first, coord.second, cur.time_step});
          }
        }
      }

      if (!pre_mov_nodes.empty() || !post_mov_nodes.empty()) {
        auto it = nodes.find(mov_dfg_id);
        if (it != nodes.end()) {
          it->second.tile_x = -1;
          it->second.tile_y = -1;
          it->second.time_step = -1;
        }

        int base = mov_dfg_id * 10000;
        for (size_t i = 0; i < pre_mov_nodes.size(); ++i) {
          int node_id = base + static_cast<int>(i);
          nodes[node_id] = pre_mov_nodes[i];
        }
        for (size_t i = 0; i < post_mov_nodes.size(); ++i) {
          int node_id = base + static_cast<int>(pre_mov_nodes.size() + i);
          nodes[node_id] = post_mov_nodes[i];
        }

        if (producer_id >= 0)
          skip_edges.insert({producer_id, mov_dfg_id});
        for (int consumer_id : consumer_ids)
          skip_edges.insert({mov_dfg_id, consumer_id});

        rewrites.push_back(HopRewriteInfo{mov_dfg_id, producer_id, consumer_ids,
                                          pre_mov_nodes, post_mov_nodes});
      }
    });

    for (const auto &edge : original_edges) {
      if (skip_edges.count(edge))
        continue;
      edges.push_back(edge);
    }

    for (const HopRewriteInfo &rewrite : rewrites) {
      int base = rewrite.mov_id * 10000;
      int previous = rewrite.producer_id;

      for (size_t i = 0; i < rewrite.pre_mov_nodes.size(); ++i) {
        int node_id = base + static_cast<int>(i);
        if (previous >= 0)
          edges.emplace_back(previous, node_id);
        previous = node_id;
      }
      if (previous >= 0)
        edges.emplace_back(previous, rewrite.mov_id);
      previous = rewrite.mov_id;
      for (size_t i = 0; i < rewrite.post_mov_nodes.size(); ++i) {
        int node_id = base + static_cast<int>(rewrite.pre_mov_nodes.size() + i);
        edges.emplace_back(previous, node_id);
        previous = node_id;
      }
      for (int consumer_id : rewrite.consumer_ids)
        edges.emplace_back(previous, consumer_id);
    }
  }

  void adjustRegisterOnlyMovCoords(DfgNodeMap &nodes,
                                   const DfgEdgeList &edges) {
    std::unordered_map<int, std::vector<int>> successors;
    for (const auto &edge : edges)
      successors[edge.first].push_back(edge.second);

    auto scanMappingKinds = [](Operation *op, bool &has_link,
                               bool &has_register, int &min_register_ts) {
      has_link = false;
      has_register = false;
      min_register_ts = -1;
      if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
        for (Attribute attr : arr) {
          auto dict = dyn_cast<DictionaryAttr>(attr);
          if (!dict)
            continue;
          auto res = dyn_cast_or_null<StringAttr>(dict.get("resource"));
          if (!res)
            continue;
          if (res.getValue() == "link") {
            has_link = true;
          } else if (res.getValue() == "register" || res.getValue() == "reg") {
            has_register = true;
            if (auto ts_attr =
                    dyn_cast_or_null<IntegerAttr>(dict.get("time_step"))) {
              int time_step = ts_attr.getInt();
              if (min_register_ts < 0 || time_step < min_register_ts)
                min_register_ts = time_step;
            }
          }
        }
      }
    };

    for (auto &entry : nodes) {
      int node_id = entry.first;
      DfgNodeInfo &node = entry.second;
      if (node.tile_x != -1 || node.tile_y != -1)
        continue;
      if (node.opcode != "DATA_MOV" && node.opcode != "CTRL_MOV")
        continue;

      Operation *op = dfg_id_to_op.lookup(node_id);
      if (!op)
        continue;

      bool has_link = false, has_register = false;
      int min_register_ts = -1;
      scanMappingKinds(op, has_link, has_register, min_register_ts);
      if (!has_register)
        continue; // if register not present, keep as is; if present, fill
                  // coords from consumer

      auto succ_it = successors.find(node_id);
      if (succ_it == successors.end())
        continue;
      for (int consumer_id : succ_it->second) {
        auto consumer_it = nodes.find(consumer_id);
        if (consumer_it == nodes.end())
          continue;
        const DfgNodeInfo &consumer_node = consumer_it->second;
        if (consumer_node.tile_x == -1 || consumer_node.tile_y == -1)
          continue;
        node.tile_x = consumer_node.tile_x;
        node.tile_y = consumer_node.tile_y;
        if (node.time_step < 0 && min_register_ts >= 0)
          node.time_step = min_register_ts;
        break;
      }
    }
  }

  void
  pruneMovNodesWithoutCoords(DfgNodeMap &nodes, DfgEdgeList &edges,
                             const std::unordered_set<int> &materialized_ids) {
    std::unordered_set<int> nodes_to_remove;
    for (const auto &entry : nodes) {
      int node_id = entry.first;
      const DfgNodeInfo &node = entry.second;
      if ((node.opcode == "DATA_MOV" || node.opcode == "CTRL_MOV") &&
          (node.tile_x == -1 && node.tile_y == -1)) {
        nodes_to_remove.insert(node_id);
      }
      if ((node.opcode == "DATA_MOV" || node.opcode == "CTRL_MOV") &&
          !materialized_ids.count(node_id)) {
        nodes_to_remove.insert(node_id);
      }
    }
    if (nodes_to_remove.empty())
      return;

    std::unordered_map<int, std::vector<int>> predecessors, successors;
    for (const auto &edge : edges) {
      successors[edge.first].push_back(edge.second);
      predecessors[edge.second].push_back(edge.first);
    }
    std::unordered_set<uint64_t> dedup_edge_set;
    auto encode = [](int from, int to) -> uint64_t {
      return (static_cast<uint64_t>(from) << 32) ^ static_cast<uint32_t>(to);
    };

    std::vector<std::pair<int, int>> new_edges;
    for (const auto &edge : edges) {
      if (nodes_to_remove.count(edge.first) ||
          nodes_to_remove.count(edge.second))
        continue;
      uint64_t key = encode(edge.first, edge.second);
      if (dedup_edge_set.insert(key).second)
        new_edges.push_back(edge);
    }

    for (int removed : nodes_to_remove) {
      const auto &preds = predecessors[removed];
      const auto &succs = successors[removed];
      for (int pred : preds) {
        for (int succ : succs) {
          if (pred == succ)
            continue;
          uint64_t key = encode(pred, succ);
          if (dedup_edge_set.insert(key).second)
            new_edges.emplace_back(pred, succ);
        }
      }
    }
    edges.swap(new_edges);
    for (int removed : nodes_to_remove)
      nodes.erase(removed);
  }

  void emitDotOutput(const DfgNodeMap &nodes, const DfgEdgeList &edges) const {
    std::error_code ec;
    llvm::raw_fd_ostream dot_out("tmp-generated-dfg.dot", ec);
    if (ec)
      return;

    dot_out
        << "digraph DFG {\n  rankdir=LR;\n  node [shape=box, style=filled];\n";
    auto color_for = [](const std::string &op) {
      if (op == "DATA_MOV")
        return "lightgreen";
      if (op == "CTRL_MOV")
        return "lightyellow";
      if (op == "CONSTANT")
        return "lightblue";
      return "white";
    };
    for (const auto &entry : nodes) {
      int id = entry.first;
      const DfgNodeInfo &node = entry.second;
      dot_out << "  n" << id << " [label=\"" << escape(node.opcode)
              << "\\nID=" << id;
      if (node.tile_x >= 0 && node.tile_y >= 0) {
        dot_out << "\\n(" << node.tile_x << "," << node.tile_y << ")";
      }
      if (node.time_step >= 0) {
        dot_out << " t=" << node.time_step;
      }
      dot_out << "\", fillcolor=" << color_for(node.opcode) << "];\n";
    }
    for (const auto &edge : edges) {
      dot_out << "  n" << edge.first << " -> n" << edge.second << ";\n";
    }
    dot_out << "}\n";
  }

  void emitYamlOutput(const DfgNodeMap &nodes, const DfgEdgeList &edges) const {
    std::error_code ec;
    llvm::raw_fd_ostream yaml_out("tmp-generated-dfg.yaml", ec);
    if (ec)
      return;

    yaml_out << "nodes:\n";
    for (const auto &entry : nodes) {
      int id = entry.first;
      const DfgNodeInfo &node = entry.second;
      yaml_out << "  - id: " << id << "\n"
               << "    opcode: \"" << escape(node.opcode) << "\"\n"
               << "    tile_x: " << node.tile_x << "\n"
               << "    tile_y: " << node.tile_y << "\n"
               << "    time_step: " << node.time_step << "\n";
    }
    yaml_out << "edges:\n";
    for (const auto &edge : edges) {
      yaml_out << "  - from: " << edge.first << "\n"
               << "    to: " << edge.second << "\n";
    }
  }

  // Ensures every materialized instruction (from
  // tmp-generated-instructions.yaml) has a corresponding DFG node keyed by
  // instruction.id. This is important for synthetic instructions that don't
  // exist as IR ops (e.g., egress DATA_MOV with id=mov_id*10000).
  void injectMaterializedInstructionNodes(DfgNodeMap &nodes) const {
    for (const auto &[tile_key, timestep_map] : tile_time_instructions) {
      int tile_x = tile_key.first;
      int tile_y = tile_key.second;
      for (const auto &[idx_per_ii, inst_vec] : timestep_map) {
        (void)idx_per_ii;
        for (const Instruction &inst : inst_vec) {
          if (inst.id < 0)
            continue;
          // Only inject if absent; for IR-backed ops (id == dfg_id) the
          // existing node is fine, but injecting is also harmless.
          auto it = nodes.find(inst.id);
          if (it == nodes.end()) {
            nodes[inst.id] =
                DfgNodeInfo{inst.opcode, tile_x, tile_y, inst.time_step};
          } else {
            // If the existing node has no coords, prefer concrete instruction
            // coords.
            if (it->second.tile_x < 0 || it->second.tile_y < 0) {
              it->second.tile_x = tile_x;
              it->second.tile_y = tile_y;
            }
            if (it->second.time_step < 0 && inst.time_step >= 0)
              it->second.time_step = inst.time_step;
          }
        }
      }
    }
  }

  struct DfgBuilder {
    GenerateCodePass &pass;
    func::FuncOp func;
    const Topology &topology;
    const std::unordered_set<int> &materialized_ids;

    void run() {
      DfgNodeMap nodes;
      DfgEdgeList edges;

      DenseMap<Value, SmallVector<Operation *, 2>> reserve_to_ctrl_movs;
      pass.collectCtrlMovReserves(func, reserve_to_ctrl_movs);

      pass.buildSsaNodesAndEdges(func, reserve_to_ctrl_movs, nodes, edges);

      std::vector<std::pair<int, int>> original_edges = edges;
      edges.clear();
      llvm::SmallDenseSet<std::pair<int, int>, 32> edges_to_skip;
      std::vector<HopRewriteInfo> hop_rewrites;

      pass.collectHopRewrites(func, topology, nodes, original_edges, edges,
                              edges_to_skip, hop_rewrites);

      // Bring in synthetic instruction nodes (egress/deposit/hops) by
      // instruction id.
      pass.injectMaterializedInstructionNodes(nodes);

      pass.adjustRegisterOnlyMovCoords(nodes, edges);

      pass.pruneMovNodesWithoutCoords(nodes, edges, materialized_ids);

      pass.emitDotOutput(nodes, edges);
      pass.emitYamlOutput(nodes, edges);

      llvm::outs() << "[generate-code] DFG (SSA-based) emitted: nodes="
                   << nodes.size() << ", edges=" << edges.size()
                   << " -> tmp-generated-dfg.dot, tmp-generated-dfg.yaml\n";
    }
  };

  // Writes DOT and JSON DFG outputs based on SSA and dfg_id attributes.
  // Applies hop-aware coordinates/middle-node insertion for DATA_MOV /
  // CTRL_MOV, and bypasses reserve nodes (no node for reserve, edges direct
  // from producer to consumer).
  void writeDfgOutputSSA(func::FuncOp func, const Topology &topology,
                         const std::unordered_set<int> &materialized_ids) {
    DfgBuilder{*this, func, topology, materialized_ids}.run();
  }

  ArrayConfig buildArrayConfig(int columns, int rows, int compiled_ii = -1) {
    ArrayConfig config{columns, rows, compiled_ii, {}};
    std::map<std::pair<int, int>, std::vector<Instruction>> tile_insts;

    // Flattens and sorts by timesteps.
    for (auto &[tile_key, timestep_map] : tile_time_instructions) {
      auto &flat = tile_insts[tile_key];
      for (auto &[timestep, instruction_vec] : timestep_map)
        for (Instruction &inst : instruction_vec)
          flat.push_back(inst);
      std::stable_sort(flat.begin(), flat.end(),
                       [](const Instruction &a, const Instruction &b) {
                         return a.time_step < b.time_step;
                       });
    }

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < columns; ++c) {
        auto it = tile_insts.find({c, r});
        if (it == tile_insts.end())
          continue;
        Tile tile(c, r, r * columns + c);
        for (Instruction &inst : it->second)
          tile.entry.instructions.push_back(inst);
        config.cores.push_back(std::move(tile));
      }
    }
    return config;
  }

  using IndexGroups = std::map<int, std::vector<const Instruction *>>;

  static IndexGroups
  groupByIndexPerIi(const std::vector<Instruction> &instructions) {
    IndexGroups index_groups;
    for (const Instruction &inst : instructions)
      index_groups[inst.index_per_ii].push_back(&inst);
    return index_groups;
  }

  void emitYamlForTile(llvm::raw_fd_ostream &yaml_out, const Tile &tile) {
    yaml_out << "    - column: " << tile.col_idx
             << "\n      row: " << tile.row_idx << "\n      core_id: \""
             << tile.core_id << "\"\n      entries:\n";

    // Groups instructions by index_per_ii.
    IndexGroups index_groups = groupByIndexPerIi(tile.entry.instructions);

    yaml_out << "        - entry_id: \"entry0\"\n          instructions:\n";
    for (const auto &index_pair : index_groups) {
      int index_per_ii = index_pair.first;
      auto operations = index_pair.second;
      std::stable_sort(operations.begin(), operations.end(),
                       [](const Instruction *a, const Instruction *b) {
                         return a->time_step < b->time_step;
                       });

      yaml_out << "            - index_per_ii: " << index_per_ii
               << "\n              operations:\n";
      for (const Instruction *inst : operations) {
        yaml_out << "                - opcode: \"" << inst->opcode << "\"\n";
        if (inst->id >= 0)
          yaml_out << "                  id: " << inst->id << "\n";
        yaml_out << "                  time_step: " << inst->time_step << "\n"
                 << "                  invalid_iterations: "
                 << inst->invalid_iterations << "\n";
        // sources.
        if (!inst->src_operands.empty()) {
          yaml_out << "                  src_operands:\n";
          for (const Operand &opnd : inst->src_operands) {
            yaml_out << "                    - operand: \"" << opnd.operand
                     << "\"\n                      color: \"" << opnd.color
                     << "\"\n";
            if (!opnd.access.empty()) {
              yaml_out << "                      access: \"" << opnd.access
                       << "\"\n                      offsets: [";
              for (size_t i = 0; i < opnd.offsets.size(); ++i) {
                if (i)
                  yaml_out << ", ";
                yaml_out << opnd.offsets[i];
              }
              yaml_out << "]\n";
            }
          }
        }
        // destinations.
        if (!inst->dst_operands.empty()) {
          yaml_out << "                  dst_operands:\n";
          for (const Operand &opnd : inst->dst_operands)
            yaml_out << "                    - operand: \"" << opnd.operand
                     << "\"\n                      color: \"" << opnd.color
                     << "\"\n";
        }
      }
    }
  }

  void writeYAMLOutput(const ArrayConfig &config) {
    std::error_code ec;
    llvm::raw_fd_ostream yaml_out("tmp-generated-instructions.yaml", ec);
    if (ec)
      return;

    yaml_out << "array_config:\n  columns: " << config.columns
             << "\n  rows: " << config.rows;
    if (config.compiled_ii >= 0) {
      yaml_out << "\n  compiled_ii: " << config.compiled_ii;
    }
    yaml_out << "\n  cores:\n";
    for (const Tile &core : config.cores) {
      emitYamlForTile(yaml_out, core);
    }
    yaml_out.close();
  }

  // Direction vs const/reg helpers.
  static bool isDirectionalOperand(const std::string &operand) {
    // Non-directional operands start with $ (registers), # (constants), or arg
    // (function arguments).
    if (operand.empty())
      return false;
    if (operand[0] == '$' || operand[0] == '#')
      return false;
    // Checks if the operand starts with "arg" followed by digits (e.g., "arg0",
    // "arg1").
    if (operand.size() >= 4 && operand.substr(0, 3) == "arg") {
      // Verifies that the rest is digits.
      bool all_digits = true;
      for (size_t i = 3; i < operand.size(); ++i) {
        if (!std::isdigit(operand[i])) {
          all_digits = false;
          break;
        }
      }
      if (all_digits)
        return false;
    }
    return true;
  }

  static std::string formatOperand(const Operand &operand) {
    if (!operand.access.empty()) {
      std::string result = "[" + operand.access + "(" + operand.operand + ", ";
      for (size_t i = 0; i < operand.offsets.size(); ++i) {
        if (i)
          result += ", ";
        result += std::to_string(operand.offsets[i]);
      }
      return result + ")]";
    }
    std::string result = "[" + operand.operand;
    if (isDirectionalOperand(operand.operand)) {
      result += ", " + operand.color;
    }
    result += "]";
    return result;
  }

  void emitAsmForTile(llvm::raw_fd_ostream &asm_out, const Tile &tile) {
    asm_out << "PE(" << tile.col_idx << "," << tile.row_idx << "):\n";

    // Groups instructions by index_per_ii.
    IndexGroups index_groups = groupByIndexPerIi(tile.entry.instructions);

    for (const auto &index_pair : index_groups) {
      int index_per_ii = index_pair.first;
      auto instructions = index_pair.second;
      std::stable_sort(instructions.begin(), instructions.end(),
                       [](const Instruction *a, const Instruction *b) {
                         return a->time_step < b->time_step;
                       });

      asm_out << "{\n";
      for (size_t i = 0; i < instructions.size(); ++i) {
        const Instruction *inst = instructions[i];
        asm_out << "  " << inst->opcode;
        for (const Operand &operand : inst->src_operands)
          asm_out << ", " << formatOperand(operand);
        if (!inst->dst_operands.empty()) {
          asm_out << " -> ";
          for (size_t j = 0; j < inst->dst_operands.size(); ++j) {
            if (j > 0)
              asm_out << ", ";
            asm_out << formatOperand(inst->dst_operands[j]);
          }
        }
        asm_out << " (t=" << inst->time_step
                << ", inv_iters=" << inst->invalid_iterations << ")\n";
      }
      asm_out << "} (idx_per_ii=" << index_per_ii << ")\n";
    }
    asm_out << "\n";
  }

  void writeAsmOutput(const ArrayConfig &config) {
    std::error_code ec;
    llvm::raw_fd_ostream asm_out("tmp-generated-instructions.asm", ec);
    if (ec)
      return;

    if (config.compiled_ii >= 0) {
      asm_out << "# Compiled II: " << config.compiled_ii << "\n\n";
    }

    for (const Tile &core : config.cores) {
      emitAsmForTile(asm_out, core);
    }
    asm_out.close();
  }

  // Endpoint deposits: on destination tiles at earliest reg time_step, move
  // [incoming_dir] -> [$reg]. CTRL_MOV paths emit CTRL_MOV deposits; DATA_MOV
  // paths emit DATA_MOV deposits.
  void placeDstDeposit(const Topology &topo, int dst_tile_id, int time_step,
                       StringRef incoming_dir, int reg_id,
                       bool asCtrlMov = false, int assigned_id = -1) {
    uint64_t signature = static_cast<uint64_t>(
        llvm::hash_combine(dst_tile_id, time_step, reg_id));
    if (!deposit_signatures.insert(signature).second)
      return; // already placed.
    auto [tile_x, tile_y] = topo.tile_location.lookup(dst_tile_id);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.id = assigned_id;
    inst.time_step = time_step;
    inst.index_per_ii = indexPerIiFromTimeStep(time_step);
    inst.invalid_iterations = syntheticInvalidIterations(time_step);
    inst.src_operands.emplace_back(incoming_dir.str(), "RED");
    inst.dst_operands.emplace_back("$" + std::to_string(reg_id), "RED");
    tile_time_instructions[{tile_x, tile_y}][inst.index_per_ii].push_back(
        std::move(inst));
  }

  // Utilities to access instruction buckets/pointers.
  std::vector<Instruction> &getInstructionBucket(int column, int row,
                                                 int time_step) {
    return tile_time_instructions[{column, row}][time_step];
  }
  Instruction *getInstructionPointer(Operation *operation) {
    auto it = operation_to_instruction_reference.find(operation);
    if (it == operation_to_instruction_reference.end())
      return nullptr;
    auto [c, r, t, idx] = it->second;
    auto &vec = tile_time_instructions[{c, r}][t];
    if (idx < 0 || idx >= (int)vec.size())
      return nullptr;
    return &vec[idx];
  }

  Instruction *getForwardingInstruction(OpResult source) {
    Operation *producer = source.getOwner();
    auto existing = forwarding_instruction_reference.find(producer);
    if (existing != forwarding_instruction_reference.end()) {
      auto [column, row, index, position] = existing->second;
      auto &instructions = tile_time_instructions[{column, row}][index];
      return &instructions[position];
    }

    Instruction *compute = getInstructionPointer(producer);
    if (!compute || compute->src_operands.empty()) {
      producer->emitError("cannot forward a MAC input before it is routed");
      timing_field_error = true;
      return nullptr;
    }

    auto reference = operation_to_instruction_reference.find(producer);
    if (reference == operation_to_instruction_reference.end()) {
      return nullptr;
    }

    auto [column, row, index, position] = reference->second;
    (void)position;
    Instruction forwarding("DATA_MOV");
    forwarding.time_step = compute->time_step;
    forwarding.index_per_ii = compute->index_per_ii;
    forwarding.invalid_iterations = compute->invalid_iterations;
    forwarding.src_operands.push_back(compute->src_operands.front());

    auto &instructions = tile_time_instructions[{column, row}][index];
    instructions.push_back(std::move(forwarding));
    int forwarding_position = static_cast<int>(instructions.size()) - 1;
    forwarding_instruction_reference[producer] =
        InstructionReference{column, row, index, forwarding_position};
    return &instructions[forwarding_position];
  }

  // Replaces the exact source slots in consumers that correspond to
  // `value_at_consumer`, or fills the first UNRESOLVED placeholder if a 1:1
  // match wasn't found.
  void setConsumerSourceExact(Operation *consumer, Value value_at_consumer,
                              const std::string &text) {
    Instruction *ci = getInstructionPointer(consumer);
    if (!ci)
      return;
    auto it = operation_to_operands.find(consumer);
    if (it == operation_to_operands.end())
      return;
    auto &ops = it->second;
    for (size_t i = 0; i < ops.size() && i < ci->src_operands.size(); ++i) {
      if (ops[i] == value_at_consumer) {
        ci->src_operands[i].operand = text;
        return;
      }
    }
    for (Operand &src : ci->src_operands)
      if (src.operand == "UNRESOLVED") {
        src.operand = text;
        return;
      }
  }

  // Appends a destination only once.
  static void setUniqueDestination(Instruction *inst, const std::string &text) {
    for (Operand &d : inst->dst_operands)
      if (d.operand == text)
        return;
    inst->dst_operands.emplace_back(text, "RED");
  }

  // Checks the supported static memref layout before interpreting element
  // offsets.
  FailureOr<MemRefType> getConfiguredMemref(KernelOp kernel, int64_t index) {
    if (index < 0 || index >= static_cast<int64_t>(kernel.getInputs().size()))
      return failure();
    Type type = kernel.getInputs()[index].getType();
    if (auto predicated = dyn_cast<PredicatedValue>(type))
      type = predicated.getValueType();
    auto memref = dyn_cast<MemRefType>(type);
    if (!memref || !memref.hasStaticShape() ||
        !memref.getLayout().isIdentity() ||
        !memref.getElementType().isInteger(32) || memref.getNumElements() <= 0)
      return failure();
    return memref;
  }

  // Keeps runtime memory references symbolic until the existing loader binds
  // them.
  Operand getConfiguredOperand(int64_t index, ArrayRef<int64_t> offsets,
                               bool address) {
    Operand operand("arg" + std::to_string(index));
    operand.access = address ? "address" : "value";
    operand.offsets.assign(offsets.begin(), offsets.end());
    return operand;
  }

  // Loads template-level bindings used by configured operands.
  LogicalResult configureKernel(KernelOp kernel) {
    current_kernel = kernel;

    bool has_mac = false;
    kernel.walk([&](MacOp) { has_mac = true; });
    if (!has_mac) {
      return success();
    }

    auto mapping_info =
        kernel->getAttrOfType<DictionaryAttr>(attr::kMappingInfo);
    auto mapping_mode = mapping_info
                            ? mapping_info.getAs<StringAttr>(attr::kMappingMode)
                            : StringAttr{};
    static_dataflow = current_compiled_ii == 1 && mapping_mode &&
                      mapping_mode.getValue() == "spatial-only";

    auto metadata = kernel->getAttrOfType<DictionaryAttr>("kernel_metadata");
    auto template_metadata = metadata
                                 ? metadata.getAs<DictionaryAttr>("template")
                                 : DictionaryAttr{};
    auto stationary =
        template_metadata
            ? template_metadata.getAs<DictionaryAttr>("stationary")
            : DictionaryAttr{};
    auto input = stationary ? stationary.getAs<IntegerAttr>("kernel_input")
                            : IntegerAttr{};
    stationary_map =
        stationary ? stationary.getAs<AffineMapAttr>("map") : AffineMapAttr{};

    if (!input || !stationary_map) {
      return kernel.emitOpError(
          "configured MAC requires stationary kernel_input and map");
    }

    stationary_input_index = input.getInt();
    if (failed(getConfiguredMemref(kernel, stationary_input_index))) {
      return kernel.emitOpError(
          "stationary input requires a static identity-layout i32 "
          "memref");
    }
    return success();
  }

  // Encodes only operands that require launch-time configuration.
  // Returning false delegates the operation to the existing generic encoder.
  bool encodeConfiguredOperands(Operation *op, Instruction &instruction,
                                SmallVectorImpl<Value> &operands) {
    if (auto mac = dyn_cast<MacOp>(op)) {
      if (!current_kernel || !stationary_map) {
        op->emitError("configured MAC is missing kernel metadata");
        timing_field_error = true;
        return true;
      }

      for (Value operand : op->getOperands()) {
        operands.push_back(operand);
        instruction.src_operands.emplace_back("UNRESOLVED", "RED");
      }

      TileLocation tile = getTileLocation(op);
      auto weights =
          getConfiguredMemref(current_kernel, stationary_input_index);
      SmallVector<Attribute> indices;
      Type index_type = IndexType::get(op->getContext());
      if (!tile.has_tile || failed(weights) ||
          failed(stationary_map.getValue().constantFold(
              {IntegerAttr::get(index_type, tile.col_idx),
               IntegerAttr::get(index_type, tile.row_idx)},
              indices)) ||
          indices.size() != static_cast<size_t>(weights->getRank())) {
        op->emitError("stationary map cannot resolve this MAC Tile");
        timing_field_error = true;
        return true;
      }

      int64_t offset = 0;
      for (auto [attribute, size] : llvm::zip(indices, weights->getShape())) {
        auto index = dyn_cast<IntegerAttr>(attribute);
        if (!index || index.getInt() < 0 || index.getInt() >= size) {
          op->emitError("stationary element is outside its memref");
          timing_field_error = true;
          return true;
        }
        offset = offset * size + index.getInt();
      }

      operands.insert(operands.begin() + 1, Value());
      instruction.src_operands.insert(
          instruction.src_operands.begin() + 1,
          getConfiguredOperand(stationary_input_index, {offset}, false));
      return true;
    }

    if (!isa<LoadOp, StoreOp>(op) || !op->hasAttr("constants")) {
      return false;
    }

    auto constants = op->getAttrOfType<DenseI64ArrayAttr>("constants");
    Value address = isa<LoadOp>(op) ? cast<LoadOp>(op).getAddr()
                                    : cast<StoreOp>(op).getAddr();

    if (auto store = dyn_cast<StoreOp>(op)) {
      operands.push_back(store.getValue());
      instruction.src_operands.emplace_back("UNRESOLVED", "RED");
    }

    if (!address) {
      Operand configured_address("#" + std::to_string(constants[0]));
      configured_address.access = "absolute_address";
      configured_address.offsets.assign(constants.asArrayRef().begin(),
                                        constants.asArrayRef().end());
      operands.push_back(Value());
      instruction.src_operands.push_back(std::move(configured_address));
      return true;
    }

    auto argument = dyn_cast<BlockArgument>(address);
    if (!current_kernel || !argument ||
        argument.getOwner() != &current_kernel.getBody().front() ||
        argument.getArgNumber() >= current_kernel.getInputs().size()) {
      op->emitError("memref address must be a kernel input block argument");
      timing_field_error = true;
      return true;
    }

    unsigned input_index = argument.getArgNumber();
    auto memref = getConfiguredMemref(current_kernel, input_index);
    if (failed(memref)) {
      op->emitError("memory base requires a static identity-layout i32 "
                    "memref");
      timing_field_error = true;
      return true;
    }

    for (int64_t offset : constants.asArrayRef()) {
      if (offset < 0 || offset >= memref->getNumElements()) {
        op->emitError("constant element offset is out of bounds");
        timing_field_error = true;
        return true;
      }
    }

    operands.push_back(Value());
    instruction.src_operands.push_back(
        getConfiguredOperand(input_index, constants.asArrayRef(), true));
    return true;
  }

  // ---------- entry point ----------.
  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto generate_for_scope = [&](Operation *scope, func::FuncOp function,
                                  KernelOp kernel) -> LogicalResult {
      clearState();
      current_compiled_ii = getCompiledII(scope);
      if (kernel && failed(configureKernel(kernel))) {
        return failure();
      }

      auto [columns, rows] = getArrayDimensions(scope);
      Topology topology = getTopologyFromArchitecture(columns, rows);

      SmallVector<Operation *> data_movs;
      SmallVector<Operation *> ctrl_movs;
      DenseMap<Value, Operation *> reserve_to_phi_map;
      indexIR(scope, data_movs, ctrl_movs, reserve_to_phi_map);
      if (timing_field_error) {
        return failure();
      }

      for (Operation *op : data_movs) {
        expandMovImpl<false>(op, topology, reserve_to_phi_map);
      }
      for (Operation *op : ctrl_movs) {
        expandMovImpl<true>(op, topology, reserve_to_phi_map);
      }
      logUnresolvedOperands();

      std::unordered_set<int> materialized_ids;
      assignInstructionIds(materialized_ids);
      ArrayConfig config = buildArrayConfig(columns, rows, current_compiled_ii);
      writeYAMLOutput(config);
      writeAsmOutput(config);
      writeDfgOutputSSA(function, topology, materialized_ids);
      return success(!timing_field_error);
    };

    SmallVector<KernelOp> mapped_kernels;
    module.walk([&](KernelOp kernel) {
      if (kernel->hasAttr(attr::kMappingInfo)) {
        mapped_kernels.push_back(kernel);
      }
    });

    if (!mapped_kernels.empty()) {
      for (KernelOp kernel : mapped_kernels) {
        auto function = kernel->getParentOfType<func::FuncOp>();
        if (!function || failed(generate_for_scope(kernel, function, kernel))) {
          signalPassFailure();
          return;
        }
      }
      return;
    }

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto accelerator =
          function->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accelerator || accelerator.getValue() != accel::kNeuraTarget) {
        continue;
      }

      if (failed(generate_for_scope(function, function, KernelOp{}))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace.

namespace mlir::neura {
std::unique_ptr<Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura.
