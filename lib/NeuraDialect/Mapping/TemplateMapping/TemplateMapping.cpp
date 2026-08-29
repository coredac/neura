#include "NeuraDialect/Mapping/TemplateMapping/TemplateMapping.h"

#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <map>

namespace mlir {
namespace neura {

namespace {

// Maps kernel inputs and results to physical boundary ports.
struct TemplatePortBindings {
  llvm::DenseMap<Value, Port *> input_ports;
  std::map<unsigned, Port *> output_ports;
};

// Gets the Tile specified by an operation's placement attribute.
Tile *getPlacedTile(Operation *op, const Architecture &architecture) {
  auto placement = op->getAttrOfType<DictionaryAttr>(attr::kPlacement);

  if (!placement) {
    op->emitError("Template mapping requires a placement attribute");
    return nullptr;
  }

  IntegerAttr x_attr = placement.getAs<IntegerAttr>(attr::kX);
  IntegerAttr y_attr = placement.getAs<IntegerAttr>(attr::kY);

  if (!x_attr || !y_attr) {
    op->emitError("Placement must contain integer x and y coordinates");
    return nullptr;
  }

  int x = static_cast<int>(x_attr.getInt());
  int y = static_cast<int>(y_attr.getInt());

  Tile *tile = architecture.getTile(x, y);
  if (!tile) {
    op->emitError() << "Placement refers to unavailable tile (" << x << ", "
                    << y << ")";
    return nullptr;
  }

  return tile;
}

// Finds the neura.kernel that contains the operations being mapped.
KernelOp
findParentKernel(const std::vector<std::pair<Operation *, int>> &operations) {
  for (const auto &[op, level] : operations) {
    (void)level;

    if (KernelOp kernel = op->getParentOfType<KernelOp>()) {
      return kernel;
    }
  }

  return KernelOp();
}

// Resolves one input_ports or output_ports entry to a physical Port.
Port *resolvePortBinding(KernelOp kernel, DictionaryAttr binding,
                         PortKind port_kind, const Architecture &architecture) {
  auto direction_attr = binding.getAs<StringAttr>("direction");
  auto x_attr = binding.getAs<IntegerAttr>(attr::kX);
  auto y_attr = binding.getAs<IntegerAttr>(attr::kY);

  if (!direction_attr || !x_attr || !y_attr) {
    kernel.emitOpError("port binding requires direction, x, and y");
    return nullptr;
  }

  std::optional<PortDirection> direction =
      parsePortDirection(direction_attr.getValue());

  if (!direction) {
    kernel.emitOpError() << "unsupported port direction \""
                         << direction_attr.getValue() << "\"";
    return nullptr;
  }

  Port *port = architecture.getPort(port_kind, *direction,
                                    static_cast<int>(x_attr.getInt()),
                                    static_cast<int>(y_attr.getInt()));

  if (!port) {
    kernel.emitOpError() << "binding does not identify an available "
                         << stringifyPortKind(port_kind) << " port";
    return nullptr;
  }

  return port;
}

// Reads input_ports and output_ports from template mapping.
bool collectTemplatePortBindings(KernelOp kernel,
                                 const Architecture &architecture,
                                 TemplatePortBindings &bindings) {
  auto metadata = kernel->getAttrOfType<DictionaryAttr>("kernel_metadata");
  if (!metadata) {
    return true;
  }

  auto kind = metadata.getAs<StringAttr>("kind");
  if (!kind || kind.getValue() != "template") {
    return true;
  }

  auto template_metadata = metadata.getAs<DictionaryAttr>("template");

  if (!template_metadata) {
    kernel.emitOpError("template kernel requires template metadata");
    return false;
  }

  // Handles input ports.
  if (auto input_ports = template_metadata.getAs<ArrayAttr>("input_ports")) {
    for (Attribute attribute : input_ports) {
      auto binding = dyn_cast<DictionaryAttr>(attribute);

      if (!binding) {
        kernel.emitOpError("input port binding must be a dictionary");
        return false;
      }

      auto index_attr = binding.getAs<IntegerAttr>("kernel_input");

      if (!index_attr) {
        kernel.emitOpError("input port binding requires kernel_input");
        return false;
      }

      int64_t input_index = index_attr.getInt();

      if (input_index < 0 ||
          input_index >= static_cast<int64_t>(kernel.getInputs().size())) {
        kernel.emitOpError("kernel_input index is out of bounds");
        return false;
      }

      Port *port =
          resolvePortBinding(kernel, binding, PortKind::Input, architecture);

      if (!port) {
        return false;
      }

      Value block_argument = kernel.getBody().front().getArgument(input_index);

      if (!bindings.input_ports.try_emplace(block_argument, port).second) {
        kernel.emitOpError("kernel input has multiple port bindings");
        return false;
      }
    }
  }

  // Handles output ports.
  if (auto output_ports = template_metadata.getAs<ArrayAttr>("output_ports")) {
    for (Attribute attribute : output_ports) {
      auto binding = dyn_cast<DictionaryAttr>(attribute);

      if (!binding) {
        kernel.emitOpError("output port binding must be a dictionary");
        return false;
      }

      auto index_attr = binding.getAs<IntegerAttr>("kernel_result");

      if (!index_attr) {
        kernel.emitOpError("output port binding requires kernel_result");
        return false;
      }

      int64_t result_index = index_attr.getInt();

      if (result_index < 0 ||
          result_index >= static_cast<int64_t>(kernel->getNumResults())) {
        kernel.emitOpError("kernel_result index is out of bounds");
        return false;
      }

      Port *port =
          resolvePortBinding(kernel, binding, PortKind::Output, architecture);

      if (!port) {
        return false;
      }

      if (!bindings.output_ports
               .emplace(static_cast<unsigned>(result_index), port)
               .second) {
        kernel.emitOpError("kernel result has multiple port bindings");
        return false;
      }
    }
  }

  return true;
}

// Checks that each kernel block argument used by the compute graph has a
// matching input_ports entry.
bool validateKernelInputBindings(
    const std::vector<std::pair<Operation *, int>> &operations,
    const TemplatePortBindings &bindings) {
  for (const auto &[op, level] : operations) {
    (void)level;

    if (is_non_materialized(op)) {
      continue;
    }

    for (Value operand : op->getOperands()) {
      Operation *operand_producer = operand.getDefiningOp();

      if (isa_and_nonnull<ReserveOp>(operand_producer)) {
        continue;
      }

      auto data_move = dyn_cast_or_null<DataMovOp>(operand_producer);

      if (!data_move) {
        op->emitError(
            "template-mapped operands must be wrapped by neura.data_mov");
        return false;
      }

      Value source = data_move.getOperand();

      // An operation result, including a promoted neura.constant, is an
      // internal source and does not require an input port.
      if (source.getDefiningOp()) {
        continue;
      }

      if (!bindings.input_ports.contains(source)) {
        data_move.emitOpError("kernel input has no port binding");
        return false;
      }
    }
  }

  return true;
}

void releaseRoutes(const std::vector<Operation *> &routes,
                   MappingState &mapping_state) {
  for (Operation *route : routes) {
    mapping_state.releaseRoute(route);
  }
}

// Routes kernel inputs from their assigned ports to the consumer Tile.
bool routeKernelInputPorts(Operation *op, const MappingLoc &target_loc,
                           const TemplatePortBindings &bindings,
                           MappingState &mapping_state,
                           std::vector<Operation *> &routed_inputs) {
  assert(routed_inputs.empty() &&
         "Routed input list should initially be empty");
  for (Value operand : op->getOperands()) {
    Operation *operand_producer = operand.getDefiningOp();

    if (isa_and_nonnull<ReserveOp>(operand_producer)) {
      continue;
    }

    auto data_move = dyn_cast_or_null<DataMovOp>(operand_producer);
    assert(data_move &&
           "Kernel input bindings must be validated before routing");

    Value source = data_move.getOperand();

    // Internal operation results are routed by placeAndRoute.
    if (source.getDefiningOp()) {
      continue;
    }

    auto binding = bindings.input_ports.find(source);

    assert(binding != bindings.input_ports.end() &&
           "Kernel input must have a Port binding");

    Port *input_port = binding->second;

    Tile *input_tile = input_port->getTile();
    Tile *consumer_tile = dyn_cast<Tile>(target_loc.resource);

    assert(consumer_tile && "Template operation must be placed on a Tile");

    int distance = std::abs(input_tile->getX() - consumer_tile->getX()) +
                   std::abs(input_tile->getY() - consumer_tile->getY());

    // Injects the value as late as possible while still allowing it to reach
    // the consumer Tile at the scheduled time.
    int input_time_step = target_loc.time_step - distance;

    if (input_time_step < 0) {
      releaseRoutes(routed_inputs, mapping_state);
      routed_inputs.clear();
      return false;
    }

    MappingLoc input_port_loc = {
        input_port,
        input_time_step,
    };

    if (!mapping_state.isAvailableAcrossTime(input_port_loc, data_move)) {
      releaseRoutes(routed_inputs, mapping_state);
      routed_inputs.clear();
      return false;
    }

    MappingLoc input_tile_loc = {
        input_tile,
        input_time_step,
    };

    std::vector<MappingLoc> tile_path;

    if (!tryRouteForwardMove(data_move, input_tile_loc, target_loc,
                             mapping_state, tile_path)) {
      releaseRoutes(routed_inputs, mapping_state);
      routed_inputs.clear();
      return false;
    }

    // The DataMovOp first occupies the input Port and then any internal links
    // or registers selected by Tile-to-Tile routing.
    std::vector<MappingLoc> complete_path = {
        input_port_loc,
    };

    complete_path.insert(complete_path.end(), tile_path.begin(),
                         tile_path.end());

    mapping_state.reserveRoute(data_move.getOperation(), complete_path);
    routed_inputs.push_back(data_move.getOperation());
  }

  return true;
}

// Routes kernel results from their producer Tiles to their assigned ports.
bool routeKernelOutputPorts(KernelOp kernel,
                            const TemplatePortBindings &bindings,
                            const Architecture &architecture,
                            MappingState &mapping_state) {
  if (bindings.output_ports.empty()) {
    return true;
  }

  auto yield = dyn_cast<YieldOp>(kernel.getBody().front().getTerminator());

  if (!yield) {
    kernel.emitOpError("template kernel requires neura.yield");
    return false;
  }

  for (const auto &[result_index, output_port] : bindings.output_ports) {
    if (result_index >= yield.getResults().size()) {
      kernel.emitOpError("kernel_result index exceeds neura.yield results");
      return false;
    }

    Value yielded_value = yield.getResults()[result_index];

    auto data_move = yielded_value.getDefiningOp<DataMovOp>();

    if (!data_move) {
      kernel.emitOpError("kernel result must be wrapped by neura.data_mov");
      return false;
    }

    Operation *producer = data_move.getOperand().getDefiningOp();

    if (!producer) {
      data_move.emitOpError("kernel result has no producer");
      return false;
    }

    const std::vector<MappingLoc> &producer_locs =
        mapping_state.getAllLocsOfOp(producer);

    if (producer_locs.empty()) {
      data_move.emitOpError("kernel result producer has not been mapped");
      return false;
    }

    MappingLoc producer_loc = producer_locs.back();

    Tile *producer_tile = dyn_cast<Tile>(producer_loc.resource);

    if (!producer_tile) {
      data_move.emitOpError("kernel result producer must be mapped to a Tile");
      return false;
    }

    Tile *output_tile = output_port->getTile();

    int distance = std::abs(producer_tile->getX() - output_tile->getX()) +
                   std::abs(producer_tile->getY() - output_tile->getY());

    int first_time_step = producer_loc.time_step + distance;

    int routing_slack =
        architecture.getPerCgraRows() + architecture.getPerCgraColumns();

    int last_time_step =
        first_time_step + routing_slack + mapping_state.getII();

    bool routed = false;

    for (int time_step = first_time_step; time_step <= last_time_step;
         ++time_step) {
      MappingLoc output_tile_loc = {
          output_tile,
          time_step,
      };

      std::vector<MappingLoc> tile_path;

      if (!tryRouteForwardMove(data_move, producer_loc, output_tile_loc,
                               mapping_state, tile_path)) {
        continue;
      }

      MappingLoc output_port_loc = {
          output_port,
          time_step,
      };

      if (!mapping_state.isAvailableAcrossTime(output_port_loc, data_move)) {
        continue;
      }

      // The output Port is the final location of the DataMovOp connected to
      // neura.yield.
      tile_path.push_back(output_port_loc);
      mapping_state.reserveRoute(data_move.getOperation(), tile_path);

      routed = true;
      break;
    }

    if (!routed) {
      data_move.emitOpError("failed to route kernel result to its output port");
      return false;
    }
  }

  return true;
}
} // namespace

bool TemplateMapping::map(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    std::set<Operation *> &critical_ops, const Architecture &architecture,
    MappingState &mapping_state) {
  // Template mapping does not currently use critical-path information.
  (void)critical_ops;

  KernelOp kernel = findParentKernel(sorted_ops_with_levels);

  TemplatePortBindings port_bindings;

  if (kernel &&
      !collectTemplatePortBindings(kernel, architecture, port_bindings)) {
    return false;
  }

  if (!validateKernelInputBindings(sorted_ops_with_levels, port_bindings)) {
    return false;
  }

  for (auto [op, level] : sorted_ops_with_levels) {
    if (is_non_materialized(op)) {
      continue;
    }

    Tile *tile = getPlacedTile(op, architecture);
    if (!tile) {
      return false;
    }

    OperationKind operation_kind = getOperationKindFromMlirOp(op);
    if (!tile->canSupportOperation(operation_kind)) {
      op->emitError() << "Operation is not supported by its placed tile";
      return false;
    }

    // Placement is fixed, but scheduling remains automatic. Start from the
    // operation's dependency level and allow enough slack for multi-hop routes.
    int first_time_step = std::max(0, level - 1);
    int routing_slack =
        architecture.getPerCgraRows() + architecture.getPerCgraColumns();
    int last_time_step =
        first_time_step + routing_slack + mapping_state.getII();

    bool mapped = false;

    for (int time_step = first_time_step; time_step <= last_time_step;
         ++time_step) {
      MappingLoc target_loc = {
          tile,
          time_step,
      };

      std::vector<Operation *> routed_inputs;

      if (!routeKernelInputPorts(op, target_loc, port_bindings, mapping_state,
                                 routed_inputs)) {
        continue;
      }

      if (placeAndRoute(op, target_loc, mapping_state)) {
        mapped = true;
        break;
      }

      // Input Port routes are created by TemplateMapping and must be released
      // before trying the next schedule time.
      releaseRoutes(routed_inputs, mapping_state);
    }

    if (!mapped) {
      llvm::errs() << "[TemplateMapping] Failed to schedule operation on Tile("
                   << tile->getX() << ", " << tile->getY() << "): " << *op
                   << "\n";
      return false;
    }
  }

  if (kernel && !routeKernelOutputPorts(kernel, port_bindings, architecture,
                                        mapping_state)) {
    return false;
  }

  // Removes the placement attribute from the input IR.
  for (const auto &[op, level] : sorted_ops_with_levels) {
    (void)level;
    op->removeAttr(attr::kPlacement);
  }

  return true;
}

} // namespace neura
} // namespace mlir
