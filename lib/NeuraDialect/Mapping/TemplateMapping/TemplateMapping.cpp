#include "NeuraDialect/Mapping/TemplateMapping/TemplateMapping.h"

#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>

namespace mlir {
namespace neura {

namespace {

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

// All dynamic data must originate in a mapped operation, including LD Tiles.
// Memrefs referenced by template configuration are not dynamic data inputs.
bool validateTemplateOperands(
    const std::vector<std::pair<Operation *, int>> &operations) {
  for (const auto &[op, level] : operations) {
    (void)level;
    if (is_non_materialized(op))
      continue;
    for (Value operand : op->getOperands()) {
      if (isConfiguredMemrefAddress(op, operand)) {
        continue;
      }
      Operation *producer = operand.getDefiningOp();
      if (isa_and_nonnull<ReserveOp>(producer))
        continue;
      auto move = dyn_cast_or_null<DataMovOp>(producer);
      if (!move) {
        op->emitError(
            "template-mapped operands must be wrapped by neura.data_mov");
        return false;
      }
      if (!move.getInput().getDefiningOp()) {
        move.emitOpError(
            "template input must be produced by a mapped operation; "
            "use a configured load for memory input");
        return false;
      }
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

  if (!validateTemplateOperands(sorted_ops_with_levels)) {
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
    // LD sources are materialized Tiles too. Keep their ALAP level so later
    // rows do not inject earlier than their incoming partial sums.
    int first_time_step = std::max(0, level);
    // Stores have no outgoing data edge. Scheduling every store at the common
    // ALAP sink level would add needless waiting to earlier output columns.
    if (isa<StoreOp>(op)) {
      first_time_step = 0;
      for (Value operand : op->getOperands()) {
        auto move = operand.getDefiningOp<DataMovOp>();
        if (!move)
          continue;
        Operation *producer = move.getInput().getDefiningOp();
        const auto &locations = mapping_state.getAllLocsOfOp(producer);
        if (locations.empty())
          continue;
        const auto &source = locations.back();
        auto *source_tile = dyn_cast<Tile>(source.resource);
        if (source_tile)
          first_time_step = std::max(
              first_time_step,
              source.time_step + std::abs(source_tile->getX() - tile->getX()) +
                  std::abs(source_tile->getY() - tile->getY()));
      }
    }
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

      if (placeAndRoute(op, target_loc, mapping_state)) {
        mapped = true;
        break;
      }
    }

    if (!mapped) {
      llvm::errs() << "[TemplateMapping] Failed to schedule operation on Tile("
                   << tile->getX() << ", " << tile->getY() << "): " << *op
                   << "\n";
      return false;
    }
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
