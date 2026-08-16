#include "NeuraDialect/Mapping/TemplateMapping/TemplateMapping.h"

#include "NeuraDialect/Mapping/mapping_util.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir {
namespace neura {

namespace {

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

  for (Tile *tile : architecture.getAllTiles()) {
    if (tile->getX() == x && tile->getY() == y) {
      return tile;
    }
  }

  op->emitError() << "Placement refers to unavailable tile (" << x << ", " << y
                  << ")";
  return nullptr;
}

} // namespace

bool TemplateMapping::map(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    std::set<Operation *> &critical_ops, const Architecture &architecture,
    MappingState &mapping_state) {
  // Template mapping does not currently use critical-path information.
  (void)critical_ops;

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
    int first_time_step = std::max(0, level);
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