#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::neura;

LogicalResult YieldOp::verify() {
  Operation *parent_op = (*this)->getParentOp();

  if (!parent_op) {
    return emitOpError("must have a parent operation.");
  }

  // Allows yield in FusedOp and KernelOp
  if (isa<FusedOp>(parent_op) || isa<KernelOp>(parent_op)) {
    return success();
  }

  // Allows yield in func.func (for dataflow mode)
  if (isa<func::FuncOp>(parent_op)) {
    return success();
  }

  return emitOpError("expects parent op to be one of 'neura.fused_op', "
                     "'neura.kernel', or 'func.func'");
}

LogicalResult PhiStartOp::verify() {
  // Checks if this phi_start is inside a fused_op.
  Operation *parent_op = getOperation()->getParentOp();
  bool inside_fused_op = false;
  while (parent_op) {
    if (isa<FusedOp>(parent_op)) {
      inside_fused_op = true;
      break;
    }
    parent_op = parent_op->getParentOp();
  }

  if (!inside_fused_op) {
    // Verifies that the reserved operand is produced by a neura.reserve
    // operation.
    Value reserved = getReserved();
    Operation *def_op = reserved.getDefiningOp();

    if (!def_op) {
      return emitOpError("reserve operand must be defined by an operation.");
    }

    if (!isa<ReserveOp>(def_op)) {
      return emitOpError("reserve operand must be produced by a neura.reserve "
                         "operation.");
    }
  }

  // Verifies that there is at least one initialization value.
  if (!getInitValue()) {
    return emitOpError("At least one initialization value is required.");
  }

  // Verifies type consistency.
  Type result_type = getResult().getType();
  Type reserved_type = getReserved().getType();

  if (result_type != reserved_type) {
    return emitOpError("Result type must match the reserved value type.");
  }

  if (getInitValue().getType() != result_type) {
    return emitOpError("All initialization values must match the result type.");
  }

  return success();
}

LogicalResult MacOp::verify() {
  StringRef stationary = getStationary();

  if (stationary != "weight" && stationary != "input" &&
      stationary != "output") {
    return emitOpError(
        "stationary must be \"weight\", \"input\", or \"output\"");
  }

  if (stationary == "output" && !getInput1()) {
    return emitOpError("output-stationary MAC requires two dynamic operands");
  }

  return success();
}