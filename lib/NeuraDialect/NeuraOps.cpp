#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::neura;

// Validates the three supported load address forms.
LogicalResult LoadOp::verify() {
  Attribute raw_constants = (*this)->getAttr("constants");
  auto constants = dyn_cast_or_null<DenseI64ArrayAttr>(raw_constants);
  if (raw_constants && (!constants || constants.size() == 0)) {
    return emitOpError("constants must be a nonempty i64 array");
  }

  Value address = getAddr();
  if (!address && !constants) {
    return emitOpError("requires an address operand or constants");
  }

  if (address) {
    Type address_type = address.getType();
    if (auto data = dyn_cast<PredicatedValue>(address_type)) {
      address_type = data.getValueType();
    }

    if (constants) {
      if (!isa<MemRefType>(address_type)) {
        return emitOpError("operand plus constants requires a memref base");
      }
      return success();
    }

    if (!isa<IntegerType, IndexType, LLVM::LLVMPointerType>(address_type)) {
      return emitOpError(
          "dynamic address must be an integer, index, or pointer");
    }
    return success();
  }

  for (int64_t constant : constants.asArrayRef()) {
    if (constant < 0) {
      return emitOpError("absolute constant addresses must be nonnegative");
    }
  }
  return success();
}

// Validates StoreOp after optional constant folding.
LogicalResult StoreOp::verify() {
  Attribute stored_constant = (*this)->getAttr("lhs_value");
  Attribute address_constant = (*this)->getAttr("rhs_value");
  Attribute raw_constants = (*this)->getAttr("constants");
  auto constants = dyn_cast_or_null<DenseI64ArrayAttr>(raw_constants);

  if (raw_constants && (!constants || constants.size() == 0)) {
    return emitOpError("constants must be a nonempty i64 array");
  }

  Value address;
  if (stored_constant) {
    if (getNumOperands() > 1) {
      return emitOpError(
          "expects at most one address operand when the stored value "
          "is folded");
    }
    if (getNumOperands() == 1) {
      address = getOperand(0);
    }
  } else {
    if (getNumOperands() == 0 || getNumOperands() > 2) {
      return emitOpError(
          "expects a stored value and an optional address operand");
    }
    if (getNumOperands() == 2) {
      address = getOperand(1);
    }
  }

  if (address_constant && (address || constants)) {
    return emitOpError("has more than one address source");
  }
  if (!address_constant && !address && !constants) {
    return emitOpError("requires an address operand or constants");
  }

  if (address_constant) {
    if (!isa<IntegerAttr, StringAttr, SymbolRefAttr>(address_constant)) {
      return emitOpError(
          "folded address must be an integer, string, or symbol");
    }
    return success();
  }

  if (address) {
    Type address_type = address.getType();
    if (auto data = dyn_cast<PredicatedValue>(address_type)) {
      address_type = data.getValueType();
    }

    if (constants) {
      if (!isa<MemRefType>(address_type)) {
        return emitOpError("operand plus constants requires a memref base");
      }
      return success();
    }

    if (!isa<IntegerType, IndexType, LLVM::LLVMPointerType>(address_type)) {
      return emitOpError(
          "dynamic address must be an integer, index, or pointer");
    }
    return success();
  }

  for (int64_t constant : constants.asArrayRef()) {
    if (constant < 0) {
      return emitOpError("absolute constant addresses must be nonnegative");
    }
  }
  return success();
}

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
