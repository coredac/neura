#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include <cstddef>

using namespace mlir;
using namespace mlir::taskflow;

//===----------------------------------------------------------------------===//
// TaskflowTaskOp
//===----------------------------------------------------------------------===//

ParseResult TaskflowTaskOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parses optional task name: @Task_0.
  StringAttr task_name;
  if (succeeded(parser.parseOptionalSymbolName(task_name))) {
    result.addAttribute("task_name", task_name);
  }

  // Parses dependency_read_in: dependency_read_in(%arg0, %arg1 : memref<?xi32>,
  // memref<?xi32>).
  SmallVector<OpAsmParser::UnresolvedOperand> read_operands;
  SmallVector<Type> read_types;
  if (succeeded(parser.parseOptionalKeyword("dependency_read_in"))) {
    if (parser.parseLParen() || parser.parseOperandList(read_operands) ||
        parser.parseColonTypeList(read_types) || parser.parseRParen())
      return failure();
  }

  // Parses dependency_write_in: dependency_write_in(%arg5 : memref<?xi32>).
  SmallVector<OpAsmParser::UnresolvedOperand> write_operands;
  SmallVector<Type> write_types;
  if (succeeded(parser.parseOptionalKeyword("dependency_write_in"))) {
    if (parser.parseLParen() || parser.parseOperandList(write_operands) ||
        parser.parseColonTypeList(write_types) || parser.parseRParen())
      return failure();
  }

  // Parses value_inputs: value_inputs(%scalar : i32).
  SmallVector<OpAsmParser::UnresolvedOperand> value_operands;
  SmallVector<Type> value_types;
  if (succeeded(parser.parseOptionalKeyword("value_inputs"))) {
    if (parser.parseLParen() || parser.parseOperandList(value_operands) ||
        parser.parseColonTypeList(value_types) || parser.parseRParen())
      return failure();
  }

  // Parses original memrefs with explicit types:
  // [original_read_memrefs(%arg0, %arg1 : type1, type2),
  // original_write_memrefs(%arg5 : type3)].
  SmallVector<OpAsmParser::UnresolvedOperand> original_read_operands;
  SmallVector<Type> original_read_types;
  SmallVector<OpAsmParser::UnresolvedOperand> original_write_operands;
  SmallVector<Type> original_write_types;

  if (succeeded(parser.parseOptionalLSquare())) {
    // original_read_memrefs with types.
    if (succeeded(parser.parseOptionalKeyword("original_read_memrefs"))) {
      if (parser.parseLParen() ||
          parser.parseOperandList(original_read_operands) ||
          parser.parseColonTypeList(original_read_types) ||
          parser.parseRParen())
        return failure();
    }

    // optional comma.
    (void)parser.parseOptionalComma();

    // original_write_memrefs with types.
    if (succeeded(parser.parseOptionalKeyword("original_write_memrefs"))) {
      if (parser.parseLParen() ||
          parser.parseOperandList(original_write_operands) ||
          parser.parseColonTypeList(original_write_types) ||
          parser.parseRParen())
        return failure();
    }

    if (parser.parseRSquare())
      return failure();
  }

  // Validates operand/type count match.
  if (read_operands.size() != read_types.size() ||
      write_operands.size() != write_types.size() ||
      value_operands.size() != value_types.size() ||
      original_read_operands.size() != original_read_types.size() ||
      original_write_operands.size() != original_write_types.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "operand and type count mismatch");
  }

  // Resolves all operands.
  if (parser.resolveOperands(read_operands, read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(write_operands, write_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(value_operands, value_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(original_read_operands, original_read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(original_write_operands, original_write_types,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  // Parses optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parses function type: : (...) -> (...).
  FunctionType func_type;
  if (parser.parseColon() || parser.parseType(func_type))
    return failure();

  // Adds result types.
  result.addTypes(func_type.getResults());

  // Parses region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*args=*/{}, /*argTypes=*/{})) {
    return failure();
  }

  // Adds operand segment sizes.
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(read_operands.size()),
           static_cast<int32_t>(write_operands.size()),
           static_cast<int32_t>(value_operands.size()),
           static_cast<int32_t>(original_read_operands.size()),
           static_cast<int32_t>(original_write_operands.size())}));

  // Adds result segment sizes.
  // dependency_write_out count always equals dependency_write_in count.
  // dependency_read_out count is the remaining MemRef results after
  // subtracting write_out. It may be less than dependency_read_in count
  // when not all read memrefs have a WAR dependency.
  size_t num_write_outputs = write_operands.size();
  size_t num_value_outputs = 0;
  size_t total_memref_results = 0;
  for (Type t : func_type.getResults()) {
    if (isa<MemRefType>(t)) {
      total_memref_results++;
    } else {
      num_value_outputs++;
    }
  }
  size_t num_read_outputs = total_memref_results - num_write_outputs;
  result.addAttribute("resultSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(num_read_outputs),
                           static_cast<int32_t>(num_write_outputs),
                           static_cast<int32_t>(num_value_outputs)}));

  return success();
}

void TaskflowTaskOp::print(OpAsmPrinter &printer) {
  // Prints task name.
  printer << " @" << getTaskName();

  // Prints dependency_read_in.
  if (!getDependencyReadIn().empty()) {
    printer << " dependency_read_in(";
    llvm::interleaveComma(getDependencyReadIn(), printer);
    printer << " : ";
    llvm::interleaveComma(getDependencyReadIn().getTypes(), printer);
    printer << ")";
  }

  // Prints dependency_write_in.
  if (!getDependencyWriteIn().empty()) {
    printer << " dependency_write_in(";
    llvm::interleaveComma(getDependencyWriteIn(), printer);
    printer << " : ";
    llvm::interleaveComma(getDependencyWriteIn().getTypes(), printer);
    printer << ")";
  }

  // Prints value_inputs.
  if (!getValueInputs().empty()) {
    printer << " value_inputs(";
    llvm::interleaveComma(getValueInputs(), printer);
    printer << " : ";
    llvm::interleaveComma(getValueInputs().getTypes(), printer);
    printer << ")";
  }

  // Prints original memrefs with types.
  if (!getOriginalReadMemrefs().empty() || !getOriginalWriteMemrefs().empty()) {
    printer << " [";

    if (!getOriginalReadMemrefs().empty()) {
      printer << "original_read_memrefs(";
      llvm::interleaveComma(getOriginalReadMemrefs(), printer);
      printer << " : ";
      llvm::interleaveComma(getOriginalReadMemrefs().getTypes(), printer);
      printer << ")";
    }

    if (!getOriginalReadMemrefs().empty() && !getOriginalWriteMemrefs().empty())
      printer << ", ";

    if (!getOriginalWriteMemrefs().empty()) {
      printer << "original_write_memrefs(";
      llvm::interleaveComma(getOriginalWriteMemrefs(), printer);
      printer << " : ";
      llvm::interleaveComma(getOriginalWriteMemrefs().getTypes(), printer);
      printer << ")";
    }

    printer << "]";
  }

  // Prints attributes (skip operandSegmentSizes, resultSegmentSizes,
  // task_name).
  SmallVector<StringRef> elidedAttrs = {"operandSegmentSizes",
                                        "resultSegmentSizes", "task_name"};
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Prints function type.
  printer << " : (";
  llvm::interleaveComma(
      llvm::concat<const Type>(getDependencyReadIn().getTypes(),
                               getDependencyWriteIn().getTypes(),
                               getValueInputs().getTypes()),
      printer);
  printer << ") -> (";
  llvm::interleaveComma(
      llvm::concat<const Type>(getDependencyReadOut().getTypes(),
                               getDependencyWriteOut().getTypes(),
                               getValueOutputs().getTypes()),
      printer);
  printer << ")";

  // Prints region.
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/true);
}

//===----------------------------------------------------------------------===//
// TaskflowYieldOp
//===----------------------------------------------------------------------===//

ParseResult TaskflowYieldOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> read_operands;
  SmallVector<Type> read_types;
  SmallVector<OpAsmParser::UnresolvedOperand> write_operands;
  SmallVector<Type> write_types;
  SmallVector<OpAsmParser::UnresolvedOperand> value_operands;
  SmallVector<Type> value_types;

  // Parses reads (WAR dependency passthrough).
  if (succeeded(parser.parseOptionalKeyword("reads"))) {
    if (parser.parseLParen() || parser.parseOperandList(read_operands) ||
        parser.parseColonTypeList(read_types) || parser.parseRParen())
      return failure();
  }

  // Parses writes.
  if (succeeded(parser.parseOptionalKeyword("writes"))) {
    if (parser.parseLParen() || parser.parseOperandList(write_operands) ||
        parser.parseColonTypeList(write_types) || parser.parseRParen())
      return failure();
  }

  // Parses values.
  if (succeeded(parser.parseOptionalKeyword("values"))) {
    if (parser.parseLParen() || parser.parseOperandList(value_operands) ||
        parser.parseColonTypeList(value_types) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(read_operands, read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(write_operands, write_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(value_operands, value_types,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(read_operands.size()),
                           static_cast<int32_t>(write_operands.size()),
                           static_cast<int32_t>(value_operands.size())}));

  return success();
}

void TaskflowYieldOp::print(OpAsmPrinter &printer) {
  if (!getReadResults().empty()) {
    printer << " reads(";
    llvm::interleaveComma(getReadResults(), printer);
    printer << " : ";
    llvm::interleaveComma(getReadResults().getTypes(), printer);
    printer << ")";
  }

  if (!getMemoryResults().empty()) {
    printer << " writes(";
    llvm::interleaveComma(getMemoryResults(), printer);
    printer << " : ";
    llvm::interleaveComma(getMemoryResults().getTypes(), printer);
    printer << ")";
  }

  if (!getValueResults().empty()) {
    printer << " values(";
    llvm::interleaveComma(getValueResults(), printer);
    printer << " : ";
    llvm::interleaveComma(getValueResults().getTypes(), printer);
    printer << ")";
  }
}