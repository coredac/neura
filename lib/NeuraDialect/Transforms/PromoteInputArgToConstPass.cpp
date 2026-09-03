#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <string>

using namespace mlir;

#define GEN_PASS_DEF_PROMOTEINPUTARGTOCONST
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// Attribute name to mark iter_arg init constants.
constexpr const char *kIterArgInitAttr = "is_iter_arg_init";

LogicalResult promoteFunctionArgsToConstants(Region &region) {
  if (region.empty()) {
    return success();
  }

  Block &entry_block = region.front();
  OpBuilder builder(&entry_block, entry_block.begin());

  // Collects all function arguments.
  SmallVector<BlockArgument, 4> args(entry_block.getArguments().begin(),
                                     entry_block.getArguments().end());

  // Creates a constant operation for each function argument.
  for (auto [idx, arg] : llvm::enumerate(args)) {
    // For constant operation, no predicate.
    auto const_op = builder.create<neura::ConstantOp>(
        arg.getLoc(), arg.getType(),
        builder.getStringAttr("\%arg" + std::to_string(idx)));
    arg.replaceAllUsesWith(const_op.getResult());
  }

  return success();
}

// Collects kernel input indices that are explicitly bound to physical input
// ports by template metadata.
//
// These arguments must remain as block arguments because TemplateMapping uses
// them as external streaming values injected through TileArray boundary ports.
LogicalResult
collectPortBoundInputIndices(neura::KernelOp kernel_op,
                             llvm::DenseSet<unsigned> &port_bound_inputs) {
  auto metadata = kernel_op->getAttrOfType<DictionaryAttr>("kernel_metadata");

  if (!metadata) {
    return success();
  }

  auto kind = metadata.getAs<StringAttr>("kind");

  // Boundary-port-aware argument preservation currently belongs to template
  // kernels.
  if (!kind || kind.getValue() != "template") {
    return success();
  }

  auto template_metadata = metadata.getAs<DictionaryAttr>("template");

  if (!template_metadata) {
    return kernel_op.emitOpError("template kernel requires template metadata");
  }

  auto input_ports = template_metadata.getAs<ArrayAttr>("input_ports");

  if (!input_ports) {
    return success();
  }

  size_t num_inputs = kernel_op.getInputs().size();

  for (Attribute attribute : input_ports) {
    auto binding = dyn_cast<DictionaryAttr>(attribute);

    if (!binding) {
      return kernel_op.emitOpError("input port binding must be a dictionary");
    }

    auto index_attr = binding.getAs<IntegerAttr>("kernel_input");

    if (!index_attr) {
      return kernel_op.emitOpError("input port binding requires kernel_input");
    }

    int64_t input_index = index_attr.getInt();

    if (input_index < 0 || input_index >= static_cast<int64_t>(num_inputs)) {
      return kernel_op.emitOpError("kernel_input index is out of bounds");
    }

    if (!port_bound_inputs.insert(static_cast<unsigned>(input_index)).second) {
      return kernel_op.emitOpError("kernel input has multiple port bindings");
    }
  }

  return success();
}

LogicalResult promoteKernelArgsToConstants(neura::KernelOp kernel_op) {
  Region &kernel_region = kernel_op.getBody();
  if (kernel_region.empty()) {
    return success();
  }

  Block &entry_block = kernel_region.front();
  OpBuilder builder(&entry_block, entry_block.begin());

  // Gets the number of inputs and iter_args from kernel operands.
  size_t num_inputs = kernel_op.getInputs().size();
  size_t num_iter_args = kernel_op.getIterArgsInit().size();
  llvm::DenseSet<unsigned> port_bound_inputs;

  if (failed(collectPortBoundInputIndices(kernel_op, port_bound_inputs))) {
    return failure();
  }

  // Verifies block arguments layout: [inputs..., iter_args...]
  SmallVector<BlockArgument> args(entry_block.getArguments().begin(),
                                  entry_block.getArguments().end());

  assert(args.size() == num_inputs + num_iter_args &&
         "Kernel block arguments size mismatch");

  // Step 1: promotes input arguments (not iter_args).
  // Block arguments layout: [input0, input1, ..., iter_arg0, iter_arg1, ...]
  for (size_t i = 0; i < num_inputs; ++i) {
    BlockArgument input_arg = args[i];

    // Inputs explicitly bound to boundary ports remain block arguments.
    // TemplateMapping later routes their DataMovOps from the selected ports.
    if (port_bound_inputs.contains(static_cast<unsigned>(i))) {
      continue;
    }

    // Metadata-only inputs do not need a materialized constant operation.
    // For example, a stationary tensor may be referenced by kernel metadata
    // without appearing as an SSA operand in the compute network.
    if (input_arg.use_empty()) {
      continue;
    }

    // All other live kernel inputs are represented as constant sources in the
    // internal Neura dataflow graph.
    std::string const_name = "%input" + std::to_string(i);
    auto const_op = builder.create<neura::ConstantOp>(
        input_arg.getLoc(), input_arg.getType(),
        builder.getStringAttr(const_name));

    // Replaces all uses of this input argument with the constant.
    input_arg.replaceAllUsesWith(const_op.getResult());
  }

  // Step 2: promotes iter_args_init to constants with special attribute.
  for (size_t i = 0; i < num_iter_args; i++) {
    BlockArgument iter_arg = args[num_inputs + i];

    // Creates a constant for this iter_arg_init value.
    std::string const_name = "%iter_arg_init" + std::to_string(i);
    auto const_op =
        builder.create<neura::ConstantOp>(iter_arg.getLoc(), iter_arg.getType(),
                                          builder.getStringAttr(const_name));

    // Marks this constant as an iter_arg init value.
    const_op->setAttr(kIterArgInitAttr, builder.getBoolAttr(true));

    // Replaces all uses of this iter_arg argument with the constant.
    iter_arg.replaceAllUsesWith(const_op.getResult());
  }

  return success();
}

struct PromoteInputArgToConstPass
    : public PassWrapper<PromoteInputArgToConstPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteInputArgToConstPass)

  StringRef getArgument() const override {
    return "promote-input-arg-to-const";
  }
  StringRef getDescription() const override {
    return "Promotes input arguments to neura constants while preserving "
           "port-bound template inputs.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    module_op.walk([&](Operation *op) {
      Region *region = nullptr;
      if (auto func_op = dyn_cast<func::FuncOp>(op)) {
        auto accel_attr =
            func_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
        if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
          return;
        }
        region = &func_op.getBody();
      } else if (auto llvm_func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        auto accel_attr =
            llvm_func->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
        if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
          return;
        }
        region = &llvm_func.getBody();
      } else {
        return;
      }

      if (!region || region->empty()) {
        return;
      }

      if (failed(promoteFunctionArgsToConstants(*region))) {
        signalPassFailure();
        return;
      }
    });

    // Processes neura.kernel input arguments.
    module_op.walk([&](neura::KernelOp kernel_op) {
      auto accel_attr =
          kernel_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        return;
      }
      if (failed(promoteKernelArgsToConstants(kernel_op))) {
        signalPassFailure();
        return;
      }
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createPromoteInputArgToConstPass() {
  return std::make_unique<PromoteInputArgToConstPass>();
}
} // namespace mlir::neura