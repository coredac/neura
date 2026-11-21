#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <string>

using namespace mlir;

#define GEN_PASS_DEF_FOLDCONSTANT
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// Attribute name to mark iter_arg init constants.
constexpr const char *kIterArgInitAttr = "is_iter_arg_init";

// =========================================
// Helper Functions
// =========================================
bool isOriginConstantOp(Value value) {
  if (!value) {
    return false;
  }
  Operation *def_op = value.getDefiningOp();
  if (!def_op || !isa<neura::ConstantOp>(def_op)) {
    return false;
  }

  // Skips constants marked as iter_arg_init.
  if (def_op->hasAttr(kIterArgInitAttr)) {
    if (auto bool_attr = def_op->getAttrOfType<BoolAttr>(kIterArgInitAttr)) {
      if (bool_attr.getValue()) {
        // This constant is an iter_arg_init, should not be folded.
        return false;
      }
    }
  }

  // Checks if the result type is the original type or the predicated type.
  Type result_type = value.getType();
  if (isa<neura::PredicatedValue>(result_type)) {
    return false;
  }

  return true;
}

Attribute getOriginConstantValue(Value value) {
  neura::ConstantOp constant_op =
      dyn_cast<neura::ConstantOp>(value.getDefiningOp());
  return constant_op->getAttr(neura::attr::kValue);
}

void addConstantAttribute(Operation *op, StringRef attr_name,
                          Attribute const_value) {
  op->setAttr(attr_name, const_value);
}

// =========================================
// Generic Constant Folding Framework
// =========================================

// Information about a single constant operand to fold.
struct ConstantOperandInfo {
  size_t index;           // Index in the original operand list.
  Attribute const_value;  // The constant value.
  Operation *defining_op; // The operation that defines this constant.
};

// Analyzes operands and returns information about constants to fold.
SmallVector<ConstantOperandInfo> analyzeOperandsForFolding(Operation *op) {
  SmallVector<ConstantOperandInfo> constants_to_fold;

  size_t num_operands = op->getNumOperands();
  if (num_operands == 0) {
    return constants_to_fold;
  }

  // First pass: Identifies which operands are constants.
  SmallVector<bool> is_const(num_operands, false);
  bool has_non_const = false;

  for (size_t i = 0; i < num_operands; ++i) {
    if (isOriginConstantOp(op->getOperand(i))) {
      is_const[i] = true;
    } else {
      has_non_const = true;
    }
  }

  // Second pass: Collects constants to fold.
  for (size_t i = 0; i < num_operands; ++i) {
    if (is_const[i]) {
      // Rule 1: If this is operand 0 and there are no other non-const operands,
      // we must keep it (MLIR operations need at least one operand).
      if (i == 0 && !has_non_const) {
        continue; // Don't fold this one.
      }

      // Rule 2: Only folds at most 1 constant.
      if (!constants_to_fold.empty()) {
        // Already have one constant to fold, skip the rest.
        continue;
      }
      // This operand will be folded.
      Value operand = op->getOperand(i);
      constants_to_fold.push_back(
          {i, getOriginConstantValue(operand), operand.getDefiningOp()});
    }
  }

  return constants_to_fold;
}

// Gets the attribute name for a given operand index.
// For binary operations, uses "lhs_value" and "rhs_value".
// For other operations, uses "operand_N_value".
std::string getAttributeNameForOperandIndex(size_t index,
                                            size_t total_operands) {
  if (total_operands == 2) {
    // Binary operation: use lhs_value/rhs_value.
    if (index == 0) {
      return neura::attr::kLhsValue.str();
    } else {
      return neura::attr::kRhsValue.str();
    }
  } else {
    // Multi-operand operation: use operand_N_value.
    return "operand_" + std::to_string(index) + "_value";
  }
}

// =========================================
// Generic Constant Folding Pattern
// =========================================
template <typename OpType>
struct GenericFuseConstantPattern : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  // Virtual function to get attribute name for a given operand index.
  // Default implementation uses binary naming (lhs/rhs) or operand_N naming.
  // Derived classes can override this for custom naming.
  virtual std::string getAttributeName(size_t operand_idx,
                                       size_t total_operands) const {
    return getAttributeNameForOperandIndex(operand_idx, total_operands);
  }

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // Gets the original number of operands before folding.
    size_t num_operands = op->getNumOperands();

    // Analyzes operands to determine which can be folded.
    SmallVector<ConstantOperandInfo> constants_to_fold =
        analyzeOperandsForFolding(op);

    // If no constant operands found, nothing to do.
    if (constants_to_fold.empty()) {
      return failure();
    }

    // Checks if any operands have already been folded.
    // Looks for any attribute ending with "_value" which indicates constant
    // folding.
    for (auto attr : op->getAttrs()) {
      StringRef attr_name = attr.getName().getValue();
      if (attr_name.ends_with("_value")) {
        return failure();
      }
    }

    // Builds list of non-constant operands.
    SmallVector<Value> non_const_operands;
    SmallVector<bool> is_folded(num_operands, false);
    for (const auto &const_info : constants_to_fold) {
      is_folded[const_info.index] = true;
    }
    for (size_t i = 0; i < num_operands; ++i) {
      if (!is_folded[i]) {
        non_const_operands.push_back(op->getOperand(i));
      }
    }

    // Creates the new operation with only non-constant operands.
    Operation *new_op =
        createOpWithFoldedConstants(op, non_const_operands, rewriter);

    if (!new_op) {
      return failure();
    }

    // Adds constant attributes for each folded operand.
    for (const auto &const_info : constants_to_fold) {
      std::string attr_name = getAttributeName(const_info.index, num_operands);
      addConstantAttribute(new_op, attr_name, const_info.const_value);
    }

    // Replaces the old operation.
    rewriter.replaceOp(op, new_op->getResults());

    // Cleans up unused constant operations.
    for (const auto &const_info : constants_to_fold) {
      if (const_info.defining_op->use_empty()) {
        rewriter.eraseOp(const_info.defining_op);
      }
    }

    return success();
  }

  // Virtual function to create the operation with folded constants.
  // Must be implemented by derived classes.
  virtual Operation *
  createOpWithFoldedConstants(OpType op, ArrayRef<Value> non_const_operands,
                              PatternRewriter &rewriter) const = 0;
};

// =========================================
// Specialized Patterns for Specific Operations
// =========================================

// Helper macro to define a pattern for a binary operation.
#define DEFINE_BINARY_OP_PATTERN(OP_NAME, OP_TYPE)                             \
  struct Fuse##OP_NAME##ConstantPattern                                        \
      : public GenericFuseConstantPattern<neura::OP_TYPE> {                    \
    using GenericFuseConstantPattern<                                          \
        neura::OP_TYPE>::GenericFuseConstantPattern;                           \
    Operation *                                                                \
    createOpWithFoldedConstants(neura::OP_TYPE op,                             \
                                ArrayRef<Value> non_const_operands,            \
                                PatternRewriter &rewriter) const override {    \
      /* Uses generic Operation create and copy attributes. */                 \
      OperationState state(op.getLoc(), op.getOperationName());                \
      state.addOperands(non_const_operands);                                   \
      state.addTypes(op->getResultTypes());                                    \
      /* Copies attributes except operandSegmentSizes (will be                 \
       * auto-generated). */                                                   \
      for (auto attr : op->getAttrs()) {                                       \
        if (attr.getName() != "operandSegmentSizes") {                         \
          state.addAttribute(attr.getName(), attr.getValue());                 \
        }                                                                      \
      }                                                                        \
      return rewriter.create(state);                                           \
    }                                                                          \
  };

// Defines patterns for all binary arithmetic operations.
//
// Note: The macro DEFINE_BINARY_OP_PATTERN expands to create a complete pattern
// class. For example, DEFINE_BINARY_OP_PATTERN(Add, AddOp) expands to:
//
//   struct FuseAddConstantPattern
//       : public GenericFuseConstantPattern<neura::AddOp> {
//     using
//     GenericFuseConstantPattern<neura::AddOp>::GenericFuseConstantPattern;
//
//     Operation *createOpWithFoldedConstants(
//         neura::AddOp op, ArrayRef<Value> non_const_operands,
//         PatternRewriter &rewriter) const override {
//       // Use generic Operation create and copy attributes.
//       OperationState state(op.getLoc(), op.getOperationName());
//       state.addOperands(non_const_operands);
//       state.addTypes(op->getResultTypes());
//       // Copy attributes except operandSegmentSizes (will be auto-generated).
//       for (auto attr : op->getAttrs()) {
//         if (attr.getName() != "operandSegmentSizes") {
//           state.addAttribute(attr.getName(), attr.getValue());
//         }
//       }
//       return rewriter.create(state);
//     }
//   };
//
// All other binary operations (Sub, Mul, Div, etc.) follow the same pattern,
// so we use the macro to avoid code duplication.
// Generates: FuseAddConstantPattern.
DEFINE_BINARY_OP_PATTERN(Add, AddOp)
// Generates: FuseSubConstantPattern.
DEFINE_BINARY_OP_PATTERN(Sub, SubOp)
// Generates: FuseMulConstantPattern.
DEFINE_BINARY_OP_PATTERN(Mul, MulOp)
// Generates: FuseDivConstantPattern.
DEFINE_BINARY_OP_PATTERN(Div, DivOp)
// Generates: FuseRemConstantPattern.
DEFINE_BINARY_OP_PATTERN(Rem, RemOp)
// Generates: FuseFAddConstantPattern.
DEFINE_BINARY_OP_PATTERN(FAdd, FAddOp)
// Generates: FuseFSubConstantPattern.
DEFINE_BINARY_OP_PATTERN(FSub, FSubOp)
// Generates: FuseFMulConstantPattern.
DEFINE_BINARY_OP_PATTERN(FMul, FMulOp)
// Generates: FuseFDivConstantPattern.
DEFINE_BINARY_OP_PATTERN(FDiv, FDivOp)
// Generates: FuseICmpConstantPattern.
// Note: ICmpOp has a cmp_type attribute that is automatically preserved.
DEFINE_BINARY_OP_PATTERN(ICmp, ICmpOp)
// Generates: FuseFMaxConstantPattern.
// Note: FMaxOp has a nan_semantic attribute that is automatically preserved.
DEFINE_BINARY_OP_PATTERN(FMax, FMaxOp)
// Generates: FuseFMinConstantPattern.
// Note: FMinOp has a nan_semantic attribute that is automatically preserved.
DEFINE_BINARY_OP_PATTERN(FMin, FMinOp)
// Generates: FuseStoreConstantPattern.
DEFINE_BINARY_OP_PATTERN(Store, StoreOp)
// Generates: FuseShlConstantPattern.
DEFINE_BINARY_OP_PATTERN(Shl, ShlOp)
// Generates: FuseAndConstantPattern.
DEFINE_BINARY_OP_PATTERN(And, AndOp)
// Generates: FuseOrConstantPattern.
DEFINE_BINARY_OP_PATTERN(Or, OrOp)

// Pattern for GEP operation (base + indices).
// Only folds the base, because the hardware GEP unit takes its constant
// operand as the base.
struct FuseGEPConstantPattern : public OpRewritePattern<neura::GEP> {
  using OpRewritePattern<neura::GEP>::OpRewritePattern;
  LogicalResult matchAndRewrite(neura::GEP op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(neura::attr::kLhsValue)) {
      return failure();
    }

    Value base = op.getBase();
    if (!base || !isOriginConstantOp(base)) {
      return failure();
    }
    auto constant_op = dyn_cast<neura::ConstantOp>(base.getDefiningOp());
    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());

    OperationState state(op.getLoc(), op.getOperationName());
    state.addOperands(indices);
    state.addTypes(op->getResultTypes());
    for (auto attr : op->getAttrs()) {
      if (attr.getName() != "operandSegmentSizes") {
        state.addAttribute(attr.getName(), attr.getValue());
      }
    }
    state.addAttribute(neura::attr::kLhsValue, getOriginConstantValue(base));
    state.addAttribute("operandSegmentSizes",
                       rewriter.getDenseI32ArrayAttr(
                           {0, static_cast<int32_t>(indices.size())}));

    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op->getResults());
    if (constant_op->use_empty()) {
      rewriter.eraseOp(constant_op);
    }
    return success();
  }
};

// Pattern for LoadIndexed operation (base + indices).
// Only folds the base, never folds indices (required by assemblyFormat).
struct FuseLoadIndexedConstantPattern
    : public OpRewritePattern<neura::LoadIndexedOp> {
  using OpRewritePattern<neura::LoadIndexedOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(neura::LoadIndexedOp op,
                                PatternRewriter &rewriter) const override {
    // Checks if already folded.
    if (op->hasAttr("lhs_value")) {
      return failure();
    }

    // Only checks if base is a constant.
    Value base = op.getBase();
    if (!base || !isOriginConstantOp(base)) {
      return failure();
    }
    auto constant_op = dyn_cast<neura::ConstantOp>(base.getDefiningOp());
    Attribute base_value = getOriginConstantValue(base);
    // Keeps all indices unchanged (never fold indices).
    SmallVector<Value> indices;
    for (Value idx : op.getIndices()) {
      indices.push_back(idx);
    }

    // Creates new LoadIndexed without base.
    OperationState state(op.getLoc(), op.getOperationName());
    state.addOperands(indices); // Only indices, no base.
    state.addTypes(op->getResultTypes());

    // Copies all attributes except operandSegmentSizes.
    for (auto attr : op->getAttrs()) {
      if (attr.getName() != "operandSegmentSizes") {
        state.addAttribute(attr.getName(), attr.getValue());
      }
    }

    // Adds the folded base value.
    state.addAttribute("lhs_value", base_value);

    // Sets operandSegmentSizes: 0 base, N indices.
    state.addAttribute("operandSegmentSizes",
                       rewriter.getDenseI32ArrayAttr(
                           {0, static_cast<int32_t>(indices.size())}));

    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op->getResults());

    // Cleans up constant if no longer used.
    if (constant_op->use_empty()) {
      rewriter.eraseOp(constant_op);
    }

    return success();
  }
};

// Pattern for StoreIndexed operation (value, base, indices...).
// Only folds value and base, never folds indices (required by assemblyFormat).
struct FuseStoreIndexedConstantPattern
    : public OpRewritePattern<neura::StoreIndexedOp> {
  using OpRewritePattern<neura::StoreIndexedOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(neura::StoreIndexedOp op,
                                PatternRewriter &rewriter) const override {
    // Checks if already folded.
    if (op->hasAttr(neura::attr::kLhsValue) ||
        op->hasAttr(neura::attr::kRhsValue)) {
      return failure();
    }

    // Checks which of value/base are constants.
    Value value = op.getValue();
    Value base = op.getBase();

    bool value_is_const = value && isOriginConstantOp(value);
    bool base_is_const = base && isOriginConstantOp(base);

    // Nothing to fold if neither is constant.
    if (!value_is_const && !base_is_const) {
      return failure();
    }

    // Special restriction: For operations with 3 or more operands,
    // only fold at most 1 constant (fold value, not base).
    size_t num_operands = op->getNumOperands();
    if (num_operands >= 3 && value_is_const && base_is_const) {
      // Both are constants, but we can only fold one.
      // Prioritize folding value (lhs), don't fold base (rhs).
      // Format: neura.store_indexed %value to %base[%idx : type] base_type :
      // value_type
      //         where %value is lhs (1st operand), %base is rhs (2nd operand).
      base_is_const = false;
    }
    // Keeps all indices unchanged (never fold indices).
    SmallVector<Value> indices;
    for (Value idx : op.getIndices()) {
      indices.push_back(idx);
    }

    // Builds the new operand list.
    SmallVector<Value> operands;
    int32_t num_value = 0;
    int32_t num_base = 0;

    if (!value_is_const && value) {
      operands.push_back(value);
      num_value = 1;
    }

    if (!base_is_const && base) {
      operands.push_back(base);
      num_base = 1;
    }

    for (Value idx : indices) {
      operands.push_back(idx);
    }
    int32_t num_indices = indices.size();

    // Creates new StoreIndexed.
    OperationState state(op.getLoc(), op.getOperationName());
    state.addOperands(operands);
    state.addTypes(op->getResultTypes());

    // Copies all attributes except operandSegmentSizes.
    for (auto attr : op->getAttrs()) {
      if (attr.getName() != "operandSegmentSizes") {
        state.addAttribute(attr.getName(), attr.getValue());
      }
    }

    // Adds folded constant attributes.
    if (value_is_const) {
      state.addAttribute(neura::attr::kLhsValue, getOriginConstantValue(value));
    }
    if (base_is_const) {
      state.addAttribute(neura::attr::kRhsValue, getOriginConstantValue(base));
    }

    // Sets operandSegmentSizes: num_value, num_base, num_indices.
    state.addAttribute(
        "operandSegmentSizes",
        rewriter.getDenseI32ArrayAttr({num_value, num_base, num_indices}));

    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op->getResults());

    // Cleans up unused constants.
    if (value_is_const) {
      auto const_op = value.getDefiningOp();
      if (const_op->use_empty()) {
        rewriter.eraseOp(const_op);
      }
    }
    if (base_is_const) {
      auto const_op = base.getDefiningOp();
      if (const_op->use_empty()) {
        rewriter.eraseOp(const_op);
      }
    }

    return success();
  }
};

// =========================================
// FuseConstantAndGrantPattern
// Valid only after transform-ctrl-to-data-flow pass.
// =========================================
struct FuseConstantAndGrantPattern
    : public OpRewritePattern<neura::ConstantOp> {
  using OpRewritePattern<neura::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::ConstantOp constant_op,
                                PatternRewriter &rewriter) const override {
    bool made_change = false;

    // Checks if the constant operation is used by a grant_once or grant_always
    // operation.
    for (auto user : constant_op->getUsers()) {
      if (isa<neura::GrantOnceOp>(user) || isa<neura::GrantAlwaysOp>(user)) {
        if (neura::GrantOnceOp grant_once_op =
                dyn_cast<neura::GrantOnceOp>(user)) {
          auto new_grant_once_op = rewriter.create<neura::GrantOnceOp>(
              grant_once_op.getLoc(), grant_once_op.getResult().getType(),
              /*value=*/nullptr, constant_op->getAttr(neura::attr::kValue));
          // Replaces the original constant operation with the new one.
          rewriter.replaceOp(grant_once_op, new_grant_once_op);
          made_change = true;
        } else if (neura::GrantAlwaysOp grant_always_op =
                       dyn_cast<neura::GrantAlwaysOp>(user)) {
          auto new_grant_always_op = rewriter.create<neura::GrantAlwaysOp>(
              grant_always_op.getLoc(), grant_always_op.getResult().getType(),
              /*value=*/nullptr, constant_op->getAttr(neura::attr::kValue));
          // Replaces the original constant operation with the new one.
          rewriter.replaceOp(grant_always_op, new_grant_always_op);
          made_change = true;
        }
      }
    }

    if (constant_op->use_empty()) {
      // If the constant operation has no users, it can be removed.
      rewriter.eraseOp(constant_op);
      made_change = true;
    }

    return success(made_change);
  }
};

// =========================================
// FuseCounterConstantPattern
// Folds constant lower_bound, upper_bound, and/or step of CounterOp
// into attributes: lower_bound_value, upper_bound_value, step_value.
// =========================================
struct FuseCounterConstantPattern : public OpRewritePattern<neura::CounterOp> {
  using OpRewritePattern<neura::CounterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::CounterOp counter_op,
                                PatternRewriter &rewriter) const override {
    // Checks if already (partially) folded.
    if (counter_op->hasAttr("lower_bound_value") ||
        counter_op->hasAttr("upper_bound_value") ||
        counter_op->hasAttr("step_value")) {
      return failure();
    }
    Value lb = counter_op.getLowerBound();
    Value ub = counter_op.getUpperBound();
    Value step = counter_op.getStep();

    bool lb_is_const = isOriginConstantOp(lb);
    bool ub_is_const = isOriginConstantOp(ub);
    bool step_is_const = isOriginConstantOp(step);

    if (!lb_is_const && !ub_is_const && !step_is_const) {
      return failure();
    }

    // Builds new operand list keeping only non-constant (or absent) bounds.
    SmallVector<Value> operands;
    int32_t lb_present = 0, ub_present = 0, step_present = 0;

    if (!lb_is_const && lb) {
      operands.push_back(lb);
      lb_present = 1;
    }
    if (!ub_is_const && ub) {
      operands.push_back(ub);
      ub_present = 1;
    }
    if (!step_is_const && step) {
      operands.push_back(step);
      step_present = 1;
    }

    OperationState state(counter_op.getLoc(), counter_op.getOperationName());
    state.addOperands(operands);
    state.addTypes(counter_op->getResultTypes());

    // Copies all existing attributes except operandSegmentSizes.
    for (auto attr : counter_op->getAttrs()) {
      if (attr.getName() != "operandSegmentSizes") {
        state.addAttribute(attr.getName(), attr.getValue());
      }
    }

    // Adds folded constant attributes.
    if (lb_is_const)
      state.addAttribute("lower_bound_value", getOriginConstantValue(lb));
    if (ub_is_const)
      state.addAttribute("upper_bound_value", getOriginConstantValue(ub));
    if (step_is_const)
      state.addAttribute("step_value", getOriginConstantValue(step));

    // Updates operandSegmentSizes for the three optional slots.
    state.addAttribute(
        "operandSegmentSizes",
        rewriter.getDenseI32ArrayAttr({lb_present, ub_present, step_present}));

    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(counter_op, new_op->getResults());

    // Cleans up now-unused constant ops.
    // Use a set to avoid double-erasing when two bounds share the same defining
    // op.
    llvm::SmallPtrSet<Operation *, 4> to_erase;
    if (lb_is_const && lb.getDefiningOp()->use_empty()) {
      to_erase.insert(lb.getDefiningOp());
    }
    if (ub_is_const && ub.getDefiningOp()->use_empty()) {
      to_erase.insert(ub.getDefiningOp());
    }
    if (step_is_const && step.getDefiningOp()->use_empty()) {
      to_erase.insert(step.getDefiningOp());
    }
    for (Operation *op : to_erase) {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

// =========================================
// FoldConstantPass Implementation
// =========================================
struct FoldConstantPass
    : public PassWrapper<FoldConstantPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConstantPass)

  StringRef getArgument() const override { return "fold-constant"; }
  StringRef getDescription() const override {
    return "Fold constant operations into operation attributes.";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());

    // Adds generic constant folding patterns for all operations.
    patterns.add<FuseAddConstantPattern>(&getContext());
    patterns.add<FuseSubConstantPattern>(&getContext());
    patterns.add<FuseMulConstantPattern>(&getContext());
    patterns.add<FuseDivConstantPattern>(&getContext());
    patterns.add<FuseRemConstantPattern>(&getContext());
    patterns.add<FuseICmpConstantPattern>(&getContext());
    patterns.add<FuseFAddConstantPattern>(&getContext());
    patterns.add<FuseFSubConstantPattern>(&getContext());
    patterns.add<FuseFMulConstantPattern>(&getContext());
    patterns.add<FuseFDivConstantPattern>(&getContext());
    patterns.add<FuseFMaxConstantPattern>(&getContext());
    patterns.add<FuseFMinConstantPattern>(&getContext());

    // Adds patterns for bitwise operations.
    patterns.add<FuseShlConstantPattern>(&getContext());
    patterns.add<FuseAndConstantPattern>(&getContext());
    patterns.add<FuseOrConstantPattern>(&getContext());

    // Adds patterns for memory operations.
    patterns.add<FuseGEPConstantPattern>(&getContext());
    patterns.add<FuseStoreConstantPattern>(&getContext());
    patterns.add<FuseLoadIndexedConstantPattern>(&getContext());
    patterns.add<FuseStoreIndexedConstantPattern>(&getContext());

    // Adds pattern for grant operations (post-transform).
    patterns.add<FuseConstantAndGrantPattern>(&getContext());

    // Adds pattern for CounterOp.
    patterns.add<FuseCounterConstantPattern>(&getContext());

    FrozenRewritePatternSet frozen(std::move(patterns));

    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](Operation *op) {
      if (!op->getRegions().empty()) {
        for (Region &region : op->getRegions()) {
          if (failed(applyPatternsGreedily(region, frozen))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFoldConstantPass() {
  return std::make_unique<FoldConstantPass>();
}
} // namespace mlir::neura
