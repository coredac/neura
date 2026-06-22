#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//=================================================================
// Affine Loop Band Structure.
//=================================================================

// A loop band can be classified into two types:

// 1) Perfect Loop Band: A sequence of perfectly nested loops where each loop
// (except the innermost) has exactly one child loop and no other operations (no
// prologue/epilogue);
// 2) Imperfect Loop Band: A sequence of nested loops that
// do not satisfy the perfect nesting condition (e.g., loops with exactly one
// child loop, but with other operations in the body).
using AffineLoopBand = SmallVector<affine::AffineForOp>;

// Checks if an operation is side-effect-free (pure computation).
static bool hasSideEffect(Operation *op) {
  // Yield operations are terminators, not computations.
  if (isa<affine::AffineYieldOp>(op)) {
    return true;
  }

  // Arithmetic and pure operations.
  if (isa<arith::ArithDialect>(op->getDialect())) {
    return false;
  }

  // affine.load or memref.load is considered side-effect-free (read-only).
  if (isa<affine::AffineLoadOp>(op) || isa<memref::LoadOp>(op)) {
    return false;
  }

  // affine.store and memref.store are side-effecting (write operations).
  if (isa<affine::AffineStoreOp, memref::StoreOp>(op)) {
    return true;
  }

  // For other operations, conservatively assumes they have side effects.
  return true;
}

static bool usesLoopResult(Operation *op, affine::AffineForOp loop) {
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    return llvm::is_contained(loop.getResults(), operand);
  });
}

static LogicalResult checkLoopBandSupported(AffineLoopBand &loop_band) {
  for (size_t i = loop_band.size() - 1; i > 0; i--) {
    affine::AffineForOp loop = loop_band[i - 1];
    affine::AffineForOp child_loop = loop_band[i];

    bool is_prologue = true;
    bool has_prologue_side_effect = false;
    bool has_epilogue_side_effect = false;

    for (Operation &op : loop.getRegion().front()) {
      if (&op == child_loop) {
        is_prologue = false;
        continue;
      }

      if (isa<affine::AffineYieldOp>(&op)) {
        continue;
      }

      if (llvm::any_of(op.getResultTypes(),
                       [](Type type) { return isa<MemRefType>(type); })) {
        llvm::errs() << "[LoopPerfection] Skipping unsupported loop band: "
                        "memref-producing operation.\n";
        return failure();
      }

      if (isa<func::CallOp>(&op)) {
        llvm::errs() << "[LoopPerfection] Skipping unsupported loop band: "
                        "function call operation.\n";
        return failure();
      }

      if (!is_prologue && usesLoopResult(&op, child_loop)) {
        llvm::errs()
            << "[LoopPerfection] Skipping unsupported loop band: epilogue "
               "operation uses a child loop result.\n";
        return failure();
      }

      if (hasSideEffect(&op)) {
        if (is_prologue) {
          has_prologue_side_effect = true;
        } else {
          has_epilogue_side_effect = true;
        }
      }
    }

    ArrayRef<affine::AffineForOp> inner_loops =
        ArrayRef<affine::AffineForOp>(loop_band).drop_front(i);

    if (has_prologue_side_effect) {
      for (affine::AffineForOp inner_loop : inner_loops) {
        if (!inner_loop.hasConstantLowerBound()) {
          llvm::errs() << "[LoopPerfection] Skipping unsupported loop band: "
                          "non-constant lower bound.\n";
          return failure();
        }
      }
    }

    if (has_epilogue_side_effect) {
      for (affine::AffineForOp inner_loop : inner_loops) {
        if (!inner_loop.getStepAsInt() || !inner_loop.hasConstantUpperBound()) {
          llvm::errs() << "[LoopPerfection] Skipping unsupported loop band: "
                          "non-constant upper bound or step.\n";
          return failure();
        }
      }
    }
  }

  return success();
}

// Collects loop bands from a function.
static void collectLoopBands(func::FuncOp func_op,
                             SmallVector<AffineLoopBand> &loop_bands) {
  func_op.walk([&](affine::AffineForOp for_op) {
    // Only processes outermost loops (skips nested loops).
    if (for_op->getParentOfType<affine::AffineForOp>()) {
      return;
    }

    AffineLoopBand current_band;
    affine::AffineForOp current_loop = for_op;

    // Follows the nesting chain to build the perfect loop band.
    while (current_loop) {
      current_band.push_back(current_loop);

      // Checks if body has exactly one nested loop (perfect nesting).
      Block &body = current_loop.getRegion().front();
      affine::AffineForOp nested_loop = nullptr;
      size_t num_loops = 0;

      for (Operation &body_op : body) {
        if (auto nested_for = dyn_cast<affine::AffineForOp>(&body_op)) {
          nested_loop = nested_for;
          num_loops++;
        }
      }

      // Loop bands condition: exactly 1 nested loop, any number of other ops
      // (other ops will be perfectized).
      if (num_loops == 1) {
        current_loop = nested_loop;
      } else {
        // Has multiple nested loops, not loop bands.
        break;
      }
    }

    if (!current_band.empty()) {
      loop_bands.push_back(current_band);
    }
  });
}

//=================================================================
// Loop Perfection Logic.
//=================================================================

// Creates a condition checking if all inner loop indices are at their lower
// bounds. Used for prologue condition.
static Value
createPrologueCondition(OpBuilder &builder, Location loc,
                        ArrayRef<affine::AffineForOp> inner_loops) {
  // Builds condition for prologue code: (i1 == lb1) && (i2 == lb2) && ...
  Value condition = nullptr;

  for (affine::AffineForOp loop : inner_loops) {
    Value idx = loop.getInductionVar();
    Value lb;

    if (loop.hasConstantLowerBound()) {
      lb = builder.create<arith::ConstantIndexOp>(loc,
                                                  loop.getConstantLowerBound());
    } else {
      llvm::errs()
          << "[LoopPerfection] Non-constant lower bound not supported.\n";
      return nullptr;
    }

    Value eq =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, idx, lb);

    if (condition) {
      condition = builder.create<arith::AndIOp>(loc, condition, eq);
    } else {
      condition = eq;
    }
  }

  return condition;
}

// Creates a condition checking if all inner loop indices are at their upper
// bounds. Used for epilogue condition.
static Value
createEpilogueCondition(OpBuilder &builder, Location loc,
                        ArrayRef<affine::AffineForOp> inner_loops) {
  // Builds condition for epilogue code: (i1 == ub1 - 1) && (i2 == ub2 - 1) &&
  // ...
  Value condition = nullptr;

  for (affine::AffineForOp loop : inner_loops) {
    Value idx = loop.getInductionVar();
    Value next_idx; // idx + step
    Value ub;

    // Gets step.
    int32_t step_val = 1;
    if (loop.getStepAsInt()) {
      step_val = loop.getStepAsInt();
    } else {
      llvm::errs() << "[LoopPerfection] Non-constant step not supported.\n";
      return nullptr;
    }

    // Computes next_idx = idx + step.
    Value step = builder.create<arith::ConstantIndexOp>(loc, step_val);
    next_idx = builder.create<arith::AddIOp>(loc, idx, step);

    if (loop.hasConstantUpperBound()) {
      ub = builder.create<arith::ConstantIndexOp>(loc,
                                                  loop.getConstantUpperBound());
    } else {
      llvm::errs()
          << "[LoopPerfection] Non-constant upper bound not supported.\n";
      return nullptr;
    }

    Value is_last = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, next_idx, ub);

    if (condition) {
      condition = builder.create<arith::AndIOp>(loc, condition, is_last);
    } else {
      condition = is_last;
    }
  }

  return condition;
}

// Applies loop perfection to a single loop band.
// Sinks all operations into the innermost loop with condition execution.
static LogicalResult applyLoopPerfection(AffineLoopBand &loop_band) {
  if (loop_band.empty()) {
    return success();
  }

  llvm::errs() << "[LoopPerfection] Processing loop band with "
               << loop_band.size() << " loops.\n";

  if (failed(checkLoopBandSupported(loop_band))) {
    return success();
  }

  affine::AffineForOp innermost_loop = loop_band.back();
  OpBuilder builder(innermost_loop);

  // Processes each loop in the band from outermost to innermost.
  for (size_t i = loop_band.size() - 1; i > 0; i--) {
    affine::AffineForOp loop = loop_band[i - 1];
    affine::AffineForOp child_loop = loop_band[i];

    // Collects prologue and epilogue operations in the current loop
    // (excluding the child loop).
    SmallVector<Operation *> prologue_ops; // Before child loop.
    SmallVector<Operation *> epilogue_ops; // After child loop.

    bool is_prologue = true;
    for (Operation &op : loop.getRegion().front()) {
      if (&op == child_loop) {
        is_prologue = false;
        continue;
      }

      if (isa<affine::AffineYieldOp>(&op)) {
        // Skips yield operations.
        continue;
      }

      // Rejects operations that cannot be perfectized.
      if (llvm::any_of(op.getResultTypes(),
                       [](Type type) { return isa<MemRefType>(type); })) {
        llvm::errs() << "[LoopPerfection] Memref-producing op cannot be "
                        "perfectized.\n";
        op.dump();
        return failure();
      }

      if (isa<func::CallOp>(&op)) {
        llvm::errs()
            << "[LoopPerfection] Function call op cannot be perfectized.\n";
        op.dump();
        return failure();
      }

      if (is_prologue) {
        prologue_ops.push_back(&op);
      } else {
        epilogue_ops.push_back(&op);
      }
    }

    if (prologue_ops.empty() && epilogue_ops.empty()) {
      // No operations to perfect, continues to next loop.
      continue;
    }

    Location loc = loop.getLoc();
    Block &innermost_body = innermost_loop.getRegion().front();

    // Gets all inner loops (from current child to innermost loop).
    ArrayRef<affine::AffineForOp> inner_loops =
        ArrayRef<affine::AffineForOp>(loop_band).drop_front(i);

    // Handles prologue operations.
    if (!prologue_ops.empty()) {
      llvm::errs() << "  Moving " << prologue_ops.size()
                   << " prologue operations\n";

      Operation *insert_point = &innermost_body.front();

      // Seperates pure and side-effecting operations in the prologue.
      SmallVector<Operation *> pure_ops;
      SmallVector<Operation *> side_effect_ops;

      for (Operation *op : prologue_ops) {
        if (hasSideEffect(op)) {
          side_effect_ops.push_back(op);
        } else {
          pure_ops.push_back(op);
        }
      }

      // Moves pure operations directly into the innermost loop (will be CSE'd
      // if redundant).
      for (Operation *op : pure_ops) {
        op->moveBefore(insert_point);
      }

      // Moves side-effecting operations into the innermost loop with
      // condition execution.
      if (!side_effect_ops.empty()) {
        builder.setInsertionPoint(insert_point);
        Value condition = createPrologueCondition(builder, loc, inner_loops);

        if (condition) {
          scf::IfOp if_op = builder.create<scf::IfOp>(loc, condition,
                                                      /*withElseRegion*/ false);

          Block *then_block = if_op.thenBlock();

          for (Operation *op : side_effect_ops) {
            op->moveBefore(then_block->getTerminator());
          }
        } else {
          // If condition creation fails, returns failure to avoid
          // incorrect transformation.
          llvm::errs()
              << "[LoopPerfection] Failed to create prologue condition.\n";
          return failure();
        }
      }
    }

    // Handles epilogue operations.
    if (!epilogue_ops.empty()) {
      llvm::errs() << "  Moving " << epilogue_ops.size()
                   << " epilogue operations\n";

      Operation *insert_point = innermost_body.getTerminator();

      // Separates pure and side-effecting operations in the epilogue.
      SmallVector<Operation *> pure_ops;
      SmallVector<Operation *> side_effect_ops;

      for (Operation *op : epilogue_ops) {
        if (hasSideEffect(op)) {
          side_effect_ops.push_back(op);
        } else {
          pure_ops.push_back(op);
        }
      }

      // Moves pure operations directly into the innermost loop (will be CSE'd
      // if redundant).
      for (Operation *op : pure_ops) {
        op->moveBefore(insert_point);
      }

      // Moves side-effecting operations into the innermost loop with
      // condition execution.
      if (!side_effect_ops.empty()) {
        builder.setInsertionPoint(insert_point);
        Value condition = createEpilogueCondition(builder, loc, inner_loops);

        if (condition) {
          scf::IfOp if_op = builder.create<scf::IfOp>(loc, condition,
                                                      /*withElseRegion*/ false);

          Block *then_block = if_op.thenBlock();

          for (Operation *op : side_effect_ops) {
            op->moveBefore(then_block->getTerminator());
          }
        } else {
          // If condition creation fails, returns failure to avoid
          // incorrect transformation.
          llvm::errs()
              << "[LoopPerfection] Failed to create epilogue condition.\n";
          return failure();
        }
      }
    }
  }

  return success();
}

//=================================================================
// Pass Implementation.
//=================================================================
struct AffineLoopPerfectionPass
    : public PassWrapper<AffineLoopPerfectionPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineLoopPerfectionPass)

  StringRef getArgument() const final { return "affine-loop-perfection"; }
  StringRef getDescription() const final {
    return "Apply loop perfection for affine loops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect,
                memref::MemRefDialect, scf::SCFDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    // Collects all loop bands in the function.
    SmallVector<AffineLoopBand> loop_bands;
    collectLoopBands(func_op, loop_bands);

    if (loop_bands.empty()) {
      llvm::errs() << "[LoopPerfection] No loop bands found in function: "
                   << func_op.getName() << "\n";
      return;
    }

    llvm::errs() << "[LoopPerfection] Found " << loop_bands.size()
                 << " loop bands in function: " << func_op.getName() << "\n";

    // Apply loop perfection to each loop band.
    for (AffineLoopBand &band : loop_bands) {
      if (failed(applyLoopPerfection(band))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::taskflow::createAffineLoopPerfectionPass() {
  return std::make_unique<AffineLoopPerfectionPass>();
}
