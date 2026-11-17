#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

#define GEN_PASS_DEF_PROMOTEREDUCETOGLOBAL
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// Replaces every local vector reduction of the kernel with a global one.
static void promoteLocalReduces(FunctionOpInterface func) {
  SmallVector<neura::VectorReduceAddOp> local_ops;
  func.walk([&](neura::VectorReduceAddOp reduce_op) {
    local_ops.push_back(reduce_op);
  });

  for (neura::VectorReduceAddOp reduce_op : local_ops) {
    OpBuilder builder(reduce_op);
    auto global_op = builder.create<neura::VectorReduceAddGlobalOp>(
        reduce_op.getLoc(), reduce_op.getResult().getType(),
        reduce_op.getInput(), /*reduce_id=*/nullptr);
    reduce_op.getResult().replaceAllUsesWith(global_op.getResult());
    reduce_op.erase();
  }
}

// Gives every global reduce instruction in the kernel its own reduce table
// entry, so the kernel can hold at most num_entries such instructions.
static LogicalResult assignReduceIds(FunctionOpInterface func,
                                     unsigned num_entries) {
  SmallVector<neura::VectorReduceAddGlobalOp> unassigned_ops;
  llvm::SmallDenseSet<uint32_t> used_ids;

  WalkResult walk_result =
      func.walk([&](neura::VectorReduceAddGlobalOp reduce_op) {
        std::optional<uint32_t> reduce_id = reduce_op.getReduceId();
        if (!reduce_id) {
          unassigned_ops.push_back(reduce_op);
          return WalkResult::advance();
        }
        if (*reduce_id >= num_entries) {
          reduce_op.emitError()
              << "reduce_id " << *reduce_id
              << " is out of range for a global reduce unit with "
              << num_entries << " entries";
          return WalkResult::interrupt();
        }
        if (!used_ids.insert(*reduce_id).second) {
          reduce_op.emitError() << "reduce_id " << *reduce_id
                                << " is used by more than one global reduce";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (walk_result.wasInterrupted())
    return failure();

  size_t num_reduce_ops = used_ids.size() + unassigned_ops.size();
  if (num_reduce_ops > num_entries) {
    return func.emitError()
           << num_reduce_ops << " global reduce operations exceed the "
           << num_entries
           << " entries of the global reduce unit (global_reduce_entries in "
              "the architecture spec)";
  }

  uint32_t next_id = 0;
  for (neura::VectorReduceAddGlobalOp reduce_op : unassigned_ops) {
    while (used_ids.contains(next_id))
      ++next_id;
    reduce_op.setReduceId(next_id);
    used_ids.insert(next_id);
  }
  return success();
}

struct PromoteReduceToGlobalPass
    : public PassWrapper<PromoteReduceToGlobalPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteReduceToGlobalPass)

  StringRef getArgument() const override { return "promote-reduce-to-global"; }
  StringRef getDescription() const override {
    return "Turns every vector reduction into a global reduce and assigns it "
           "a global reduce table entry.";
  }

  void runOnOperation() override {
    unsigned num_entries =
        mlir::neura::getArchitecture().getGlobalReduceEntries();
    getOperation().walk([&](FunctionOpInterface func) {
      promoteLocalReduces(func);
      if (failed(assignReduceIds(func, num_entries)))
        signalPassFailure();
    });
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createPromoteReduceToGlobalPass() {
  return std::make_unique<PromoteReduceToGlobalPass>();
}
} // namespace mlir::neura
