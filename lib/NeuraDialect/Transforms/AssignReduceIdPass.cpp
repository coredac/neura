#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

#define GEN_PASS_DEF_ASSIGNREDUCEID
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

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

struct AssignReduceIdPass
    : public PassWrapper<AssignReduceIdPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignReduceIdPass)

  StringRef getArgument() const override { return "assign-reduce-id"; }
  StringRef getDescription() const override {
    return "Assigns a global reduce table entry to every global reduce "
           "operation.";
  }

  void runOnOperation() override {
    unsigned num_entries =
        mlir::neura::getArchitecture().getGlobalReduceEntries();
    getOperation().walk([&](FunctionOpInterface func) {
      if (failed(assignReduceIds(func, num_entries)))
        signalPassFailure();
    });
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createAssignReduceIdPass() {
  return std::make_unique<AssignReduceIdPass>();
}
} // namespace mlir::neura
