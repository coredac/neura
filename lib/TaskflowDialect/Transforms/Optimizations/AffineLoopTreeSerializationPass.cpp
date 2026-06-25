#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "TaskflowDialect/TaskflowTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//==============================================================================
// Static Affine Loop Tree (SALT) Node.
//==============================================================================
struct SALTNode {
  affine::AffineForOp loop_op;
  int64_t lower_bound;
  int64_t upper_bound;
  int64_t step;

  SALTNode *parent = nullptr;
  SmallVector<SALTNode *> children;

  // Operations that are NOT nested loops (the actual computation at this
  // level).
  SmallVector<Operation *> body_operations;

  bool isLeaf() const { return children.empty(); }
  bool isRoot() const { return parent == nullptr; }
};

//==============================================================================
// Loop Chain - Path from Root to Leaf.
//==============================================================================
struct LoopChain {
  SmallVector<SALTNode *> nodes; // Ordered from root to leaf.

  SALTNode *getRoot() const { return nodes.front(); }
  SALTNode *getLeaf() const { return nodes.back(); }
};

//==============================================================================
// SALT Builder.
//==============================================================================
class SALTBuilder {
public:
  SmallVector<SALTNode *> build(func::FuncOp func_op) {
    SmallVector<SALTNode *> roots;

    for (Block &block : func_op.getBlocks()) {
      for (Operation &op : block) {
        if (affine::AffineForOp for_op = dyn_cast<affine::AffineForOp>(&op)) {
          if (for_op.hasConstantLowerBound() &&
              for_op.hasConstantUpperBound()) {
            SALTNode *root = buildNodeRecursively(for_op, nullptr);
            if (root) {
              roots.push_back(root);
            }
          }
        }
      }
    }

    return roots;
  }

  const SmallVector<std::unique_ptr<SALTNode>> &getAllNodes() const {
    return all_nodes;
  }

private:
  SmallVector<std::unique_ptr<SALTNode>> all_nodes;

  SALTNode *buildNodeRecursively(affine::AffineForOp for_op, SALTNode *parent) {
    auto node = std::make_unique<SALTNode>();
    node->loop_op = for_op;
    node->lower_bound = for_op.getConstantLowerBound();
    node->upper_bound = for_op.getConstantUpperBound();
    node->step = for_op.getStepAsInt();
    node->parent = parent;

    SALTNode *node_ptr = node.get();
    all_nodes.push_back(std::move(node));

    Block &body = for_op.getRegion().front();
    for (Operation &op : body) {
      if (auto nested_for = dyn_cast<affine::AffineForOp>(&op)) {
        if (nested_for.hasConstantLowerBound() &&
            nested_for.hasConstantUpperBound()) {
          SALTNode *child = buildNodeRecursively(nested_for, node_ptr);
          if (child) {
            node_ptr->children.push_back(child);
          }
        } else {
          node_ptr->body_operations.push_back(&op);
        }
      } else if (!isa<affine::AffineYieldOp>(&op)) {
        node_ptr->body_operations.push_back(&op);
      }
    }

    return node_ptr;
  }
};

//==============================================================================
// Loop Chain Extractor (DFS).
//==============================================================================
class LoopChainExtractor {
public:
  SmallVector<LoopChain> extract(const SmallVector<SALTNode *> &roots) {
    SmallVector<LoopChain> chains;

    for (SALTNode *root : roots) {
      SmallVector<SALTNode *> current_path;
      dfs(root, current_path, chains);
    }

    return chains;
  }

private:
  void dfs(SALTNode *node, SmallVector<SALTNode *> &current_path,
           SmallVector<LoopChain> &chains) {
    current_path.push_back(node);

    if (node->isLeaf()) {
      LoopChain chain;
      chain.nodes = current_path;
      chains.push_back(chain);
    } else {
      for (SALTNode *child : node->children) {
        dfs(child, current_path, chains);
      }
    }

    current_path.pop_back();
  }
};

//==============================================================================
// MCT Builder.
//
// MCT stands for "Minimized Canonicalized Task": a standalone affine loop nest
// rebuilt from one root-to-leaf LoopChain extracted from the original SALT. If
// a loop tree has multiple leaf loops, this pass serializes it into multiple
// MCTs, one per root-to-leaf path.
//==============================================================================
class MCTBuilder {
public:
  MCTBuilder(OpBuilder &builder, Location loc) : builder(builder), loc(loc) {}

  // Entry point for building one MCT. `buildNode` does the recursive work while
  // this function owns the old-to-new value mapping for the whole chain.
  FailureOr<affine::AffineForOp> build(const LoopChain &chain) {
    // Mapping from old values to new values.
    IRMapping mapping;
    return buildNode(chain, /*index=*/0, mapping, builder,
                     chain.getRoot()->loop_op.getOperation());
  }

private:
  OpBuilder &builder;
  Location loc;

  static bool isDefinedInside(Operation *root_op, Value value) {
    if (auto block_arg = dyn_cast<BlockArgument>(value)) {
      Region *region = block_arg.getParentRegion();
      Operation *owner = region ? region->getParentOp() : nullptr;
      while (owner) {
        if (owner == root_op) {
          return true;
        }
        owner = owner->getParentOp();
      }
      return false;
    }

    Operation *def = value.getDefiningOp();
    while (def) {
      if (def == root_op) {
        return true;
      }
      def = def->getParentOp();
    }
    return false;
  }

  LogicalResult checkOperandsMapped(Operation *op, Operation *root_op,
                                    const IRMapping &mapping) {
    for (Value operand : op->getOperands()) {
      if (isDefinedInside(root_op, operand) && !mapping.contains(operand)) {
        return failure();
      }
    }
    return success();
  }

  FailureOr<Value> lookupMappedValue(Operation *op, Value value,
                                     Operation *root_op,
                                     const IRMapping &mapping) {
    if (isDefinedInside(root_op, value) && !mapping.contains(value)) {
      return failure();
    }
    return mapping.lookupOrDefault(value);
  }

  // Recursively rebuilds `chain[index]` and its selected child.
  //
  // The original loop body order matters. A parent loop body may contain a
  // child reduction followed by an operation that consumes the child result:
  //
  //   %sum = affine.for ... -> f32 { ... }
  //   affine.store %sum, ...
  //
  // Rebuilding the selected child exactly where it appeared ensures `%sum` is
  // mapped before later operations are cloned.
  FailureOr<affine::AffineForOp> buildNode(const LoopChain &chain, size_t index,
                                           IRMapping &mapping,
                                           OpBuilder &insert_builder,
                                           Operation *root_op) {
    SALTNode *node = chain.nodes[index];
    // The next node in the LoopChain is the only nested loop cloned at this
    // level. Other sibling loops are serialized into their own MCTs.
    SALTNode *selected_child =
        (index + 1 < chain.nodes.size()) ? chain.nodes[index + 1] : nullptr;

    SmallVector<Value> iter_args_init_values;
    for (Value init : node->loop_op.getInits()) {
      FailureOr<Value> mapped_init =
          lookupMappedValue(node->loop_op, init, root_op, mapping);
      if (failed(mapped_init)) {
        return failure();
      }
      iter_args_init_values.push_back(*mapped_init);
    }

    auto new_loop = insert_builder.create<affine::AffineForOp>(
        loc, node->lower_bound, node->upper_bound, node->step,
        iter_args_init_values);

    mapping.map(node->loop_op.getInductionVar(), new_loop.getInductionVar());

    for (auto [old_arg, new_arg] : llvm::zip(node->loop_op.getRegionIterArgs(),
                                             new_loop.getRegionIterArgs())) {
      mapping.map(old_arg, new_arg);
    }

    for (auto [old_result, new_result] :
         llvm::zip(node->loop_op.getResults(), new_loop.getResults())) {
      mapping.map(old_result, new_result);
    }

    Block *body = new_loop.getBody();
    if (!body->empty() && isa<affine::AffineYieldOp>(body->back())) {
      body->back().erase();
    }

    // Clone the body in source order. This is what keeps child-loop results
    // available before cloning later non-loop operations that consume them.
    OpBuilder body_builder = OpBuilder::atBlockEnd(body);

    for (Operation &op : node->loop_op.getBody()->getOperations()) {
      // Rebuild yields explicitly because the default yield was removed above.
      if (auto yield_op = dyn_cast<affine::AffineYieldOp>(&op)) {
        SmallVector<Value> yielded_values;
        for (Value operand : yield_op.getOperands()) {
          FailureOr<Value> mapped_operand =
              lookupMappedValue(yield_op, operand, root_op, mapping);
          if (failed(mapped_operand)) {
            new_loop.erase();
            return failure();
          }
          yielded_values.push_back(*mapped_operand);
        }
        body_builder.create<affine::AffineYieldOp>(loc, yielded_values);
        continue;
      }

      if (auto nested_for = dyn_cast<affine::AffineForOp>(&op)) {
        // Only clone the selected child for this root-to-leaf chain.
        if (selected_child && nested_for == selected_child->loop_op) {
          if (failed(buildNode(chain, index + 1, mapping, body_builder,
                               root_op))) {
            new_loop.erase();
            return failure();
          }
          body_builder = OpBuilder::atBlockEnd(body);
        }
        continue;
      }

      // Non-loop operations belong to this MCT and are cloned with the current
      // mapping. Their results are then available to later operations.
      if (failed(checkOperandsMapped(&op, root_op, mapping))) {
        new_loop.erase();
        return failure();
      }
      Operation *new_op = body_builder.clone(op, mapping);
      for (auto [old_res, new_res] :
           llvm::zip(op.getResults(), new_op->getResults())) {
        mapping.map(old_res, new_res);
      }
    }

    return new_loop;
  }
};

//==============================================================================
// Pass Implementation
//==============================================================================
struct AffineLoopTreeSerializationPass
    : public PassWrapper<AffineLoopTreeSerializationPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineLoopTreeSerializationPass)

  StringRef getArgument() const final {
    return "affine-loop-tree-serialization";
  }

  StringRef getDescription() const final {
    return "Serialize Affine loop trees into a linear sequence of loop nests "
           "for MCT construction.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TaskflowDialect, affine::AffineDialect, func::FuncDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    WalkResult result = module.walk([&](func::FuncOp func_op) {
      if (failed(convertFunction(func_op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

private:
  LogicalResult convertFunction(func::FuncOp func_op) {
    Location loc = func_op.getLoc();

    // Builds static affine loop tree.
    SALTBuilder salt_builder;
    SmallVector<SALTNode *> roots = salt_builder.build(func_op);

    if (roots.empty()) {
      return success();
    }

    llvm::errs() << "=== SALT Structure ===\n";
    for (SALTNode *root : roots) {
      printSALT(root, 0);
    }

    // Extracts loop chains.
    LoopChainExtractor extractor;
    SmallVector<LoopChain> chains = extractor.extract(roots);

    llvm::errs() << "=== Extracted " << chains.size() << " MCT(s) ===\n";
    for (size_t i = 0; i < chains.size(); ++i) {
      llvm::errs() << "MCT " << i << ": ";
      for (SALTNode *node : chains[i].nodes) {
        llvm::errs() << "[" << node->lower_bound << "," << node->upper_bound
                     << ") ";
      }
      llvm::errs() << "\n";
    }

    // LoopChainExtractor iterates roots in order of SALTBuilder (order
    // of appearance). So we can iterate through roots, and for each root, build
    // its chains, replace root with chains.

    for (SALTNode *root : roots) {
      OpBuilder builder(root->loop_op);

      // Finds chains originating from this root.
      SmallVector<LoopChain> root_chains;
      for (const auto &chain : chains) {
        if (chain.getRoot() == root) {
          root_chains.push_back(chain);
        }
      }

      // Builds new chains.
      bool serialized_root = true;
      SmallVector<affine::AffineForOp> new_loops;
      for (const LoopChain &chain : root_chains) {
        MCTBuilder mct_builder(builder, loc);
        FailureOr<affine::AffineForOp> new_loop = mct_builder.build(chain);
        if (failed(new_loop)) {
          serialized_root = false;
          break;
        }
        new_loops.push_back(*new_loop);
      }

      if (!serialized_root) {
        for (auto it = new_loops.rbegin(); it != new_loops.rend(); ++it) {
          it->erase();
        }
        continue;
      }

      // If the original root loop had results (iter_args), and the new loop
      // has matching results, we must replace the uses of the original
      // results with the new ones. NOTE: This assumes that for a loop
      // defining values, there is a corresponding single chain that produces
      // all the values (or at least the one we process). If a root with
      // results is split into multiple chains, this simple logic might loop
      // over them. However, for a reduction loop that is a single chain, this
      // works.
      if (root->loop_op.getNumResults() > 0 && new_loops.size() == 1 &&
          new_loops.front().getNumResults() == root->loop_op.getNumResults()) {
        root->loop_op.replaceAllUsesWith(new_loops.front().getResults());
      }

      // Erases the original root loop.
      root->loop_op.erase();
    }

    return success();
  }

  void printSALT(SALTNode *node, int depth) {
    for (int i = 0; i < depth; ++i) {
      llvm::errs() << "  ";
    }
    llvm::errs() << "Loop [" << node->lower_bound << "," << node->upper_bound
                 << ") step=" << node->step
                 << " | body_ops=" << node->body_operations.size()
                 << " | children=" << node->children.size() << "\n";
    for (SALTNode *child : node->children) {
      printSALT(child, depth + 1);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::taskflow::createAffineLoopTreeSerializationPass() {
  return std::make_unique<AffineLoopTreeSerializationPass>();
}
