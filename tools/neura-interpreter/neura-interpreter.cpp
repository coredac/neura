#include <iterator>
#include <map>
#include <stdlib.h>
#include <string>
#include <unordered_map>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <vector>

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// Predicated data structure that stores scalar/vector values and related
// metadata.
struct PredicatedData {
  /* Scalar floating-point value (valid when is_vector is false). */
  float value;
  /* Validity flag: true means the value is valid, false means it should be
   * ignored. */
  bool predicate;
  /* Indicates if it's a vector: true for vector, false for scalar. */
  bool is_vector;
  /* Vector data (valid when is_vector is true). */
  std::vector<float> vector_data;
  /* Reserve flag (may be used for memory reservation or temporary storage
   * marking). */
  bool is_reserve;
  /* Update flag (indicates whether the data has been modified). */
  bool is_updated;

  /**
   * @brief Constructs a new PredicatedData object with default values.
   *
   * Initializes all member variables to their default states: scalar value
   * 0.0f, valid predicate, scalar type, empty vector data, and flags set to
   * false.
   */
  PredicatedData()
      : value{0.0f}, predicate{true}, is_vector{false}, vector_data{},
        is_reserve{false}, is_updated{false} {}

  /**
   * @brief Compares this PredicatedData instance with another to check for
   * updates.
   *
   * Determines whether any part of the current PredicatedData differs from
   * another instance. Compares scalar values, predicate flags, vector flags,
   * and vector contents (if applicable). Useful in dataflow analysis to
   * determine if a value change should trigger downstream updates.
   *
   * @param other   The PredicatedData instance to compare against.
   * @return bool   True if any field differs (indicating an update); false if
   * all fields are equal.
   */
  bool isUpdatedComparedTo(const PredicatedData &other) const;
};

bool PredicatedData::isUpdatedComparedTo(const PredicatedData &other) const {
  if (value != other.value) {
    return true;
  }
  if (predicate != other.predicate) {
    return true;
  }
  if (is_vector != other.is_vector) {
    return true;
  }
  if (is_vector && vector_data != other.vector_data) {
    return true;
  }
  return false;
}

/**
 * @brief Structure that holds operation handling results and control flow
 * information.
 */
struct OperationHandleResult {
  /* Indicates if the operation was processed successfully. */
  bool success;
  /* Indicates if execution should terminate (e.g., after return operations).
   */
  bool is_terminated;
  /* Indicates if the operation is a branch (requires special index handling).
   */
  bool is_branch;
};

/**< Tracks which kernels have exhausted their counters. */
static llvm::DenseSet<Operation *> exhausted_kernels;

/**
 * @brief Handler for neura.load_indexed.
 *
 * Simulates an indexed memory load. Since the interpreter doesn't have real
 * memory, we use a simplified flat memory model keyed by a string combining
 * the lhs_value/rhs_value attribute and the index values.
 */
static std::unordered_map<std::string, float> simulated_memory;

/**
 * @brief Extract the memref SSA value from a neura.load_indexed or
 * neura.store_indexed operation.
 *
 * Prefers the lhs_value / rhs_value %inputN attribute (set by
 * FoldConstantPass), and falls back to the memref operand of the op (first
 * operand for load_indexed, the target memref for store_indexed).  Block
 * arguments are resolved through resolveKernelBlockArg so that the same
 * underlying memref SSA value is returned across kernels.
 *
 * Look up the memref SSA value that corresponds to a given %inputN string
 * within the enclosing neura::KernelOp. The %inputN convention (set by
 * PromoteInputArgToConstPass) is: N is a 0-based index into ALL kernel
 * inputs (not just memref-typed ones). This function returns the Value at
 * that index only if it is a memref; otherwise it returns Value().
 *
 * @param op         The load/store op to resolve the memref for.
 * @param input_str  The %inputN string to look up.
 * @return The memref SSA Value, or Value() if not found.
 */
static Value getMemRefForInputN(Operation *op, const std::string &input_str) {
  // Parse the number from "%inputN"
  if (input_str.size() < 7 || input_str.substr(0, 6) != "%input")
    return Value();
  int n = 0;
  try {
    n = std::stoi(input_str.substr(6));
  } catch (...) {
    return Value();
  }

  // Walk up to enclosing kernel
  Operation *parent = op->getParentOp();
  while (parent && !isa<neura::KernelOp>(parent))
    parent = parent->getParentOp();
  if (!parent)
    return Value();
  auto kernel = cast<neura::KernelOp>(parent);

  // Use 0-based indexing into ALL kernel inputs (matching generator
  // convention).
  auto inputs = kernel.getInputs();
  if (n >= 0 && n < static_cast<int>(inputs.size())) {
    Value input = *(inputs.begin() + n);
    if (isa<MemRefType>(input.getType()))
      return input;
  }
  return Value();
}

static constexpr const char *INPUT_PREFIX = "%input";
static constexpr size_t INPUT_PREFIX_LEN = 6;
static constexpr size_t INPUT_MIN_LEN = 7; // prefix + at least 1 digit

/**
 * @brief If a Value is a block argument inside a neura::KernelOp, resolve
 * it to the corresponding kernel input operand (the SSA value that was
 * passed into the kernel).
 *
 * Returns the input as-is for all other Values. In dataflow mode, kernel
 * body ops are flattened and block arguments have no producing op, so direct
 * lookup in value_to_predicated_data_map fails. This helper lets handlers
 * resolve the block arg to the kernel's input value which IS tracked in the
 * map.
 *
 * @param val  The Value to resolve.
 * @return The resolved kernel input operand, or @p val unchanged.
 */
static Value resolveKernelBlockArg(Value val) {
  // Handle neura.data_mov wrappers (inserted by InsertDataMovPass) that
  // wrap the actual memref value — just follow through.
  if (auto dm_op = val.getDefiningOp<neura::DataMovOp>()) {
    Value operand = dm_op.getOperand();
    if (operand)
      return resolveKernelBlockArg(operand);
  }
  // Handle neura.constant wrappers (e.g., {value = "%inputN"})
  // that wrap kernel input references in DF-lowered IR.
  if (auto const_op = val.getDefiningOp<neura::ConstantOp>()) {
    if (auto attr = const_op->getAttrOfType<mlir::StringAttr>("value")) {
      std::string ref = attr.getValue().str();
      // Resolve %inputN → get the N-th kernel input
      if (ref.size() >= INPUT_MIN_LEN &&
          ref.substr(0, INPUT_PREFIX_LEN) == INPUT_PREFIX) {
        Operation *parent = const_op->getParentOp();
        while (parent && !isa<neura::KernelOp>(parent))
          parent = parent->getParentOp();
        if (auto kernel = dyn_cast_or_null<neura::KernelOp>(parent)) {
          int n = std::stoi(ref.substr(INPUT_PREFIX_LEN));
          auto inputs = kernel.getInputs();
          if (n >= 0 && n < static_cast<int>(inputs.size())) {
            Value resolved = *(inputs.begin() + n);
            // Recurse to handle block args
            return resolveKernelBlockArg(resolved);
          }
        }
      }
    }
  }

  auto block_arg = dyn_cast<BlockArgument>(val);
  if (!block_arg)
    return val;
  Block *block = block_arg.getOwner();
  if (!block)
    return val;
  Operation *parent = block->getParentOp();
  if (!parent || !isa<neura::KernelOp>(parent))
    return val;
  auto kernel = cast<neura::KernelOp>(parent);
  auto inputs = kernel.getInputs();
  unsigned arg_num = block_arg.getArgNumber();
  if (arg_num < inputs.size())
    return inputs[arg_num];
  return val;
}

/**
 * @brief Extracts the memref SSA value from a neura.load_indexed or
 * neura.store_indexed operation.
 *
 * First attempts attribute-based resolution via the lhs_value / rhs_value
 * string attributes (which name the input memref), falling back to the
 * operation's getBase() operand. The resolved value is passed through
 * resolveKernelBlockArg() to unwrap indirect block-argument references.
 *
 * @param op  A neura::LoadIndexedOp or neura::StoreIndexedOp operation.
 * @return The resolved memref SSA Value, or a null Value on failure.
 */
static Value getMemRefFromLoadStoreOp(Operation *op) {
  // Try attribute-based resolution first.
  std::string attr_str;
  if (auto load_op = dyn_cast<neura::LoadIndexedOp>(op)) {
    if (auto attr = load_op->getAttrOfType<mlir::StringAttr>("lhs_value"))
      attr_str = attr.getValue().str();
    else if (auto attr = load_op->getAttrOfType<mlir::StringAttr>("rhs_value"))
      attr_str = attr.getValue().str();
  } else if (auto store_op = dyn_cast<neura::StoreIndexedOp>(op)) {
    if (auto attr = store_op->getAttrOfType<mlir::StringAttr>("rhs_value"))
      attr_str = attr.getValue().str();
    else if (auto attr = store_op->getAttrOfType<mlir::StringAttr>("lhs_value"))
      attr_str = attr.getValue().str();
  }

  if (!attr_str.empty()) {
    Value memref_val = getMemRefForInputN(op, attr_str);
    if (memref_val)
      return resolveKernelBlockArg(memref_val);
  }

  // Fallback: use the memref operand directly
  // For both load_indexed and store_indexed, getBase() returns the memref.

  if (auto load_op = dyn_cast<neura::LoadIndexedOp>(op)) {
    Value base = load_op.getBase();
    if (base)
      return resolveKernelBlockArg(base);
  }

  if (auto store_op = dyn_cast<neura::StoreIndexedOp>(op)) {
    Value base = store_op.getBase();
    if (base)
      return resolveKernelBlockArg(base);
  }

  return Value();
}

/**
 * @brief Structure that represents the dependency graph of operations.
 */
struct DependencyGraph {
  // Tracks the number of pending producer operations that each consumer
  // operation is waiting for.
  llvm::DenseMap<Operation *, int> consumer_pending_producers;
  // Records operations that have been executed.
  llvm::DenseSet<Operation *> executed_ops;
  // Stores the initial dependency counts for resetting purposes.
  llvm::DenseMap<Operation *, int> initial_dependency_counts;
  // Memory dependency tracking: maps a producer operation to the consumer
  // operations that have a memory dependency on it (same memref RAW/WAW/WAR).
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> memory_dependents;

  /**
   * @brief Builds a dependency graph for operations, calculating the initial
   * count of producer operations that each consumer operation depends on.
   * This count represents how many preceding producer operations within the
   * same sequence the current consumer operation relies on.
   *
   * @param op_sequence The sequence of operations for which to build the
   * dependency graph.
   */
  void build(const std::vector<Operation *> &op_sequence);

  /**
   * @brief Determines if an operation can be executed based on its count of
   * pending producers. An operation is executable when it has zero pending
   * producers (all dependencies have been satisfied) and it hasn't been
   * executed yet.
   *
   * @param op The operation to check for execution eligibility.
   * @return true if the operation can be executed; false otherwise.
   */
  bool canExecute(Operation *op);

  /**
   * @brief Updates the dependency graph after an operation has been executed.
   * This involves marking the operation as executed and decrementing the
   * pending producer count for all consumer operations that depend on it.
   *
   * @param executed_op The operation that has been executed (acts as a
   * producer).
   */
  void updateAfterExecution(Operation *executed_op);

  /**
   * @brief Retrieves a list of operations that are ready to be executed,
   * meaning they have no pending producers and have not yet been executed.
   *
   * @return A vector of operations that can be executed.
   */
  std::vector<Operation *> getReadyToExecuteOperations();

  /**
   * @brief Retrieves a list of operations that depend on a specific
   * operation.
   *
   * @param op The operation to check for dependents.
   * @return A vector of operations that depend on the specified operation.
   */
  std::vector<Operation *> getReadyToExecuteConsumerOperations(Operation *op);

  /**
   * @brief Resets the dependency graph for the next iteration of execution.
   */
  void resetForNextIteration();

  /**
   * @brief Checks if there are any unexecuted operations in the graph.
   *
   * @return true if there are unexecuted operations; false otherwise.
   */
  bool hasUnexecutedOperations();

  /**
   * @brief Gets all operations in the dependency graph.
   *
   * @return A vector of all operations.
   */
  std::vector<Operation *> getAllOperations();

  /**
   * @brief Marks an operation as executed.
   *
   * @param op The operation to mark as executed.
   */
  void markAsExecuted(Operation *op);
};

void DependencyGraph::build(const std::vector<Operation *> &op_sequence) {
  // First pass: build SSA use-def dependencies.
  for (Operation *consumer_op : op_sequence) {
    int required_producers = 0;
    // Counts how many producer operations this consumer depends on.
    for (Value operand : consumer_op->getOperands()) {
      if (Operation *producer_op = operand.getDefiningOp()) {
        // Counts only producers within the same operation sequence.
        if (std::find(op_sequence.begin(), op_sequence.end(), producer_op) !=
            op_sequence.end()) {
          required_producers++;
        }
      }
    }
    consumer_pending_producers[consumer_op] = required_producers;
    initial_dependency_counts[consumer_op] = required_producers;
  }

  // Second pass: add memory dependencies for load_indexed / store_indexed
  // on the same memref. The DF interpreter flattens all kernel body ops into
  // a single sequence, but store_indexed has no SSA result, so a subsequent
  // load_indexed on the same memref has no SSA edge to the store.
  // Likewise, consecutive stores to the same memref and loads followed by
  // stores all need ordering to preserve RAW, WAW and WAR semantics.
  //
  // We add a dependency link from the last operation that touched a memref
  // to the current one, creating a per-memref total order that is
  // conservative (even index-collisions are serialised) but sound.
  llvm::DenseMap<Value, Operation *> memref_last_op;

  for (Operation *op : op_sequence) {
    if (!isa<neura::LoadIndexedOp>(op) && !isa<neura::StoreIndexedOp>(op))
      continue;

    Value memref_val = getMemRefFromLoadStoreOp(op);
    if (!memref_val)
      continue;

    auto it = memref_last_op.find(memref_val);
    if (it != memref_last_op.end()) {
      Operation *prev_op = it->second;
      // Current op depends on the previous op that touched this memref.
      consumer_pending_producers[op]++;
      initial_dependency_counts[op]++;
      // Record reverse edge so updateAfterExecution can find it.
      memory_dependents[prev_op].push_back(op);
    }

    memref_last_op[memref_val] = op;
  }
}

bool DependencyGraph::canExecute(Operation *op) {
  auto it = consumer_pending_producers.find(op);
  // Operation is executable if:
  // 1. It exists in the dependency graph.
  // 2. It has no pending producers (all dependencies satisfied).
  return it != consumer_pending_producers.end() && it->second == 0;
}

void DependencyGraph::updateAfterExecution(Operation *executed_op) {
  // Marks the completed operation as executed (it acts as a producer).
  executed_ops.insert(executed_op);

  // Updates all SSA consumer operations that depend on this producer.
  for (Value result : executed_op->getResults()) {
    for (Operation *consumer_op : result.getUsers()) {
      auto it = consumer_pending_producers.find(consumer_op);
      // Decrements pending producer count for valid consumers.
      if (it != consumer_pending_producers.end() && it->second > 0) {
        it->second--;
      }
    }
  }

  // Updates all memory-dependent consumer operations.
  auto mem_it = memory_dependents.find(executed_op);
  if (mem_it != memory_dependents.end()) {
    for (Operation *consumer_op : mem_it->second) {
      auto it = consumer_pending_producers.find(consumer_op);
      if (it != consumer_pending_producers.end() && it->second > 0) {
        it->second--;
      }
    }
  }
}

std::vector<Operation *> DependencyGraph::getReadyToExecuteOperations() {
  std::vector<Operation *> executable_ops;
  for (const auto &entry : this->consumer_pending_producers) {
    if (entry.second == 0 && !this->executed_ops.count(entry.first)) {
      executable_ops.push_back(entry.first);
    }
  }
  return executable_ops;
}

std::vector<Operation *>
DependencyGraph::getReadyToExecuteConsumerOperations(Operation *op) {
  std::vector<Operation *> dependent_ops;
  // Collect SSA consumers.
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (this->consumer_pending_producers.count(user) &&
          !this->executed_ops.count(user)) {
        dependent_ops.push_back(user);
      }
    }
  }

  if (neura::CtrlMovOp ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(op)) {
    Value target = ctrl_mov_op.getTarget();
    for (Operation *user : target.getUsers()) {
      if (user != ctrl_mov_op && this->consumer_pending_producers.count(user) &&
          !this->executed_ops.count(user)) {
        dependent_ops.push_back(user);
      }
    }
  }

  // Collect memory-dependent consumers (load_indexed / store_indexed on the
  // same memref).
  auto mem_it = memory_dependents.find(op);
  if (mem_it != memory_dependents.end()) {
    for (Operation *consumer : mem_it->second) {
      if (this->consumer_pending_producers.count(consumer) &&
          !this->executed_ops.count(consumer)) {
        dependent_ops.push_back(consumer);
      }
    }
  }

  return dependent_ops;
}

void DependencyGraph::resetForNextIteration() {
  this->executed_ops.clear();
  for (const auto &entry : initial_dependency_counts) {
    consumer_pending_producers[entry.first] = entry.second;
  }
}

bool DependencyGraph::hasUnexecutedOperations() {
  return this->executed_ops.size() < this->consumer_pending_producers.size();
}

std::vector<Operation *> DependencyGraph::getAllOperations() {
  std::vector<Operation *> all_ops;
  for (const auto &entry : consumer_pending_producers) {
    all_ops.push_back(entry.first);
  }
  return all_ops;
}

void DependencyGraph::markAsExecuted(Operation *op) {
  if (executed_ops.insert(op).second) {
    // Operation was newly inserted, decrement pending count for all SSA
    // dependents.
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (consumer_pending_producers.count(user)) {
          consumer_pending_producers[user]--;
        }
      }
    }
    // Also decrement for memory-dependent consumers.
    auto mem_it = memory_dependents.find(op);
    if (mem_it != memory_dependents.end()) {
      for (Operation *consumer : mem_it->second) {
        if (consumer_pending_producers.count(consumer)) {
          consumer_pending_producers[consumer]--;
        }
      }
    }
  }
}

static llvm::SmallVector<Operation *, 16>
    pending_operation_queue; /* List of operations to execute in current
                                iteration. */
static llvm::DenseMap<Operation *, bool>
    is_operation_enqueued; /* Marks whether an operation is already in
                              pending_operation_queue. */

static bool verbose = false;  /* Verbose logging mode switch: outputs debug
                                 information when true. */
static bool dataflow = false; /* Dataflow analysis mode switch: enables
                                 dataflow-related analysis logic when true. */
inline void setDataflowMode(bool v) { dataflow = v; }

inline bool isDataflowMode() { return dataflow; }

inline void setVerboseMode(bool v) { verbose = v; }

inline bool isVerboseMode() { return verbose; }

/**
 * @brief Global mapping from memref SSA Value to a unique memref ID.
 *
 * Ensures the same physical memref (same SSA value) gets the same key prefix
 * across all kernels that reference it, preserving aliasing.
 */
static llvm::DenseMap<Value, int> memref_value_to_id;
static int next_memref_id = 0; ///< monotonically increasing ID counter

/**
 * @brief Assign a unique global ID to a memref SSA Value if not already
 * mapped.
 *
 * @param memref_val  The memref SSA Value to look up or register.
 * @return The assigned unique ID.
 */
static int getOrAssignMemRefId(Value memref_val) {
  if (memref_value_to_id.count(memref_val))
    return memref_value_to_id[memref_val];
  int id = next_memref_id++;
  memref_value_to_id[memref_val] = id;
  return id;
}

/**
 * @brief Per-counter storage for neura.counter, keyed by SSA result value.
 *
 * Bounds and step are read from op attributes and initialized on first
 * encounter. Each counter's current index is stored in counter_state;
 * loop bounds in counter_lower / counter_upper; traversal direction in
 * counter_step.
 */
static llvm::DenseMap<Value, int64_t> counter_state; ///< current index
static llvm::DenseMap<Value, int64_t> counter_lower; ///< lower bound
static llvm::DenseMap<Value, int64_t> counter_upper; ///< upper bound
static llvm::DenseMap<Value, int64_t> counter_step;  ///< step (direction)

/** Counter hierarchy level, keyed by SSA result value:
 *  "root" (outermost) = 0, "relay" = 1, "leaf" (innermost) = 2. */
static llvm::DenseMap<Value, int> counter_hierarchy_level;

/**
 * @brief Handles the execution of an arithmetic constant operation
 * (arith.constant) by parsing its value and storing it in the value map.
 *
 * This function processes MLIR's arith.constant operations, which represent
 * constant values. It extracts the constant value from the operation's
 * attribute, converts it to a floating-point representation (supporting
 * floats, integers, and booleans), and stores it in the value map with a
 * predicate set to true (since constants are always valid). Unsupported
 * constant types result in an error.
 *
 * @param op                             The arith.constant operation to
 * handle
 * @param value_to_predicated_data_map   Reference to the map where the parsed
 * constant value will be stored, keyed by the operation's result value
 * @return bool                          True if the constant is successfully
 * parsed and stored; false if the constant type is unsupported
 */
bool handleArithConstantOp(
    mlir::arith::ConstantOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  auto attr = op.getValue();
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing arith.constant:\n";
  }

  PredicatedData val;

  // Handles floating-point constants (convert to double-precision float).
  if (auto float_attr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    val.value = float_attr.getValueAsDouble();
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Parsed float constant : "
                   << llvm::format("%.6f", val.value) << "\n";
    }
  }
  // Handles integer constants (including booleans, which are 1-bit integers).
  else if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    if (int_attr.getType().isInteger(1)) {
      val.value = int_attr.getInt() != 0 ? 1.0f : 0.0f;
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  └─ Parsed boolean constant : "
                     << (val.value ? "true" : "false") << "\n";
      }
    } else {
      val.value = static_cast<float>(int_attr.getInt());
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  └─ Parsed integer constant : "
                     << llvm::format("%.6f", val.value) << "\n";
      }
    }
  }
  // Handles unsupported constant types.
  else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported constant type in "
                      "arith.constant\n";
    }
    return false;
  }

  // In dataflow mode, constant ops may be re-executed across DFG iterations.
  // If the value already exists, just skip (idempotent).
  if (value_to_predicated_data_map.count(op.getResult()) == 0) {
    value_to_predicated_data_map[op.getResult()] = val;
  }
  return true;
}

/**
 * @brief Handles the execution of a Neura constant operation (neura.constant)
 * by parsing its value (scalar or vector) and storing it in the value map.
 *
 * This function processes Neura's custom constant operations, which can
 * represent floating-point scalars, integer scalars, or floating-point
 * vectors. It extracts the constant value from the operation's attribute,
 * converts it to the appropriate format, and stores it in the value map. The
 * predicate for the constant can be explicitly set via an attribute
 * (defaulting to true if not specified). Unsupported types or vector element
 * types result in an error.
 *
 * @param op                             The neura.constant operation to
 * handle
 * @param value_to_predicated_data_map   Reference to the map where the parsed
 * constant value will be stored, keyed by the operation's result value
 * @return bool                          True if the constant is successfully
 * parsed and stored; false if the constant type or vector element type is
 * unsupported
 */
bool handleNeuraConstantOp(
    neura::ConstantOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  auto attr = op.getValue();

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.constant:\n";
  }
  // Handles floating-point scalar constants.
  if (auto float_attr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    PredicatedData val;
    val.value = float_attr.getValueAsDouble();
    val.predicate = true;
    val.is_vector = false;

    if (auto pred_attr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = pred_attr.getValue();
    }

    if (value_to_predicated_data_map.count(op.getResult()) == 0) {
      value_to_predicated_data_map[op.getResult()] = val;
    }
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Constant  : value = "
                   << val.value << " [pred = " << val.predicate << "]\n";
    }
  }
  // Handles integer scalar constants.
  else if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    PredicatedData val;
    val.value = static_cast<float>(int_attr.getInt());
    val.predicate = true;
    val.is_vector = false;

    if (auto pred_attr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = pred_attr.getValue();
    }

    if (value_to_predicated_data_map.count(op.getResult()) == 0) {
      value_to_predicated_data_map[op.getResult()] = val;
    }
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Constant  : value = "
                   << val.value << " [pred = " << val.predicate << "]\n";
    }
  }
  // Handles vector constants (dense element attributes).
  else if (auto dense_attr = llvm::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!dense_attr.getElementType().isF32()) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported vector element "
                        "type in neura.constant\n";
      }
      return false;
    }

    PredicatedData val;
    val.is_vector = true;
    val.predicate = true;

    size_t vector_size = dense_attr.getNumElements();
    val.vector_data.resize(vector_size);

    auto float_values = dense_attr.getValues<float>();
    std::copy(float_values.begin(), float_values.end(),
              val.vector_data.begin());

    if (auto pred_attr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = pred_attr.getValue();
    }

    if (value_to_predicated_data_map.count(op.getResult()) == 0) {
      value_to_predicated_data_map[op.getResult()] = val;
    }

    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Constant  : pred = "
                   << val.predicate << "]\n";
      llvm::outs() << "[neura-interpreter]  └─ Parsed vector constant of size: "
                   << vector_size << "\n";
    }
  }
  // Handles unsupported constant types.
  else if (auto str_attr = llvm::dyn_cast<mlir::StringAttr>(attr)) {
    // String constants are Neura input references (e.g. "%input0").
    // Resolve the reference to the enclosing kernel's argument value.
    std::string ref_str = str_attr.getValue().str();
    PredicatedData val;
    val.value = 0.0f;
    val.predicate = true; // the referenced kernel arg is always valid
    val.is_vector = false;

    // Try to resolve %inputN → kernel argument number N (0-based,
    // counting ALL kernel inputs, matching PromoteInputArgToConstPass).
    if (ref_str.rfind("%input", 0) == 0) { // starts with "%input"
      int n = 0;
      try {
        n = std::stoi(ref_str.substr(6));
      } catch (...) {
        n = -1;
      }
      if (n >= 0) {
        // Walk up to enclosing kernel.
        Operation *parent = op->getParentOp();
        while (parent && !isa<neura::KernelOp>(parent))
          parent = parent->getParentOp();
        if (parent) {
          auto kernel = cast<neura::KernelOp>(parent);
          auto inputs = kernel.getInputs();
          if (n < static_cast<int>(inputs.size())) {
            Value arg_val = *(inputs.begin() + n);
            if (value_to_predicated_data_map.count(arg_val))
              val = value_to_predicated_data_map[arg_val];
          }
        }
      }
    }

    value_to_predicated_data_map[op.getResult()] = val;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Input ref : "
                   << str_attr.getValue() << " → value = " << val.value
                   << " [pred = " << val.predicate << "]\n";
    }
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported constant type in "
                      "neura.constant\n";
    }
    return false;
  }
  return true;
}

/**
 * @brief Handles the execution of an arithmetic constant operation
 * (arith.constant) by parsing its value and storing it in the value map.
 *
 * This function processes MLIR's arith.constant operations, which represent
 * constant values. It extracts the constant value from the operation's
 * attribute, converts it to a floating-point representation (supporting
 * floats, integers, and booleans), and stores it in the value map with a
 * predicate set to true (since constants are always valid). Unsupported
 * constant types result in an error.
 *
 * @param op                             The arith.constant operation to
 * handle
 * @param value_to_predicated_data_map   Reference to the map where the parsed
 * constant value will be stored, keyed by the operation's result value
 * @return bool                          True if the constant is successfully
 * parsed and stored; false if the constant type is unsupported
 */
bool handleAddOp(
    neura::AddOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.add:\n";
  }

  if (op.getNumOperands() < 1) {
    return false;
  }

  auto lhs = value_to_predicated_data_map[op.getLhs()];
  int64_t lhs_int = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhs_int;
  bool final_predicate = lhs.predicate;

  if (op.getNumOperands() >= 2 && op.getRhs()) {
    auto rhs = value_to_predicated_data_map[op.getRhs()];
    rhs_int = static_cast<int64_t>(std::round(rhs.value));
    final_predicate = final_predicate && rhs.predicate;
  } else if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("rhs_value")) {
    rhs_int = attr.getInt();
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.add: no rhs operand or "
                      "rhs_value attribute\n";
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : " << lhs_int
                 << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : " << rhs_int << "\n";
  }

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
  }

  int64_t sum = lhs_int + rhs_int;

  PredicatedData result;
  result.value = static_cast<float>(sum);
  result.predicate = final_predicate;
  result.is_vector = false;

  value_to_predicated_data_map[op.getResult()] = result;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura subtraction operation (neura.sub)
 * by computing the difference of integer operands.
 *
 * This function processes Neura's subtraction operations, which take 2-3
 * operands: two integer inputs (LHS and RHS) and an optional predicate
 * operand. It computes the difference of the integer values (LHS - RHS),
 * combines the predicates of all operands (including the optional predicate
 * if present), and stores the result in the value map. The operation requires
 * at least two operands; fewer will result in an error.
 *
 * @param op                             The neura.sub operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the subtraction is
 * successfully computed; false if there are fewer than 2 operands
 */
bool handleSubOp(
    neura::SubOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.sub:\n";
  }

  if (op.getNumOperands() < 1) {
    return false;
  }

  auto lhs = value_to_predicated_data_map[op.getOperand(0)];
  int64_t lhs_int = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhs_int;
  bool final_predicate = lhs.predicate;

  if (op.getNumOperands() >= 2 && op.getRhs()) {
    auto rhs = value_to_predicated_data_map[op.getOperand(1)];
    rhs_int = static_cast<int64_t>(std::round(rhs.value));
    final_predicate = final_predicate && rhs.predicate;
  } else if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("rhs_value")) {
    rhs_int = attr.getInt();
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.sub: no rhs operand or "
                      "rhs_value attribute\n";
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : " << lhs_int
                 << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : " << rhs_int << "\n";
  }

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
  }

  int64_t result_int = lhs_int - rhs_int;

  PredicatedData result;
  result.value = static_cast<float>(result_int);
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura integer multiplication operation
 *        (neura.mul) by computing the product of integer operands.
 *
 * Supports both binary (lhs * rhs) and unary-with-attribute forms:
 *   - %r = "neura.mul"(%a, %b) : ...         → lhs * rhs
 *   - %r = "neura.mul"(%a) {rhs_value = 2}   → lhs * rhs_value
 */
bool handleMulOp(
    neura::MulOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.mul:\n";
  }

  if (op.getNumOperands() < 1) {
    return false;
  }

  auto lhs = value_to_predicated_data_map[op.getLhs()];
  int64_t lhs_int = static_cast<int64_t>(std::round(lhs.value));

  int64_t rhs_int;
  if (op.getNumOperands() >= 2 && op.getRhs()) {
    auto rhs = value_to_predicated_data_map[op.getRhs()];
    rhs_int = static_cast<int64_t>(std::round(rhs.value));
  } else if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("rhs_value")) {
    rhs_int = attr.getInt();
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.mul: no rhs operand or "
                      "rhs_value attribute\n";
    return false;
  }

  bool final_predicate = lhs.predicate;
  if (op.getNumOperands() >= 2 && op.getRhs()) {
    auto rhs = value_to_predicated_data_map[op.getRhs()];
    final_predicate = final_predicate && rhs.predicate;
  }

  int64_t product = lhs_int * rhs_int;

  PredicatedData result;
  result.value = static_cast<float>(product);
  result.predicate = final_predicate;
  result.is_vector = false;

  value_to_predicated_data_map[op.getResult()] = result;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands\n"
                 << "[neura-interpreter]  │  ├─ LHS  : " << lhs_int
                 << " [pred = " << lhs.predicate << "]\n"
                 << "[neura-interpreter]  │  └─ RHS  : " << rhs_int << "\n"
                 << "[neura-interpreter]  └─ Result: " << product
                 << " [pred = " << final_predicate << "]\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura integer division operation
 *        (neura.div) by computing the quotient of integer operands.
 */
bool handleDivOp(
    neura::DivOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.div:\n";
  }

  if (op.getNumOperands() < 1) {
    return false;
  }

  auto lhs = value_to_predicated_data_map[op.getLhs()];
  int64_t lhs_int = static_cast<int64_t>(std::round(lhs.value));

  int64_t rhs_int;
  if (op.getNumOperands() >= 2 && op.getRhs()) {
    auto rhs = value_to_predicated_data_map[op.getRhs()];
    rhs_int = static_cast<int64_t>(std::round(rhs.value));
  } else if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("rhs_value")) {
    rhs_int = attr.getInt();
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.div: no rhs operand or "
                      "rhs_value attribute\n";
    return false;
  }

  if (rhs_int == 0) {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.div: division by zero\n";
    return false;
  }

  bool final_predicate = lhs.predicate;
  if (op.getNumOperands() >= 2 && op.getRhs()) {
    auto rhs = value_to_predicated_data_map[op.getRhs()];
    final_predicate = final_predicate && rhs.predicate;
  }

  int64_t quotient = lhs_int / rhs_int;

  PredicatedData result;
  result.value = static_cast<float>(quotient);
  result.predicate = final_predicate;
  result.is_vector = false;

  value_to_predicated_data_map[op.getResult()] = result;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result: " << quotient
                 << " [pred = " << final_predicate << "]\n";
  }

  return true;
}

/**
 * @brief Resolves an operand value from either the operand list or a folded
 *        attribute (lhs_value / rhs_value).
 *
 * When FoldConstantPass or PromoteInputArgToConstPass folds an operand into an
 * attribute, binary ops like neura.fadd / neura.fdiv may have only one operand,
 * with the other stored in the \"rhs_value\" or \"lhs_value\" attribute.
 * This attribute can be:
 *   - FloatAttr:  a literal constant (e.g. 2.0 : f32)
 *   - StringAttr: a kernel input reference (e.g. \"%input1\") that must be
 *                 resolved by walking up to the enclosing neura::KernelOp
 *                 and looking up the corresponding argument value.
 *
 * @param op        The operation that may carry the attribute
 * @param attrName  Attribute name (\"rhs_value\" or \"lhs_value\")
 * @param value_map The interpreter's value-to-data map
 * @return std::optional<PredicatedData>  Resolved value, or std::nullopt if
 *         neither a matching operand nor a resolvable attribute exists
 */
static std::optional<PredicatedData>
resolveAttributeValue(Operation *op, const char *attrName,
                      llvm::DenseMap<Value, PredicatedData> &value_map) {
  // --- StringAttr: kernel input reference, e.g. "%input1" ---------------
  if (auto str_attr = op->getAttrOfType<mlir::StringAttr>(attrName)) {
    std::string ref = str_attr.getValue().str();
    if (ref.rfind("%input", 0) == 0) {
      int n = 0;
      try {
        n = std::stoi(ref.substr(6));
      } catch (...) {
        n = -1;
      }
      if (n >= 0) {
        Operation *parent = op->getParentOp();
        while (parent && !isa<neura::KernelOp>(parent))
          parent = parent->getParentOp();
        if (auto kernel = dyn_cast_or_null<neura::KernelOp>(parent)) {
          auto inputs = kernel.getInputs();
          if (n < static_cast<int>(inputs.size())) {
            Value arg_val = *(inputs.begin() + n);
            if (value_map.count(arg_val))
              return value_map[arg_val];
          }
        }
      }
    }
    return PredicatedData{}; // unresolved reference → default (0.0, true)
  }

  // --- FloatAttr: literal constant, e.g. 2.500000e+00 : f32 ------------
  if (auto float_attr = op->getAttrOfType<mlir::FloatAttr>(attrName)) {
    PredicatedData val;
    val.value = static_cast<float>(float_attr.getValueAsDouble());
    val.predicate = true;
    val.is_vector = false;
    return val;
  }

  return std::nullopt;
}

/**
 * @brief Handles the execution of a Neura floating-point addition operation
 *        (neura.fadd) by computing the sum of floating-point operands.
 *
 * This function processes Neura's floating-point addition operations, which
 * take 2-3 operands: two floating-point inputs (LHS and RHS) and an optional
 * predicate operand. It computes the sum of the floating-point values,
 * combines the predicates of all operands (including the optional predicate
 * if present), and stores the result in the value map. The operation requires
 * at least two operands; fewer will result in an error.
 *
 * @param op                             The neura.fadd operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the floating-point addition
 * is successfully computed; false if there are fewer than 2 operands
 */
bool handleFAddOp(
    neura::FAddOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fadd:\n";
  }

  PredicatedData lhs, rhs;

  // Resolve LHS: prefer operand, fall back to lhs_value attribute.
  if (op.getNumOperands() >= 1 && op.getLhs()) {
    lhs = value_to_predicated_data_map[op.getLhs()];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "lhs_value",
                                       value_to_predicated_data_map)) {
    lhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fadd: no LHS operand "
                      "or lhs_value attribute\n";
    return false;
  }

  // Resolve RHS: prefer operand, fall back to rhs_value attribute.
  if (op.getNumOperands() >= 2 && op.getRhs()) {
    rhs = value_to_predicated_data_map[op.getRhs()];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "rhs_value",
                                       value_to_predicated_data_map)) {
    rhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fadd: no RHS operand "
                      "or rhs_value attribute\n";
    return false;
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value
                 << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value
                 << " [pred = " << rhs.predicate << "]\n";
  }

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value
                   << " [pred = " << pred.predicate << "]\n";
    }
  }

  PredicatedData result;
  result.value = lhs.value + rhs.value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point subtraction
 * operation (neura.fsub) by computing the difference of floating-point
 * operands.
 *
 * This function processes Neura's floating-point subtraction operations,
 * which take 2-3 operands: two floating-point inputs (LHS and RHS) and an
 * optional predicate operand. It calculates the difference of the
 * floating-point values (LHS - RHS), combines the predicates of all operands
 * (including the optional predicate if present), and stores the result in the
 * value map. The operation requires at least two operands; fewer will result
 * in an error.
 *
 * @param op                             The neura.fsub operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the floating-point
 * subtraction is successfully computed; false if there are fewer than 2
 * operands
 */
bool handleFSubOp(
    neura::FSubOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fsub:\n";
  }

  PredicatedData lhs, rhs;

  // Resolve LHS: prefer operand, fall back to lhs_value attribute.
  if (op.getNumOperands() >= 1 && op.getLhs()) {
    lhs = value_to_predicated_data_map[op.getLhs()];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "lhs_value",
                                       value_to_predicated_data_map)) {
    lhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fsub: no LHS operand "
                      "or lhs_value attribute\n";
    return false;
  }

  // Resolve RHS: prefer operand, fall back to rhs_value attribute.
  if (op.getNumOperands() >= 2 && op.getRhs()) {
    rhs = value_to_predicated_data_map[op.getRhs()];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "rhs_value",
                                       value_to_predicated_data_map)) {
    rhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fsub: no RHS operand "
                      "or rhs_value attribute\n";
    return false;
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value
                 << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value
                 << " [pred = " << rhs.predicate << "]\n";
  }

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value
                   << " [pred = " << pred.predicate << "]\n";
    }
  }

  PredicatedData result;
  result.value = lhs.value - rhs.value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point multiplication
 * operation (neura.fmul) by computing the product of floating-point operands.
 *
 * This function processes Neura's floating-point multiplication operations,
 * which take 2-3 operands: two floating-point inputs (LHS and RHS) and an
 * optional predicate operand. It calculates the product of the floating-point
 * values (LHS * RHS), combines the predicates of all operands (including the
 * optional predicate if present), and stores the result in the value map. The
 * operation requires at least two operands; fewer will result in an error.
 *
 * @param op                             The neura.fmul operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the floating-point
 * multiplication is successfully computed; false if there are fewer than 2
 * operands
 */
bool handleFMulOp(
    neura::FMulOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fmul:\n";
  }

  PredicatedData lhs, rhs;

  // Resolve LHS: prefer operand, fall back to lhs_value attribute.
  if (op.getNumOperands() >= 1) {
    lhs = value_to_predicated_data_map[op.getOperand(0)];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "lhs_value",
                                       value_to_predicated_data_map)) {
    lhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fmul: no LHS operand "
                      "or lhs_value attribute\n";
    return false;
  }

  // Resolve RHS: prefer operand, fall back to rhs_value attribute.
  if (op.getNumOperands() >= 2) {
    rhs = value_to_predicated_data_map[op.getOperand(1)];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "rhs_value",
                                       value_to_predicated_data_map)) {
    rhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fmul: no RHS operand "
                      "or rhs_value attribute\n";
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value
                 << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value
                 << " [pred = " << rhs.predicate << "]\n";
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value
                   << " [pred = " << pred.predicate << "]\n";
    }
  }

  float lhs_float = static_cast<float>(lhs.value);
  float rhs_float = static_cast<float>(rhs.value);
  float result_float = lhs_float * rhs_float;

  PredicatedData result;
  result.value = result_float;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point division operation
 *        (neura.fdiv) by computing the quotient of floating-point operands.
 *
 * This function processes Neura's floating-point division operations, which
 * take 2-3 operands: two floating-point inputs (dividend/LHS and divisor/RHS)
 * and an optional predicate operand. It calculates the quotient of the
 * floating-point values (LHS / RHS), handles division by zero by returning
 * NaN, combines the predicates of all operands (including the optional
 * predicate if present), and stores the result in the value map. The
 * operation requires at least two operands; fewer will result in an error.
 *
 * @param op                             The neura.fdiv operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the floating-point division
 * is successfully computed; false if there are fewer than 2 operands
 */
bool handleFDivOp(
    neura::FDivOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fdiv:\n";
  }

  PredicatedData lhs, rhs;

  // Resolve LHS: prefer operand, fall back to lhs_value attribute.
  if (op.getNumOperands() >= 1) {
    lhs = value_to_predicated_data_map[op.getOperand(0)];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "lhs_value",
                                       value_to_predicated_data_map)) {
    lhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fdiv: no LHS operand "
                      "or lhs_value attribute\n";
    return false;
  }

  // Resolve RHS: prefer operand, fall back to rhs_value attribute.
  if (op.getNumOperands() >= 2) {
    rhs = value_to_predicated_data_map[op.getOperand(1)];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "rhs_value",
                                       value_to_predicated_data_map)) {
    rhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fdiv: no RHS operand "
                      "or rhs_value attribute\n";
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value
                 << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value
                 << " [pred = " << rhs.predicate << "]\n";
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value
                   << " [pred = " << pred.predicate << "]\n";
    }
  }

  float result_float = 0.0f;
  float rhs_float = static_cast<float>(rhs.value);

  if (rhs_float == 0.0f) {
    // Returns quiet NaN for division by zero to avoid runtime errors.
    result_float = std::numeric_limits<float>::quiet_NaN();
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Warning: Division by zero, "
                      "result is NaN\n";
    }
  } else {
    float lhs_float = static_cast<float>(lhs.value);
    result_float = lhs_float / rhs_float;
  }

  PredicatedData result;
  result.value = result_float;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point maximum operation
 *        (neura.fmax) by computing the maximum of floating-point operands.
 *
 * This function processes Neura's floating-point maximum operations, which
 * take 2-3 operands: two floating-point inputs (LHS and RHS) and an optional
 * predicate operand. It calculates the maximum of the floating-point values
 * (max(LHS, RHS)), combines the predicates of all operands (including the
 * optional predicate if present), and stores the result in the value map. The
 * operation requires at least two operands; fewer will result in an error.
 *
 * @param op                             The neura.fmax operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the floating-point maximum
 * is successfully computed; false if there are fewer than 2 operands
 */
bool handleFMaxOp(
    neura::FMaxOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fmax:\n";
  }

  PredicatedData lhs, rhs;

  // Resolve LHS: prefer operand, fall back to lhs_value attribute.
  if (op.getNumOperands() >= 1) {
    lhs = value_to_predicated_data_map[op.getOperand(0)];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "lhs_value",
                                       value_to_predicated_data_map)) {
    lhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fmax: no LHS operand "
                      "or lhs_value attribute\n";
    return false;
  }

  // Resolve RHS: prefer operand, fall back to rhs_value attribute.
  if (op.getNumOperands() >= 2) {
    rhs = value_to_predicated_data_map[op.getOperand(1)];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "rhs_value",
                                       value_to_predicated_data_map)) {
    rhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fmax: no RHS operand "
                      "or rhs_value attribute\n";
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value
                 << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value
                 << " [pred = " << rhs.predicate << "]\n";
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value
                   << " [pred = " << pred.predicate << "]\n";
    }
  }

  float lhs_float = static_cast<float>(lhs.value);
  float rhs_float = static_cast<float>(rhs.value);

  // Get NaN semantic attribute (default is "maxnum")
  std::string nan_semantic = op.getNanSemantic().str();
  float result_float;

  if (nan_semantic == "maxnum") {
    // maxnum semantic: return non-NaN value when one operand is NaN
    if (std::isnan(lhs_float) && !std::isnan(rhs_float)) {
      result_float = rhs_float;
    } else if (std::isnan(rhs_float) && !std::isnan(lhs_float)) {
      result_float = lhs_float;
    } else {
      result_float = std::max(lhs_float, rhs_float);
    }
  } else { // "maximum"
    // maximum semantic: propagate NaN when any operand is NaN
    if (std::isnan(lhs_float) || std::isnan(rhs_float)) {
      result_float = std::nan("");
    } else {
      result_float = std::max(lhs_float, rhs_float);
    }
  }

  PredicatedData result;
  result.value = result_float;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ NaN semantic: " << nan_semantic
                 << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point minimum operation
 *        (neura.fmin) by computing the minimum of floating-point operands.
 *
 * This function processes Neura's floating-point minimum operations, which
 * take 2-3 operands: two floating-point inputs (LHS and RHS) and an optional
 * predicate operand. It calculates the minimum of the floating-point values
 * (min(LHS, RHS)), combines the predicates of all operands (including the
 * optional predicate if present), and stores the result in the value map. The
 * operation requires at least two operands; fewer will result in an error.
 *
 * @param op                             The neura.fmin operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the floating-point minimum
 * is successfully computed; false if there are fewer than 2 operands
 */
bool handleFMinOp(
    neura::FMinOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fmin:\n";
  }

  PredicatedData lhs, rhs;

  // Resolve LHS: prefer operand, fall back to lhs_value attribute.
  if (op.getNumOperands() >= 1) {
    lhs = value_to_predicated_data_map[op.getOperand(0)];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "lhs_value",
                                       value_to_predicated_data_map)) {
    lhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fmin: no LHS operand "
                      "or lhs_value attribute\n";
    return false;
  }

  // Resolve RHS: prefer operand, fall back to rhs_value attribute.
  if (op.getNumOperands() >= 2) {
    rhs = value_to_predicated_data_map[op.getOperand(1)];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "rhs_value",
                                       value_to_predicated_data_map)) {
    rhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fmin: no RHS operand "
                      "or rhs_value attribute\n";
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value
                 << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value
                 << " [pred = " << rhs.predicate << "]\n";
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value
                   << " [pred = " << pred.predicate << "]\n";
    }
  }

  float lhs_float = static_cast<float>(lhs.value);
  float rhs_float = static_cast<float>(rhs.value);

  // Get NaN semantic attribute (default is "minnum")
  std::string nan_semantic = op.getNanSemantic().str();
  float result_float;

  if (nan_semantic == "minnum") {
    // minnum semantic: return non-NaN value when one operand is NaN
    if (std::isnan(lhs_float) && !std::isnan(rhs_float)) {
      result_float = rhs_float;
    } else if (std::isnan(rhs_float) && !std::isnan(lhs_float)) {
      result_float = lhs_float;
    } else {
      result_float = std::min(lhs_float, rhs_float);
    }
  } else { // "minimum"
    // minimum semantic: propagate NaN when any operand is NaN
    if (std::isnan(lhs_float) || std::isnan(rhs_float)) {
      result_float = std::nan("");
    } else {
      result_float = std::min(lhs_float, rhs_float);
    }
  }

  PredicatedData result;
  result.value = result_float;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ NaN semantic: " << nan_semantic
                 << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura vector floating-point
 * multiplication operation (neura.vfmul) by computing element-wise products
 * of vector operands.
 *
 * This function processes Neura's vector floating-point multiplication
 * operations, which take 2-3 operands: two vector inputs (LHS and RHS) and an
 * optional scalar predicate operand. It validates that both primary operands
 * are vectors of equal size, computes element-wise products, combines the
 * predicates of all operands (including the optional scalar predicate if
 * present), and stores the resulting vector in the value map. Errors are
 * returned for invalid operand types (non-vectors), size mismatches, or
 * vector predicates.
 *
 * @param op                             The neura.vfmul operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 * resulting vector will be stored, keyed by the operation's result value
 * @return bool                          True if the vector multiplication is
 *                                       successfully computed; false if there
 * are invalid operands, size mismatches, or other errors
 */
bool handleVFMulOp(
    neura::VFMulOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.vfmul:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.vfmul expects at least "
                      "two operands\n";
    }
    return false;
  }

  auto lhs = value_to_predicated_data_map[op.getLhs()];
  auto rhs = value_to_predicated_data_map[op.getRhs()];

  if (!lhs.is_vector || !rhs.is_vector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.vfmul requires both "
                      "operands to be vectors\n";
    }
    return false;
  }

  auto print_vector = [](ArrayRef<float> vec) {
    llvm::outs() << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      llvm::outs() << vec[i];
      if (i != vec.size() - 1) {
        llvm::outs() << ", ";
      }
    }
    llvm::outs() << "]";
  };

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : vector size = "
                 << lhs.vector_data.size() << ", ";
    print_vector(lhs.vector_data);
    llvm::outs() << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : vector size = "
                 << rhs.vector_data.size() << ", ";
    print_vector(rhs.vector_data);
    llvm::outs() << ", [pred = " << rhs.predicate << "]\n";
  }

  if (lhs.vector_data.size() != rhs.vector_data.size()) {
    if (isVerboseMode()) {
      llvm::errs()
          << "[neura-interpreter]  └─ Vector size mismatch in neura.vfmul\n";
    }
    return false;
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    if (pred.is_vector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Predicate operand must be a "
                        "scalar in neura.vfmul\n";
      }
      return false;
    }
    final_predicate = final_predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value
                   << " [pred = " << pred.predicate << "]\n";
    }
  }

  PredicatedData result;
  result.is_vector = true;
  result.predicate = final_predicate;
  result.vector_data.resize(lhs.vector_data.size());
  // Computes element-wise multiplication.
  for (size_t i = 0; i < lhs.vector_data.size(); ++i) {
    result.vector_data[i] = lhs.vector_data[i] * rhs.vector_data[i];
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : " << "vector size = "
                 << result.vector_data.size() << ", ";
    print_vector(result.vector_data);
    llvm::outs() << ", [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura chained floating-point addition
 *        operation (neura.fadd_fadd) by computing a three-operand sum.
 *
 * This function processes Neura's chained floating-point addition operations,
 * which take 3-4 operands: three floating-point inputs (A, B, C) and an
 * optional predicate operand. It calculates the sum using the order ((A + B)
 * + C), combines the predicates of all operands (including the optional
 * predicate if present), and stores the result in the value map. The
 * operation requires at least three operands; fewer will result in an error.
 *
 * @param op                             The neura.fadd_fadd operation to
 * handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the chained floating-point
 *                                       addition is successfully computed;
 * false if there are fewer than 3 operands
 */
bool handleFAddFAddOp(
    neura::FAddFAddOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fadd_fadd:\n";
  }

  if (op.getNumOperands() < 3) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fadd_fadd expects at "
                      "least three operands\n";
    }
    return false;
  }

  auto a = value_to_predicated_data_map[op.getA()];
  auto b = value_to_predicated_data_map[op.getB()];
  auto c = value_to_predicated_data_map[op.getC()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand A : value = " << a.value
                 << ", [pred = " << a.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand B : value = " << b.value
                 << ", [pred = " << b.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Operand C : value = " << c.value
                 << ", [pred = " << c.predicate << "]\n";
  }

  bool final_predicate = a.predicate && b.predicate && c.predicate;

  if (op.getNumOperands() > 3) {
    auto pred_operand = value_to_predicated_data_map[op.getOperand(3)];
    final_predicate = final_predicate && pred_operand.predicate &&
                      (pred_operand.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = "
                   << pred_operand.value
                   << " [pred = " << pred_operand.predicate << "]\n";
    }
  }

  // Computes the chained sum: ((A + B) + C).
  float result_value = (a.value + b.value) + c.value;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Calculation  : (" << a.value
                 << " + " << b.value << ") + " << c.value << " = "
                 << result_value << "\n";
  }

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result       : value = "
                 << result_value << ", [pred = " << final_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura fused multiply-add operation
 *        (neura.fmul_fadd) by computing (A * B) + C.
 *
 * This function processes Neura's fused multiply-add operations, which take
 * 3-4 operands: three floating-point inputs (A, B, C) and an optional
 * predicate operand. It calculates the result using the formula (A * B) + C,
 * combines the predicates of all operands (including the optional predicate
 * if present), and stores the result in the value map. The operation requires
 * at least three operands; fewer will result in an error.
 *
 * @param op                             The neura.fmul_fadd operation to
 * handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the fused multiply-add is
 *                                       successfully computed; false if there
 * are fewer than 3 operands
 */
bool handleFMulFAddOp(
    neura::FMulFAddOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fmul_fadd:\n";
  }
  if (op.getNumOperands() < 3) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fmul_fadd expects at "
                      "least three operands\n";
    }
    return false;
  }

  auto a = value_to_predicated_data_map[op.getA()];
  auto b = value_to_predicated_data_map[op.getB()];
  auto c = value_to_predicated_data_map[op.getC()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand A : value = " << a.value
                 << ", [pred = " << a.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand B : value = " << b.value
                 << ", [pred = " << b.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Operand C : value = " << c.value
                 << ", [pred = " << c.predicate << "]\n";
  }

  bool final_predicate = a.predicate && b.predicate && c.predicate;

  if (op.getNumOperands() > 3) {
    auto pred_operand = value_to_predicated_data_map[op.getOperand(3)];
    final_predicate = final_predicate && pred_operand.predicate &&
                      (pred_operand.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = "
                   << pred_operand.value
                   << ", [pred = " << pred_operand.predicate << "]\n";
    }
  }
  // Computes the fused multiply-add: (A * B) + C.
  float result_value = 0.0f;
  float mul_result = a.value * b.value;
  result_value = mul_result + c.value;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Calculation  : (" << a.value
                 << " * " << b.value << ") + " << c.value << " = " << mul_result
                 << " + " << c.value << " = " << result_value << "\n";
  }

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result       : value = "
                 << result_value << ", [pred = " << final_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a function return operation (func.return)
 * by outputting the return value (if any).
 *
 * This function processes MLIR's standard function return operations, which
 * may optionally return a single value. It retrieves the return value from
 * the value map (if present) and prints it in a human-readable format—either
 * as a scalar or a vector. For vector values, it formats elements as a
 * comma-separated list. If no return value is present (void return), it
 * indicates a void output.
 *
 * @param op                             The func.return operation to handle
 * @param value_to_predicated_data_map   Reference to the map storing
 * predicated data for values (used to retrieve the return value)
 * @return bool                          True if the return operation is
 * processed successfully; false only if the operation is invalid (nullptr)
 */
bool handleFuncReturnOp(
    func::ReturnOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map,
    bool &has_valid_result) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing func.return:\n";
  }
  has_valid_result = false;
  if (!op && isVerboseMode()) {
    llvm::errs() << "[neura-interpreter]  └─ Expected func.return but got "
                    "something else\n";
    return false;
  }

  if (op.getNumOperands() == 0) {
    llvm::outs() << "[neura-interpreter]  → Output: (void)\n";
    return true;
  }

  auto result = value_to_predicated_data_map[op.getOperand(0)];
  if (value_to_predicated_data_map[op.getOperand(0)].predicate) {
    has_valid_result = true;
  }
  // Prints vector return value if the result is a vector.
  if (result.is_vector) {
    llvm::outs() << "[neura-interpreter]  → Output: [";
    for (size_t i = 0; i < result.vector_data.size(); ++i) {
      float val = result.predicate ? result.vector_data[i] : 0.0f;
      llvm::outs() << llvm::format("%.6f", val);
      if (i != result.vector_data.size() - 1) {
        llvm::outs() << ", ";
      }
    }
    llvm::outs() << "]\n";
  } else {
    float val = result.predicate ? result.value : 0.0f;
    llvm::outs() << "[neura-interpreter]  → Output: "
                 << llvm::format("%.6f", val) << "\n";
  }
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point comparison operation
 *        (neura.fcmp) by evaluating a specified comparison between two
 * operands.
 *
 * This function processes Neura's floating-point comparison operations, which
 * take 2-3 operands: two floating-point inputs (LHS and RHS) and an optional
 * execution predicate. It evaluates the comparison based on the specified
 * type (e.g., "eq" for equality, "lt" for less than), combines the predicates
 * of all operands (including the optional predicate if present), and stores
 * the result as a boolean scalar (1.0f for true, 0.0f for false) in the value
 * map. Errors are returned for insufficient operands or unsupported
 * comparison types.
 *
 * @param op                             The neura.fcmp operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 * comparison result will be stored, keyed by the operation's result value
 * @return bool                          True if the comparison is
 * successfully evaluated; false if there are insufficient operands or an
 * unsupported comparison type
 */
bool handleFCmpOp(
    neura::FCmpOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fcmp:\n";
  }

  PredicatedData lhs, rhs;

  // Resolve LHS: prefer operand, fall back to lhs_value attribute.
  if (op.getNumOperands() >= 1 && op.getLhs()) {
    lhs = value_to_predicated_data_map[op.getLhs()];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "lhs_value",
                                       value_to_predicated_data_map)) {
    lhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fcmp: no LHS operand "
                      "or lhs_value attribute\n";
    return false;
  }

  // Resolve RHS: prefer operand, fall back to rhs_value attribute.
  if (op.getNumOperands() >= 2 && op.getRhs()) {
    rhs = value_to_predicated_data_map[op.getRhs()];
  } else if (auto resolved =
                 resolveAttributeValue(op.getOperation(), "rhs_value",
                                       value_to_predicated_data_map)) {
    rhs = *resolved;
  } else {
    if (isVerboseMode())
      llvm::errs() << "[neura-interpreter]  └─ neura.fcmp: no RHS operand "
                      "or rhs_value attribute\n";
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS               : value = "
                 << lhs.value << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS               : value = "
                 << rhs.value << ", [pred = " << rhs.predicate << "]\n";
  }

  // Predication is handled through the operands' PredicatedData:
  // if either operand's predicate is false (NaN/disabled), the result
  // predicate will also be false.
  bool pred = true;

  bool fcmp_result = false;
  StringRef cmp_type = op.getCmpType();
  // Evaluates the comparison based on the specified type.
  // Supports both ordered (ogt, olt, ...) and unordered (ugt, ult, ...)
  // variants. Ordered variants return false if any operand is NaN.
  if (cmp_type == "eq" || cmp_type == "oeq" || cmp_type == "ueq") {
    fcmp_result = (lhs.value == rhs.value);
  } else if (cmp_type == "ne" || cmp_type == "une") {
    fcmp_result = (lhs.value != rhs.value);
  } else if (cmp_type == "le" || cmp_type == "ole" || cmp_type == "ule") {
    fcmp_result = (lhs.value <= rhs.value);
  } else if (cmp_type == "lt" || cmp_type == "olt" || cmp_type == "ult") {
    fcmp_result = (lhs.value < rhs.value);
  } else if (cmp_type == "ge" || cmp_type == "oge" || cmp_type == "uge") {
    fcmp_result = (lhs.value >= rhs.value);
  } else if (cmp_type == "gt" || cmp_type == "ogt" || cmp_type == "ugt") {
    fcmp_result = (lhs.value > rhs.value);
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported comparison type: "
                   << cmp_type << "\n";
    }
    return false;
  }

  bool final_predicate = lhs.predicate && rhs.predicate && pred;
  float result_value = fcmp_result ? 1.0f : 0.0f;

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Comparison type   : "
                 << op.getCmpType() << "\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Comparison result : "
                 << (fcmp_result ? "true" : "false") << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result               : value = "
                 << result_value << ", [pred = " << final_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura integer comparison operation
 *        (neura.icmp) by evaluating signed or unsigned comparisons between
 *        integer operands.
 *
 * This function processes Neura's integer comparison operations, which take
 * 2-3 operands: two integer inputs (LHS and RHS, stored as floats) and an
 * optional execution predicate. It converts the floating-point stored values
 * to integers, evaluates the comparison based on the specified type (e.g.,
 * "eq" for equality, "slt" for signed less than, "ult" for unsigned less
 * than), combines the predicates of all operands (including the optional
 * predicate if present), and stores the result as a boolean scalar (1.0f for
 * true, 0.0f for false) in the value map. Errors are returned for
 * insufficient operands or unsupported comparison types.
 *
 * @param op                             The neura.icmp operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 *                                       comparison result will be stored,
 *                                       keyed by the operation's result value
 * @return bool                          True if the comparison is
 * successfully evaluated; false if there are insufficient operands or an
 * unsupported comparison type
 */
bool handleICmpOp(
    neura::ICmpOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.icmp:\n";
  }
  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.icmp expects at least two "
                      "operands\n";
    }
    return false;
  }

  auto lhs = value_to_predicated_data_map[op.getLhs()];
  auto rhs = value_to_predicated_data_map[op.getRhs()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS               : value = "
                 << lhs.value << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS               : value = "
                 << rhs.value << ", [pred = " << rhs.predicate << "]\n";
  }

  // Predication is handled through the operands' PredicatedData.
  bool pred = true;
  // Converts stored floating-point values to signed integers (rounded to
  // nearest integer).
  int64_t s_lhs = static_cast<int64_t>(std::round(lhs.value));
  int64_t s_rhs = static_cast<int64_t>(std::round(rhs.value));

  auto signed_to_unsigned = [](int64_t val) {
    return val >= 0 ? static_cast<uint64_t>(val)
                    : static_cast<uint64_t>(UINT64_MAX + val + 1);
  };
  // Converts signed integers to unsigned for unsigned comparisons.
  uint64_t u_lhs = signed_to_unsigned(s_lhs);
  uint64_t u_rhs = signed_to_unsigned(s_rhs);

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Signed values     : LHS = "
                 << s_lhs << ", RHS = " << s_rhs << "\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Unsigned values   : LHS = "
                 << u_lhs << ", RHS = " << u_rhs << "\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Comparison type   : "
                 << op.getCmpType() << "\n";
  }

  bool icmp_result = false;
  StringRef cmp_type = op.getCmpType();
  // Evaluates the comparison based on the specified type (signed, unsigned,
  // or equality).
  if (cmp_type == "eq") {
    icmp_result = (s_lhs == s_rhs);
  } else if (cmp_type == "ne") {
    icmp_result = (s_lhs != s_rhs);
  } else if (cmp_type.starts_with("s")) {
    if (cmp_type == "slt") {
      icmp_result = (s_lhs < s_rhs);
    } else if (cmp_type == "sle") {
      icmp_result = (s_lhs <= s_rhs);
    } else if (cmp_type == "sgt") {
      icmp_result = (s_lhs > s_rhs);
    } else if (cmp_type == "sge") {
      icmp_result = (s_lhs >= s_rhs);
    } else {
      if (isVerboseMode()) {
        llvm::errs()
            << "[neura-interpreter]  └─ Unsupported signed comparison type: "
            << cmp_type << "\n";
      }
      return false;
    }
  }
  // Handles unsigned comparisons.
  else if (cmp_type.starts_with("u")) {
    if (cmp_type == "ult") {
      icmp_result = (u_lhs < u_rhs);
    } else if (cmp_type == "ule") {
      icmp_result = (u_lhs <= u_rhs);
    } else if (cmp_type == "ugt") {
      icmp_result = (u_lhs > u_rhs);
    } else if (cmp_type == "uge") {
      icmp_result = (u_lhs >= u_rhs);
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported unsigned "
                        "comparison type: "
                     << cmp_type << "\n";
      }
      return false;
    }
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported comparison type: "
                   << cmp_type << "\n";
    }
    return false;
  }

  bool final_predicate = lhs.predicate && rhs.predicate && pred;
  float result_value = icmp_result ? 1.0f : 0.0f;

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  │  └─ Comparison result : "
                 << (icmp_result ? "true" : "false") << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result               : value = "
                 << result_value << ", [pred = " << final_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura logical OR operation (neura.or)
 *        for scalar boolean values.
 *
 * This function processes Neura's logical OR operations, which take 2-3
 * operands: two scalar boolean inputs (LHS and RHS, represented as floats)
 * and an optional execution predicate. It computes the logical OR of the
 * two operands (true if at least one is non-zero), combines the predicates
 * of all operands (including the optional predicate if present), and stores
 * the result as a boolean scalar (1.0f for true, 0.0f for false) in the
 * value map. Errors are returned for insufficient operands or invalid
 * operand types (e.g., vectors).
 *
 * @param op                             The neura.or operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 *                                       logical OR result will be stored,
 *                                       keyed by the operation's result value
 * @return bool                          True if the logical OR is
 * successfully executed; false if there are invalid operands or insufficient
 * inputs
 */
bool handleOrOp(
    neura::OrOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.or (logical OR):\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.or (logical) expects at "
                      "least two operands\n";
    }
    return false;
  }

  // Retrieves left and right operands.
  auto lhs = value_to_predicated_data_map[op.getOperand(0)];
  auto rhs = value_to_predicated_data_map[op.getOperand(1)];

  if (lhs.is_vector || rhs.is_vector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.or (logical) requires "
                      "scalar operands\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS        : value = "
                 << lhs.value
                 << " (boolean: " << (lhs.value != 0.0f ? "true" : "false")
                 << "), [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS        : value = "
                 << rhs.value
                 << " (boolean: " << (rhs.value != 0.0f ? "true" : "false")
                 << "), [pred = " << rhs.predicate << "]\n";
  }

  // Converts operands to boolean (non-zero = true).
  bool lhs_bool = (lhs.value != 0.0f);
  bool rhs_bool = (rhs.value != 0.0f);
  // Logical OR result: true if either operand is true.
  bool result_bool = lhs_bool || rhs_bool;

  // Computes final validity predicate (combines operand predicates and
  // optional predicate).
  bool final_predicate = lhs.predicate && rhs.predicate;
  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = "
                   << pred.value << ", [pred = " << pred.predicate << "]\n";
    }
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Logical OR : " << lhs_bool
                 << " || " << rhs_bool << " = " << result_bool << "\n";
  }

  PredicatedData result;
  result.value = result_bool ? 1.0f : 0.0f;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result     : value = "
                 << result.value
                 << " (boolean: " << (result_bool ? "true" : "false")
                 << "), [pred = " << final_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura logical AND operation (neura.and)
 *        for scalar boolean values.
 *
 * This function processes Neura's logical AND operations, which take 2-3
 * operands: two scalar boolean inputs (LHS and RHS, represented as floats)
 * and an optional execution predicate. It computes the logical AND of the
 * two operands (true if both are non-zero), combines the predicates of all
 * operands (including the optional predicate if present), and stores the
 * result as a boolean scalar (1.0f for true, 0.0f for false) in the value
 * map. Errors are returned for insufficient operands or invalid operand
 * types (e.g., vectors).
 *
 * @param op                             The neura.and operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 *                                       logical AND result will be stored,
 *                                       keyed by the operation's result value
 * @return bool                          True if the logical AND is
 * successfully executed; false if there are invalid operands or insufficient
 * inputs
 */
bool handleAndOp(
    neura::AndOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.and (logical AND):\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.and (logical) expects at "
                      "least two operands\n";
    }
    return false;
  }

  // Retrieves left and right operands.
  auto lhs = value_to_predicated_data_map[op.getOperand(0)];
  auto rhs = value_to_predicated_data_map[op.getOperand(1)];

  if (lhs.is_vector || rhs.is_vector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.and (logical) requires "
                      "scalar operands\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS        : value = "
                 << lhs.value
                 << " (boolean: " << (lhs.value != 0.0f ? "true" : "false")
                 << "), [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS        : value = "
                 << rhs.value
                 << " (boolean: " << (rhs.value != 0.0f ? "true" : "false")
                 << "), [pred = " << rhs.predicate << "]\n";
  }

  // Converts operands to boolean (non-zero = true).
  bool lhs_bool = (lhs.value != 0.0f);
  bool rhs_bool = (rhs.value != 0.0f);
  // Logical AND result: true if both operands are true.
  bool result_bool = lhs_bool && rhs_bool;

  // Computes final validity predicate (combines operand predicates and
  // optional predicate).
  bool final_predicate = lhs.predicate && rhs.predicate;
  if (op.getNumOperands() > 2) {
    auto pred = value_to_predicated_data_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = "
                   << pred.value << ", [pred = " << pred.predicate << "]\n";
    }
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Logical AND: " << lhs_bool
                 << " && " << rhs_bool << " = " << result_bool << "\n";
  }

  PredicatedData result;
  result.value = result_bool ? 1.0f : 0.0f;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result     : value = "
                 << result.value
                 << " (boolean: " << (result_bool ? "true" : "false")
                 << "), [pred = " << final_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura logical NOT operation (neura.not)
 *        by computing the inverse of a boolean input.
 *
 * This function processes Neura's logical NOT operations, which take a single
 * boolean operand (represented as a floating-point value). It converts the
 * input value to an integer (rounded to the nearest whole number), applies
 * the logical NOT operation (inverting true/false), and stores the result as
 * a floating-point value (1.0f for true, 0.0f for false) in the value map.
 * The result's predicate is inherited from the input operand's predicate.
 *
 * @param op                             The neura.not operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          Always returns true since the
 * operation executes successfully with valid input
 */
bool handleNotOp(
    neura::NotOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  auto input = value_to_predicated_data_map[op.getOperand()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.not:\n";
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Input       : value = "
                 << input.value << ", [pred = " << input.predicate << "]\n";
  }

  // Converts the input floating-point value to an integer (rounded to nearest
  // whole number).
  int64_t inputInt = static_cast<int64_t>(std::round(input.value));
  // Applies logical NOT: 0 (false) becomes 1 (true), non-zero (true) becomes
  // 0 (false).
  int64_t result_int = !inputInt;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Logical NOT : !" << inputInt;
    llvm::outs() << "\n";
  }

  PredicatedData result;
  result.value = static_cast<float>(result_int);
  result.predicate = input.predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result         : value = "
                 << result.value << ", [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura selection operation (neura.sel)
 *        by choosing between two values based on a condition.
 *
 * This function processes Neura's selection operations, which take exactly
 * three operands: a condition, a value to use if the condition is true, and
 * a value to use if the condition is false. It evaluates the condition
 * (non-zero values with a true predicate are treated as true), selects the
 * corresponding value ("if_true" or "if_false"), and combines the predicate
 * of the condition with the predicate of the chosen value. The result is
 * marked as a vector only if both input values are vectors. Errors are
 * returned if the operand count is not exactly three.
 *
 * @param op                             The neura.sel operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 *                                       selection result will be stored,
 *                                       keyed by the operation's result value
 * @return bool                          True if the selection is successfully
 *                                       computed; false if the operand count
 *                                       is invalid
 */
bool handleSelOp(
    neura::SelOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.sel:\n";
  }

  if (op.getNumOperands() != 3) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.sel expects exactly 3 "
                      "operands (cond, if_true, if_false)\n";
    }
    return false;
  }

  auto cond =
      value_to_predicated_data_map[op.getCond()]; /* Condition to evaluate */
  auto if_true =
      value_to_predicated_data_map[op.getIfTrue()]; /* Value if condition is
                                                       true */
  auto if_false =
      value_to_predicated_data_map[op.getIfFalse()]; /* Value if condition is
                                                        false */
  // Evaluates the condition: true if the value is non-zero and its predicate
  // is true.
  bool cond_value = (cond.value != 0.0f) && cond.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Condition : value = "
                 << cond.value << ", [pred = " << cond.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ If true   : value = "
                 << if_true.value << ", [pred = " << if_true.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ If false  : value = "
                 << if_false.value << ", [pred = " << if_false.predicate
                 << "]\n";
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation \n";
  }

  PredicatedData result;
  // Prepares the result by selecting the appropriate value based on the
  // condition.
  if (cond_value) {
    result.value = if_true.value;
    result.predicate = if_true.predicate && cond.predicate;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Condition is true, selecting "
                      "'if_true' branch\n";
    }
  } else {
    result.value = if_false.value;
    result.predicate = if_false.predicate && cond.predicate;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Condition is false, "
                      "selecting 'if_false' branch\n";
    }
  }

  result.is_vector = if_true.is_vector && if_false.is_vector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result      : value = "
                 << result.value << ", predicate = " << result.predicate
                 << "\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura type conversion operation
 *        (neura.cast) by converting an input value between supported types.
 *
 * This function processes Neura's type conversion operations, which take
 * one or two operands: an input value to convert, and an optional predicate
 * operand. It supports multiple conversion types (e.g., float to integer,
 * integer to boolean) and validates that the input type matches the
 * conversion requirements. The result's predicate is combined with the
 * optional predicate (if present), and the result inherits the input's
 * vector flag. Errors are returned for invalid operand counts, unsupported
 * conversion types, or mismatched input types.
 *
 * @param op                             The neura.cast operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 *                                       converted result will be stored,
 *                                       keyed by the operation's result value
 * @return bool                          True if the conversion is
 * successfully computed; false if operands are invalid, the type is
 * unsupported, or the input type does not match the conversion
 */
bool handleCastOp(
    neura::CastOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.cast:\n";
  }
  if (op.getOperation()->getNumOperands() != 1) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cast expects 1 operand\n";
    }
    return false;
  }

  auto input = value_to_predicated_data_map[op.getOperand()];
  std::string cast_type = op.getCastType().str();

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Input  : value = "
                 << input.value << ", [pred = " << input.predicate << "]\n";
  }

  bool final_predicate = input.predicate;
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
    llvm::outs() << "[neura-interpreter]  ├─ Cast type : " << cast_type << "\n";
  }

  float result_value = 0.0f;
  auto input_type = op.getOperand().getType();
  // Handles specific conversion types with input type validation.
  if (cast_type == "f2i") {
    if (!input_type.isF32()) {
      if (isVerboseMode()) {
        llvm::errs()
            << "[neura-interpreter]  └─ Cast type 'f2i' requires f32 input\n";
      }
      return false;
    }
    int64_t int_value = static_cast<int64_t>(std::round(input.value));
    result_value = static_cast<float>(int_value);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Converting float to integer "
                   << input.value << " -> " << int_value << "\n";
    }

  } else if (cast_type == "i2f") {
    if (!input_type.isInteger()) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Cast type 'i2f' requires "
                        "integer input\n";
      }
      return false;
    }
    int64_t int_value = static_cast<int64_t>(input.value);
    result_value = static_cast<float>(int_value);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Converting integer to float "
                   << int_value << " -> " << result_value << "\n";
    }
  } else if (cast_type == "bool2i" || cast_type == "bool2f") {
    if (!input_type.isInteger(1)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Cast type '" << cast_type
                     << "' requires i1 (boolean) input\n";
      }
      return false;
    }
    bool bool_value = (input.value != 0.0f);
    result_value = bool_value ? 1.0f : 0.0f;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Converting boolean to number "
                   << (bool_value ? "true" : "false") << " -> " << result_value
                   << "\n";
    }
  } else if (cast_type == "i2bool" || cast_type == "f2bool") {
    if (!input_type.isInteger() && !input_type.isF32()) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Cast type '" << cast_type
                     << "' requires integer or f32 input\n";
      }
      return false;
    }
    bool bool_value = (input.value != 0.0f);
    result_value = bool_value ? 1.0f : 0.0f;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Converting number to boolean "
                   << input.value << " -> " << (bool_value ? "true" : "false")
                   << " (stored as " << result_value << ")\n";
    }
  } else if (cast_type == "truncf") {
    // f64 → f32 truncation: all interpreter values are f32, passthrough.
    result_value = input.value;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ f64→f32 truncf passthrough: "
                   << input.value << "\n";
    }
  } else if (cast_type == "extf") {
    // f32 → f64 extension: all interpreter values are f32, passthrough.
    result_value = input.value;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ f32→f64 extf passthrough: "
                   << input.value << "\n";
    }
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported cast type: "
                   << cast_type << "\n";
    }
    return false;
  }

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = input.is_vector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result    : value = "
                 << result_value << ", [pred = " << final_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura reciprocal square root operation
 *        (neura.rsqrt) by computing 1/sqrt(input) for floating-point values.
 *
 * This function processes Neura's rsqrt operation, which takes a single
 * floating-point operand and computes the reciprocal square root: result =
 * 1/sqrt(input). The input's predicate is passed through to the result. The
 * operation requires exactly one operand; zero or more will result in an
 * error.
 *
 * @param op                             The neura.rsqrt operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the rsqrt is successfully
 * computed; false if there are insufficient operands
 */
bool handleExpOp(
    neura::ExpOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.exp:\n";
  }

  auto input = value_to_predicated_data_map[op.getInput()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Input  : value = " << input.value
                 << " [pred = " << input.predicate << "]\n";
  }

  // Compute exp(input). Clamp to avoid overflow.
  float clamped = std::max(-88.0f, std::min(88.0f, input.value));
  float exp_result = expf(clamped);

  PredicatedData result;
  result.value = exp_result;
  result.predicate = input.predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

bool handleRsqrtOp(
    neura::RsqrtOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.rsqrt:\n";
  }

  auto input = value_to_predicated_data_map[op.getInput()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Input  : value = " << input.value
                 << " [pred = " << input.predicate << "]\n";
  }

  // Compute 1/sqrt(input). Guard against zero/negative input.
  float rsqrt_result;
  if (input.value <= 0.0f) {
    rsqrt_result = (input.value == 0.0f) ? INFINITY : NAN;
  } else {
    rsqrt_result = 1.0f / sqrtf(input.value);
  }

  PredicatedData result;
  result.value = rsqrt_result;
  result.predicate = input.predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result : value = " << result.value
                 << " [pred = " << result.predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura Get Element Pointer operation
 *        (neura.gep) by computing a memory address from a base address
 *        and one or more indices.
 *
 * This function processes Neura's GEP operations, which calculate a target
 * memory address by adding an offset to a base address. The offset is
 * computed using multi-dimensional indices and corresponding strides
 * (step sizes between elements in each dimension). The operation takes
 * one or more operands: a base address, optional indices (one per dimension),
 * and an optional predicate operand (if present, it must be the last one).
 * The "strides" attribute specifies the stride for each dimension, which
 * determines how much to multiply each index by when calculating the offset.
 *
 * The operation validates operand counts, extracts and applies the optional
 * predicate, computes the total offset as the sum of (index * stride) for
 * each dimension, combines the predicates of the base address and the
 * optional predicate, and produces the final address (base + offset) with
 * the combined predicate.
 *
 * @param op                             The neura.gep operation to handle
 * @param value_to_predicated_data_map   Reference to the map storing
 *                                       predicated data for values (base
 *                                       address, indices, predicate)
 * @return bool                          True if the address is successfully
 *                                       computed; false if operands are
 *                                       invalid, strides are missing, or
 *                                       indices do not match the strides
 */
bool handleGEPOp(
    neura::GEP op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.gep:\n";
  }

  if (op.getOperation()->getNumOperands() < 1) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.gep expects at least 1 "
                      "operand (base address)\n";
    }
    return false;
  }

  auto base_val = value_to_predicated_data_map[op.getOperand(0)];
  size_t base_addr = static_cast<size_t>(base_val.value);
  bool final_predicate = base_val.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Base address: value = "
                 << base_addr << ", [pred = " << base_val.predicate << "]\n";
  }

  unsigned num_operands = op.getOperation()->getNumOperands();
  bool has_predicate = false;
  unsigned index_count = num_operands - 1;

  if (num_operands > 1) {
    auto last_operand_type = op.getOperand(num_operands - 1).getType();
    if (last_operand_type.isInteger(1)) {
      has_predicate = true;
      index_count -= 1;
    }
  }

  auto strides_attr = op->getAttrOfType<mlir::ArrayAttr>("strides");
  if (!strides_attr) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.gep requires 'strides' "
                      "attribute\n";
    }
    return false;
  }

  // Converts strides attribute to a vector of size_t (scaling factors for
  // indices).
  std::vector<size_t> strides;
  for (auto s : strides_attr) {
    auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(s);
    if (!int_attr) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Invalid type in 'strides' "
                        "attribute (expected integer)\n";
      }
      return false;
    }
    strides.push_back(static_cast<size_t>(int_attr.getInt()));
  }

  if (index_count != strides.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ GEP index count (" << index_count
                   << ") mismatch with strides size (" << strides.size()
                   << ")\n";
    }
    return false;
  }

  // Calculates total offset by scaling each index with its stride and
  // summing.
  size_t offset = 0;
  for (unsigned i = 0; i < index_count; ++i) {
    auto idx_val = value_to_predicated_data_map[op.getOperand(i + 1)];
    if (!idx_val.predicate) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ GEP index " << i
                     << " has false predicate\n";
      }
      return false;
    }

    size_t idx = static_cast<size_t>(idx_val.value);
    offset += idx * strides[i];
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Index " << i
                   << ": value = " << idx << ", stride = " << strides[i]
                   << ", cumulative offset = " << offset << "\n";
    }
  }

  if (has_predicate) {
    auto pred_val =
        value_to_predicated_data_map[op.getOperand(num_operands - 1)];
    final_predicate =
        final_predicate && pred_val.predicate && (pred_val.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Predicate operand: value = "
                   << pred_val.value << ", [pred = " << pred_val.predicate
                   << "]\n";
    }
  }

  size_t final_addr = base_addr + offset;

  PredicatedData result;
  result.value = static_cast<float>(final_addr);
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Final GEP result: base = "
                 << base_addr << ", total offset = " << offset
                 << ", final address = " << final_addr
                 << ", [pred = " << final_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura unconditional branch operation
 *        (neura.br) by transferring control to a target block.
 *
 * This function processes Neura's unconditional branch operations, which
 * always transfer control flow to a specified target block. It validates
 * that the target block exists, checks that the number of branch arguments
 * matches the target block's parameters, and copies the argument values to
 * the target's parameters. Finally, it updates the current block and the
 * last visited block to reflect the control transfer.
 *
 * @param op                             The neura.br operation to handle
 * @param value_to_predicated_data_map   Reference to the map storing branch
 *                                       arguments and where target parameters
 *                                       will be updated
 * @param current_block                  Reference to the current block;
 *                                       updated to the target block after
 *                                       the branch
 * @param last_visited_block             Reference to the last visited block;
 *                                       updated to the previous current block
 *                                       after the branch
 * @return bool                          True if the branch is successfully
 *                                       executed; false if the target block
 *                                       is invalid or argument/parameter
 *                                       counts mismatch
 */
bool handleBrOp(
    neura::Br op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map,
    Block *&current_block, Block *&last_visited_block) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.br:\n";
  }

  // Gets the target block of the unconditional branch.
  Block *dest_block = op.getDest();
  if (!dest_block) {
    if (isVerboseMode()) {
      llvm::errs()
          << "[neura-interpreter]  └─ neura.br: Target block is null\n";
    }
    return false;
  }

  // Retrieves all successor blocks of the current block.
  auto current_succs_range = current_block->getSuccessors();
  std::vector<Block *> succ_blocks(current_succs_range.begin(),
                                   current_succs_range.end());

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Block Information\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Current block    : Block@"
                 << current_block << "\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Successor blocks : \n";
    for (unsigned int i = 0; i < succ_blocks.size(); ++i) {
      if (i < succ_blocks.size() - 1)
        llvm::outs() << "[neura-interpreter]  │  │  ├─ [" << i << "] Block@"
                     << succ_blocks[i] << "\n";
      else
        llvm::outs() << "[neura-interpreter]  │  │  └─ [" << i << "] Block@"
                     << succ_blocks[i] << "\n";
    }
    llvm::outs() << "[neura-interpreter]  │  └─ Target block : Block@"
                 << dest_block << "\n";
    llvm::outs() << "[neura-interpreter]  ├─ Pass Arguments\n";
  }

  // Gets branch arguments and target block parameters.
  const auto &args = op.getArgs();
  const auto &dest_params = dest_block->getArguments();

  if (args.size() != dest_params.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.br: Argument count "
                      "mismatch (passed "
                   << args.size() << ", target expects " << dest_params.size()
                   << ")\n";
    }
    return false;
  }

  // Copies argument values to target block parameters in the value map.
  for (size_t i = 0; i < args.size(); ++i) {
    Value dest_param = dest_params[i];
    Value src_arg = args[i];

    if (!value_to_predicated_data_map.count(src_arg)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ neura.br: Argument " << i
                     << " (source value) not found in value map\n";
      }
      return false;
    }

    // Transfers argument value to target parameter.
    value_to_predicated_data_map[dest_param] =
        value_to_predicated_data_map[src_arg];
    if (isVerboseMode() && i < dest_params.size() - 1) {
      llvm::outs() << "[neura-interpreter]  │  ├─ Param[" << i << "]: value = "
                   << value_to_predicated_data_map[src_arg].value << "\n";
    } else if (isVerboseMode() && i == dest_params.size() - 1) {
      llvm::outs() << "[neura-interpreter]  │  └─ Param[" << i << "]: value = "
                   << value_to_predicated_data_map[src_arg].value << "\n";
    }
  }

  // Updates control flow state: last visited = previous current block;
  // current = target block.
  last_visited_block = current_block;
  current_block = dest_block;
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Control Transfer\n";
    llvm::outs() << "[neura-interpreter]     └─ Jump successfully to Block@ "
                 << dest_block << "\n";
  }
  return true;
}

/**
 * @brief Handles the execution of a Neura conditional branch operation
 *        (neura.cond_br) by transferring control to one of two target
 *        blocks based on a boolean condition.
 *
 * This function processes Neura's conditional branch operations, which
 * direct control flow to either a "true" target block or a "false" target
 * block depending on the value of a boolean condition. It validates the
 * operands (one mandatory condition and one optional predicate), checks
 * that the condition is a boolean (i1 type), and computes the final
 * predicate to determine if the branch is valid. If valid, it selects the
 * target block based on the condition’s value, copies the branch arguments
 * to the target block’s parameters, and updates the current block and last
 * visited block to reflect the control transfer.
 *
 * @param op                             The neura.cond_br operation to handle
 * @param value_to_predicated_data_map   Reference to the map storing values
 *                                       (including the condition and optional
 *                                       predicate)
 * @param current_block                  Reference to the current block;
 *                                       updated to the target block after
 *                                       the branch
 * @param last_visited_block             Reference to the last visited block;
 *                                       updated to the previous current block
 *                                       after the branch
 * @return bool                          True if the branch is successfully
 *                                       executed; false if operands are
 * invalid, values are missing, types mismatch, or the predicate is invalid
 */
bool handleCondBrOp(
    neura::CondBr op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map,
    Block *&current_block, Block *&last_visited_block) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.cond_br:\n";
  }

  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br expects 1 or 2 "
                      "operands (condition + optional predicate)\n";
    }
    return false;
  }

  auto cond_value = op.getCondition();
  if (!value_to_predicated_data_map.count(cond_value)) {
    if (isVerboseMode()) {
      llvm::errs()
          << "[neura-interpreter]  └─ cond_br: condition value not found in "
             "value_to_predicated_data_map! (SSA name missing)\n";
    }
    return false;
  }
  auto cond_data = value_to_predicated_data_map[op.getCondition()];

  if (!op.getCondition().getType().isInteger(1)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: condition must "
                      "be of type i1 (boolean)\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Condition     : value = "
                 << cond_data.value << ", [pred = " << cond_data.predicate
                 << "]\n";
  }

  // Computes final predicate (combines condition's predicate and optional
  // predicate operand).
  bool final_predicate = cond_data.predicate;
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
  }

  // Retrieves successor blocks (targets of the conditional branch).
  auto current_succs_range = current_block->getSuccessors();
  std::vector<Block *> succ_blocks(current_succs_range.begin(),
                                   current_succs_range.end());

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Block Information\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Current block : Block@"
                 << current_block << "\n";
    llvm::outs() << "[neura-interpreter]  ├─ Branch Targets\n";
    for (unsigned int i = 0; i < succ_blocks.size(); ++i) {
      if (i < succ_blocks.size() - 1) {
        llvm::outs() << "[neura-interpreter]  │  ├─ True block    : Block@"
                     << succ_blocks[i] << "\n";
      } else {
        llvm::outs() << "[neura-interpreter]  │  └─ False block   : Block@"
                     << succ_blocks[i] << "\n";
      }
    }
  }

  if (!final_predicate) {
    llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: condition or "
                    "predicate is invalid\n";
    return false;
  }

  // Determines target block based on condition value (non-zero = true
  // branch).
  bool is_true_branch = (cond_data.value != 0.0f);
  Block *target_block = is_true_branch ? op.getTrueDest() : op.getFalseDest();
  const auto &branch_args =
      is_true_branch ? op.getTrueArgs() : op.getFalseArgs();
  const auto &target_params = target_block->getArguments();

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Condition is "
                 << (cond_data.value != 0.0f ? "true" : "false")
                 << " → selecting '" << (is_true_branch ? "true" : "false")
                 << "' branch\n";
  }

  if (branch_args.size() != target_params.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: argument count "
                      "mismatch for "
                   << (is_true_branch ? "true" : "false")
                   << " branch (expected " << target_params.size() << ", got "
                   << branch_args.size() << ")\n";
    }
    return false;
  }

  // Passes branch arguments to target block parameters (update
  // value_to_predicated_data_map).
  if (!branch_args.empty()) {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Pass Arguments\n";
    }
  }

  for (size_t i = 0; i < branch_args.size(); ++i) {
    value_to_predicated_data_map[target_params[i]] =
        value_to_predicated_data_map[branch_args[i]];
    if (i < branch_args.size() - 1) {
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  ├─ param[" << i
                     << "]: value = "
                     << value_to_predicated_data_map[branch_args[i]].value
                     << "\n";
      }
    } else {
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ param[" << i
                     << "]: value = "
                     << value_to_predicated_data_map[branch_args[i]].value
                     << "\n";
      }
    }
  }

  // Updates control flow state: last visited block = previous current block;
  // current block = target.
  last_visited_block = current_block;
  current_block = target_block;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Control Transfer\n";
    llvm::outs() << "[neura-interpreter]     └─ Jump successfully to Block@"
                 << target_block << "\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura phi_start operation
 * (neura.phi_start), specifically designed for loop initialization with a
 * reserve operand.
 *
 * In control flow mode, phi_start behaves the same as phi: selects input based
 * on the predecessor block.
 *
 * In data flow mode, phi_start has special semantics:
 * - On first execution: selects from init_values (the non-reserve operands)
 * - On subsequent executions: selects the operand with a true predicate
 *
 * @param op                             The neura.phi_start operation to handle
 * @param value_to_predicated_data_map   Reference to the map storing input
 *                                       values and where the result will be
 * stored
 * @param current_block                  [ControlFlow only] The block containing
 *                                       the phi_start operation (nullptr in
 * DataFlow mode)
 * @param last_visited_block             [ControlFlow only] The most recently
 *                                       visited predecessor block (nullptr in
 * DataFlow mode)
 * @return bool                          True if the phi_start is successfully
 *                                       executed; false if validation fails
 */
bool handlePhiStartOp(
    neura::PhiStartOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map,
    Block *current_block = nullptr, Block *last_visited_block = nullptr) {
  // Static map to track if this phi_start has been executed before (for
  // dataflow mode)
  static llvm::DenseMap<Operation *, bool> phi_start_executed;

  if (isVerboseMode()) {
    if (!isDataflowMode()) {
      llvm::outs() << "[neura-interpreter]  Executing neura.phi_start:\n";
    } else {
      llvm::outs()
          << "[neura-interpreter]  Executing neura.phi_start(dataflow):\n";
    }
  }

  // Collects all inputs: reserved operand and init values
  Value reserved = op.getReserved();
  auto init_value = op.getInitValue();

  SmallVector<Value> all_inputs;
  all_inputs.push_back(reserved);
  all_inputs.push_back(init_value);

  size_t input_count = all_inputs.size();
  if (input_count < 2) {
    if (isVerboseMode()) {
      llvm::errs()
          << "[neura-interpreter]  └─ Error: phi_start requires at least "
             "reserved + one init value\n";
    }
    return false;
  }

  PredicatedData selected_input_data;
  bool selection_success = false;

  if (!isDataflowMode()) {
    // ControlFlow mode: same logic as phi - select based on predecessor block.
    if (!current_block || !last_visited_block) {
      if (isVerboseMode()) {
        llvm::errs()
            << "[neura-interpreter]  └─ Error: ControlFlow mode requires "
               "current_block and last_visited_block\n";
      }
      return false;
    }

    auto predecessors_range = current_block->getPredecessors();
    std::vector<Block *> predecessors(predecessors_range.begin(),
                                      predecessors_range.end());
    size_t pred_count = predecessors.size();
    if (pred_count == 0) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Error: No predecessors for "
                        "current block\n";
      }
      return false;
    }

    if (input_count != pred_count) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Error: Input count ("
                     << input_count << ") != predecessor count (" << pred_count
                     << ")\n";
      }
      return false;
    }

    size_t pred_index = 0;
    bool found = false;
    for (auto pred : predecessors) {
      if (pred == last_visited_block) {
        found = true;
        break;
      }
      ++pred_index;
    }
    if (!found) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Error: Last visited block not "
                        "found among predecessors\n";
      }
      return false;
    }

    if (pred_index >= input_count) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Error: Predecessor index out "
                        "of bounds\n";
      }
      return false;
    }

    Value selected_input = all_inputs[pred_index];
    if (!value_to_predicated_data_map.count(selected_input)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Error: Selected input not "
                        "found in value map\n";
      }
      return false;
    }
    selected_input_data = value_to_predicated_data_map[selected_input];
    selection_success = true;

    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Predecessor blocks ("
                   << pred_count << ")\n";
      for (size_t i = 0; i < pred_count; ++i) {
        llvm::outs() << "[neura-interpreter]  │  "
                     << (i < pred_count - 1 ? "├─" : "└─") << " Block@"
                     << predecessors[i]
                     << (i == pred_index ? " (selected)" : "") << "\n";
      }
    }
  } else {
    // DataFlow mode: special logic for phi_start.
    bool is_first_execution = !phi_start_executed[op.getOperation()];

    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution state: "
                   << (is_first_execution ? "FIRST" : "SUBSEQUENT") << "\n";
    }

    if (is_first_execution) {
      // First execution: selects from init_values (prefer one with true
      // predicate)
      bool found_valid_init = false;
      auto init_data = value_to_predicated_data_map[init_value];
      if (init_data.predicate) {
        selected_input_data = init_data;
        found_valid_init = true;
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  ├─ First execution: Selected "
                          "init value "
                       << "with value = " << init_data.value
                       << ", [pred = " << init_data.predicate << "]\n";
        }
      }

      // If no valid init value found on first execution, this is an error.
      if (!found_valid_init) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Error: First execution of "
                          "phi_start requires at least one init value with "
                          "predicate = true\n";
        }
        return false;
      }

      // Mark as executed for subsequent iterations
      phi_start_executed[op.getOperation()] = true;
      selection_success = true;
    } else {
      // Subsequent executions: select the operand with true predicate from ALL
      // inputs.
      bool found_valid_input = false;
      for (size_t i = 0; i < all_inputs.size(); ++i) {
        Value input = all_inputs[i];
        if (!value_to_predicated_data_map.count(input)) {
          continue;
        }
        auto input_data = value_to_predicated_data_map[input];
        if (input_data.predicate) {
          selected_input_data = input_data;
          found_valid_input = true;
          if (isVerboseMode()) {
            llvm::outs()
                << "[neura-interpreter]  ├─ Subsequent execution: Selected "
                   "input["
                << i << "] " << (i == 0 ? "(reserved)" : "(init)")
                << " with value = " << input_data.value
                << ", [pred = " << input_data.predicate << "]\n";
          }
          break;
        }
      }

      // If no valid input found, this is also an error.
      if (!found_valid_input) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Error: Subsequent execution "
                          "of phi_start requires at least one operand with "
                          "predicate = true\n";
        }
        return false;
      }

      selection_success = true;
    }

    // Logs input details in dataflow mode.
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Input values:\n";
      llvm::outs() << "[neura-interpreter]  │  ├─ Reserved: ";
      if (value_to_predicated_data_map.count(reserved)) {
        auto &data = value_to_predicated_data_map[reserved];
        llvm::outs() << "value = " << data.value
                     << ", [pred = " << data.predicate
                     << ", is_reserve = " << data.is_reserve << "]\n";
      } else {
        llvm::outs() << "(not in map)\n";
      }

      if (value_to_predicated_data_map.count(init_value)) {
        auto &data = value_to_predicated_data_map[init_value];
        llvm::outs() << "value = " << data.value
                     << ", [pred = " << data.predicate << "]\n";
      } else {
        llvm::outs() << "(not in map)\n";
      }
    }
  }

  // Checks for changes
  bool is_updated = false;
  if (value_to_predicated_data_map.count(op.getResult())) {
    auto old_result = value_to_predicated_data_map[op.getResult()];
    is_updated = selected_input_data.isUpdatedComparedTo(old_result);
  } else {
    is_updated = true;
  }

  PredicatedData result = selected_input_data;
  result.is_updated = is_updated;
  result.is_reserve = false;
  value_to_predicated_data_map[op.getResult()] = result;

  if (isVerboseMode()) {
    if (!isDataflowMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Result    : " << op.getResult()
                   << "\n";
      llvm::outs() << "[neura-interpreter]     └─ Value : " << result.value
                   << ", [Pred = " << result.predicate << "]\n";
    } else {
      llvm::outs() << "[neura-interpreter]  └─ Execution "
                   << (selection_success ? "succeeded" : "partially succeeded")
                   << " | Result: value = " << result.value
                   << ", pred = " << result.predicate
                   << ", is_updated = " << result.is_updated << "\n";
    }
  }

  return selection_success;
}

/**
 * @brief Handles the execution of a Neura phi operation (neura.phi),
 *        supporting both control flow and data flow modes.
 *
 * In control flow mode, the phi operation selects its result based on the
 * most recently visited predecessor block. It validates the block
 * relationships and ensures that the number of incoming values matches
 * the number of predecessors.
 *
 * In data flow mode, the phi operation merges inputs by selecting the first
 * input with a true predicate. If no valid input exists, it falls back to
 * the first input with a false predicate. The result is tracked to enable
 * propagation through the data flow network.
 *
 * @param op                             The neura.phi operation to handle
 * @param value_to_predicated_data_map   Reference to the map storing input
 *                                       values and where the result will be
 *                                       stored
 * @param current_block                  [ControlFlow only] The block
 *                                       containing the phi operation
 *                                       (nullptr in DataFlow mode)
 * @param last_visited_block             [ControlFlow only] The most
 * recently visited predecessor block (nullptr in DataFlow mode)
 * @return bool                          True if the phi is successfully
 *                                       executed; false if validation fails
 *                                       in control flow mode (always true
 * in data flow mode)
 */
bool handlePhiOp(
    neura::PhiOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map,
    Block *current_block = nullptr, Block *last_visited_block = nullptr) {
  if (isVerboseMode()) {
    if (!isDataflowMode()) {
      llvm::outs() << "[neura-interpreter]  Executing neura.phi:\n";
    } else {
      llvm::outs() << "[neura-interpreter]  Executing neura.phi(dataflow):\n";
    }
  }

  auto inputs = op.getInputs();
  size_t input_count = inputs.size();
  if (input_count == 0) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: No inputs provided "
                      "(execution failed)\n";
    }
    return false; // No inputs is a failure in both modes.
  }

  // Stores the finally selected input data.
  PredicatedData selected_input_data;
  bool selection_success = false;

  // --------------------------
  // Mode-specific logic: Input selection and validation.
  // --------------------------
  if (!isDataflowMode()) {
    // ControlFlow mode: Validate required block parameters.
    if (!current_block || !last_visited_block) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ ControlFlow mode requires "
                        "current_block and last_visited_block\n";
      }
      return false;
    }

    // ControlFlow mode: Get predecessors and validate count.
    auto predecessors_range = current_block->getPredecessors();
    std::vector<Block *> predecessors(predecessors_range.begin(),
                                      predecessors_range.end());
    size_t pred_count = predecessors.size();
    if (pred_count == 0) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ neura.phi: Current block has "
                        "no predecessors\n";
      }
      return false;
    }

    // ControlFlow mode: Validate input count matches predecessor count.
    if (input_count != pred_count) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ neura.phi: Input count ("
                     << input_count << ") != predecessor count (" << pred_count
                     << ")\n";
      }
      return false;
    }

    // ControlFlow mode: Finds index of last visited block among predecessors.
    size_t pred_index = 0;
    bool found = false;
    for (auto pred : predecessors) {
      if (pred == last_visited_block) {
        found = true;
        break;
      }
      ++pred_index;
    }
    if (!found) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ neura.phi: Last visited block "
                        "not found in predecessors\n";
      }
      return false;
    }

    // ControlFlow mode: Validates index and retrieves selected input.
    if (pred_index >= input_count) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ neura.phi: Invalid "
                        "predecessor index ("
                     << pred_index << ")\n";
      }
      return false;
    }
    Value selected_input = inputs[pred_index];
    if (!value_to_predicated_data_map.count(selected_input)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ neura.phi: Selected input not "
                        "found in value map\n";
      }
      return false;
    }
    selected_input_data = value_to_predicated_data_map[selected_input];
    selection_success = true;

    // ControlFlow mode: Log predecessor details.
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Predecessor blocks ("
                   << pred_count << ")\n";
      for (size_t i = 0; i < pred_count; ++i) {
        if (i < pred_count - 1) {
          llvm::outs() << "[neura-interpreter]  │  ├─ [" << i
                       << "]: " << "Block@" << predecessors[i];
        } else {
          llvm::outs() << "[neura-interpreter]  │  └─ [" << i
                       << "]: " << "Block@" << predecessors[i];
        }
        if (i == pred_index) {
          llvm::outs() << " (→ current path)\n";
        } else {
          llvm::outs() << "\n";
        }
      }
    }

  } else { // DataFlow mode
    // DataFlow mode: Selects first valid input (with true predicate).
    bool found_valid_input = false;
    for (size_t i = 0; i < input_count; ++i) {
      Value input = inputs[i];
      if (!value_to_predicated_data_map.count(input)) {
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  ├─ Input [" << i
                       << "] not found, skipping\n";
        }
        continue;
      }
      auto input_data = value_to_predicated_data_map[input];
      if (input_data.predicate) {
        selected_input_data = input_data;
        found_valid_input = true;
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  ├─ Selected input [" << i
                       << "] (latest valid)\n";
        }
        break;
      }
    }

    // DataFlow mode: Falls back to the first input with a false predicate if
    // there are no valid inputs.
    if (!found_valid_input) {
      Value first_input = inputs[0];
      if (value_to_predicated_data_map.count(first_input)) {
        auto first_data = value_to_predicated_data_map[first_input];
        selected_input_data.value = first_data.value;
        selected_input_data.is_vector = first_data.is_vector;
        selected_input_data.vector_data = first_data.vector_data;
        selected_input_data.predicate = false;
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  ├─ No valid input, using first "
                          "input with pred=false\n";
        }
      } else {
        // Edge case: First input is undefined; initializes default values.
        selected_input_data.value = 0.0f;
        selected_input_data.predicate = false;
        selected_input_data.is_vector = false;
        selected_input_data.vector_data = {};
      }
    }
    selection_success =
        true; // DataFlow mode always considers selection successful.

    // DataFlow mode: Logs input values.
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Input values (" << input_count
                   << ")\n";
      for (size_t i = 0; i < input_count; ++i) {
        Value input = inputs[i];
        if (value_to_predicated_data_map.count(input)) {
          auto input_data = value_to_predicated_data_map[input];
          const std::string prefix = (i < input_count - 1) ? "│ ├─" : "│ └─";
          llvm::outs() << "[neura-interpreter]  " << prefix << "[" << i
                       << "]:" << "value = " << input_data.value << ","
                       << "pred = " << input_data.predicate << "\n";
        } else {
          const std::string prefix = (i < input_count - 1) ? "│ ├─" : "│ └─";
          llvm::outs() << "[neura-interpreter]  " << prefix << "[" << i
                       << "]: <undefined>\n";
        }
      }
    }
  }

  // Checks for changes.
  bool is_updated = false;
  if (value_to_predicated_data_map.count(op.getResult())) {
    auto old_result = value_to_predicated_data_map[op.getResult()];
    is_updated = selected_input_data.isUpdatedComparedTo(old_result);
  } else {
    is_updated = true;
  }

  PredicatedData result = selected_input_data;
  result.is_updated = is_updated;
  result.is_reserve = false;
  value_to_predicated_data_map[op.getResult()] = result;

  if (isVerboseMode()) {
    if (!isDataflowMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Result    : " << op.getResult()
                   << "\n";
      llvm::outs() << "[neura-interpreter]     └─ Value : " << result.value
                   << ", [Pred = " << result.predicate << "]\n";
    } else {
      llvm::outs() << "[neura-interpreter]  └─ Execution "
                   << (selection_success ? "succeeded" : "partially succeeded")
                   << " | Result: value = " << result.value
                   << ", pred = " << result.predicate
                   << ", is_updated = " << result.is_updated << "\n";
    }
  }

  return selection_success;
}

/**
 * @brief Handles the execution of a Neura reservation operation
 * (neura.reserve) by creating a placeholder value for future use.
 *
 * The reserve operation allocates a placeholder value with predefined initial
 * properties: a numeric value of 0.0f, a predicate set to false (initially
 * invalid), and the is_reserve flag set to indicate that this value is
 * reserved. It is typically used to allocate a value slot that will be
 * updated later with actual data during execution.
 *
 * @param op                             The neura.reserve operation to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 * reserved placeholder will be stored, keyed by the operation’s result value
 * @return bool                          Always returns true, as reservation
 * is guaranteed to succeed
 */
bool handleReserveOp(
    neura::ReserveOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.reserve:\n";
  }
  Value result = op.getResult();

  auto it = value_to_predicated_data_map.find(result);
  if (it != value_to_predicated_data_map.end()) {
    it->second.is_reserve = true;

    if (isVerboseMode()) {
      llvm::outs()
          << "[neura-interpreter]  └─ Preserving existing placeholder: "
          << result << "\n";
      llvm::outs() << "[neura-interpreter]     ├─ Current value     : "
                   << it->second.value << "\n";
      llvm::outs() << "[neura-interpreter]     ├─ Current predicate : "
                   << it->second.predicate << "\n";
      llvm::outs() << "[neura-interpreter]     └─ Type              : "
                   << result.getType() << "\n";
    }
  } else {
    PredicatedData placeholder;
    placeholder.value = 0.0f; /* Initial value sets to 0.0f. */
    placeholder.predicate =
        false; /* Initially marked as invalid (predicate false). */
    placeholder.is_reserve =
        true; /* Flag to indicate this is a reserved placeholder. */

    value_to_predicated_data_map[result] = placeholder;

    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Created placeholder  : "
                   << result << "\n";
      llvm::outs() << "[neura-interpreter]     ├─ Initial value     : 0.0f\n";
      llvm::outs() << "[neura-interpreter]     ├─ Initial predicate : false\n";
      llvm::outs() << "[neura-interpreter]     └─ Type              : "
                   << result.getType() << "\n";
    }
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura control move operation
 *        (neura.ctrl_mov), supporting both control flow and data flow modes.
 *
 * This function processes neura.ctrl_mov operations by copying data from a
 * source value to a reserved target placeholder, with behavior determined
 * by the CtrlMovMode parameter:
 * - In control flow mode, the source is unconditionally copied to the target
 *   after validating type matching, ensuring strict consistency. The
 * operation fails on critical validation errors (e.g., missing source/target,
 * type mismatch).
 * - In data flow mode, the copy occurs only if the source predicate is true.
 *   Updates are tracked via the `is_updated` flag, computed by comparing the
 *   new value with the target’s previous state, to propagate changes to
 *   dependent operations.
 *
 * Both modes validate that the source exists in the value map and that the
 * target is a reserved placeholder (created via neura.reserve). Vector and
 * scalar type consistency is preserved, with vector data copied only if
 * the source is a vector.
 *
 * @param op                             The neura.ctrl_mov operation to
 * handle
 * @param value_to_predicated_data_map   Reference to the map storing source
 *                                       and target values along with metadata
 *                                       (value, predicate, type flags)
 * @return bool                          True if the operation succeeds; false
 *                                       only for critical errors (e.g.,
 * invalid target) in control flow mode
 */
bool handleCtrlMovOp(
    neura::CtrlMovOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.ctrl_mov"
                 << (isDataflowMode() ? "(dataflow)" : "") << ":\n";
  }

  Value source = op.getValue();
  Value target = op.getTarget();

  if (!value_to_predicated_data_map.count(source)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Source value not found "
                      "in value map\n";
    }
    return false;
  }

  if (!value_to_predicated_data_map.count(target) ||
      !value_to_predicated_data_map[target].is_reserve) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Target is not a reserve "
                      "placeholder\n";
    }
    return false;
  }

  const auto &source_data = value_to_predicated_data_map[source];
  auto &target_data = value_to_predicated_data_map[target];

  if (!isDataflowMode() && source.getType() != target.getType()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Type mismatch (source="
                   << source.getType() << ", target = " << target.getType()
                   << ")\n";
    }
    return false;
  }

  const PredicatedData old_target_data = target_data;
  const bool should_update =
      isDataflowMode() ? (source_data.predicate == 1) : true;

  if (should_update) {
    target_data.value = source_data.value;
    target_data.predicate = source_data.predicate;
    target_data.is_vector = source_data.is_vector;
    if (source_data.is_vector) {
      target_data.vector_data = source_data.vector_data;
    }
  } else if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Skip update: Source predicate "
                    "invalid (pred="
                 << source_data.predicate << ")\n";
  }

  target_data.is_updated = target_data.isUpdatedComparedTo(old_target_data);

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Source: " << source
                 << ", value = " << source_data.value
                 << ", [pred = " << source_data.predicate << "]\n"
                 << "[neura-interpreter]  ├─ Target: " << target
                 << ", value = " << target_data.value
                 << ", [pred = " << target_data.predicate << "]\n"
                 << "[neura-interpreter]  └─ Updated (is_updated = "
                 << target_data.is_updated << ")\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura return operation (neura.return)
 *        by processing and outputting return values.
 *
 * This function processes Neura's return operations, which return zero or
 * more values from a function. It retrieves each return value from the
 * value map, validates its existence, and prints it in a human-readable
 * format in verbose mode. Scalar values are printed directly, while vector
 * values are formatted as a comma-separated list, with elements having an
 * invalid predicate displayed as 0.0f. If no return values are present,
 * the operation indicates a void return.
 *
 * @param op                             The neura.return operation to handle
 * @param value_to_predicated_data_map   Reference to the map storing the
 *                                       return values to be processed
 * @return bool                          True if all return values are
 * successfully processed; false if any value is missing from the value map
 */
bool handleNeuraReturnOp(
    neura::ReturnOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map,
    bool &has_valid_result) {
  has_valid_result = false;
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.return:\n";
  }

  auto return_values = op.getValues();
  // Handles void return (no values).
  if (return_values.empty()) {
    llvm::outs() << "[neura-interpreter]  → Output: (void)\n";
    return true;
  }

  // Collects and validates return values from the value map.
  std::vector<PredicatedData> results;
  for (Value val : return_values) {
    has_valid_result = true;
    if (!value_to_predicated_data_map.count(val)) {
      llvm::errs()
          << "[neura-interpreter]  └─ Return value not found in value map\n";
      return false;
    }
    results.push_back(value_to_predicated_data_map[val]);
    if (!value_to_predicated_data_map[val].predicate) {
      has_valid_result = false;
      break; // If any return value is invalid, do not terminate.
    }
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Return values:\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto &data = results[i];
      // Prints vector values with predicate check (0.0f if predicate is
      // false).
      if (data.is_vector) {
        llvm::outs() << "[neura-interpreter]  │  └─ vector = [";
        for (size_t j = 0; j < data.vector_data.size(); ++j) {
          float val = data.predicate ? data.vector_data[j] : 0.0f;
          llvm::outs() << llvm::format("%.6f", val);
          if (j != data.vector_data.size() - 1)
            llvm::outs() << ", ";
        }
        llvm::outs() << "]";
      } else {
        // Prints scalar value with predicate check (0.0f if predicate is
        // false).
        float val = data.predicate ? data.value : 0.0f;
        llvm::outs() << "[neura-interpreter]  │  └─"
                     << llvm::format("%.6f", val);
      }
      llvm::outs() << ", [pred = " << data.predicate << "]\n";
    }
    llvm::outs()
        << "[neura-interpreter]  └─ Execution terminated successfully\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura conditional grant operation
 *        (neura.grant_predicate) by updating a value's predicate based on
 *        a new condition.
 *
 * This function processes Neura's grant_predicate operations, which take
 * exactly two operands: a source value and a new predicate value. It updates
 * the source's predicate to be the logical AND of the source's original
 * predicate, the new predicate's validity, and whether the new predicate
 * value is non-zero (true). The numeric value is retained, while the
 * computed predicate is applied. Errors are returned for invalid operand
 * counts or missing operands in the value map.
 *
 * @param op                             The neura.grant_predicate operation
 * to handle
 * @param value_to_predicated_data_map   Reference to the map where the
 * updated result will be stored, keyed by the operation's result value
 * @return bool                          True if the operation is successfully
 *                                       executed; false if operands are
 * invalid or missing from the value map
 */
bool handleGrantPredicateOp(
    neura::GrantPredicateOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.grant_predicate:\n";
  }

  if (op.getOperation()->getNumOperands() != 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.grant_predicate expects "
                      "exactly 2 operands (value, new_predicate)\n";
    }
    return false;
  }

  if (!value_to_predicated_data_map.count(op.getValue()) ||
      !value_to_predicated_data_map.count(op.getPredicate())) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Source or new predicate "
                      "not found in value_to_predicated_data_map\n";
    }
    return false;
  }

  auto source = value_to_predicated_data_map[op.getValue()];
  auto new_pred = value_to_predicated_data_map[op.getPredicate()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Source: value = "
                 << source.value << ", [pred = " << source.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ New predicate: value = "
                 << new_pred.value << ", [pred = " << new_pred.predicate
                 << "]\n";
  }
  // Evaluates new predicate (non-zero value = true) and computes combined
  // result predicate.
  bool is_new_pred_true = (new_pred.value != 0.0f);
  bool result_predicate =
      source.predicate && new_pred.predicate && is_new_pred_true;

  PredicatedData result = source;
  result.predicate = result_predicate;
  result.is_vector = source.is_vector;

  if (isVerboseMode()) {
    std::string grant_status =
        result_predicate ? "Granted access" : "Denied access (predicate false)";
    llvm::outs() << "[neura-interpreter]  ├─ " << grant_status << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value
                 << ", [pred = " << result_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura one-time grant operation
 *        (neura.grant_once) by granting validity to a value exactly once.
 *
 * This function processes Neura's grant_once operations, which take either
 * a source value operand or a constant value attribute (but not both). It
 * grants validity (sets predicate to true) on the first execution, and denies
 * validity (sets predicate to false) on all subsequent executions. The
 * numeric value is retained, while the one-time predicate is applied. Errors
 * are returned for invalid operand/attribute combinations or unsupported
 * constant types.
 *
 * @param op                             The neura.grant_once operation to
 * handle
 * @param value_to_predicated_data_map   Reference to the map where the result
 *                                       will be stored, keyed by the
 * operation's result value
 * @return bool                          True if the operation is successfully
 *                                       executed; false for invalid inputs
 *                                       or unsupported types
 */
bool handleGrantOnceOp(
    neura::GrantOnceOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.grant_once:\n";
  }
  // Checks if either a value operand or constant attribute is provided.
  bool has_value = op.getValue() != nullptr;
  bool has_constant = op.getConstantValue().has_value();

  if (has_value == has_constant) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ grant_once requires exactly one "
                      "of (value operand, constant_value attribute)\n";
    }
    return false;
  }

  PredicatedData source;
  if (has_value) {
    Value input_value = op.getValue();
    if (!value_to_predicated_data_map.count(input_value)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Source value not found in "
                        "value_to_predicated_data_map\n";
      }
      return false;
    }
    source = value_to_predicated_data_map[input_value];

    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Source value: "
                   << source.value << ", [pred = " << source.predicate << "]\n";
    }
  } else {
    // Extracts and converts constant value from attribute.
    Attribute constant_attr = op.getConstantValue().value();
    if (auto int_attr = mlir::dyn_cast<IntegerAttr>(constant_attr)) {
      source.value = int_attr.getInt();
      source.predicate = false;
      source.is_vector = false;

    } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(constant_attr)) {
      source.value = float_attr.getValueAsDouble();
      source.predicate = false;
      source.is_vector = false;
    } else {
      if (isVerboseMode()) {
        llvm::errs()
            << "[neura-interpreter]  └─ Unsupported constant_value type\n";
      }
      return false;
    }

    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Constant value: " << source.value
                   << "\n";
    }
  }
  // Tracks if this operation has already granted access (static to persist
  // across calls).
  static llvm::DenseMap<Operation *, bool> granted;
  bool has_granted = granted[op.getOperation()];
  bool result_predicate = !has_granted;

  if (!has_granted) {
    granted[op.getOperation()] = true;
    if (isVerboseMode()) {
      llvm::outs()
          << "[neura-interpreter]  ├─ First access - granting predicate\n";
    }
  } else {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Subsequent access - denying "
                      "predicate\n";
    }
  }

  PredicatedData result = source;
  result.predicate = result_predicate;
  result.is_vector = source.is_vector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value
                 << ", [pred = " << result_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura unconditional grant operation
 *        (neura.grant_always) by unconditionally validating a value's
 * predicate.
 *
 * This function processes Neura's grant_always operations, which take exactly
 * one operand (a source value) and return a copy of that value with its
 * predicate set to true, regardless of the original predicate. It effectively
 * "grants" validity unconditionally. Errors are returned if the operand count
 * is not exactly one.
 *
 * @param op                             The neura.grant_always operation to
 * handle
 * @param value_to_predicated_data_map   Reference to the map where the
 * granted result will be stored, keyed by the operation's result value
 * @return bool                          True if the operation is successfully
 *                                       executed; false if the operand count
 *                                       is invalid
 */
bool handleGrantAlwaysOp(
    neura::GrantAlwaysOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.grant_always:\n";
  }

  if (op.getOperation()->getNumOperands() != 1) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.grant_always expects "
                      "exactly 1 operand (value)\n";
    }
    return false;
  }

  auto source = value_to_predicated_data_map[op.getValue()];
  // Sets the result predicate to true unconditionally.
  bool result_predicate = true;
  PredicatedData result = source;
  result.predicate = result_predicate;
  result.is_vector = source.is_vector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Source value: " << source.value
                 << ", [pred = " << source.predicate << "]\n";
    llvm::outs()
        << "[neura-interpreter]  ├─ Granting predicate unconditionally\n";
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value
                 << ", [pred = " << result_predicate << "]\n";
  }

  value_to_predicated_data_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles neura.counter: initializes bounds/step from op attributes
 * on first encounter, then produces the current index as a predicated value.
 *
 * The counter is in-bounds while its step direction has not yet reached
 * the upper bound. Counter state persists across iterations via global maps
 * (counter_state, counter_lower, counter_upper, counter_step,
 * counter_hierarchy_level).
 *
 * @param op                             The neura.counter operation
 * @param value_to_predicated_data_map   Map of SSA values to predicated data
 * @return true on success, false on error
 */
bool handleCounterOp(
    neura::CounterOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.counter:\n";
  }

  Value result = op.getCurrentIndex();

  // Initialize counter state on first encounter.
  if (!counter_state.count(result)) {
    // Read bounds from attributes (constant_bound counters store bounds as
    // attrs).
    int64_t lower = 0, upper = 1, step = 1;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("lower_bound_value"))
      lower = attr.getInt();
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("upper_bound_value"))
      upper = attr.getInt();
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("step_value"))
      step = attr.getInt();

    // Also check SSA operands for dynamic bounds.
    if (auto lb = op.getLowerBound()) {
      if (value_to_predicated_data_map.count(lb))
        lower = static_cast<int64_t>(value_to_predicated_data_map[lb].value);
    }
    if (auto ub = op.getUpperBound()) {
      if (value_to_predicated_data_map.count(ub))
        upper = static_cast<int64_t>(value_to_predicated_data_map[ub].value);
    }
    if (auto st = op.getStep()) {
      if (value_to_predicated_data_map.count(st))
        step = static_cast<int64_t>(value_to_predicated_data_map[st].value);
    }

    counter_lower[result] = lower;
    counter_upper[result] = upper;
    counter_step[result] = step;
    counter_state[result] = lower;

    // Track counter hierarchy for nested loop advancement.
    if (auto hier_attr =
            op->getAttrOfType<mlir::StringAttr>("counter_hierarchy")) {
      auto hier = hier_attr.getValue();
      if (hier == "root")
        counter_hierarchy_level[result] = 0;
      else if (hier == "relay")
        counter_hierarchy_level[result] = 1;
      else if (hier == "leaf")
        counter_hierarchy_level[result] = 2;
      else
        counter_hierarchy_level[result] = 2; // default to innermost
    } else {
      counter_hierarchy_level[result] = 2; // default to innermost
    }

    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Counter initialized: lower="
                   << lower << ", upper=" << upper << ", step=" << step
                   << ", hierarchy=" << counter_hierarchy_level[result] << "\n";
    }
  }

  int64_t current = counter_state[result];
  bool in_bounds = (counter_step[result] > 0)
                       ? (current < counter_upper[result])
                       : (current > counter_upper[result]);

  PredicatedData val;
  val.value = static_cast<float>(current);
  val.predicate = in_bounds;
  val.is_vector = false;

  value_to_predicated_data_map[result] = val;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Counter index: " << current
                 << " [pred = " << in_bounds << "]\n";
  }

  // Counter advancement: in both CF and DF modes, advancement is handled
  // centrally with nested-loop carry propagation.
  // See advanceCountersNested() below.

  return true;
}

/**
 * @brief Advances a single counter by its step value, resetting to lower
 * bound on overflow.
 *
 * @param cv  The counter SSA value to advance.
 * @return true if overflow occurred (counter wrapped around).
 */
static bool advanceOneCounter(Value cv) {
  if (!cv || !counter_state.count(cv))
    return true;
  counter_state[cv] += counter_step[cv];
  int64_t cur = counter_state[cv];
  int64_t upper = counter_upper[cv];
  int64_t step = counter_step[cv];
  bool overflow = (step > 0) ? (cur >= upper) : (cur <= upper);
  // Reset counter_state to lower_bound on overflow so that the next iteration
  // starts from the beginning of the inner loop.
  if (overflow)
    counter_state[cv] = counter_lower[cv];
  return overflow;
}

/**
 * @brief Advances a set of counter Values (one neura.kernel's counters) with
 * nested-loop carry propagation: leaf → relay → root.
 *
 * @param counter_values  The counter SSA values to advance.
 * @return true if at least one more loop iteration remains, false if all
 *         outer loops are exhausted.
 */
static bool advanceCountersNested(const SmallVector<Value> &counter_values) {
  // Classify counters by hierarchy level.
  Value root_val, relay_val, leaf_val;
  for (Value cv : counter_values) {
    if (!counter_state.count(cv))
      continue;
    int h = counter_hierarchy_level.count(cv) ? counter_hierarchy_level[cv] : 2;
    if (h == 0)
      root_val = cv;
    else if (h == 1)
      relay_val = cv;
    else
      leaf_val = cv;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  advanceCountersNested: root="
                 << (root_val ? counter_state[root_val] : -1)
                 << ", relay=" << (relay_val ? counter_state[relay_val] : -1)
                 << ", leaf=" << (leaf_val ? counter_state[leaf_val] : -1)
                 << "\n";
  }

  // Advance leaf first.
  bool overflow = advanceOneCounter(leaf_val);
  bool outermost_overflow = false;

  if (overflow) {
    if (relay_val) {
      overflow = advanceOneCounter(relay_val);
      if (overflow && root_val) {
        outermost_overflow = advanceOneCounter(root_val);
      } else if (overflow && !root_val) {
        outermost_overflow = true; // relay overflow w/o root → loop done
      }
    } else if (root_val) {
      outermost_overflow = advanceOneCounter(root_val);
    } else {
      outermost_overflow = true; // leaf overflow w/o outer → loop done
    }
  }

  if (isVerboseMode()) {
    llvm::outs()
        << "[neura-interpreter]  advanceCountersNested result: overflow="
        << overflow << ", outermost_overflow=" << outermost_overflow
        << ", has_more=" << !outermost_overflow << "\n";
  }

  return !outermost_overflow;
}

/**
 * @brief Checks if a kernel's outer-most (hierarchy 0) counter is exhausted.


/**
 * @brief Collects the mapping from kernel-scoped "%inputN" names to their
 *        memref shapes by scanning neura.kernel ops.
 *
 * Each kernel gets a unique scope ID, producing keys like "k0/%input0".
 * This prevents collisions when different kernels reuse the same %inputN name
 * for different physical memrefs.
 *
 * @param func  The top-level function containing neura.kernel ops.
 * @return Map from scoped input key to memref shape (dimension sizes).
 */
static std::map<std::string, std::vector<int64_t>>
collectInputMemRefShapes(func::FuncOp func) {
  std::map<std::string, std::vector<int64_t>> input_shapes;

  func.walk([&](neura::KernelOp kernel) {
    auto inputs = kernel.getInputs();
    int memref_idx = 0;
    for (Value input : inputs) {
      auto memref_ty = dyn_cast<MemRefType>(input.getType());
      if (!memref_ty) {
        continue;
      }
      // Use global memref ID as key prefix so the same physical memref
      // gets the same key across all kernels (preserving aliasing).
      int mid = getOrAssignMemRefId(input);
      std::string input_name = "m" + std::to_string(mid) + "/";
      std::vector<int64_t> shape_vec;
      for (int64_t s : memref_ty.getShape()) {
        shape_vec.push_back(s);
      }
      input_shapes[input_name] = std::move(shape_vec);
      memref_idx++;
    }
  });

  return input_shapes;
}

/**
 * @brief Fills simulated_memory with deterministic pseudo-random data for
 *        every coordinate of every memref input identified by "%inputN".
 *
 * @param input_shapes     Map from scoped "%inputN" key to memref shape.
 * @param global_constants Optional map from key to flat float data used to
 *                         override the pseudo-random seed for matched keys.
 */
static void seedInputMemRef(
    const std::map<std::string, std::vector<int64_t>> &input_shapes,
    const std::map<std::string, std::vector<float>> &global_constants =
        std::map<std::string, std::vector<float>>()) {
  for (auto &kv : input_shapes) {
    const std::string &base_key = kv.first;
    const std::vector<int64_t> &shape = kv.second;

    // Check if this input has a global constant override.
    bool has_global = global_constants.count(base_key) > 0;
    const std::vector<float> &global_data =
        has_global ? global_constants.at(base_key) : std::vector<float>();

    // Recursively fill all coordinates in the multi-dimensional shape.
    std::function<void(std::vector<int64_t> &, int, size_t &)> fill_rec;
    fill_rec = [&](std::vector<int64_t> &coords, int dim, size_t &flat_idx) {
      if (dim == static_cast<int>(shape.size())) {
        std::string key = base_key;
        for (int64_t c : coords) {
          key += "[" + std::to_string(c) + "]";
        }
        if (has_global && flat_idx < global_data.size()) {
          simulated_memory[key] = global_data[flat_idx++];
        } else {
          uint64_t hash = 0x9e3779b9;
          for (int64_t c : coords) {
            hash ^= static_cast<uint64_t>(c) + 0x9e3779b9 + (hash << 6) +
                    (hash >> 2);
          }
          // Deterministic float in [-1, 1) from hash.
          float val =
              static_cast<float>(static_cast<int32_t>(hash & 0xFFFFFFFF)) /
              2147483648.0f;
          simulated_memory[key] = val;
        }
        return;
      }
      for (int64_t i = 0; i < shape[dim]; ++i) {
        coords.push_back(i);
        fill_rec(coords, dim + 1, flat_idx);
        coords.pop_back();
      }
    };

    std::vector<int64_t> coords;
    size_t flat_idx = 0;
    fill_rec(coords, 0, flat_idx);
  }
}

/**
 * @brief Initializes simulated memory for all memref function arguments with
 *        deterministic pseudo-random data so that load_indexed operations
 *        see meaningful (non-zero) values.
 *
 * Scans neura.kernel ops to discover memref inputs, correlates them with
 * "%inputN" names used by neura.constant / load_indexed / store_indexed,
 * and seeds simulated_memory for every coordinate in each memref shape.
 *
 * Also registers function arguments in the value map.
 *
 * @param func                           The function whose memref args to seed
 * @param value_to_predicated_data_map   Value map (updated with PredicatedData
 *                                       entries for each argument Value)
 */
static void initMemRefArgs(
    func::FuncOp func,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  // Collect kernel-scoped "%inputN" → shape mapping from neura.kernel ops.
  auto input_shapes = collectInputMemRefShapes(func);

  //  Build memref-ID-based key → flat float data map from
  //  memref.global constants.
  std::map<std::string, std::vector<float>> global_constants;
  auto module = func->getParentOfType<ModuleOp>();
  if (module) {
    // First pass: index all memref.global by symbol name → flat float data.
    std::map<std::string, std::vector<float>> global_data_map;
    for (auto global_op : module.getOps<memref::GlobalOp>()) {
      std::string name = global_op.getSymName().str();
      auto init_attr = global_op.getInitialValueAttr();
      if (auto dense_attr =
              llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(init_attr)) {
        std::vector<float> data;
        auto vals = dense_attr.getValues<float>();
        data.assign(vals.begin(), vals.end());
        global_data_map[name] = std::move(data);
      } else if (auto res_attr =
                     llvm::dyn_cast_or_null<mlir::DenseF32ResourceElementsAttr>(
                         init_attr)) {
        auto data = res_attr.tryGetAsArrayRef();
        if (data) {
          global_data_map[name] =
              std::vector<float>(data->begin(), data->end());
        }
      } else if (auto res_attr =
                     llvm::dyn_cast_or_null<mlir::DenseF64ResourceElementsAttr>(
                         init_attr)) {
        auto data = res_attr.tryGetAsArrayRef();
        if (data) {
          global_data_map[name] =
              std::vector<float>(data->begin(), data->end());
        }
      }
    }

    // Second pass: for each neura.kernel, map memref inputs to memref-ID keys
    // and check if they come from memref.get_global.
    func.walk([&](neura::KernelOp kernel) {
      auto inputs = kernel.getInputs();
      int memref_idx = 0;
      for (Value input : inputs) {
        auto memref_ty = dyn_cast<MemRefType>(input.getType());
        if (!memref_ty) {
          continue;
        }
        int mid = getOrAssignMemRefId(input);
        std::string input_name = "m" + std::to_string(mid) + "/";
        // Check if this input is defined by memref.get_global.
        if (auto get_global = input.getDefiningOp<memref::GetGlobalOp>()) {
          std::string global_name = get_global.getName().str();
          if (global_data_map.count(global_name) &&
              !global_constants.count(input_name)) {
            global_constants[input_name] = global_data_map[global_name];
          }
        }
        memref_idx++;
      }
    });
  }

  // Seed simulated_memory for each input memref (with global overrides).
  seedInputMemRef(input_shapes, global_constants);

  //  Register function arguments in value map.
  for (Value arg : func.getArguments()) {
    PredicatedData pd;
    pd.value = 0.0f;
    pd.predicate = true;
    value_to_predicated_data_map[arg] = pd;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Initialized simulated memory with "
                 << simulated_memory.size() << " entries for "
                 << input_shapes.size() << " memref input(s)";
    if (!global_constants.empty()) {
      llvm::outs() << " (" << global_constants.size()
                   << " from global constants)";
    }
    llvm::outs() << "\n";
  }
}

bool handleLoadIndexedOp(
    neura::LoadIndexedOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.load_indexed:\n";
  }

  Value result = op.getResult();

  // Build a memory key with global memref ID prefix (preserves aliasing
  // across kernels) + index values.
  std::string mem_key;
  std::string attr_str;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("lhs_value"))
    attr_str = attr.getValue().str();
  else if (auto attr = op->getAttrOfType<mlir::StringAttr>("rhs_value"))
    attr_str = attr.getValue().str();

  Value memref_val = getMemRefForInputN(op.getOperation(), attr_str);
  // Fallback: use getBase() for accurate memref identification when
  // lhs_value/rhs_value attributes are absent (common in DF-lowered IR).
  if (!memref_val) {
    Value base = op.getBase();
    if (base)
      memref_val = resolveKernelBlockArg(base);
  }
  // Ensure the memref has a global ID (lazy assignment for alloc'd memrefs).
  if (memref_val && !memref_value_to_id.count(memref_val))
    getOrAssignMemRefId(memref_val);
  if (memref_val && memref_value_to_id.count(memref_val))
    mem_key = "m" + std::to_string(memref_value_to_id[memref_val]) + "/";
  else
    mem_key = "m?/"; // fallback

  // Append index values to the key.
  for (Value idx : op.getIndices()) {
    if (value_to_predicated_data_map.count(idx)) {
      mem_key += "[" +
                 std::to_string(static_cast<int64_t>(
                     value_to_predicated_data_map[idx].value)) +
                 "]";
    }
  }

  PredicatedData val;
  if (simulated_memory.count(mem_key)) {
    val.value = simulated_memory[mem_key];
    val.predicate = true;
  } else {
    // First access: initialize with 0.0.
    val.value = 0.0f;
    val.predicate = true;
  }
  val.is_vector = false;

  value_to_predicated_data_map[result] = val;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Load from " << mem_key
                 << ": value = " << val.value << " [pred = " << val.predicate
                 << "]\n";
  }

  return true;
}

/**
 * @brief Handler for neura.store_indexed.
 *
 * Simulates an indexed memory store. Writes the value operand into the
 * simulated memory model.
 */
bool handleStoreIndexedOp(
    neura::StoreIndexedOp op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.store_indexed:\n";
  }

  // Get the value to store.
  Value store_val = op.getValue();
  if (!value_to_predicated_data_map.count(store_val)) {
    // In dataflow mode the value may be a kernel block argument that
    // refers to a kernel input.  Resolve it so we can find the
    // input's PredicatedData.
    store_val = resolveKernelBlockArg(store_val);
    if (!value_to_predicated_data_map.count(store_val)) {
      if (isVerboseMode()) {
        llvm::errs()
            << "[neura-interpreter]  └─ Store value not found in map\n";
      }
      return false;
    }
  }

  PredicatedData data = value_to_predicated_data_map[store_val];

  // Respect predicate: only write when the store is actually enabled.
  if (!data.predicate) {
    if (isVerboseMode()) {
      llvm::outs()
          << "[neura-interpreter]  └─ Store skipped: predicate is false\n";
    }
    return true;
  }

  // Build a memory key with global memref ID prefix (preserves aliasing
  // across kernels) + index values.
  //
  // FoldConstantPass convention:
  //   rhs_value = folded BASE  operand (target memref) — try this first
  //   lhs_value = folded VALUE operand (data to store)   — fallback
  std::string mem_key;
  std::string attr_str;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("rhs_value"))
    attr_str = attr.getValue().str();
  else if (auto attr = op->getAttrOfType<mlir::StringAttr>("lhs_value"))
    attr_str = attr.getValue().str();

  Value memref_val = getMemRefForInputN(op.getOperation(), attr_str);

  // Fallback: use getBase() for accurate memref identification when
  // lhs_value/rhs_value attributes are absent (common in DF-lowered IR).
  if (!memref_val) {
    Value base = op.getBase();
    if (base)
      memref_val = resolveKernelBlockArg(base);
  }
  // Ensure the memref has a global ID (lazy assignment for alloc'd memrefs).
  if (memref_val && !memref_value_to_id.count(memref_val))
    getOrAssignMemRefId(memref_val);

  if (memref_val && memref_value_to_id.count(memref_val))
    mem_key = "m" + std::to_string(memref_value_to_id[memref_val]) + "/";
  else
    mem_key = "m?/"; // fallback

  for (Value idx : op.getIndices()) {
    if (value_to_predicated_data_map.count(idx)) {
      mem_key += "[" +
                 std::to_string(static_cast<int64_t>(
                     value_to_predicated_data_map[idx].value)) +
                 "]";
    }
  }

  simulated_memory[mem_key] = data.value;

  llvm::outs() << "[neura-interpreter]  Store to " << mem_key
               << ": value = " << data.value << "\n";

  return true;
}

/**
 * @brief Recursively collect operations from a region (including nested
 * neura.kernel bodies) into a flat op sequence for dataflow analysis.
 *
 * @param region  The MLIR region to traverse.
 * @param op_seq  Output vector to append operations into.
 */
static void collectOpsFromRegion(mlir::Region &region,
                                 std::vector<Operation *> &op_seq) {
  for (auto &block : region) {
    for (auto &op : block) {
      op_seq.push_back(&op);
      // Recurse into neura.kernel and taskflow.task regions.
      if (auto kernel_op = dyn_cast<neura::KernelOp>(op)) {
        collectOpsFromRegion(kernel_op.getBody(), op_seq);
      }
      if (auto task_op = dyn_cast<taskflow::TaskflowTaskOp>(op)) {
        collectOpsFromRegion(task_op.getBody(), op_seq);
      }
    }
  }
}

/**
 * @brief Generic operation handling function that unifies type checking for
 * both execution modes
 *
 * Processes core logic for all operation types and returns handling results.
 * This function abstracts the common operation processing logic while
 * accommodating mode-specific requirements through optional parameters for
 * control flow information.
 *
 * @param op                              The operation to process
 * @param value_to_predicated_data_map    Reference to map storing predicated
 * data for values
 * @param current_block                   Pointer to current block (control
 * flow mode only)
 * @param last_visited_block              Pointer to previously visited block
 * (control flow mode only)
 * @return OperationHandleResult          Structure containing processing
 * status and control flags
 */
OperationHandleResult handleOperation(
    Operation *op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map,
    Block **current_block = nullptr, Block **last_visited_block = nullptr) {
  OperationHandleResult result{true, false, false};

  if (auto const_op = dyn_cast<mlir::arith::ConstantOp>(op)) {
    result.success =
        handleArithConstantOp(const_op, value_to_predicated_data_map);
  } else if (auto const_op = dyn_cast<neura::ConstantOp>(op)) {
    result.success =
        handleNeuraConstantOp(const_op, value_to_predicated_data_map);
  } else if (auto mov_op = dyn_cast<neura::DataMovOp>(op)) {
    value_to_predicated_data_map[mov_op.getResult()] =
        value_to_predicated_data_map[mov_op.getOperand()];
  } else if (auto counter_op = dyn_cast<neura::CounterOp>(op)) {
    result.success = handleCounterOp(counter_op, value_to_predicated_data_map);
  } else if (auto load_op = dyn_cast<neura::LoadIndexedOp>(op)) {
    result.success = handleLoadIndexedOp(load_op, value_to_predicated_data_map);
  } else if (auto store_op = dyn_cast<neura::StoreIndexedOp>(op)) {
    result.success =
        handleStoreIndexedOp(store_op, value_to_predicated_data_map);
  } else if (auto add_op = dyn_cast<neura::AddOp>(op)) {
    result.success = handleAddOp(add_op, value_to_predicated_data_map);
  } else if (auto sub_op = dyn_cast<neura::SubOp>(op)) {
    result.success = handleSubOp(sub_op, value_to_predicated_data_map);
  } else if (auto mul_op = dyn_cast<neura::MulOp>(op)) {
    result.success = handleMulOp(mul_op, value_to_predicated_data_map);
  } else if (auto div_op = dyn_cast<neura::DivOp>(op)) {
    result.success = handleDivOp(div_op, value_to_predicated_data_map);
  } else if (auto fadd_op = dyn_cast<neura::FAddOp>(op)) {
    result.success = handleFAddOp(fadd_op, value_to_predicated_data_map);
  } else if (auto fsub_op = dyn_cast<neura::FSubOp>(op)) {
    result.success = handleFSubOp(fsub_op, value_to_predicated_data_map);
  } else if (auto fmul_op = dyn_cast<neura::FMulOp>(op)) {
    result.success = handleFMulOp(fmul_op, value_to_predicated_data_map);
  } else if (auto fdiv_op = dyn_cast<neura::FDivOp>(op)) {
    result.success = handleFDivOp(fdiv_op, value_to_predicated_data_map);
  } else if (auto fmax_op = dyn_cast<neura::FMaxOp>(op)) {
    result.success = handleFMaxOp(fmax_op, value_to_predicated_data_map);
  } else if (auto fmin_op = dyn_cast<neura::FMinOp>(op)) {
    result.success = handleFMinOp(fmin_op, value_to_predicated_data_map);
  } else if (auto vfmul_op = dyn_cast<neura::VFMulOp>(op)) {
    result.success = handleVFMulOp(vfmul_op, value_to_predicated_data_map);
  } else if (auto fadd_fadd_op = dyn_cast<neura::FAddFAddOp>(op)) {
    result.success =
        handleFAddFAddOp(fadd_fadd_op, value_to_predicated_data_map);
  } else if (auto fmul_fadd_op = dyn_cast<neura::FMulFAddOp>(op)) {
    result.success =
        handleFMulFAddOp(fmul_fadd_op, value_to_predicated_data_map);
  } else if (auto ret_op = dyn_cast<func::ReturnOp>(op)) {
    bool has_valid_result = false;
    result.success = handleFuncReturnOp(ret_op, value_to_predicated_data_map,
                                        has_valid_result);
    if (isDataflowMode()) {
      result.is_terminated = has_valid_result;
    } else {
      result.is_terminated = true;
    }
  } else if (auto fcmp_op = dyn_cast<neura::FCmpOp>(op)) {
    result.success = handleFCmpOp(fcmp_op, value_to_predicated_data_map);
  } else if (auto icmp_op = dyn_cast<neura::ICmpOp>(op)) {
    result.success = handleICmpOp(icmp_op, value_to_predicated_data_map);
  } else if (auto or_op = dyn_cast<neura::OrOp>(op)) {
    result.success = handleOrOp(or_op, value_to_predicated_data_map);
  } else if (auto and_op = dyn_cast<neura::AndOp>(op)) {
    result.success = handleAndOp(and_op, value_to_predicated_data_map);
  } else if (auto not_op = dyn_cast<neura::NotOp>(op)) {
    result.success = handleNotOp(not_op, value_to_predicated_data_map);
  } else if (auto sel_op = dyn_cast<neura::SelOp>(op)) {
    result.success = handleSelOp(sel_op, value_to_predicated_data_map);
  } else if (auto cast_op = dyn_cast<neura::CastOp>(op)) {
    result.success = handleCastOp(cast_op, value_to_predicated_data_map);
  } else if (auto rsqrt_op = dyn_cast<neura::RsqrtOp>(op)) {
    result.success = handleRsqrtOp(rsqrt_op, value_to_predicated_data_map);
  } else if (auto exp_op = dyn_cast<neura::ExpOp>(op)) {
    result.success = handleExpOp(exp_op, value_to_predicated_data_map);
  } else if (auto gep_op = dyn_cast<neura::GEP>(op)) {
    result.success = handleGEPOp(gep_op, value_to_predicated_data_map);
  } else if (auto br_op = dyn_cast<neura::Br>(op)) {
    // Branch operations only need block handling in control flow mode.
    if (current_block && last_visited_block) {
      result.success = handleBrOp(br_op, value_to_predicated_data_map,
                                  *current_block, *last_visited_block);
      result.is_branch = true; // Marks as branch to reset index.
    } else {
      result.success = false;
    }
  } else if (auto cond_br_op = dyn_cast<neura::CondBr>(op)) {
    if (current_block && last_visited_block) {
      result.success = handleCondBrOp(cond_br_op, value_to_predicated_data_map,
                                      *current_block, *last_visited_block);
      result.is_branch = true;
    } else {
      result.success = false;
    }
  } else if (auto phi_op = dyn_cast<neura::PhiOp>(op)) {
    // Phi operations need block information in control flow mode, but not in
    // data flow.
    if (current_block && last_visited_block) {
      result.success = handlePhiOp(phi_op, value_to_predicated_data_map,
                                   *current_block, *last_visited_block);
    } else {
      result.success = handlePhiOp(phi_op, value_to_predicated_data_map);
    }
  } else if (auto phi_start_op = dyn_cast<neura::PhiStartOp>(op)) {
    // PhiStart operations need block information in control flow mode, but
    // not in data flow.
    if (current_block && last_visited_block) {
      result.success =
          handlePhiStartOp(phi_start_op, value_to_predicated_data_map,
                           *current_block, *last_visited_block);
    } else {
      result.success =
          handlePhiStartOp(phi_start_op, value_to_predicated_data_map);
    }
  } else if (auto reserve_op = dyn_cast<neura::ReserveOp>(op)) {
    result.success = handleReserveOp(reserve_op, value_to_predicated_data_map);
  } else if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(op)) {
    result.success = handleCtrlMovOp(ctrl_mov_op, value_to_predicated_data_map);
  } else if (auto return_op = dyn_cast<neura::ReturnOp>(op)) {
    bool has_valid_result = false;
    result.success = handleNeuraReturnOp(
        return_op, value_to_predicated_data_map, has_valid_result);
    if (isDataflowMode()) {
      result.is_terminated = has_valid_result;
    } else {
      result.is_terminated = true;
    }
  } else if (auto grant_pred_op = dyn_cast<neura::GrantPredicateOp>(op)) {
    result.success =
        handleGrantPredicateOp(grant_pred_op, value_to_predicated_data_map);
  } else if (auto grant_once_op = dyn_cast<neura::GrantOnceOp>(op)) {
    result.success =
        handleGrantOnceOp(grant_once_op, value_to_predicated_data_map);
  } else if (auto grant_always_op = dyn_cast<neura::GrantAlwaysOp>(op)) {
    result.success =
        handleGrantAlwaysOp(grant_always_op, value_to_predicated_data_map);
  } else if (isa<neura::KernelOp>(op)) {
    // neura.kernel is a container op; its body ops are already collected
    // and handled individually. The kernel op itself is a no-op.
    result.success = true;
  } else if (auto yield_op = dyn_cast<neura::YieldOp>(op)) {
    // neura.yield is the terminator for neura.kernel.
    // In control-flow mode, it triggers a loop-back to the kernel body start
    // if any counter is still in bounds; otherwise it exits the kernel.
    result.success = true;
    // The actual loop decision is made in the main control-flow loop
    // based on the is_branch flag and counter predicates.
    // We mark it as a branch so the main loop resets op_index.
    result.is_branch = true;
  } else if (isa<memref::AllocOp, memref::DeallocOp, memref::GetGlobalOp>(op)) {
    // Buffer allocation/deallocation are no-ops in interpreter mode.
    // memref.get_global data is pre-seeded into simulated_memory during init.
    result.success = true;
  } else if (isa<taskflow::TaskflowTaskOp>(op)) {
    // taskflow.task is a container op wrapping neura.kernel; its body is
    // already collected separately. Treat as no-op.
    result.success = true;
  } else if (isa<taskflow::TaskflowYieldOp>(op)) {
    // taskflow.yield is a terminator; treated as no-op.
    result.success = true;
  } else if (isa<taskflow::TaskflowCounterOp>(op)) {
    // taskflow.counter wraps neura.counter; treated as no-op.
    result.success = true;
  } else {
    llvm::errs() << "[neura-interpreter]  Unhandled op: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
    result.success = false;
  }

  return result;
}

/**
 * @brief Executes a single operation in data flow mode with update
 * propagation
 *
 * Processes the operation using the generic handler, then checks for value
 * updates and propagates changes to dependent operations. This function
 * contains data flow specific logic for dependency management and pending
 * operation queue updates.
 *
 * @param op                              The operation to execute
 * @param value_to_predicated_data_map    Reference to map storing predicated
 * data for valuess
 * @param next_pending_operation_queue    Queue to receive dependent
 * operations needing processing
 * @return bool                           True if operation executed
 * successfully, false otherwise
 */
bool executeOperation(
    Operation *op,
    llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map,
    bool &should_terminate) {
  // Saves current values to detect updates after execution.
  llvm::DenseMap<Value, PredicatedData> old_values;
  for (Value result : op->getResults()) {
    if (value_to_predicated_data_map.count(result)) {
      old_values[result] = value_to_predicated_data_map[result];
    }
  }

  // Special handling for neura.ctrl_mov target updates.
  Value ctrl_mov_target;
  if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(op)) {
    ctrl_mov_target = ctrl_mov_op.getTarget();
    if (value_to_predicated_data_map.count(ctrl_mov_target)) {
      old_values[ctrl_mov_target] =
          value_to_predicated_data_map[ctrl_mov_target];
    }
  }

  // Processes the operation using the generic handler (no block info needed
  // for data flow).
  auto handle_result = handleOperation(op, value_to_predicated_data_map);
  if (!handle_result.success) {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Operation failed, no propagation\n";
    }
    return false;
  }

  if (handle_result.is_terminated) {
    should_terminate = true;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Termination signal received\n";
    }
  }

  for (Value result : op->getResults()) {
    if (!value_to_predicated_data_map.count(result)) {
      continue;
    }

    PredicatedData &new_data = value_to_predicated_data_map[result];
    if (!old_values.count(result) ||
        old_values[result].isUpdatedComparedTo(new_data)) {
      new_data.is_updated = true;

      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  Result updated: " << result
                     << ", value = " << new_data.value
                     << ", [pred = " << new_data.predicate << "]\n";
      }
    }
  }

  // Special handling for neura.ctrl_mov target updates.
  if (ctrl_mov_target.getImpl() &&
      value_to_predicated_data_map.count(ctrl_mov_target)) {
    PredicatedData &target_data = value_to_predicated_data_map[ctrl_mov_target];

    if (!old_values.count(ctrl_mov_target) ||
        old_values[ctrl_mov_target].isUpdatedComparedTo(target_data)) {
      target_data.is_updated = true;

      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  CtrlMov target updated: "
                     << ctrl_mov_target << ", value = " << target_data.value
                     << ", [pred = " << target_data.predicate << "]\n";
      }
    }
  }

  return true;
}

/**
 * @brief Unified execution entry point supporting both data flow and control
 * flow modes
 *
 * Executes a function using either data flow or control flow semantics based
 * on the specified mode. Data flow mode processes operations when their
 * dependencies are satisfied, while control flow mode follows traditional
 * sequential execution with branch handling.
 *
 * @param func                           The function to execute
 * @param value_to_predicated_data_map   Reference to map storing predicated
 * data for values
 * @return int                           0 if execution completes
 * successfully, 1 on error
 */
int run(func::FuncOp func,
        llvm::DenseMap<Value, PredicatedData> &value_to_predicated_data_map) {
  // Initialize simulated memory for memref arguments with pseudo-random data.
  initMemRefArgs(func, value_to_predicated_data_map);

  if (isDataflowMode()) {
    // Data flow mode execution logic.
    //
    // Kernels are executed SEQUENTIALLY (in IR order) to prevent
    // interleaved execution across kernels.  In a flat DFG where all
    // kernels' counters advance simultaneously, a later kernel (e.g.
    // matmul layer 2) could read partially-computed values from an
    // earlier kernel (e.g. matmul layer 1) that hasn't finished all
    // its iterations yet.  Sequential execution guarantees that kernel
    // N+1 only sees the fully-computed output of kernel N.

    // Collect all ops, separating non-kernel ops from per-kernel body ops.
    std::vector<Operation *> non_kernel_ops;
    struct KernelGroup {
      Operation *kernel_op;              // the neura.kernel op
      std::vector<Operation *> body_ops; // body ops including yield
      SmallVector<Value> counter_values; // counter result Values
    };
    SmallVector<KernelGroup> kernel_groups;
    Operation *return_op = nullptr;

    for (auto &block : func.getBody()) {
      for (auto &op : block) {
        if (isa<func::ReturnOp>(op)) {
          return_op = &op;
          continue;
        }
        if (isa<neura::KernelOp>(op)) {
          KernelGroup group;
          group.kernel_op = &op;
          collectOpsFromRegion(cast<neura::KernelOp>(op).getBody(),
                               group.body_ops);
          // Collect counter values for this kernel
          for (Operation *body_op : group.body_ops) {
            if (auto ctr = dyn_cast<neura::CounterOp>(body_op))
              group.counter_values.push_back(ctr.getCurrentIndex());
          }
          kernel_groups.push_back(std::move(group));
        } else {
          non_kernel_ops.push_back(&op);
        }
      }
    }

    // Execute non-kernel ops once (arith.constant, memref.alloc,
    // memref.get_global, etc.).  These are pure init ops with no counters.
    if (!non_kernel_ops.empty()) {
      DependencyGraph init_graph;
      init_graph.build(non_kernel_ops);
      std::vector<Operation *> ready = init_graph.getReadyToExecuteOperations();
      while (!ready.empty()) {
        std::vector<Operation *> next_ready;
        for (Operation *op : ready) {
          bool ignored_terminate = false;
          if (!executeOperation(op, value_to_predicated_data_map,
                                ignored_terminate)) {
            return EXIT_FAILURE;
          }
          init_graph.updateAfterExecution(op);
          for (Operation *dep :
               init_graph.getReadyToExecuteConsumerOperations(op)) {
            if (init_graph.canExecute(dep) &&
                std::find(next_ready.begin(), next_ready.end(), dep) ==
                    next_ready.end())
              next_ready.push_back(dep);
          }
        }
        ready = std::move(next_ready);
      }
    }

    // Execute each kernel sequentially to exhaustion.
    for (auto &group : kernel_groups) {
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  ==================== "
                        "Starting kernel: ";
        group.kernel_op->print(llvm::outs());
        llvm::outs() << " (" << group.body_ops.size() << " ops, "
                     << group.counter_values.size() << " counters)\n";
      }

      // Map this kernel's block arguments to its input values.
      auto kernel = cast<neura::KernelOp>(group.kernel_op);
      auto inputs = kernel.getInputs();
      Block &body = kernel.getBody().front();
      unsigned num_args = body.getNumArguments();
      for (unsigned i = 0; i < num_args && i < inputs.size(); ++i) {
        Value block_arg = body.getArgument(i);
        Value input_val = inputs[i];
        if (value_to_predicated_data_map.count(input_val)) {
          value_to_predicated_data_map[block_arg] =
              value_to_predicated_data_map[input_val];
        }
      }

      DependencyGraph dep_graph;
      dep_graph.build(group.body_ops);
      std::vector<Operation *> ready = dep_graph.getReadyToExecuteOperations();

      int dfg_count = 0;

      while (!ready.empty() || dep_graph.hasUnexecutedOperations()) {
        std::vector<Operation *> next_ready;
        for (Operation *op : ready) {
          bool ignored = false;
          if (!executeOperation(op, value_to_predicated_data_map, ignored))
            return EXIT_FAILURE;

          dep_graph.updateAfterExecution(op);

          for (Operation *consumer :
               dep_graph.getReadyToExecuteConsumerOperations(op)) {
            if (dep_graph.canExecute(consumer) &&
                std::find(next_ready.begin(), next_ready.end(), consumer) ==
                    next_ready.end())
              next_ready.push_back(consumer);
          }
        }
        ready = std::move(next_ready);

        if (ready.empty()) {
          // Advance counters for this kernel
          bool has_iterations = advanceCountersNested(group.counter_values);

          if (!has_iterations)
            break;

          constexpr int MAX_DFG_ITERATIONS = 100000;
          if (dfg_count >= MAX_DFG_ITERATIONS) {
            if (isVerboseMode()) {
              llvm::outs()
                  << "[neura-interpreter]  Reached max DFG iterations ("
                  << MAX_DFG_ITERATIONS << "), stopping kernel\n";
            }
            break;
          }
          dfg_count++;

          // Reset non-constant, non-reserve values for the next iteration.
          for (auto &entry : value_to_predicated_data_map) {
            Value val = entry.first;
            // Skip kernel block arguments and function-level constants.
            if (isa<BlockArgument>(val))
              continue;
            if (val.getDefiningOp() &&
                (isa<neura::ConstantOp>(val.getDefiningOp()) ||
                 isa<mlir::arith::ConstantOp>(val.getDefiningOp()) ||
                 isa<memref::AllocOp>(val.getDefiningOp()) ||
                 isa<memref::GetGlobalOp>(val.getDefiningOp())))
              continue;
            entry.second.is_updated = false;
            entry.second.predicate = false;
          }

          // Re-map block args for next iteration
          for (unsigned i = 0; i < num_args && i < inputs.size(); ++i) {
            Value block_arg = body.getArgument(i);
            Value input_val = inputs[i];
            if (value_to_predicated_data_map.count(input_val)) {
              value_to_predicated_data_map[block_arg] =
                  value_to_predicated_data_map[input_val];
            }
          }

          dep_graph.resetForNextIteration();
          ready = dep_graph.getReadyToExecuteOperations();
          continue;
        }
      }

      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  Kernel finished after "
                     << dfg_count << " DFG iterations\n";
      }
    }

    // Execute func.return if present.
    if (return_op) {
      auto handle_result =
          handleOperation(return_op, value_to_predicated_data_map);
      if (!handle_result.success) {
        return EXIT_FAILURE;
      }
    }
  } else {
    // Control flow mode execution logic
    Block *current_block = &func.getBody().front();
    Block *last_visited_block = nullptr;
    size_t op_index = 0;
    bool is_terminated = false;

    // Stack for handling nested container ops (taskflow.task, neura.kernel).
    struct Frame {
      Block *block;
      size_t index;
      Block *last_block;
      // For kernel loops: set of counter result Values to check at yield.
      SmallVector<Value> counter_values;
      // The kernel body block to jump back to on loop continuation.
      Block *kernel_body;
    };
    SmallVector<Frame> frame_stack;

    // Main loop: processes operations sequentially through blocks.
    while (!is_terminated && current_block) {
      auto &operations = current_block->getOperations();
      if (op_index >= operations.size()) {
        // Reached end of block — pop from stack if we were in a nested block.
        if (!frame_stack.empty()) {
          Frame parent = frame_stack.pop_back_val();
          current_block = parent.block;
          op_index = parent.index;
          last_visited_block = parent.last_block;
          // Advance past the container op.
          op_index++;
          continue;
        }
        break;
      }

      Operation &op = *std::next(operations.begin(), op_index);

      // Enter nested containers to execute their body ops inline.
      if (auto kernel_op = dyn_cast<neura::KernelOp>(op)) {
        if (!kernel_op.getBody().empty()) {
          Frame frame;
          frame.block = current_block;
          frame.index = op_index;
          frame.last_block = last_visited_block;
          frame.kernel_body = &kernel_op.getBody().front();
          // Collect counter values from kernel body for loop predicate check.
          for (auto &body_op : kernel_op.getBody().front()) {
            if (isa<neura::CounterOp>(body_op)) {
              frame.counter_values.push_back(body_op.getResult(0));
            }
          }
          frame_stack.push_back(frame);
          current_block = frame.kernel_body;
          last_visited_block = nullptr;
          op_index = 0;
          continue;
        }
      }
      if (auto task_op = dyn_cast<taskflow::TaskflowTaskOp>(op)) {
        if (!task_op.getBody().empty()) {
          Frame frame;
          frame.block = current_block;
          frame.index = op_index;
          frame.last_block = last_visited_block;
          frame.kernel_body = nullptr; // taskflow.task doesn't loop
          frame_stack.push_back(frame);
          current_block = &task_op.getBody().front();
          last_visited_block = nullptr;
          op_index = 0;
          continue;
        }
      }

      // Processes operation with block information for control flow handling.
      auto handle_result = handleOperation(&op, value_to_predicated_data_map,
                                           &current_block, &last_visited_block);

      if (!handle_result.success) {
        return EXIT_FAILURE;
      }
      if (handle_result.is_terminated) {
        is_terminated = true;
        op_index++;
      } else if (handle_result.is_branch) {
        // Check if this is a neura.yield that needs loop-back handling.
        if (isa<neura::YieldOp>(op) && !frame_stack.empty() &&
            frame_stack.back().kernel_body != nullptr) {
          // Advance counters with nested-loop carry propagation
          // (leaf → relay → root).
          bool more_iterations =
              advanceCountersNested(frame_stack.back().counter_values);
          // Update predicates in the value map so that the next
          // handleCounterOp call reflects the new counter_state.
          for (Value cv : frame_stack.back().counter_values) {
            if (!value_to_predicated_data_map.count(cv))
              continue;
            if (!counter_state.count(cv))
              continue;
            int64_t cur = counter_state[cv];
            int64_t upper = counter_upper[cv];
            int64_t step = counter_step[cv];
            bool in_bounds = (step > 0) ? (cur < upper) : (cur > upper);
            value_to_predicated_data_map[cv].predicate = in_bounds;
            value_to_predicated_data_map[cv].value = static_cast<float>(cur);
          }
          if (more_iterations) {
            // Loop back to start of kernel body.
            op_index = 0;
          } else {
            // All counters exhausted; pop frame to exit kernel.
            Frame parent = frame_stack.pop_back_val();
            current_block = parent.block;
            op_index = parent.index + 1; // advance past kernel op
            last_visited_block = parent.last_block;
          }
        } else {
          // Normal branch: reset index to start of new block.
          op_index = 0;
        }
      } else {
        // Regular operations increment to next operation in block.
        op_index++;
      }
    }
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  // Parses command line arguments.
  for (int i = 0; i < argc; ++i) {
    if (std::string(argv[i]) == "--verbose") {
      setVerboseMode(true);
    } else if (std::string(argv[i]) == "--dataflow") {
      setDataflowMode(true);
    }
  }

  if (argc < 2) {
    llvm::errs() << "[neura-interpreter]  Usage: neura-interpreter "
                    "<input.mlir> [--verbose] [--dataflow]\n";
    return EXIT_FAILURE;
  }

  // Initializes MLIR context and dialects.
  DialectRegistry registry;
  registry.insert<neura::NeuraDialect, func::FuncDialect, arith::ArithDialect,
                  memref::MemRefDialect, taskflow::TaskflowDialect,
                  math::MathDialect>();

  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects();

  // Loads and parses input MLIR file.
  // Pre-process: strip ml_program dialect ops which are not registered.
  llvm::SourceMgr source_mgr;
  auto file_or_err = mlir::openInputFile(argv[1]);
  if (!file_or_err) {
    llvm::errs() << "[neura-interpreter]  Error opening file\n";
    return EXIT_FAILURE;
  }

  std::string content = file_or_err->getBuffer().str();
  // Remove ml_program.global lines (from torch-mlir export)
  {
    std::stringstream filtered;
    std::stringstream input(content);
    std::string line;
    while (std::getline(input, line)) {
      if (line.find("ml_program.global") == std::string::npos)
        filtered << line << '\n';
    }
    content = filtered.str();
  }
  auto filtered_buf = llvm::MemoryBuffer::getMemBufferCopy(content);
  source_mgr.AddNewSourceBuffer(std::move(filtered_buf), llvm::SMLoc());

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(source_mgr, &context);
  if (!module) {
    llvm::errs() << "[neura-interpreter]  Failed to parse MLIR input file\n";
    return EXIT_FAILURE;
  }

  // Initializes data structures.
  llvm::DenseMap<Value, PredicatedData> value_to_predicated_data_map;

  for (auto func : module->getOps<func::FuncOp>()) {
    if (run(func, value_to_predicated_data_map)) {
      llvm::errs() << "[neura-interpreter]  Execution failed\n";
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}