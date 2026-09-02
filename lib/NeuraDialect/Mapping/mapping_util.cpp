#include <deque>
#include <queue>

#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;
using namespace mlir::neura;

// Constants for award calculation.
constexpr int kAwardProximityScale = 1;
constexpr int kAwardBackwardProximityScale = 1;
constexpr int kAwardBaseMultiplier = 1;
constexpr int kAwardCriticalBonusDiv = 1;

// Congestion penalty coefficients (tunable).
constexpr int kStrongCongestionPenalty = 60; // used for high fan-in ops (>=3)
constexpr int kWeakCongestionPenalty = 15;   // used for low fan-in ops

namespace mlir {
namespace neura {
OperationKind getOperationKindFromMlirOp(Operation *op) {
  // Integer arithmetic operations
  if (isa<neura::AddOp>(op))
    return IAdd;
  if (isa<neura::SubOp>(op))
    return ISub;
  if (isa<neura::MulOp>(op))
    return IMul;
  if (isa<neura::DivOp>(op))
    return IDiv;
  if (isa<neura::RemOp>(op))
    return IRem;

  // Floating-point arithmetic operations
  if (isa<neura::FAddOp>(op))
    return FAdd;
  if (isa<neura::FSubOp>(op))
    return FSub;
  if (isa<neura::FMulOp>(op))
    return FMul;
  if (isa<neura::FDivOp>(op))
    return FDiv;

  // Memory operations
  if (isa<neura::LoadOp>(op))
    return ILoad;
  if (isa<neura::StoreOp>(op))
    return IStore;
  if (isa<neura::LoadIndexedOp>(op))
    return ILoadIndexed;
  if (isa<neura::StoreIndexedOp>(op))
    return IStoreIndexed;
  if (isa<neura::AllocaOp>(op))
    return IAlloca;

  // Logical operations
  if (isa<neura::OrOp>(op))
    return IOr;
  if (isa<neura::NotOp>(op))
    return INot;
  if (isa<neura::ICmpOp>(op))
    return ICmp;
  if (isa<neura::FCmpOp>(op))
    return FCmp;
  if (isa<neura::SelOp>(op))
    return ISel;

  // Type conversion operations
  if (isa<neura::CastOp>(op))
    return ICast;
  if (isa<neura::SExtOp>(op))
    return ISExt;
  if (isa<neura::ZExtOp>(op))
    return IZExt;
  if (isa<neura::ShlOp>(op))
    return IShl;

  // Vector operations
  if (isa<neura::VFMulOp>(op))
    return VFMul;

  // Fused operations
  if (isa<neura::FAddFAddOp>(op))
    return FAddFAdd;
  if (isa<neura::FMulFAddOp>(op))
    return FMulFAdd;

  // Control flow operations
  if (isa<neura::ReturnOp>(op))
    return IReturn;
  if (isa<neura::PhiOp>(op))
    return IPhi;

  // Data movement operations
  if (isa<neura::DataMovOp>(op))
    return IDataMov;
  if (isa<neura::CtrlMovOp>(op))
    return ICtrlMov;

  // Predicate operations
  if (isa<neura::ReserveOp>(op))
    return IReserve;
  if (isa<neura::GrantPredicateOp>(op))
    return IGrantPredicate;
  if (isa<neura::GrantOnceOp>(op))
    return IGrantOnce;
  if (isa<neura::GrantAlwaysOp>(op))
    return IGrantAlways;

  // Loop control operations
  if (isa<neura::LoopControlOp>(op))
    return ILoopControl;

  // Constant operations
  if (isa<neura::ConstantOp>(op))
    return IConstant;

  // Default fallback
  return IAdd;
}

// Returns true if the operation does not need CGRA tile placement.
bool is_non_materialized(Operation *op) {
  // Returns true if the operation does not need CGRA tile placement.
  return mlir::isa<neura::ReserveOp, neura::CtrlMovOp, neura::DataMovOp,
                   neura::YieldOp>(op);
}

} // namespace neura
} // namespace mlir

namespace {
// Traverses (backward) the operation graph starting from the given operation
// towards reserve_value.
void traverseAlongPath(Operation *op, Value reserve_value,
                       std::deque<Operation *> &current_path,
                       DenseSet<Operation *> &visited_in_path,
                       SmallVector<RecurrenceCycle, 4> &collected_paths) {
  if (!op || visited_in_path.contains(op)) {
    if (visited_in_path.contains(op)) {
      llvm::errs() << "Skipping already visited operation: " << *op << "\n";
    }
    return;
  }
  visited_in_path.insert(op);
  current_path.push_front(op);

  for (Value operand : op->getOperands()) {
    if (operand == reserve_value) {
      Operation *res_op = reserve_value.getDefiningOp();
      if (res_op) {
        current_path.push_front(res_op);
      }

      int effective_length = 0;
      for (Operation *op : current_path) {
        // Skips the non-materialized ops when counting the cycle length.
        if (!is_non_materialized(op)) {
          ++effective_length;
        }
      }
      collected_paths.push_back(
          RecurrenceCycle{/* operations = */ SmallVector<Operation *>(
                              current_path.begin(), current_path.end()),
                          /* length = */ static_cast<int>(effective_length)});

      if (res_op) {
        current_path.pop_front();
      }
      continue;
    }

    if (Operation *def_op = operand.getDefiningOp()) {
      traverseAlongPath(def_op, reserve_value, current_path, visited_in_path,
                        collected_paths);
    }
  }

  current_path.pop_front();
  visited_in_path.erase(op);
}

} // namespace

SmallVector<RecurrenceCycle, 4>
mlir::neura::collectRecurrenceCycles(Region &region) {
  SmallVector<RecurrenceCycle, 4> recurrence_cycles;

  region.walk([&](neura::CtrlMovOp ctrl_mov_op) {
    Value target = ctrl_mov_op.getTarget();
    neura::ReserveOp reserve_op = target.getDefiningOp<neura::ReserveOp>();
    if (!reserve_op) {
      return;
    }

    Value reserve_value = reserve_op.getResult();
    Value ctrl_mov_from = ctrl_mov_op.getValue();

    Operation *parent_op = ctrl_mov_from.getDefiningOp();
    if (!parent_op) {
      return;
    }

    std::deque<Operation *> current_path;
    SmallVector<RecurrenceCycle, 4> collected_paths;
    DenseSet<Operation *> visited_in_path;
    llvm::errs() << "Collecting recurrence cycles from back edge: parent_op "
                 << *parent_op << "->" << reserve_op << "\n";
    traverseAlongPath(parent_op, reserve_value, current_path, visited_in_path,
                      collected_paths);

    for (RecurrenceCycle &cycle : collected_paths) {
      cycle.operations.push_back(ctrl_mov_op);
      recurrence_cycles.push_back(std::move(cycle));
    }
  });

  return recurrence_cycles;
}

int mlir::neura::calculateResMii(Region &region,
                                 const Architecture &architecture) {
  int num_ops = 0;

  // Count all "compute" operations (non-terminators, non-block ops).
  region.walk([&](Operation *op) {
    // Skips non-materialized ops.
    if (isa<func::FuncOp>(op) ||
        isa<neura::CtrlMovOp, neura::DataMovOp, neura::ReserveOp>(op)) {
      return;
    }
    // Skips operations inside fused_op regions
    Operation *parent_op = op->getParentOp();
    if (isa<neura::FusedOp>(parent_op)) {
      return;
    }
    ++num_ops;
  });

  llvm::errs() << "[calculateResMii] Total operations: " << num_ops << "\n";

  // Avoid divide-by-zero
  int num_tiles = std::max(1, architecture.getNumTiles());

  return llvm::divideCeil(num_ops, num_tiles);
}

std::vector<Operation *>
mlir::neura::getTopologicallySortedOps(Region &region) {
  std::vector<Operation *> sorted_ops;
  llvm::DenseMap<Operation *, int> pending_deps;
  std::deque<Operation *> ready_queue;

  // Collects recurrence cycle ops.
  SmallVector<RecurrenceCycle> recurrence_cycles =
      collectRecurrenceCycles(region);
  llvm::DenseSet<Operation *> recurrence_ops;
  for (const RecurrenceCycle &cycle : recurrence_cycles) {
    for (Operation *op : cycle.operations) {
      recurrence_ops.insert(op);
    }
  }
  // Counts unresolved dependencies for each op.
  region.walk([&](Operation *op) {
    // if (op == func_op) {
    //   return;
    // }
    int dep_count = 0;
    for (Value operand : op->getOperands()) {
      if (operand.getDefiningOp()) {
        ++dep_count;
      }
    }
    pending_deps[op] = dep_count;
    if (dep_count == 0) {
      // TODO: Prioritize recurrence ops. But cause compiled II regression.
      // https://github.com/coredac/dataflow/issues/59.
      if (recurrence_ops.contains(op)) {
        // ready_queue.push_front(op);
        ready_queue.push_back(op);
      } else {
        ready_queue.push_back(op);
      }
    }
  });

  // BFS-style topological sort with recurrence priority.
  while (!ready_queue.empty()) {
    Operation *op = ready_queue.front();
    ready_queue.pop_front();
    sorted_ops.push_back(op);

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (--pending_deps[user] == 0) {
          // TODO: Prioritize recurrence ops. But cause compiled II regression.
          // https://github.com/coredac/dataflow/issues/59.
          if (recurrence_ops.contains(user)) {
            // ready_queue.push_front(user);
            ready_queue.push_back(user);
          } else {
            ready_queue.push_back(user);
          }
        }
      }
    }
  }

  return sorted_ops;
}

std::vector<std::vector<Operation *>>
mlir::neura::getOpsInAlapLevels(const std::vector<Operation *> &sorted_ops,
                                const std::set<Operation *> &critical_ops) {
  llvm::DenseMap<Operation *, int> op_level;
  int max_level = 0;

  // Step 1: Computes raw ALAP level: longest path to any sink.
  for (auto it = sorted_ops.rbegin(); it != sorted_ops.rend(); ++it) {
    Operation *op = *it;

    int level = 0;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!op_level.count(user)) {
          continue;
        }

        int user_level = op_level[user];

        // Increments level only for materialized ops.
        if (!is_non_materialized(user)) {
          level = std::max(level, user_level + 1);
        } else {
          level = std::max(level, user_level);
        }
      }
    }

    op_level[op] = level;
    max_level = std::max(max_level, level);
  }

  // Step 2: Reverses the level so the earliest op gets level 0.
  for (Operation *op : sorted_ops) {
    int raw_level = op_level[op];
    int normalized_level = max_level - raw_level;
    op_level[op] = normalized_level;
  }

  // Step 3: Overwrites critical ops with ASAP schedule: shortest path from
  // source.
  for (Operation *op : sorted_ops) {
    if (!critical_ops.count(op)) {
      continue;
    }

    int level = -1;
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !op_level.count(def)) {
        continue;
      }

      int def_level = op_level[def];

      assert(def_level <= op_level[op] &&
             "Critical op should not have a lower level than its operands");
      // Increments level only for materialized ops.
      if (!is_non_materialized(op)) {
        level = std::max(level, def_level + 1);
      } else {
        level = std::max(level, def_level);
      }
    }

    if (level != -1) {
      // If there exists operand, moves the critical op earlier.
      op_level[op] = level;
    }
  }

  // Step 4: Assembles the ops into level buckets.
  std::vector<std::vector<Operation *>> level_buckets(max_level + 1);

  for (Operation *op : sorted_ops) {
    level_buckets[op_level[op]].push_back(op);
  }

  return level_buckets;
}

std::vector<std::pair<Operation *, int>> mlir::neura::flatten_level_buckets(
    const std::vector<std::vector<Operation *>> &level_buckets,
    const std::set<Operation *> &critical_ops) {
  std::vector<std::pair<Operation *, int>> result;

  for (int level = 0; level < static_cast<int>(level_buckets.size()); ++level) {
    // Collects ops with their current index to ensure stable sorting.
    std::vector<std::pair<Operation *, int>> ops_with_index;
    for (int i = 0; i < (int)level_buckets[level].size(); ++i) {
      ops_with_index.push_back({level_buckets[level][i], i});
    }

    // Sorts with criticality as PRIMARY criterion within the same ALAP level.
    // This addresses tancheng's feedback: critical ops should map before
    // high-degree non-critical ops in the same level.
    std::sort(ops_with_index.begin(), ops_with_index.end(),
              [&critical_ops](const std::pair<Operation *, int> &a_pair,
                              const std::pair<Operation *, int> &b_pair) {
                Operation *a = a_pair.first;
                Operation *b = b_pair.first;

                bool a_is_critical = critical_ops.count(a) > 0;
                bool b_is_critical = critical_ops.count(b) > 0;

                // Priority 1: Critical ops come first (within same ALAP level).
                if (a_is_critical != b_is_critical)
                  return a_is_critical > b_is_critical;

                // Priority 2: Degree (connectivity) - higher degree first.
                int degree_a = a->getNumOperands();
                int degree_b = b->getNumOperands();
                for (Value res : a->getResults()) {
                  degree_a += std::distance(res.getUsers().begin(),
                                            res.getUsers().end());
                }
                for (Value res : b->getResults()) {
                  degree_b += std::distance(res.getUsers().begin(),
                                            res.getUsers().end());
                }
                if (degree_a != degree_b)
                  return degree_a > degree_b;

                // Priority 3: Original index (stability tie-breaker).
                return a_pair.second < b_pair.second;
              });

    for (const auto &p : ops_with_index) {
      result.emplace_back(p.first, level);
    }
  }

  return result;
}

mlir::Operation *mlir::neura::getMaterializedBackwardUser(Operation *op) {
  assert(isa<neura::CtrlMovOp>(op) && "Expected a ctrl_mov operation");
  auto ctrl_mov = dyn_cast<neura::CtrlMovOp>(op);
  Value target = ctrl_mov.getTarget();

  assert(isa<neura::ReserveOp>(target.getDefiningOp()) &&
         "Expected the user of ctrl_mov target to be a reserve operation");
  neura::ReserveOp reserve_op =
      dyn_cast<neura::ReserveOp>(target.getDefiningOp());

  // Skip ctrl_mov users of reserve; return the first materialized user.
  for (Operation *user : reserve_op.getResult().getUsers()) {
    if (isMaterializedReserveUser(user)) {
      return user;
    }
  }

  // print info
  llvm::errs()
      << "No materialized backward user (i.e., phi) found for ctrl_mov: " << *op
      << "\n";
  llvm::errs() << "Target: " << *target.getDefiningOp() << "\n";
  llvm::errs() << "\n";

  assert(false &&
         "No materialized backward user (i.e., phi) found for ctrl_mov");
}

// This struct represents a pending data_mov/ctrl_mov that is being routed,
// along with its routing path.
struct PendingRoute {
  Operation *mov_op;
  std::vector<MappingLoc> path;
};

bool hasSafeOperandIterationAtConsume(
    Operation *op, const std::vector<PendingRoute> &operand_routes, int ii) {
  assert(ii > 0 && "II should be positive");

  if (operand_routes.empty()) {
    return true;
  }

  for (const PendingRoute &route : operand_routes) {
    // Records the time range that each register is occupied on this route.
    // <Register*, <min_time, max_time>>, this means the register is occupied
    // from min_time to max_time.
    DenseMap<Register *, std::pair<int, int>> reg_time_range;
    for (const MappingLoc &loc : route.path) {
      Register *reg = dyn_cast<Register>(loc.resource);
      if (!reg) {
        continue;
      }

      // For each register, tracks its live interval on this path by keeping
      // the earliest and latest time it appears.
      // Inserts a new entry if the register is seen for the first time.
      auto [it, inserted] = reg_time_range.try_emplace(
          reg, std::make_pair(loc.time_step, loc.time_step));

      // If this register has been seen before, updates the time range to
      // include the new time step.
      if (!inserted) {
        // Updates the min_time for this seen register if the new time step is
        // earlier.
        it->second.first = std::min(it->second.first, loc.time_step);
        // Updates the max_time for this seen register if the new time step is
        // later.
        it->second.second = std::max(it->second.second, loc.time_step);
      }
    }

    // Register occupancy is tracked in per-cycle slots (half-open in routing
    // builders), so max_t - min_t + 1 corresponds to the hold duration.
    for (const auto &entry : reg_time_range) {
      Register *reg = entry.first;
      int min_t = entry.second.first;
      int max_t = entry.second.second;
      int occupancy = max_t - min_t + 1;
      if (occupancy > ii) {
        llvm::errs() << "[DEBUG] Reject schedule due to register hold >= next "
                        "iteration window. op="
                     << *op << ", II=" << ii << ", reg=#" << reg->getId()
                     << ", hold_start=" << min_t << ", hold_end=" << max_t
                     << ", hold_len=" << occupancy
                     << ", mov_op=" << *route.mov_op << "\n";
        return false;
      }
    }
  }

  return true;
}

llvm::SmallVector<mlir::Operation *>
mlir::neura::getMaterializedUserOps(Operation *op) {
  llvm::SmallVector<Operation *> result;
  llvm::DenseSet<Operation *> visited;
  visited.insert(op);
  llvm::errs() << "Starting to collect materialized users for: " << *op << "\n";
  llvm::SmallVector<Operation *> worklist(op->getUsers().begin(),
                                          op->getUsers().end());

  while (!worklist.empty()) {
    Operation *curr = worklist.pop_back_val();
    llvm::errs() << "Visiting operation: " << *curr << "\n";
    if (!visited.insert(curr).second) {
      llvm::errs() << "Already visited, so skip: " << *curr << "\n";
      continue;
    }

    if (isa<neura::DataMovOp>(curr)) {
      for (Operation *next : curr->getUsers()) {
        if (visited.insert(next).second) {
          // Only adds the next operation if it hasn't been visited yet.
          worklist.push_back(next);
        }
      }
      continue;
    }

    // Specially handles the ctrl_mov, i.e., the second operand of ctrl_mov is
    // treated as a target/destination/user in terms of dataflow.
    if (neura::CtrlMovOp ctrl_mov = dyn_cast<neura::CtrlMovOp>(curr)) {
      Value target = ctrl_mov.getTarget();
      for (Operation *user : target.getUsers()) {
        if (visited.insert(user).second) {
          worklist.push_back(user);
        }
      }
      continue;
    }

    // Materialized op
    result.push_back(curr);
  }

  for (Operation *res : result) {
    llvm::errs() << "Materialized user: " << *res << "\n";
  }
  return result;
}

bool mlir::neura::tryRouteForwardMove(Operation *mov_op, MappingLoc src_loc,
                                      MappingLoc dst_loc,
                                      const MappingState &state,
                                      std::vector<MappingLoc> &path_out) {
  return tryRouteDataMove(mov_op, src_loc, dst_loc, false, state, path_out);
}

bool mlir::neura::tryRouteBackwardMove(Operation *mov_op, MappingLoc src_loc,
                                       MappingLoc dst_loc,
                                       const MappingState &state,
                                       std::vector<MappingLoc> &path_out) {
  llvm::errs() << "[tryRouteBackwardMove] src_loc: "
               << src_loc.resource->getType() << "#"
               << src_loc.resource->getId() << " @t=" << src_loc.time_step
               << ", dst_loc: " << dst_loc.resource->getType() << "#"
               << dst_loc.resource->getId() << " @t=" << dst_loc.time_step
               << "\n";
  return tryRouteDataMove(mov_op, src_loc, dst_loc, true, state, path_out);
}

Register *mlir::neura::getAvailableRegister(const MappingState &state,
                                            Tile *tile, int start_time,
                                            int exclusive_end_time,
                                            neura::DataMovOp move_op) {
  // Finds the first register that is free across the requested range AND
  // satisfies the single-port cluster constraints:
  //   - Write port: at start_time, only one register per RegisterFile may be
  //     written (same-register writes are allowed/idempotent).
  //   - Read port:  at (exclusive_end_time - 1), only one register per
  //     RegisterFile may be read (same-register reads are shared/allowed).
  // Read and write ports are independent — one read AND one write can happen
  // simultaneously on the same RegisterFile.
  //
  // NOTE: Multiple DataMovOps carrying the same materialized source value are
  // allowed to share the same physical register (see
  // MappingState::isAvailableAcrossTime() and isAvailableForOccupyStatus()).
  // This is correct because all such DataMovOps read the identical value from
  // the register, so the single register read port broadcasts to all consumers.
  for (Register *reg : tile->getRegisters()) {
    if (!state.isAvailableAcrossTimeInRange(reg, start_time, exclusive_end_time,
                                            move_op)) {
      continue;
    }
    // Check cluster write-port constraint at the write time step.
    if (!state.isRegisterWriteAvailableAcrossTime(reg, start_time)) {
      continue;
    }
    // Check cluster read-port constraint at the read time step.
    if (!state.isRegisterReadAvailableAcrossTime(reg, exclusive_end_time)) {
      continue;
    }
    return reg;
  }
  return nullptr;
}

bool mlir::neura::tryRouteDataMove(Operation *mov_op, MappingLoc src_loc,
                                   MappingLoc dst_loc, bool is_backward_move,
                                   const MappingState &state,
                                   std::vector<MappingLoc> &path_out) {
  assert(path_out.empty() && "Output path should be empty");

  // Gets the source tile and destination tile.
  Tile *src_tile = dyn_cast<Tile>(src_loc.resource);
  Tile *dst_tile = dyn_cast<Tile>(dst_loc.resource);

  assert(src_tile && dst_tile &&
         "Source and destination locations must be tiles");

  // Calculates the deadline time step (adds II for backward moves).
  int exclusive_deadline_step = dst_loc.time_step;
  if (is_backward_move) {
    exclusive_deadline_step += state.getII();
  }

  llvm::outs() << "[tryRouteDataMove] Routing from Tile#" << src_tile->getId()
               << " @t=" << src_loc.time_step << " to Tile#"
               << dst_tile->getId() << " @t=" << exclusive_deadline_step
               << "\n";

  // Special case: source tile and destination tile are the same.
  if (src_tile == dst_tile) {
    // Uses register as routing resource within the same tile.
    // Finds an available register to store the data.
    Register *available_reg = getAvailableRegister(
        state, src_tile, src_loc.time_step, exclusive_deadline_step,
        dyn_cast_or_null<neura::DataMovOp>(mov_op));
    if (!available_reg) {
      llvm::outs()
          << "[tryRouteDataMove] Cannot find available register on Tile#"
          << src_tile->getId() << " for time range: t=" << src_loc.time_step
          << " to t=" << exclusive_deadline_step << "\n";
      return false;
    }

    // Builds path: uses register to store data for the specified time period.
    for (int t = src_loc.time_step; t < exclusive_deadline_step; ++t) {
      path_out.push_back({available_reg, t});
    }

    llvm::outs() << "[tryRouteDataMove] Successfully routed on same tile using "
                    "Register #"
                 << available_reg->getId() << "\n";
    return true;
  }

  // Search state: records current tile, time step, and path to reach this
  // state.
  struct SearchState {
    Tile *current_tile; // Current tile location.
    int current_time;   // Current time step.
    std::vector<MappingLoc>
        path; // Routing resource path to reach current state.
  };

  // BFS search.
  std::queue<SearchState> search_queue;
  std::set<std::pair<Tile *, int>>
      visited; // Records visited (tile, time) combinations.

  // Initial state: starts from source tile.
  search_queue.push({src_tile, src_loc.time_step, {}});
  visited.insert({src_tile, src_loc.time_step});

  while (!search_queue.empty()) {
    SearchState current_state = search_queue.front();
    search_queue.pop();

    // Checks if destination tile is reached with appropriate timing.
    if (current_state.current_tile == dst_tile) {
      // The link/register between producer tile and consumer tile is belonging
      // to the producer tile with same time step.
      if (current_state.current_time <= exclusive_deadline_step) {
        if (current_state.current_time == exclusive_deadline_step) {
          // Arrives exactly at deadline, no additional register needed.
          path_out = current_state.path;
          return true;
        } else {
          // Arrives early, needs register on destination tile to wait.
          Register *wait_reg =
              getAvailableRegister(state, dst_tile, current_state.current_time,
                                   exclusive_deadline_step,
                                   dyn_cast_or_null<neura::DataMovOp>(mov_op));
          if (!wait_reg) {
            llvm::outs() << "[tryRouteDataMove] Cannot find available waiting"
                            "register on destination Tile#"
                         << dst_tile->getId() << "\n";
            continue; // Tries other paths.
          }

          // Builds complete path.
          path_out = current_state.path;
          for (int t = current_state.current_time; t < exclusive_deadline_step;
               ++t) {
            path_out.push_back({wait_reg, t});
          }
          return true;
        }
      } else {
        // Arrives too late, skips this path.
        continue;
      }
    }

    // Skips if current time already exceeds deadline.
    if (current_state.current_time >= exclusive_deadline_step) {
      continue;
    }

    // Option 1: Moves to adjacent tile through link.
    for (Link *out_link : current_state.current_tile->getOutLinks()) {
      MappingLoc link_loc = {out_link, current_state.current_time};

      // Checks if link is available at current time step.
      if (!state.isAvailableAcrossTime(
              link_loc, dyn_cast_or_null<neura::DataMovOp>(mov_op))) {
        continue;
      }

      Tile *next_tile = out_link->getDstTile();
      int next_time = current_state.current_time + 1;

      // Checks if this (tile, time) combination has been visited.
      if (visited.insert({next_tile, next_time}).second) {
        std::vector<MappingLoc> new_path = current_state.path;
        new_path.push_back(link_loc);

        search_queue.push({next_tile, next_time, new_path});
      }
    }

    // Option 2: Uses register on current tile to wait one time step.
    Register *wait_register = getAvailableRegister(
        state, current_state.current_tile, current_state.current_time,
        current_state.current_time + 1,
        dyn_cast_or_null<neura::DataMovOp>(mov_op));
    if (wait_register) {
      int next_time = current_state.current_time + 1;
      // Checks if this(tile, time) combination has been visited.
      // Though theoretically we can revisit a tile at different time steps
      // to explore alternative routing paths, we disallow this during the
      // routing search to prevent exponential search complexity and ensure
      // algorithm termination within reasonable time bounds.
      if (visited.insert({current_state.current_tile, next_time}).second) {
        std::vector<MappingLoc> new_path = current_state.path;
        new_path.push_back({wait_register, current_state.current_time});

        search_queue.push({current_state.current_tile, next_time, new_path});
      }
    }
  }

  // Search failed.
  llvm::outs() << "[tryRouteDataMove] Cannot find routing path from Tile#"
               << src_tile->getId() << " @t=" << src_loc.time_step
               << " to Tile#" << dst_tile->getId()
               << " @t=" << exclusive_deadline_step << "\n";
  return false;
}

Operation *mlir::neura::getMaterializedProducer(Value operand) {
  Operation *producer = operand.getDefiningOp();

  // ReserveOp is not wrapped by DataMovOp (see InsertDataMovPass).
  // Return it directly as it represents the loop-carried dependency
  // placeholder.
  if (isa<neura::ReserveOp>(producer)) {
    return producer;
  }

  // For operations wrapped by DataMovOp, find the actual producer.
  assert(
      isa<neura::DataMovOp>(producer) &&
      "Expected a DataMovOp as operand producer for non-ReserveOp operations");
  neura::DataMovOp mov_op = dyn_cast<neura::DataMovOp>(producer);
  Operation *materialized_producer = mov_op.getOperand().getDefiningOp();
  return materialized_producer;
}

int mlir::neura::getPhysicalHops(const std::vector<Operation *> &producers,
                                 Tile *tile,
                                 const MappingState &mapping_state) {

  // Counts the number of physical hops from the producers to the tile.
  int hops = 0;

  for (Operation *producer : producers) {
    // Get the last location of the producer.
    std::vector<MappingLoc> producer_locs =
        mapping_state.getAllLocsOfOp(producer);
    assert(!producer_locs.empty() && "No locations found for producer");

    MappingLoc producer_loc = producer_locs.back();
    Tile *producer_tile = dyn_cast<Tile>(producer_loc.resource);
    assert(producer_tile && "Producer location must be a Tile");
    hops += std::abs(producer_tile->getX() - tile->getX()) +
            std::abs(producer_tile->getY() - tile->getY());
  }
  return hops;
}

bool mlir::neura::canReachLocInTime(const std::vector<Operation *> &producers,
                                    const MappingLoc &target_loc,
                                    int deadline_step,
                                    const MappingState &mapping_state) {

  for (Operation *producer : producers) {
    // Get the last location of the producer.
    std::vector<MappingLoc> producer_locs =
        mapping_state.getAllLocsOfOp(producer);
    assert(!producer_locs.empty() && "No locations found for producer");

    MappingLoc producer_loc = producer_locs.back();
    if (!canReachLocInTime(producer_loc, target_loc, deadline_step,
                           mapping_state)) {
      return false;
    }
  }
  return true;
}

bool mlir::neura::canReachLocInTime(const MappingLoc &src_loc,
                                    const MappingLoc &dst_loc,
                                    int deadline_step,
                                    const MappingState &mapping_state) {
  // Checks if the destination is reachable from the source within the given
  // time window.
  if (src_loc.resource == dst_loc.resource &&
      src_loc.time_step < deadline_step && dst_loc.time_step <= deadline_step) {
    return true;
  }

  // Checks if the destination is reachable from the source tile within given
  // steps. This uses BFS similar to tryRouteDataMove, considering both link
  // traversal and register-based waiting.
  assert(isa<Tile>(src_loc.resource));
  assert(isa<Tile>(dst_loc.resource));

  Tile *dst_tile = dyn_cast<Tile>(dst_loc.resource);

  std::queue<std::pair<Tile *, int>> queue;
  // Tracks visited (tile, time) states to allow revisiting the same tile at
  // different time steps (needed to faithfully model waiting in registers).
  std::set<std::pair<Tile *, int>> visited;

  queue.push({dyn_cast<Tile>(src_loc.resource), src_loc.time_step});
  visited.insert({dyn_cast<Tile>(src_loc.resource), src_loc.time_step});

  while (!queue.empty()) {
    auto [current_tile, current_step] = queue.front();
    queue.pop();

    // If we reach the destination tile within deadline
    if (current_tile == dst_tile && current_step <= deadline_step) {
      return true;
    }

    if (current_step >= deadline_step) {
      continue;
    }

    int next_step = current_step + 1;

    // Option 1: Move to adjacent tile through link.
    for (Link *out_link : current_tile->getOutLinks()) {
      MappingLoc link_loc = {out_link, current_step};

      // Checks if link is available at current time step.
      if (!mapping_state.isAvailableAcrossTime(link_loc)) {
        continue;
      }

      Tile *next_tile = out_link->getDstTile();
      if (visited.insert({next_tile, next_step}).second) {
        queue.push({next_tile, next_step});
      }
    }

    // Option 2: Wait on current tile using register (if available).
    Register *wait_register = getAvailableRegister(
        mapping_state, current_tile, current_step, current_step + 1);
    if (wait_register) {
      if (visited.insert({current_tile, next_step}).second) {
        queue.push({current_tile, next_step});
      }
    }
  }

  return false;
}

bool mlir::neura::isMaterializedReserveUser(Operation *user) {
  if (isa<neura::InvariantOp>(user)) {
    return true;
  }
  if (isa<neura::CarryOp>(user)) {
    return true;
  }
  if (isa<neura::FusedOp>(user)) {
    return true;
  }
  if (isa<neura::PhiStartOp>(user)) {
    return true;
  }
  if (isa<neura::PhiOp>(user)) {
    return true;
  }
  return false;
}

void mlir::neura::updateAward(std::map<MappingLoc, int> &locs_with_award,
                              MappingLoc loc, int award) {
  // Updates the award of the top element in the priority queue.
  if (locs_with_award.find(loc) != locs_with_award.end()) {
    locs_with_award[loc] += award;
  } else {
    locs_with_award[loc] = award;
  }
}

std::vector<MappingLoc>
mlir::neura::calculateAward(Operation *op, std::set<Operation *> &critical_ops,
                            int target_level, const Architecture &architecture,
                            const MappingState &mapping_state) {
  // Early exit if the operation is not supported by all the tiles.
  bool op_can_be_supported = false;
  for (Tile *tile : architecture.getAllTiles()) {
    if (tile->canSupportOperation(getOperationKindFromMlirOp(op))) {
      op_can_be_supported = true;
    }
  }
  if (!op_can_be_supported) {
    llvm::errs() << "[calculateAward] Operation: " << *op
                 << " is not supported by any tile.\n";
    return {};
  }

  // A heap of locations with their associated award. Note that we use a
  // max-heap to prioritize locations with higher awards.
  std::map<MappingLoc, int> locs_with_award;

  // Assembles all the producers.
  std::vector<Operation *> producers;
  for (Value operand : op->getOperands()) {
    if (isa<neura::ReserveOp>(operand.getDefiningOp())) {
      // Skips Reserve ops (backward ctrl move) when calculating award.
      continue;
    }
    Operation *producer = getMaterializedProducer(operand);
    assert(producer && "Expected a materialized producer");
    producers.push_back(producer);
  }

  // Assembles all the backward users if exist.
  std::vector<Operation *> backward_users;
  for (Operation *user : getCtrlMovUsers(op)) {
    neura::CtrlMovOp ctrl_mov = dyn_cast<neura::CtrlMovOp>(user);
    assert(ctrl_mov && "Expected user to be a CtrlMovOp");
    mlir::Operation *materialized_backward_op =
        getMaterializedBackwardUser(ctrl_mov);
    assert(isMaterializedReserveUser(materialized_backward_op) &&
           "Expected materialized operation of ctrl_mov to be a "
           "PhiOp/InvariantOp/CarryOp.");
    backward_users.push_back(materialized_backward_op);
  }

  llvm::errs() << "[calculateAward] Operation: " << *op
               << "; Producers: " << producers.size() << "\n";

  for (Tile *tile : architecture.getAllTiles()) {
    if (!tile->canSupportOperation(getOperationKindFromMlirOp(op))) {
      llvm::errs() << "[calculateAward] Tile<" << tile->getX() << ", "
                   << tile->getY() << "> does not support operation: " << *op
                   << "\n";
      continue; // Skip tiles that cannot support the operation.
    }
    int earliest_start_time_step = target_level;
    for (Operation *producer : producers) {
      std::vector<MappingLoc> producer_locs =
          mapping_state.getAllLocsOfOp(producer);
      assert(!producer_locs.empty() && "No locations found for producer");

      MappingLoc producer_loc = producer_locs.back();
      earliest_start_time_step =
          std::max(earliest_start_time_step, producer_loc.time_step + 1);
    }
    int latest_end_time_step = earliest_start_time_step + mapping_state.getII();
    std::vector<MappingLoc> backward_users_locs;
    for (Operation *user : backward_users) {
      std::vector<MappingLoc> user_locs = mapping_state.getAllLocsOfOp(user);
      assert(!user_locs.empty() && "No locations found for backward user");

      MappingLoc backward_user_loc = user_locs.back();
      latest_end_time_step =
          std::min(latest_end_time_step,
                   backward_user_loc.time_step + mapping_state.getII());
      backward_users_locs.push_back(backward_user_loc);
    }

    // === Tile-based award (independent of time) ===
    int tile_award = 0;

    // Computes proximity bonus to producers. Closer tiles get higher scores.
    int hops_to_producers = getPhysicalHops(producers, tile, mapping_state);
    // Calculates the maximum possible distance.
    int kMaxDist =
        (architecture.getPerCgraRows() + architecture.getPerCgraColumns() - 2);
    int max_hops = static_cast<int>(producers.size()) * kMaxDist;
    int proximity_bonus =
        std::max(0, max_hops - hops_to_producers) * kAwardProximityScale;
    tile_award += proximity_bonus;

    // Computes proximity bonus to backward users. Closer is better for
    // recurrence routing.
    for (MappingLoc &backward_user_loc : backward_users_locs) {
      Tile *backward_tile = dyn_cast<Tile>(backward_user_loc.resource);
      if (backward_tile) {
        int backward_hops = std::abs(backward_tile->getX() - tile->getX()) +
                            std::abs(backward_tile->getY() - tile->getY());
        tile_award += std::max(0, (kMaxDist - backward_hops) *
                                      kAwardBackwardProximityScale);
      }
    }

    // Grants critical ops higher base award and routing flexibility bonus.
    if (critical_ops.count(op)) {
      // Keep the original critical bonuses but allow tuning via division.
      tile_award += (mapping_state.getII() +
                     static_cast<int>(tile->getDstTiles().size())) /
                    std::max(1, kAwardCriticalBonusDiv);
    }

    // Apply base multiplier to amplify or dampen tile-based award.
    tile_award *= kAwardBaseMultiplier;

    // === Time-based award ===
    for (int t = earliest_start_time_step; t < latest_end_time_step; t += 1) {
      MappingLoc tile_loc_candidate = {tile, t};
      // Considers the tile at time `t` for mapping if available.
      if (mapping_state.isAvailableAcrossTime(tile_loc_candidate)) {
        bool meet_producer_constraint =
            producers.empty() ||
            canReachLocInTime(producers, tile_loc_candidate, t, mapping_state);
        bool meet_backward_user_constraint = true;
        for (MappingLoc &backward_user_loc : backward_users_locs) {
          // Checks if the location can reach all backward users.
          if (!canReachLocInTime(tile_loc_candidate, backward_user_loc,
                                 backward_user_loc.time_step +
                                     mapping_state.getII(),
                                 mapping_state)) {
            meet_backward_user_constraint = false;
            break;
          }
        }
        // Grants reward if all constraints are satisfied.
        if (meet_producer_constraint && meet_backward_user_constraint) {
          // Earlier time steps get higher scores.
          int time_bonus = latest_end_time_step - t;

          // === Balanced Link congestion penalty ===
          // A conservative penalty to guide the mapper away from hotspots
          // without being too restrictive for small IIs.
          int total_in = tile->getInLinks().size();
          int total_out = tile->getOutLinks().size();
          int occupied_in = 0;
          int occupied_out = 0;

          for (Link *link : tile->getInLinks()) {
            if (!mapping_state.isAvailableAcrossTime({link, t})) {
              occupied_in++;
            }
          }
          for (Link *link : tile->getOutLinks()) {
            if (!mapping_state.isAvailableAcrossTime({link, t})) {
              occupied_out++;
            }
          }

          float in_ratio = (total_in > 0) ? (float)occupied_in / total_in : 0;
          float out_ratio =
              (total_out > 0) ? (float)occupied_out / total_out : 0;

          // Adaptive penalty strategy:
          // - Use very strong penalty (60) only for high fan-in ops (>= 3
          // producers)
          // - Use weak penalty (15) for low fan-in ops
          // This optimizes fuse-pattern (II=11 target) without breaking
          // iter-merge
          int base_penalty_coeff = (producers.size() >= 3)
                                       ? kStrongCongestionPenalty
                                       : kWeakCongestionPenalty;

          int congestion_penalty =
              static_cast<int>(in_ratio * in_ratio * base_penalty_coeff) +
              static_cast<int>(out_ratio * out_ratio * base_penalty_coeff);

          int total_award = tile_award + time_bonus - congestion_penalty;
          updateAward(locs_with_award, tile_loc_candidate, total_award);
        }
      }
    }
  }

  // Copies map entries into a vector of pairs for sorting.
  std::vector<std::pair<MappingLoc, int>> locs_award_vec(
      locs_with_award.begin(), locs_with_award.end());

  // Sorts by award (descending). Use stable sort/tie-breaker logic
  // to minimize noise in mapping results.
  std::sort(locs_award_vec.begin(), locs_award_vec.end(),
            [](const std::pair<MappingLoc, int> &a,
               const std::pair<MappingLoc, int> &b) {
              if (a.second != b.second)
                return a.second > b.second;
              // Tie-breaker: earlier time step first.
              return a.first.time_step < b.first.time_step;
            });
  // TODO: Needs to handle tie case and prioritize lower resource utilization,
  // however, compiled II becomes worse after adding this tie-breaker:
  // https://github.com/coredac/dataflow/issues/59.
  // std::sort(locs_award_vec.begin(), locs_award_vec.end(),
  //           [&](const std::pair<MappingLoc, int> &a, const
  //           std::pair<MappingLoc, int> &b) {
  //               if (a.second != b.second) {
  //                 return a.second > b.second;
  //               }
  //               // Tie-breaker: prioritizes lower resource utilization and
  //               // earlier time step.
  //               if (a.first.time_step != b.first.time_step) {
  //                 return a.first.time_step > b.first.time_step;
  //               }
  //               const bool is_resource_a_lower_utilized =
  //                   mapping_state.countOpsAtResource(a.first.resource) >
  //                   mapping_state.countOpsAtResource(b.first.resource);
  //               return is_resource_a_lower_utilized;
  //             });

  // Extracts just the MappingLocs, already sorted by award.
  std::vector<MappingLoc> sorted_locs;
  sorted_locs.reserve(locs_award_vec.size());
  for (const auto &pair : locs_award_vec)
    sorted_locs.push_back(pair.first);

  return sorted_locs;
}

llvm::SmallVector<Operation *> mlir::neura::getCtrlMovUsers(Operation *op) {
  llvm::SmallVector<Operation *> results;
  for (Operation *user : op->getUsers()) {
    if (isa<neura::CtrlMovOp>(user)) {
      results.push_back(user);
    }
  }
  return results;
}

bool mlir::neura::placeAndRoute(Operation *op, const MappingLoc &target_loc,
                                MappingState &mapping_state) {
  // Get the latency of the operation to determine if it's multi-cycle
  int latency = getOpLatency(op);
  bool is_multi_cycle = latency > 1;

  bool bind_success = false;
  if (is_multi_cycle) {
    // For multi-cycle ops, bind across multiple time steps with pipeline status
    bind_success = mapping_state.bindMultiCycleOp(
        target_loc.resource, target_loc.time_step, latency, op);
    if (bind_success) {
      llvm::errs() << "[DEBUG] Bound multi-cycle op (latency=" << latency
                   << ") " << *op
                   << " onto loc: " << target_loc.resource->getType() << "#"
                   << target_loc.resource->getId()
                   << " @t=" << target_loc.time_step
                   << " to t=" << (target_loc.time_step + latency - 1) << "\n";
    }
  } else {
    // For single-cycle ops, use default SINGLE_OCCUPY binding
    bind_success = mapping_state.bindOp(target_loc, op);
    if (bind_success) {
      llvm::errs() << "[DEBUG] Schedule op " << *op
                   << " onto loc: " << target_loc.resource->getType() << "#"
                   << target_loc.resource->getId()
                   << " @t=" << target_loc.time_step << "\n";
    }
  }

  if (bind_success) {
    std::vector<PendingRoute> pending_operand_routes;
    std::vector<Operation *> routed_operands;
    std::vector<Operation *> routed_ctrl_movs;
    // Tries to route the data move operations.
    for (Value operand : op->getOperands()) {
      llvm::errs() << "Processing operand: " << operand << "\n";
      if (isa<neura::ReserveOp>(operand.getDefiningOp())) {
        // Skips Reserve ops (backward ctrl move) when routing.
        continue;
      }
      Operation *data_move = operand.getDefiningOp();

      // ReserveOp is not wrapped by DataMovOp (see InsertDataMovPass).
      // Skip routing for ReserveOp as it represents loop-carried dependency.
      if (isa<neura::ReserveOp>(data_move)) {
        llvm::errs() << "Skipping unwrapped operand: " << *data_move << "\n";
        continue;
      }

      assert(isa<neura::DataMovOp>(data_move) &&
             "Expected a DataMovOp as operand for non-ReserveOp operations");

      Operation *producer = getMaterializedProducer(operand);
      MappingLoc src_loc = mapping_state.getAllLocsOfOp(producer).back();

      std::vector<MappingLoc> route_path;
      if (tryRouteForwardMove(data_move, src_loc, target_loc, mapping_state,
                              route_path)) {
        mapping_state.reserveRoute(data_move, route_path);
        pending_operand_routes.push_back({data_move, std::move(route_path)});
        routed_operands.push_back(data_move);
        llvm::errs() << "[DEBUG] Successfully routed data move: " << *data_move
                     << " from " << src_loc.resource->getType() << "#"
                     << src_loc.resource->getId() << " @t=" << src_loc.time_step
                     << " to " << target_loc.resource->getType() << "#"
                     << target_loc.resource->getId()
                     << " @t=" << target_loc.time_step << "\n";
        continue;
      }
      llvm::errs() << "[DEBUG] Failed to route data move: " << *data_move
                   << " from " << src_loc.resource->getType() << "#"
                   << src_loc.resource->getId() << " @t=" << src_loc.time_step
                   << " to " << target_loc.resource->getType() << "#"
                   << target_loc.resource->getId()
                   << " @t=" << target_loc.time_step << "; so unschedule op\n";
      mapping_state.unbindOp(op);
      for (Operation *routed_op : routed_operands) {
        mapping_state.releaseRoute(routed_op);
      }
      return false;
    }

    if (!hasSafeOperandIterationAtConsume(op, pending_operand_routes,
                                          mapping_state.getII())) {
      llvm::errs() << "[DEBUG] Operand iteration shift at consume time; "
                      "unschedule op\n";
      mapping_state.unbindOp(op);
      for (Operation *routed_op : routed_operands) {
        mapping_state.releaseRoute(routed_op);
      }
      return false;
    }
    // Checks whether the operation's user is a ctrl_mov.
    for (Operation *user : getCtrlMovUsers(op)) {
      neura::CtrlMovOp ctrl_mov = dyn_cast<neura::CtrlMovOp>(user);
      llvm::errs() << "[DEBUG] Found ctrl_mov user: " << *ctrl_mov << "\n";
      assert(ctrl_mov && "Expected user to be a CtrlMovOp");
      mlir::Operation *materialized_backward_op =
          getMaterializedBackwardUser(ctrl_mov);
      assert(isMaterializedReserveUser(materialized_backward_op) &&
             "Expected materialized operation of ctrl_mov to be a "
             "PhiOp/InvariantOp/CarryOp");
      // Gets the last location of the materialized operation.
      MappingLoc backward_loc =
          mapping_state.getAllLocsOfOp(materialized_backward_op).back();
      // Routes the ctrl_mov to the phi location.
      std::vector<MappingLoc> route_path;
      if (tryRouteBackwardMove(ctrl_mov, target_loc, backward_loc,
                               mapping_state, route_path)) {
        mapping_state.reserveRoute(ctrl_mov, route_path);
        routed_ctrl_movs.push_back(ctrl_mov);
        llvm::errs() << "[DEBUG] Successfully routed ctrl_mov: " << *ctrl_mov
                     << " to " << backward_loc.resource->getType() << "#"
                     << backward_loc.resource->getId()
                     << " @t=" << backward_loc.time_step << "\n";
        continue;
      }
      llvm::errs() << "[DEBUG] Failed to route ctrl_mov: " << *ctrl_mov
                   << " from " << target_loc.resource->getType() << "#"
                   << target_loc.resource->getId()
                   << " @t=" << target_loc.time_step << " to "
                   << backward_loc.resource->getType() << "#"
                   << backward_loc.resource->getId()
                   << " @t=" << backward_loc.time_step
                   << "; so unschedule op\n";
      mapping_state.unbindOp(op);
      for (Operation *routed_ctrl_mov : routed_ctrl_movs) {
        llvm::errs() << "[DEBUG] Releasing route for routed ctrl_mov: "
                     << *routed_ctrl_mov << "\n";
        mapping_state.releaseRoute(routed_ctrl_mov);
      }

      for (Operation *routed_op : routed_operands) {
        llvm::errs() << "[DEBUG] Releasing route for routed operand: "
                     << *routed_op << "\n";
        mapping_state.releaseRoute(routed_op);
      }
      return false;
    }
    return true;
  }
  return false;
}

int mlir::neura::getOpLatency(Operation *op) {
  // Try to get the latency attribute from the operation
  if (IntegerAttr latency_attr = op->getAttrOfType<IntegerAttr>("latency")) {
    return latency_attr.getInt();
  }
  // Default to single-cycle if no latency attribute is present
  return 1;
}

bool mlir::neura::isMultiCycleOp(Operation *op) { return getOpLatency(op) > 1; }