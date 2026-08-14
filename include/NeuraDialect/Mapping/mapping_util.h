#pragma once

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace neura {
// Returns the kind of operation from the MLIR operation, or IUnknown when the
// op has no modelled functional unit. IUnknown is NOT a member of any FU class,
// so fuClassOf reports "other" for it and every consumer must treat it as
// unconstrained (runs on any tile), never as infeasible. It used to fall
// through to IAdd, which made ~20 op kinds -- counter, phi_start, and,
// return_value, extract_predicate, carry, merge, invariant, the steers, fmax /
// fmin, mul_add, the vector ops, memset, gather -- claim the adder class and be
// priced against the adder capacity.
OperationKind getOperationKindFromMlirOp(Operation *op);

// Returns true if the operation does not need CGRA tile placement.
bool is_non_materialized(Operation *op);

// Returns true if the op occupies a tile/FU and must be placed by the mapper
// (not a routing/structural op or a fused-region interior). NOTE: this is a
// STRICTER predicate than !is_non_materialized -- besides the routing movs it
// also excludes container ops (func/module/kernel) and fused-region interior
// ops, so do not substitute a negation of is_non_materialized for it. Shared by
// DumpDfgJsonPass (which emits these ops in walk order) and MapToAcceleratorPass's
// import-mapping (which replays onto them), so their op indices stay aligned.
bool occupiesFU(Operation *op);

// FU class name an op maps to (the inverse of Architecture's
// kFuTypesToOperations). A fused op reports its pattern_name (or "fused" if it
// carries none); an op with no known class reports "other". Shared by the
// analytical cost model and --dump-dfg-json so both bucket ops identically.
std::string fuClassOf(Operation *op);

// Tiles that physically provide `fu_class`.
//
// A class the FU table does not describe (absent, or present with no
// OperationKinds) is UNCONSTRAINED: every tile is returned, because nothing is
// known about which FU would run it. A class the table DOES describe but that
// no tile provides returns EMPTY -- that is a real capacity of zero, i.e. the
// op cannot be placed anywhere on this architecture. The two cases are
// deliberately distinct, and --dump-dfg-json mirrors them: it emits no key for
// an undescribed class (so the exact mapper's `fu_class_tiles.get(c, all)`
// falls back to every tile) and an empty list for a described-but-unprovided
// one. Single source of truth for the analytical cost model, --dump-dfg-json
// and anything else that asks "how many tiles can run this class".
llvm::SmallVector<Tile *, 16> tilesProvidingFuClass(const Architecture &arch,
                                                    llvm::StringRef fu_class);

// The placed ops of `region`, in the order --dump-dfg-json emits them.
//
// THIS ORDER IS A CONTRACT: op index i in the emitted JSON is placed_ops[i]
// here, and --import-mapping replays the solver's placements onto exactly this
// sequence. It is the region walk order filtered by occupiesFU; do not sort,
// filter or otherwise reorder it.
std::vector<Operation *> collectPlacedOps(Region &region);

// The placed producer of `value` as the dependence graph sees it: unwraps one
// data_mov to the op that really produced the value, and returns null for a
// reserve (the loop-carried placeholder, which the omega=1 edges carry
// instead). Non-asserting counterpart of getMaterializedProducer.
Operation *getPlacedProducer(Value value);

// One dependence edge over collectPlacedOps indices. `omega` is the iteration
// distance: 0 for an intra-iteration operand edge, 1 for a loop-carried
// ctrl_mov/reserve edge.
struct DependenceEdge {
  int src;
  int dst;
  int omega;
};

// The dependence edges over `placed_ops` (which must come from
// collectPlacedOps on the same region): forward operand edges with omega=0
// plus ctrl_mov/reserve back edges with omega=1.
//
// Edges are de-duplicated on (src, dst, omega). An op that reads the same
// value twice (`x + x`) has two data_movs to one producer, but that is ONE
// dependence net -- one value, one route -- and since an edge's delay is a
// function of its source alone, the repeats are identical records. Emitting
// them twice would double-book the shared link on import, and would put
// parallel arcs in the cost model's recurrence graph that cannot change its
// maximum cycle ratio. Shared by --dump-dfg-json (which hands these edges to
// the exact mapper) and the analytical RecMII, so both reason about the same
// graph.
std::vector<DependenceEdge>
buildDfgEdges(Region &region, const std::vector<Operation *> &placed_ops);

// Returns true if the operation is a steering-mode operation that doesn't
// require DataMovOp wrapping (e.g., constants, carry, invariant, etc.).
bool is_steering_unwrapped_op(Operation *op);

// Returns true if the operation is a materialized reserve user, i.e.,
// phi, invariant, carry.
bool isMaterializedReserveUser(Operation *op);

// Represents a recurrence cycle rooted at a reserve operation and closed by
// ctrl_mov.
struct RecurrenceCycle {
  // Ordered list of operations in the cycle.
  SmallVector<Operation *> operations;
  // Number of operations excluding reserve/ctrl_mov.
  int length = 0;
};

// Collects recurrence cycles rooted at reserve and closed by ctrl_mov.
SmallVector<RecurrenceCycle, 4> collectRecurrenceCycles(Region &region);

// Calculates ResMII: ceil(#ops / #tiles).
int calculateResMii(Region &region, const Architecture &architecture);

// Returns topologically sorted operations in region.
std::vector<Operation *> getTopologicallySortedOps(Region &region);

// Given the sorted operations, returns a vector of pairs where each pair
// contains a vector of operations at the same ALAP (as late as possible)
// level and the level number.
std::vector<std::vector<Operation *>>
getOpsInAlapLevels(const std::vector<Operation *> &sorted_ops,
                   const std::set<Operation *> &critical_ops);

// Flattens the level buckets into a vector of pairs (operation, level).
// Within each ALAP level, critical ops are prioritized before non-critical ops.
std::vector<std::pair<Operation *, int>> flatten_level_buckets(
    const std::vector<std::vector<Operation *>> &level_buckets,
    const std::set<Operation *> &critical_ops);

// Gets the physical hops from the producers to the tile, which is used for
// estimating the award of a location for placement.
int getPhysicalHops(const std::vector<Operation *> &producers, Tile *tile,
                    const MappingState &mapping_state);

Operation *getMaterializedProducer(Value operand);

// Collects the real users of an operation, excluding ctrl_mov and data_mov.
llvm::SmallVector<mlir::Operation *> getMaterializedUserOps(Operation *op);

// Gets the last materialized backward user of an operation, which is expected
// to be a phi operation.
Operation *getMaterializedBackwardUser(Operation *op);

// Attempts to route a data move operation from src_loc to dst_loc.
bool tryRouteDataMove(Operation *mov, MappingLoc src_loc, MappingLoc dst_loc,
                      bool is_backward_move, const MappingState &mapping_state,
                      std::vector<MappingLoc> &path_out);

bool tryRouteForwardMove(Operation *mov_op, MappingLoc src_loc,
                         MappingLoc dst_loc, const MappingState &state,
                         std::vector<MappingLoc> &path_out);

bool tryRouteBackwardMove(Operation *mov_op, MappingLoc src_loc,
                          MappingLoc dst_loc, const MappingState &state,
                          std::vector<MappingLoc> &path_out);

// Gets the ctrl_mov users of an operation, empty vector is returned if no
// ctrl_mov users found.
llvm::SmallVector<Operation *> getCtrlMovUsers(Operation *op);

// Identifies operations on the critical path (i.e., operations with zero
// slack). Returns pair of: (critical_ops_set, asap_level_map)
std::pair<std::set<Operation *>, llvm::DenseMap<Operation *, int>>
identifyCriticalPathOps(const std::vector<Operation *> &sorted_ops);

// Maps a materialized operation to the accelerator, and routes the dataflow
// from the producers to the given op.
bool placeAndRoute(Operation *op, const MappingLoc &target_loc,
                   MappingState &mapping_state);

// Calculates the award of mapping locations for a given op, the returned
// locations are sorted based on the award.
std::vector<MappingLoc> calculateAward(Operation *op,
                                       std::set<Operation *> &critical_ops,
                                       int target_level,
                                       const Architecture &architecture,
                                       const MappingState &mapping_state);

void updateAward(std::map<MappingLoc, int> &locs_with_award, MappingLoc loc,
                 int award);

bool canReachLocInTime(const MappingLoc &src_loc, const MappingLoc &dst_loc,
                       int deadline_step, const MappingState &mapping_state);

bool canReachLocInTime(const std::vector<Operation *> &producers,
                       const MappingLoc &target_loc, int deadline_step,
                       const MappingState &mapping_state);

// Gets an available register (for the given time range) in the given tile.
// The end_time is exclusive, meaning the register should be available
// until end_time - 1. Returns nullptr if no available register found.
//
// The optional `move_op` parameter is the DataMovOp being routed. It is
// forwarded to MappingState::isAvailableAcrossTime() /
// isAvailableForOccupyStatus() so that multiple DataMovOps carrying the same
// materialized source value can share a single physical register. When
// `move_op` is non-null the availability check recognises that two DataMovOps
// reading the identical value do not actually conflict, because the single
// register read port broadcasts the value to all consumers. Passing nullptr
// disables this sharing and falls back to the strict one-occupant-per-register
// rule.
Register *getAvailableRegister(const MappingState &mapping_state, Tile *tile,
                               int start_time, int exclusive_end_time,
                               neura::DataMovOp move_op = nullptr);

// Gets the execution latency of an operation from its "latency" attribute.
// Returns 1 (single-cycle) if the attribute is not present.
int getOpLatency(Operation *op);

// Checks if an operation is a multi-cycle operation (latency > 1).
bool isMultiCycleOp(Operation *op);

} // namespace neura
} // namespace mlir
