//===- analytical_cost_model.h - Parametric single-task II model ---------===//
//
// Analytical (parametric) cost model that predicts the modulo-scheduling
// Initiation Interval (II) of a single Neura task from compile-time structural
// parameters ONLY — the DFG and the CGRA `Architecture`. It never invokes the
// mapper. The mapper is used solely as an offline validation oracle.
//
// The sound floor is the maximum of ComputeMII, RecMII, MemMII and ResMII.
// RouteMII and RegMII are placement-dependent diagnostics and do not
// participate in the result because their aggregate estimates can over-predict
// a valid placement.
//
// Every bound is a ceil(demand / capacity)-style expression over interpretable
// quantities, so the dominant bound and its numerator/denominator are printable
// and derivable. No training data, no regression, no lookup tables, and no
// dependence on any workload / task name is used.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "NeuraDialect/Architecture/Architecture.h"
#include "mlir/IR/Region.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace neura {

// A resource estimate with the raw quantities that produced it.
struct IIBound {
  int value = 1;           // ceil(demand / capacity), or zero if infeasible.
  long long demand = 0;    // numerator (work / accesses / moves / live vals).
  long long capacity = 1;  // denominator (FU count / ports / links / regs).
  std::string detail;      // human-readable dominant sub-term (e.g. FU class).
  bool infeasible = false; // True when nonzero demand has zero capacity.
};

// Full per-bound breakdown of the analytical II prediction for one region.
struct AnalyticalIIBreakdown {
  IIBound compute; // ComputeMII — per-FU-class functional-unit throughput.
  IIBound rec;     // RecMII   — loop-carried recurrence latency / distance.
  IIBound mem;     // MemMII   — load/store port + memory-FU contention.
  IIBound route;   // RouteMII — routed edge demand vs link capacity.
  IIBound reg; // RegMII   — simultaneously-live values vs register capacity.
  IIBound res; // ResMII — tile issue-slot occupancy.

  int final_ii = 1; // max(sound bounds), or zero when infeasible.
  int max_ii = 0;   // architecture II ceiling (ctrl_mem_items).
  bool infeasible = false;
  std::string infeasible_reason;
  std::string dominant; // name of the largest sound bound.

  // Prints the structured diagnostic block:
  //   [cost-model-analytical]
  //   compute_mii=.. rec_mii=.. mem_mii=.. route_mii=.. reg_mii=..
  //   res_mii=.. final_ii=..  (dominant=..)
  void print(llvm::raw_ostream &output_stream) const;
};

// ===-- Individual bounds (each independently testable) --===================
// //

// ComputeMII: partitions placed ops by FU class (add/mul/mem/cmp/... per
// kFuTypesToOperations), counts one start issue slot per op, and divides by the
// number of tiles that physically provide that FU class. Pipeline latency is
// handled by the shared mapper recurrence bound, not by holding a FU occupied
// for every cycle.
//   ComputeMII = max_class ceil(op_count(class) / #tiles_supporting(class))
IIBound calculateComputeMiiPerClass(Region &region, const Architecture &arch);

// RecMII: for every loop-carried recurrence cycle (reserve -> ... -> ctrl_mov),
// sums the execution latency of the materialized ops on the cycle and divides
// by the dependence distance (1 for a single reserve/ctrl_mov pair in this IR).
//   RecMII = max_cycle ceil( sum_latency(cycle) / distance(cycle) )
// MemMII: load/store demand against the tiles that provide memory FUs.
//   MemMII = max( ceil(mem_ops / #mem_tiles),
//                 ceil(indexed_mem_ops / #mem_indexed_tiles) )
// Indexed / indirect accesses are treated conservatively (never discounted).
IIBound calculateMemMii(Region &region, const Architecture &arch);

// RouteMII: aggregate inter-op move demand against total link capacity. This is
// a placement-dependent diagnostic and is excluded from the sound floor.
//   RouteMII = ceil( sum_edges ceil(bits(edge)/link_bw) / #links )
// Fanout is counted as N moves (one per real consumer), never a single edge.
IIBound calculateRouteMii(Region &region, const Architecture &arch);

// RegMII: peak number of simultaneously-live SSA values (via ASAP levels)
// against the total register capacity of the array. This is a diagnostic only:
// ASAP pressure is not necessarily the pressure of the selected schedule.
//   RegMII = ceil( max_level(#live values) / sum_tiles(#registers) )
IIBound calculateRegMii(Region &region, const Architecture &arch);

// Computes all bounds and combines them into the final prediction.
AnalyticalIIBreakdown computeAnalyticalII(Region &region,
                                          const Architecture &arch);

} // namespace neura
} // namespace mlir
