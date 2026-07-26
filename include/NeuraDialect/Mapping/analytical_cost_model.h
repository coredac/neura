//===- analytical_cost_model.h - Parametric single-task II model ---------===//
//
// Analytical (parametric) cost model that predicts the modulo-scheduling
// Initiation Interval (II) of a single Neura task from compile-time structural
// parameters ONLY — the DFG and the CGRA `Architecture`. It never invokes the
// mapper. The mapper is used solely as an offline validation oracle.
//
// The prediction is the maximum of independent resource lower bounds:
//
//   II = clamp( max(ResMII, RecMII, MemMII, RouteMII, RegMII, IssueMII),
//               1, max_ii )
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

// A single resource lower bound with the raw quantities that produced it, so
// the contribution of every term is explainable / printable.
struct IIBound {
  int value = 1;             // ceil(demand / capacity), floored at 1.
  long long demand = 0;      // numerator (work / accesses / moves / live vals).
  long long capacity = 1;    // denominator (FU count / ports / links / regs).
  std::string detail;        // human-readable dominant sub-term (e.g. FU class).
};

// Full per-bound breakdown of the analytical II prediction for one region.
struct AnalyticalIIBreakdown {
  IIBound res;    // ResMII   — per-FU-class functional-unit throughput.
  IIBound rec;    // RecMII   — loop-carried recurrence latency / distance.
  IIBound mem;    // MemMII   — load/store port + memory-FU contention.
  IIBound route;  // RouteMII — routed edge demand vs link capacity.
  IIBound reg;    // RegMII   — simultaneously-live values vs register capacity.
  IIBound issue;  // IssueMII — tile issue-slot occupancy.

  int final_ii = 1;          // clamp(max(all bounds), 1, max_ii).
  int max_ii = 0;            // architecture II ceiling (ctrl_mem_items).
  bool clamped = false;      // true if max(bounds) exceeded max_ii.
  std::string dominant;      // name of the bound equal to final_ii.

  // Prints the structured diagnostic block:
  //   [cost-model-analytical]
  //   res_mii=..  rec_mii=..  mem_mii=..  route_mii=..  reg_mii=..  issue_mii=..
  //   final_ii=..  (dominant=..)
  void print(llvm::raw_ostream &os) const;
};

// ===-- Individual bounds (each independently testable) --=================== //

// ResMII: partitions materialized ops by FU class (add/mul/mem/cmp/... per
// kFuTypesToOperations), weights each by its execution latency, and divides by
// the number of tiles that physically provide that FU class.
//   ResMII = max_class ceil( sum_latency(class) / #tiles_supporting(class) )
IIBound calculateResMiiPerClass(Region &region, const Architecture &arch);

// RecMII: for every loop-carried recurrence cycle (reserve -> ... -> ctrl_mov),
// sums the execution latency of the materialized ops on the cycle and divides
// by the dependence distance (1 for a single reserve/ctrl_mov pair in this IR).
//   RecMII = max_cycle ceil( sum_latency(cycle) / distance(cycle) )
IIBound calculateRecMiiWeighted(Region &region, const Architecture &arch);

// MemMII: load/store demand against the tiles that provide memory FUs.
//   MemMII = max( ceil(mem_ops / #mem_tiles),
//                 ceil(indexed_mem_ops / #mem_indexed_tiles) )
// Indexed / indirect accesses are treated conservatively (never discounted).
IIBound calculateMemMii(Region &region, const Architecture &arch);

// RouteMII: total inter-op move demand against total link capacity.
//   RouteMII = ceil( sum_edges ceil(bits(edge)/link_bw) / #links )
// Fanout is counted as N moves (one per real consumer), never a single edge.
IIBound calculateRouteMii(Region &region, const Architecture &arch);

// RegMII: peak number of simultaneously-live SSA values (via ASAP levels)
// against the total register capacity of the array.
//   RegMII = ceil( max_level(#live values) / sum_tiles(#registers) )
IIBound calculateRegMii(Region &region, const Architecture &arch);

// IssueMII: total materialized-op issue slots against total tile issue
// bandwidth (1 issue / tile / cycle here). Coincides with the crude
// ceil(#ops / #tiles) baseline and is kept separate so multi-issue tiles are
// expressible.
IIBound calculateIssueMii(Region &region, const Architecture &arch);

// Computes all bounds and combines them into the final prediction.
AnalyticalIIBreakdown computeAnalyticalII(Region &region,
                                          const Architecture &arch);

} // namespace neura
} // namespace mlir
