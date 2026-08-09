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
  int value = 1;          // ceil(demand / capacity), floored at 1.
  long long demand = 0;   // numerator (work / accesses / moves / live vals).
  long long capacity = 1; // denominator (FU count / ports / links / regs).
  std::string detail;     // human-readable dominant sub-term (e.g. FU class).
};

// Full per-bound breakdown of the analytical II prediction for one region.
struct AnalyticalIIBreakdown {
  IIBound res;   // ResMII   — per-FU-class functional-unit throughput.
  IIBound rec;   // RecMII   — loop-carried recurrence latency / distance.
  IIBound mem;   // MemMII   — load/store port + memory-FU contention.
  IIBound route; // RouteMII — routed edge demand vs link capacity.
  IIBound reg;   // RegMII   — simultaneously-live values vs register capacity.
  IIBound issue; // IssueMII — tile issue-slot occupancy.

  // RouteHopMII — the same routed demand as RouteMII, but charged for DISTANCE.
  //
  // This is deliberately NOT a lower bound, and it is kept out of `final_ii`
  // for that reason. RouteMII divides moves by links, which silently prices
  // every transfer at one link; a value that crosses h tiles holds h links for
  // the residue, so the honest denominator is `links / mean_hops`. Using the
  // MEAN hop makes the term shape-aware -- the identical DFG costs more on a
  // 1x16 strip than on a 4x4 block, which is the one thing every ceil(demand /
  // capacity) bound here is blind to -- but a placement that clusters
  // communicating ops can beat the mean, so as a bound it would be unsound.
  //
  // Measured against the exact CP-SAT mapper, the sound floor alone is what
  // under-predicts: `plain_gemm` on 4x4 schedules at II=1 but cannot ROUTE
  // there, and the true optimum is 2. That gap is what this term exists to
  // close.
  IIBound route_hop;

  int final_ii = 1;     // clamp(max(sound bounds), 1, max_ii). Never over-predicts.
  int predicted_ii = 1; // clamp(max(final_ii, route_hop), 1, max_ii).
  int max_ii = 0;       // architecture II ceiling (ctrl_mem_items).
  bool clamped = false; // true if max(bounds) exceeded max_ii.
  std::string dominant; // name of the largest-demand bound (== final_ii unless
                        // clamped to max_ii).
  double mean_hops = 0.0; // mean tile-to-tile distance over the link graph.

  // Prints the structured diagnostic block:
  //   [cost-model-analytical]
  //   res_mii=..  rec_mii=..  mem_mii=..  route_mii=..  reg_mii=.. issue_mii=..
  //   final_ii=..  (dominant=..)  predicted_ii=..
  void print(llvm::raw_ostream &os) const;
};

// ===-- Individual bounds (each independently testable) --===================
// //

// ResMII: partitions placed ops by FU class (add/mul/mem/cmp/... per
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

// IssueMII: total placed-op issue slots against total tile issue
// bandwidth (1 issue / tile / cycle here). Coincides with the crude
// ceil(#ops / #tiles) baseline and is kept separate so multi-issue tiles are
// expressible.
IIBound calculateIssueMii(Region &region, const Architecture &arch);

// Mean directed tile-to-tile distance, in link hops, over the REAL link graph
// (BFS from every tile, averaged over reachable ordered pairs). Works for any
// topology, including the irregular L/T blocks the shape search proposes, so a
// non-rectangular shape is priced by its actual connectivity rather than by its
// bounding box. Returns 0 for a fabric with no inter-tile links.
double meanHopDistance(const Architecture &arch);

// RouteHopMII: RouteMII's demand charged for distance -- each move is assumed
// to hold `meanHopDistance` links for its residue instead of one.
//   RouteHopMII = ceil( demand * mean_hops / #links )
// Shape-aware, and NOT a lower bound; see AnalyticalIIBreakdown::route_hop.
IIBound calculateRouteHopMii(Region &region, const Architecture &arch);

// Computes all bounds and combines them into the final prediction.
AnalyticalIIBreakdown computeAnalyticalII(Region &region,
                                          const Architecture &arch);

} // namespace neura
} // namespace mlir
