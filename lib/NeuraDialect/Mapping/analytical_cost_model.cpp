//===- analytical_cost_model.cpp - Parametric single-task II model -------===//
//
// See analytical_cost_model.h. Implements each resource lower bound purely from
// the DFG and the CGRA Architecture; no mapper is invoked.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/Mapping/analytical_cost_model.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::neura;

namespace {

// occupiesFU (which ops need a tile/FU), fuClassOf (an op's FU class),
// tilesProvidingFuClass (which tiles run a class) and buildDfgEdges (the
// dependence graph) all live in mapping_util.h -- the single source of truth
// shared with --dump-dfg-json and the mapper. Reusing them keeps the cost
// model's op set, FU bucketing, FU capacities and edge set identical to the
// instance the exact mapper solves; do not re-define any of them here.

// Number of tiles that physically provide a given FU class. Zero means NO tile
// provides it (the kernel is unmappable on this architecture); see
// tilesProvidingFuClass for the undescribed-class case, which is unconstrained
// rather than empty.
int tilesSupportingClass(const Architecture &arch,
                         const std::string &fu_class) {
  return (int)tilesProvidingFuClass(arch, fu_class).size();
}

// Bit width carried by an SSA value (for routing channel demand).
int valueBitWidth(Value value) {
  Type type = value.getType();
  if (type.isIntOrFloat()) {
    return static_cast<int>(type.getIntOrFloatBitWidth());
  }
  return 32; // predicated / opaque types: conservative single-channel default.
}

int ceilDiv(long long numerator, long long denominator) {
  if (denominator <= 0) {
    denominator = 1;
  }
  return static_cast<int>((numerator + denominator - 1) / denominator);
}

} // namespace

namespace mlir {
namespace neura {

//===----------------------------------------------------------------------===//
// ResMII — per-FU-class, latency-weighted.
//
//   work(c)  = sum over placed ops of class c of max(1, latency(op))
//   fus(c)   = #tiles that physically provide FU class c
//   ResMII   = max over classes c of ceil( work(c) / fus(c) )
//
// Why not reuse neura::calculateResMii (mapping_util.cpp)? That one is just
// ceil(#ops / #tiles): it treats every tile as interchangeable and every op as
// unit-cost, so it under-predicts whenever an FU class is scarce (e.g. 6 muls
// but only 2 mul-capable tiles) or ops are multi-cycle. Reusing it would defeat
// the point of a sharper model. Its exact formula is preserved as IssueMII
// below, and the legacy value is still max()'d in as a redundant floor by the
// caller, so nothing is lost by superseding it here.
//===----------------------------------------------------------------------===//
IIBound calculateResMiiPerClass(Region &region, const Architecture &arch) {
  std::map<std::string, long long> work_by_class; // class -> sum of latencies.
  region.walk([&](Operation *op) {
    if (!occupiesFU(op)) {
      return;
    }
    work_by_class[fuClassOf(op)] += std::max(1, getOpLatency(op));
  });

  IIBound bound;
  bound.value = 1;
  // Classes this kernel uses that the arch describes but that NO tile provides.
  std::vector<std::string> unprovidable_classes;
  for (const auto &[fu_class, class_work] : work_by_class) {
    int fu_count = tilesSupportingClass(arch, fu_class);
    if (fu_count == 0) {
      // A class no tile provides is capacity ZERO, not one. Clamping the
      // divisor to 1 here used to fabricate "some tile has it" and report a
      // finite II for a kernel that cannot be placed at all on a pruned
      // (L/T-shaped) architecture -- while the exact mapper, reading the same
      // empty tile list, correctly rejects it. Record it and flag the bound.
      unprovidable_classes.push_back(fu_class);
      continue;
    }
    int class_ii = ceilDiv(class_work, fu_count);
    if (class_ii > bound.value) {
      bound.value = class_ii;
      bound.demand = class_work;
      bound.capacity = fu_count;
      bound.detail = "class=" + fu_class +
                     " work=" + std::to_string(class_work) +
                     " fus=" + std::to_string(fu_count);
    }
  }
  if (!unprovidable_classes.empty()) {
    // No II makes this kernel mappable, so every finite floor is vacuously
    // valid; report the architecture's II ceiling, the largest representable
    // one, and let the flag (not the number) carry "unmappable".
    bound.infeasible = true;
    bound.demand = work_by_class[unprovidable_classes.front()];
    bound.capacity = 0;
    bound.value = std::max(bound.value, std::max(1, arch.getMaxCtrlMemItems()));
    bound.detail = "INFEASIBLE: no tile provides fu class";
    for (const std::string &fu_class : unprovidable_classes) {
      bound.detail += " " + fu_class;
    }
  }
  if (bound.detail.empty()) {
    bound.detail = "no placed ops";
  }
  return bound;
}

//===----------------------------------------------------------------------===//
// RecMII — loop-carried recurrence latency / distance (the critical circuit).
//
//   RecMII = max over ALL cycles K of ceil( lat(K) / omega(K) )
//     lat(K)   = sum of producer latencies of the edges around K
//     omega(K) = sum of iteration distances (1 per loop-carried back-edge) on K
//
// This is the maximum cycle ratio ("critical circuit") of the dependence graph,
// computed exactly via binary search + a positive-cycle (Bellman-Ford) test —
// NOT by enumerating cycles. The old version summed the ops that
// collectRecurrenceCycles returned, but that helper walks one ctrl_mov/reserve
// at a time, so it only sees single-recurrence-variable cycles and misses
// circuits that thread through more than one loop-carried variable. On bicg,
// for example, the enumerated bound was 5 while the true critical circuit (and
// the mapper's proven optimum) is 9; this version returns 9. Because it is the
// exact max cycle ratio it dominates any enumerated cycle, so nothing is lost.
//
// The node and edge set come from neura::collectPlacedOps / neura::buildDfgEdges
// (mapping_util), the same call --dump-dfg-json makes, so this is the exact
// instance the mapper solves rather than a second construction that happens to
// agree: forward operand edges (omega=0) plus ctrl_mov/reserve back edges
// (omega=1). Edge "delay" is the producer's latency, matching the
// scheduler's precedence  t[dst] >= t[src] + lat(src) + hop - omega*II  with
// hop dropped (a lower bound, hop >= 0).
//===----------------------------------------------------------------------===//
IIBound calculateRecMiiWeighted(Region &region, const Architecture &arch) {
  (void)arch;
  IIBound bound;
  bound.value = 1;
  bound.detail = "no recurrence";

  // Nodes and edges from the shared builder -- literally the graph
  // --dump-dfg-json hands the exact mapper, so the model and the mapper reason
  // about one instance. (De-duplicated on (src, dst, omega): repeated operand
  // edges are one dependence net, and parallel arcs cannot change a maximum
  // cycle ratio anyway.)
  std::vector<Operation *> placed_ops = collectPlacedOps(region);
  if (placed_ops.empty()) {
    return bound;
  }
  std::vector<DependenceEdge> dfg_edges = buildDfgEdges(region, placed_ops);

  struct Arc {
    int dst;
    int delay;
    int omega;
  };
  std::vector<std::vector<Arc>> out(placed_ops.size());
  long long total_delay = 0;
  for (const DependenceEdge &edge : dfg_edges) {
    // Edge delay is the producer's latency, matching the scheduler's
    // precedence constraint.
    int delay = std::max(1, getOpLatency(placed_ops[edge.src]));
    out[edge.src].push_back({edge.dst, delay, edge.omega});
    total_delay += delay;
  }

  // Max cycle ratio via binary search on r: a cycle has ratio > r iff the graph
  // with edge weight (delay - r*omega) has a positive cycle. Every cycle uses
  // >=1 omega=1 back-edge (forward edges form a DAG), so the ratio is finite.
  const int num_nodes = (int)placed_ops.size();
  auto hasPositiveCycle = [&](double r) {
    std::vector<double> dist(num_nodes, 0.0);
    for (int iter = 0; iter < num_nodes; ++iter) {
      bool updated = false;
      for (int u = 0; u < num_nodes; ++u) {
        for (const Arc &arc : out[u]) {
          double relaxed = dist[u] + (arc.delay - r * arc.omega);
          if (relaxed > dist[arc.dst] + 1e-9) {
            dist[arc.dst] = relaxed;
            updated = true;
          }
        }
      }
      if (!updated) {
        return false; // settled with no positive cycle
      }
    }
    return true; // still relaxing after |V| passes => positive cycle
  };
  double lo = 0.0, hi = (double)total_delay + 1.0;
  if (hasPositiveCycle(0.0)) { // a recurrence exists at all
    for (int iter = 0; iter < 100; ++iter) {
      double mid = 0.5 * (lo + hi);
      if (hasPositiveCycle(mid)) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    int rec_ii = std::max(1, (int)std::ceil(lo - 1e-6));
    bound.value = rec_ii;
    bound.demand = rec_ii;
    bound.capacity = 1;
    bound.detail = "critical circuit ratio=" + std::to_string(lo);
  }
  return bound;
}

//===----------------------------------------------------------------------===//
// MemMII — load/store contention on memory FUs.
//
//   MemMII = max( ceil( #plain_loads_stores / #mem_tiles ),
//                 ceil( #indexed_loads_stores / #mem_indexed_tiles ) )
//
// Plain and indexed memory ops are counted against their own tile pools, since
// a tile may support one class but not the other.
//===----------------------------------------------------------------------===//
IIBound calculateMemMii(Region &region, const Architecture &arch) {
  long long num_mem_ops = 0, num_indexed_ops = 0;
  region.walk([&](Operation *op) {
    if (!occupiesFU(op)) {
      return;
    }
    OperationKind kind = getOperationKindFromMlirOp(op);
    if (kind == ILoad || kind == IStore) {
      ++num_mem_ops;
    } else if (kind == ILoadIndexed || kind == IStoreIndexed) {
      ++num_indexed_ops;
    }
  });

  int mem_tiles = tilesSupportingClass(arch, "mem");
  int indexed_tiles = tilesSupportingClass(arch, "mem_indexed");
  // Same rule as ResMII: a memory op with no memory tile to run on is
  // infeasible, not "one tile's worth of contention". Only the divisor is
  // clamped, and only so the ratio below stays defined.
  bool infeasible = (num_mem_ops > 0 && mem_tiles == 0) ||
                    (num_indexed_ops > 0 && indexed_tiles == 0);
  int mem_ii = ceilDiv(num_mem_ops, std::max(1, mem_tiles));
  int indexed_ii = ceilDiv(num_indexed_ops, std::max(1, indexed_tiles));

  IIBound bound;
  if (indexed_ii >= mem_ii) {
    bound.value = std::max(1, indexed_ii);
    bound.demand = num_indexed_ops;
    bound.capacity = indexed_tiles;
    bound.detail = "indexed=" + std::to_string(num_indexed_ops) +
                   " mem_indexed_fus=" + std::to_string(indexed_tiles);
  } else {
    bound.value = std::max(1, mem_ii);
    bound.demand = num_mem_ops;
    bound.capacity = mem_tiles;
    bound.detail = "loads+stores=" + std::to_string(num_mem_ops) +
                   " mem_fus=" + std::to_string(mem_tiles);
  }
  if (infeasible) {
    bound.infeasible = true;
    bound.value = std::max(bound.value, std::max(1, arch.getMaxCtrlMemItems()));
    bound.detail = "INFEASIBLE: no memory tile provides it -- " + bound.detail;
  }
  return bound;
}

//===----------------------------------------------------------------------===//
// RouteMII — routed move demand vs total link capacity.
//
//   demand  = sum over data_mov m of ceil( bitwidth(m) / link_bandwidth )
//   links   = total #links in the mesh
//   RouteMII = ceil( demand / links )
//
// Each data_mov is one physical transfer (fanout is pre-expanded into one
// data_mov per consumer). This is an aggregate link-throughput bound; it ignores
// placement, so it under-predicts when routing is topologically constrained.
//===----------------------------------------------------------------------===//
IIBound calculateRouteMii(Region &region, const Architecture &arch) {
  auto links = arch.getAllLinks();
  long long total_links = static_cast<long long>(links.size());

  // A single-tile / link-less arch needs no inter-tile routing, so RouteMII does
  // not bind — dividing intra-tile moves by a phantom link would fabricate a
  // bound. Returning 1 keeps it a valid (non-over-predicting) lower bound.
  if (total_links <= 0) {
    IIBound bound;
    bound.value = 1;
    bound.detail = "no inter-tile links; RouteMII not applied";
    return bound;
  }

  // Charge each move against the WIDEST link bandwidth read from the arch spec
  // (Link::getBandwidth, populated from architecture.yaml / defaults). Since
  // RouteMII must be a lower bound, the most generous channel (fewest sub-
  // channels per move) is the safe, non-over-predicting choice on a heterogeneous
  // mesh; on the default homogeneous mesh every link is identical so it is exact.
  int link_bandwidth = 0;
  for (Link *link : links) {
    link_bandwidth = std::max(link_bandwidth, link->getBandwidth());
  }
  if (link_bandwidth <= 0) {
    // The arch specifies no usable link bandwidth; rather than invent one, treat
    // routing bandwidth as unmodeled so RouteMII does not bind (safe for a LB).
    IIBound bound;
    bound.value = 1;
    bound.detail = "no link bandwidth in arch spec; RouteMII not applied";
    return bound;
  }

  // Each neura.data_mov is one physical move; fanout is already expanded into
  // one data_mov per real consumer, so replicated traffic is counted here.
  long long channel_demand = 0;
  long long num_moves = 0;
  region.walk([&](neura::DataMovOp move) {
    ++num_moves;
    channel_demand += ceilDiv(valueBitWidth(move.getResult()), link_bandwidth);
  });

  IIBound bound;
  bound.value = std::max(1, ceilDiv(channel_demand, total_links));
  bound.demand = channel_demand;
  bound.capacity = total_links;
  bound.detail = "moves=" + std::to_string(num_moves) +
                 " channel_demand=" + std::to_string(channel_demand) +
                 " links=" + std::to_string(total_links) +
                 " link_bw=" + std::to_string(link_bandwidth);
  return bound;
}

//===----------------------------------------------------------------------===//
// meanHopDistance — mean directed tile-to-tile distance over the link graph.
//
// BFS from every tile over the real directed links, averaged over the ordered
// pairs that are actually reachable. Unreachable pairs are skipped rather than
// charged a sentinel: a disconnected shape would otherwise get an enormous mean
// and be ranked last for a reason that has nothing to do with its traffic.
// Self-pairs are excluded -- a value produced and consumed on one tile crosses
// no link and is not what this term prices.
//
// The taskflow allocator used to carry a second implementation of this
// (`averageHop`) that fed the same objective through `predictedCost`, and the
// two disagreed on both points above: it counted dist[source]==0 in the pair
// count, reading ~6.25% low on a connected 16-tile fabric, and it enumerated
// tiles from getAllTiles() so an ISOLATED tile contributed a zero row and made
// a worse-connected shape score BETTER. Excluding self-pairs settles both:
// once dist[source] is skipped, a tile with no links contributes no pair at
// all, so enumerating the sources from getAllTiles() (done here, as the
// architecture -- not the link list -- is what defines the tile set) is
// identical to enumerating them from the link endpoints.
//
// Why the shape-blind MII family needs this at all: `ResMII = ceil(ops/tiles)`
// sees only the tile count and `RouteMII = moves/links` is invariant under
// transposing the array, so two arrays of equal area score identically where
// the real mapper does not (fir on 8x4 -> II 6 vs 4x8 -> II 5). This is the
// topology feature that turns a sound floor into a shape-aware prediction; the
// floor itself is left untouched.
//===----------------------------------------------------------------------===//
double meanHopDistance(const Architecture &arch, bool mem_weighted) {
  std::vector<Tile *> tiles = arch.getAllTiles();
  if (tiles.size() < 2) {
    return 0.0;
  }

  // Adjacency straight off the link list, so any topology works -- mesh,
  // strip, or the irregular blocks the shape search proposes.
  llvm::DenseSet<Tile *> present(tiles.begin(), tiles.end());
  llvm::DenseMap<Tile *, llvm::SmallVector<Tile *, 8>> neighbours;
  for (Link *link : arch.getAllLinks()) {
    Tile *src = link->getSrcTile();
    Tile *dst = link->getDstTile();
    if (!src || !dst) {
      continue;
    }
    if (!present.contains(src) || !present.contains(dst)) {
      continue;
    }
    neighbours[src].push_back(dst);
  }

  llvm::DenseMap<Tile *, int> distance;
  llvm::SmallVector<Tile *, 64> queue;
  auto bfs = [&](Tile *source) {
    distance.clear();
    queue.clear();
    distance[source] = 0;
    queue.push_back(source);
    for (size_t head = 0; head < queue.size(); ++head) {
      Tile *current = queue[head];
      int next_distance = distance[current] + 1;
      auto found = neighbours.find(current);
      if (found == neighbours.end()) {
        continue;
      }
      for (Tile *neighbour : found->second) {
        if (distance.insert({neighbour, next_distance}).second) {
          queue.push_back(neighbour);
        }
      }
    }
  };

  // Memory-capable tiles: the sources/sinks of every live value.
  if (mem_weighted) {
    llvm::DenseSet<Tile *> mem_tiles;
    for (Tile *tile : tiles) {
      if (tile->canSupportOperation(ILoadIndexed) ||
          tile->canSupportOperation(IStoreIndexed) ||
          tile->canSupportOperation(ILoad) ||
          tile->canSupportOperation(IStore)) {
        mem_tiles.insert(tile);
      }
    }
    // No memory FU anywhere (or all of them): the weighting carries no signal,
    // so fall through to the plain pairwise mean rather than returning a
    // constant.
    if (!mem_tiles.empty() && mem_tiles.size() != tiles.size()) {
      // Mean over tiles of the distance to the nearest memory tile. A memory
      // tile is at distance 0 from itself: it already holds what it needs.
      long long total = 0;
      long long counted = 0;
      for (Tile *tile : tiles) {
        bfs(tile);
        int best = INT_MAX;
        for (Tile *mem_tile : mem_tiles) {
          auto found = distance.find(mem_tile);
          if (found != distance.end()) {
            best = std::min(best, found->second);
          }
        }
        if (best != INT_MAX) {
          total += best;
          ++counted;
        }
      }
      return counted ? (double)total / (double)counted : 0.0;
    }
  }

  long long total_hops = 0, pairs = 0;
  for (Tile *source : tiles) {
    bfs(source);
    for (auto &[tile, hops] : distance) {
      if (tile == source) {
        continue;
      }
      total_hops += hops;
      ++pairs;
    }
  }
  return pairs ? (double)total_hops / (double)pairs : 0.0;
}

//===----------------------------------------------------------------------===//
// RouteHopMII — RouteMII's demand, charged for distance.
//
//   RouteHopMII = ceil( channel_demand * mean_hops / #links )
//
// RouteMII prices every transfer at one link, which is why it cannot tell a
// 4x4 block from a 1x16 strip carrying the same traffic. A value crossing h
// tiles occupies h links for its residue, so multiplying the demand by the mean
// hop distance restores the topology signal. Not a lower bound -- a clustered
// placement can beat the mean -- so it is reported separately and never folded
// into `final_ii`.
//===----------------------------------------------------------------------===//
IIBound calculateRouteHopMii(Region &region, const Architecture &arch) {
  IIBound bound = calculateRouteMii(region, arch);
  const double mean_hops = meanHopDistance(arch);
  // A fabric with no links, or one where RouteMII already declined to apply,
  // has nothing to scale: keep whatever RouteMII decided rather than inventing
  // a distance for a transfer that never crosses a link.
  if (mean_hops <= 1.0 || bound.capacity <= 0 || bound.demand <= 0) {
    bound.detail += " (route_hop: mean_hops=" + std::to_string(mean_hops) +
                    ", not applied)";
    return bound;
  }
  const long long scaled_demand =
      (long long)std::llround((double)bound.demand * mean_hops);
  bound.value = std::max<int>(1, ceilDiv(scaled_demand, bound.capacity));
  bound.detail += " mean_hops=" + std::to_string(mean_hops) +
                  " hop_demand=" + std::to_string(scaled_demand);
  bound.demand = scaled_demand;
  return bound;
}

//===----------------------------------------------------------------------===//
// RegMII — peak simultaneously-live values vs total register capacity.
//
//   For each placed value v with an ASAP def level and a later use, it is
//   live over [def_level(v), last_use_level(v)); peak_live = max over levels of
//   the count of values whose live range covers that level (via a +1/-1 sweep).
//   regs     = total registers across all tiles
//   RegMII   = ceil( peak_live / regs )
//
// ASAP levels come from the topological order; the reserve op breaks the
// loop-carried back-edge so the traversal is a finite DAG.
//===----------------------------------------------------------------------===//
IIBound calculateRegMii(Region &region, const Architecture &arch) {
  // ASAP levels over the whole op graph (data_mov/reserve are their own nodes).
  // Uses the direct SSA producer — getTopologicallySortedOps guarantees
  // producers precede consumers, and the reserve op (no operands) breaks the
  // loop-carried back-edge so this is a finite DAG traversal.
  std::vector<Operation *> topo_ops = getTopologicallySortedOps(region);
  llvm::DenseMap<Operation *, int> asap_level;
  for (Operation *op : topo_ops) {
    int op_level = 0;
    for (Value operand : op->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (producer && asap_level.count(producer)) {
        op_level = std::max(op_level, asap_level[producer] + 1);
      }
    }
    asap_level[op] = op_level;
  }

  // Live range of each placed value: [def_level, last_use].
  // Accumulate +1 at each value's def level and -1 at its last-use level, then
  // prefix-sum to find the peak number of simultaneously-live values.
  int max_level = 0;
  for (auto &entry : asap_level) {
    max_level = std::max(max_level, entry.second);
  }
  std::vector<long long> live_delta(max_level + 2, 0);

  region.walk([&](Operation *op) {
    if (!occupiesFU(op) || op->getNumResults() == 0) {
      return;
    }
    if (!asap_level.count(op)) {
      return;
    }
    int def_level = asap_level[op];
    int last_use_level = def_level;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        // Unwrap routing users (data_mov) to the placed consumer.
        Operation *placed_user = user;
        if (is_non_materialized(user)) {
          for (Operation *router_user : user->getUsers()) {
            if (asap_level.count(router_user)) {
              placed_user = router_user;
            }
          }
        }
        if (asap_level.count(placed_user)) {
          last_use_level =
              std::max(last_use_level, asap_level[placed_user]);
        }
      }
    }
    if (last_use_level > def_level) {
      live_delta[def_level] += 1;
      live_delta[last_use_level] -= 1;
    }
  });

  long long peak_live = 0, running_live = 0;
  int peak_level = 0;
  for (int level = 0; level <= max_level; ++level) {
    running_live += live_delta[level];
    if (running_live > peak_live) {
      peak_live = running_live;
      peak_level = level;
    }
  }

  long long total_registers = 0;
  for (Tile *tile : arch.getAllTiles()) {
    total_registers += static_cast<long long>(tile->getRegisters().size());
  }

  IIBound bound;
  if (total_registers <= 0) {
    // No register files are modeled on this arch (e.g. a config with fewer than
    // one regfile's worth of registers, where Architecture rounds down to zero).
    // A register-pressure lower bound is meaningless here, so don't let RegMII
    // bind — reporting peak_live/1 would grossly OVER-predict and wrongly reject
    // otherwise-mappable shapes. Under-binding is the safe direction for a lower
    // bound.
    bound.value = 1;
    bound.demand = peak_live;
    bound.capacity = 0;
    bound.detail = "no register files modeled; RegMII not applied";
    return bound;
  }

  // NOTE: peak_live is the peak concurrency of a single ASAP schedule, which
  // MAXIMISES overlap; a different valid schedule at the same II may realise
  // lower register pressure. So this is a strong heuristic estimate, not a
  // certified lower bound — it can over-predict on wide non-reconverging fan-out.
  // It rarely dominates (registers are usually plentiful), and top-k + mapper
  // verification absorbs the residual.
  bound.value = std::max(1, ceilDiv(peak_live, total_registers));
  bound.demand = peak_live;
  bound.capacity = total_registers;
  bound.detail = "peak_live=" + std::to_string(peak_live) +
                 " @level=" + std::to_string(peak_level) +
                 " regs=" + std::to_string(total_registers);
  return bound;
}

//===----------------------------------------------------------------------===//
// IssueMII — tile issue-slot occupancy (crude ceil(#ops / #tiles) baseline).
//
//   IssueMII = ceil( #placed_ops / #tiles )
//
// FU-agnostic floor: every op needs some tile-slot per iteration regardless of
// class. This is exactly the legacy calculateResMii formula, kept as its own
// bound so nothing is lost by replacing that coarse ResMII with the per-class one.
//===----------------------------------------------------------------------===//
IIBound calculateIssueMii(Region &region, const Architecture &arch) {
  long long num_ops = 0;
  region.walk([&](Operation *op) {
    if (occupiesFU(op)) {
      ++num_ops;
    }
  });
  int num_tiles = std::max(1, arch.getNumTiles());
  IIBound bound;
  bound.value = std::max(1, ceilDiv(num_ops, num_tiles));
  bound.demand = num_ops;
  bound.capacity = num_tiles;
  bound.detail =
      "ops=" + std::to_string(num_ops) + " tiles=" + std::to_string(num_tiles);
  return bound;
}

//===----------------------------------------------------------------------===//
// Combine.
//
//   II = clamp( max(ResMII, RecMII, MemMII, RouteMII, RegMII, IssueMII),
//               1, ctrl_mem_items )
//
// Every bound is an independent lower bound on the achievable II, so their max
// is the tightest analytical lower bound. ctrl_mem_items is the hardware cap on
// schedulable II (control-memory depth); exceeding it means "not mappable at
// this shape", recorded via clamped=true.
//===----------------------------------------------------------------------===//
AnalyticalIIBreakdown computeAnalyticalII(Region &region,
                                          const Architecture &arch) {
  AnalyticalIIBreakdown breakdown;
  breakdown.res = calculateResMiiPerClass(region, arch);
  breakdown.rec = calculateRecMiiWeighted(region, arch);
  breakdown.mem = calculateMemMii(region, arch);
  breakdown.route = calculateRouteMii(region, arch);
  breakdown.reg = calculateRegMii(region, arch);
  breakdown.issue = calculateIssueMii(region, arch);

  struct NamedBound {
    const char *name;
    int value;
  };
  NamedBound named_bounds[] = {
      {"res", breakdown.res.value}, {"rec", breakdown.rec.value},
      {"mem", breakdown.mem.value}, {"route", breakdown.route.value},
      {"reg", breakdown.reg.value}, {"issue", breakdown.issue.value}};
  int max_bound_ii = 1;
  const char *dominant_name = "res";
  for (const NamedBound &named_bound : named_bounds) {
    if (named_bound.value > max_bound_ii) {
      max_bound_ii = named_bound.value;
      dominant_name = named_bound.name;
    }
  }
  breakdown.dominant = dominant_name;

  breakdown.max_ii = arch.getMaxCtrlMemItems();
  // A capacity-zero FU class is infeasibility, not a large II: no II maps this
  // kernel on this architecture. Report the ceiling (every floor is vacuously
  // valid then) and let `infeasible` carry the verdict, so a caller can tell
  // "does not fit" from "fits, expensively". Marked clamped too, since that is
  // the flag existing callers already read as "not mappable at this shape".
  breakdown.infeasible = breakdown.res.infeasible || breakdown.mem.infeasible;
  if (breakdown.infeasible) {
    breakdown.final_ii =
        breakdown.max_ii > 0 ? breakdown.max_ii : std::max(1, max_bound_ii);
    breakdown.clamped = true;
  } else if (breakdown.max_ii > 0 && max_bound_ii > breakdown.max_ii) {
    breakdown.final_ii = breakdown.max_ii;
    breakdown.clamped = true;
  } else {
    breakdown.final_ii = std::max(1, max_bound_ii);
  }

  // The shape-aware prediction sits on top of the sound floor and never below
  // it. `final_ii` is left exactly as it was so every existing consumer,
  // including the pruning proofs that rely on it not over-predicting, is
  // unaffected; a caller that wants accuracy rather than soundness reads
  // `predicted_ii`.
  breakdown.route_hop = calculateRouteHopMii(region, arch);
  breakdown.mean_hops = meanHopDistance(arch);
  int predicted = std::max(breakdown.final_ii, breakdown.route_hop.value);
  if (breakdown.max_ii > 0 && predicted > breakdown.max_ii) {
    predicted = breakdown.max_ii;
  }
  breakdown.predicted_ii = std::max(1, predicted);
  return breakdown;
}

void AnalyticalIIBreakdown::print(llvm::raw_ostream &os) const {
  os << "[cost-model-analytical]\n";
  os << "  res_mii=" << res.value << " (" << res.detail << ")\n";
  os << "  rec_mii=" << rec.value << " (" << rec.detail << ")\n";
  os << "  mem_mii=" << mem.value << " (" << mem.detail << ")\n";
  os << "  route_mii=" << route.value << " (" << route.detail << ")\n";
  os << "  reg_mii=" << reg.value << " (" << reg.detail << ")\n";
  os << "  issue_mii=" << issue.value << " (" << issue.detail << ")\n";
  os << "  route_hop_mii=" << route_hop.value << " (" << route_hop.detail
     << ")\n";
  os << "  final_ii=" << final_ii << " (dominant=" << dominant;
  if (infeasible) {
    os << ", INFEASIBLE: an fu class this kernel needs is provided by no tile";
  }
  if (clamped) {
    os << ", clamped-to-max_ii=" << max_ii;
  }
  os << ")\n";
  os << "  predicted_ii=" << predicted_ii << " (mean_hops=" << mean_hops
     << ")\n";
}

} // namespace neura
} // namespace mlir
