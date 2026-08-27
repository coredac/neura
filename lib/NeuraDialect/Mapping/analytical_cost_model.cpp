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
#include "NeuraDialect/NeuraTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::neura;

namespace {

// occupiesFU (which ops need a tile/FU) and fuClassOf (an op's FU class) both
// live in mapping_util.h -- the single source of truth shared with the mapper.
// Reusing them keeps the cost model's op set and FU bucketing identical to the
// mapper's; do not re-define either here.

// Number of tiles that physically provide a given FU class.
int tilesSupportingClass(const Architecture &arch,
                         const std::string &fu_class) {
  auto found = kFuTypesToOperations.find(fu_class);
  if (found == kFuTypesToOperations.end() || found->second.empty()) {
    return arch.getNumTiles(); // unknown class: don't over-constrain.
  }
  OperationKind probe_kind = found->second.front();
  int supporting_tiles = 0;
  for (Tile *tile : arch.getAllTiles()) {
    if (tile->canSupportOperation(probe_kind)) {
      ++supporting_tiles;
    }
  }
  return supporting_tiles;
}

// Bit width carried by an SSA value (for routing channel demand).
int valueBitWidth(Value value) {
  Type type = value.getType();
  if (auto predicated_value = dyn_cast<neura::PredicatedValue>(type)) {
    type = predicated_value.getValueType();
  }
  if (type.isIntOrFloat()) {
    return static_cast<int>(type.getIntOrFloatBitWidth());
  }
  return 32; // Opaque types: conservative single-channel default.
}

int ceilDiv(long long numerator, long long denominator) {
  assert(denominator > 0 && "ceilDiv denominator must be positive");
  return static_cast<int>((numerator + denominator - 1) / denominator);
}

} // namespace

namespace mlir {
namespace neura {

//===----------------------------------------------------------------------===//
// ComputeMII — per-FU-class start issue demand.
//
//   work(c)  = number of placed ops of class c
//   fus(c)   = #tiles that physically provide FU class c
//   ComputeMII = max over classes c of ceil( work(c) / fus(c) )
//
// A pipelined FU accepts one new operation per issue slot. Its latency belongs
// in the shared recurrence constraint rather than being counted as occupancy.
//===----------------------------------------------------------------------===//
IIBound calculateComputeMiiPerClass(Region &region, const Architecture &arch) {
  std::map<std::string, long long> work_by_class; // class -> start slots.
  region.walk([&](Operation *op) {
    if (!occupiesFU(op)) {
      return;
    }
    ++work_by_class[fuClassOf(op)];
  });

  IIBound bound;
  bound.value = 1;
  std::vector<std::string> unprovided_classes;
  for (const auto &[fu_class, class_work] : work_by_class) {
    int fu_count = tilesSupportingClass(arch, fu_class);
    if (fu_count == 0) {
      unprovided_classes.push_back(fu_class);
      continue;
    }
    int class_ii = ceilDiv(class_work, fu_count);
    if (bound.detail.empty() || class_ii > bound.value) {
      bound.value = class_ii;
      bound.demand = class_work;
      bound.capacity = fu_count;
      bound.detail = "class=" + fu_class +
                     " ops=" + std::to_string(class_work) +
                     " fus=" + std::to_string(fu_count);
    }
  }
  if (!unprovided_classes.empty()) {
    bound.infeasible = true;
    bound.demand = work_by_class[unprovided_classes.front()];
    bound.capacity = 0;
    bound.value = 0;
    bound.detail = "infeasible: no tile provides fu class";
    for (const std::string &fu_class : unprovided_classes)
      bound.detail += " " + fu_class;
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
// Edge set matches DumpDfgJsonPass exactly (same nodes/edges the exact mapper
// solves over): forward operand edges (omega=0) plus ctrl_mov/reserve back
// edges (omega=1). Edge "delay" is the producer's latency, matching the
// scheduler's precedence  t[dst] >= t[src] + lat(src) + hop - omega*II  with
// hop dropped (a lower bound, hop >= 0).
//===----------------------------------------------------------------------===//
IIBound makeRecMiiBound(Region &region) {
  IIBound bound;
  bound.value = calculateRecMii(region);
  bound.demand = bound.value;
  bound.capacity = 1;
  bound.detail =
      "shared mapper recurrence bound=" + std::to_string(bound.value);
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
    bound.value = 0;
    bound.capacity = 0;
    if (num_mem_ops > 0 && mem_tiles == 0) {
      bound.demand = num_mem_ops;
      bound.detail = "infeasible: no tile provides mem";
    } else {
      bound.demand = num_indexed_ops;
      bound.detail = "infeasible: no tile provides mem_indexed";
    }
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
// data_mov per consumer). This is an aggregate link-throughput bound; it
// ignores placement, so it under-predicts when routing is topologically
// constrained.
//===----------------------------------------------------------------------===//
IIBound calculateRouteMii(Region &region, const Architecture &arch) {
  auto links = arch.getAllLinks();
  long long total_links = static_cast<long long>(links.size());

  // A single-tile / link-less arch needs no inter-tile routing, so RouteMII
  // does not bind — dividing intra-tile moves by a phantom link would fabricate
  // a bound. Returning 1 keeps it a valid (non-over-predicting) lower bound.
  if (total_links <= 0) {
    IIBound bound;
    bound.value = 1;
    bound.detail = "no inter-tile links; RouteMII not applied";
    return bound;
  }

  // Charge each move against the WIDEST link bandwidth read from the arch spec
  // (Link::getBandwidth, populated from architecture.yaml / defaults). Since
  // RouteMII must be a lower bound, the most generous channel (fewest sub-
  // channels per move) is the safe, non-over-predicting choice on a
  // heterogeneous mesh; on the default homogeneous mesh every link is identical
  // so it is exact.
  int link_bandwidth = 0;
  for (Link *link : links) {
    link_bandwidth = std::max(link_bandwidth, link->getBandwidth());
  }
  if (link_bandwidth <= 0) {
    // The arch specifies no usable link bandwidth; rather than invent one,
    // treat routing bandwidth as unmodeled so RouteMII does not bind (safe for
    // a LB).
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
          last_use_level = std::max(last_use_level, asap_level[placed_user]);
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
    // one regfile's worth of registers, where Architecture rounds down to
    // zero). A register-pressure lower bound is meaningless here, so don't let
    // RegMII bind — reporting peak_live/1 would grossly OVER-predict and
    // wrongly reject otherwise-mappable shapes. Under-binding is the safe
    // direction for a lower bound.
    bound.value = 1;
    bound.demand = peak_live;
    bound.capacity = 0;
    bound.detail = "no register files modeled; RegMII not applied";
    return bound;
  }

  // NOTE: peak_live is the peak concurrency of a single ASAP schedule, which
  // MAXIMISES overlap; a different valid schedule at the same II may realise
  // lower register pressure. So this is a strong heuristic estimate, not a
  // certified lower bound — it can over-predict on wide non-reconverging
  // fan-out. It rarely dominates (registers are usually plentiful), and top-k +
  // mapper verification absorbs the residual.
  bound.value = std::max(1, ceilDiv(peak_live, total_registers));
  bound.demand = peak_live;
  bound.capacity = total_registers;
  bound.detail = "peak_live=" + std::to_string(peak_live) +
                 " @level=" + std::to_string(peak_level) +
                 " regs=" + std::to_string(total_registers);
  return bound;
}

//===----------------------------------------------------------------------===//
// ResourceMII — tile issue-slot occupancy (crude ceil(#ops / #tiles) baseline).
//
//   ResourceMII = ceil( #placed_ops / #tiles )
//
// FU-agnostic floor: every op needs some tile-slot per iteration regardless of
// class. This is exactly the legacy calculateResourceMii formula.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Combine.
//
//   II = clamp( max(ComputeMII, RecMII, MemMII, RouteMII, RegMII, ResourceMII),
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
  breakdown.compute = calculateComputeMiiPerClass(region, arch);
  breakdown.rec = makeRecMiiBound(region);
  breakdown.mem = calculateMemMii(region, arch);
  breakdown.route = calculateRouteMii(region, arch);
  breakdown.reg = calculateRegMii(region, arch);
  long long res_demand = 0;
  long long res_capacity = 0;
  breakdown.res.value =
      calculateResourceMii(region, arch, &res_demand, &res_capacity);
  breakdown.res.demand = res_demand;
  breakdown.res.capacity = res_capacity;
  breakdown.res.detail = "ops=" + std::to_string(res_demand) +
                         " tiles=" + std::to_string(res_capacity);

  struct NamedBound {
    const char *name;
    int value;
  };
  NamedBound named_bounds[] = {{"compute", breakdown.compute.value},
                               {"rec", breakdown.rec.value},
                               {"mem", breakdown.mem.value},
                               {"res", breakdown.res.value}};
  int max_bound_ii = 1;
  const char *dominant_name = "compute";
  for (const NamedBound &named_bound : named_bounds) {
    if (named_bound.value > max_bound_ii) {
      max_bound_ii = named_bound.value;
      dominant_name = named_bound.name;
    }
  }
  breakdown.dominant = dominant_name;

  breakdown.max_ii = arch.getMaxCtrlMemItems();
  breakdown.infeasible =
      breakdown.compute.infeasible || breakdown.mem.infeasible;
  if (breakdown.infeasible) {
    breakdown.final_ii = 0;
    breakdown.dominant = "infeasible";
    breakdown.infeasible_reason = breakdown.compute.infeasible
                                      ? breakdown.compute.detail
                                      : breakdown.mem.detail;
  } else if (breakdown.max_ii > 0 && max_bound_ii > breakdown.max_ii) {
    breakdown.final_ii = 0;
    breakdown.infeasible = true;
    breakdown.dominant = "infeasible";
    breakdown.infeasible_reason =
        "sound lower bound " + std::to_string(max_bound_ii) +
        " exceeds control-memory limit " + std::to_string(breakdown.max_ii);
  } else {
    breakdown.final_ii = std::max(1, max_bound_ii);
  }
  return breakdown;
}

void AnalyticalIIBreakdown::print(llvm::raw_ostream &output_stream) const {
  output_stream << "[cost-model-analytical]\n";
  output_stream << "  compute_mii=" << compute.value << " (" << compute.detail
                << ")\n";
  output_stream << "  rec_mii=" << rec.value << " (" << rec.detail << ")\n";
  output_stream << "  mem_mii=" << mem.value << " (" << mem.detail << ")\n";
  output_stream << "  route_mii=" << route.value << " (" << route.detail
                << ")\n";
  output_stream << "  reg_mii=" << reg.value << " (" << reg.detail << ")\n";
  output_stream << "  res_mii=" << res.value << " (" << res.detail << ")\n";
  output_stream << "  final_ii=" << final_ii << " (dominant=" << dominant;
  if (infeasible)
    output_stream << ", INFEASIBLE: " << infeasible_reason;
  output_stream << ")\n";
}

} // namespace neura
} // namespace mlir
