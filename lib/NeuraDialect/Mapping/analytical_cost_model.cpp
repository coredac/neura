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
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::neura;

namespace {

// An op occupies a functional unit / tile (and thus contributes to resource
// demand) unless it is a pure routing / structural op or lives inside a fused
// region (the fused op itself is what gets mapped).
bool isMaterializedForCost(Operation *op) {
  if (isa<func::FuncOp, ModuleOp>(op)) {
    return false;
  }
  if (isa<neura::KernelOp>(op)) {
    return false;
  }
  if (is_non_materialized(op)) { // reserve / ctrl_mov / data_mov / yield
    return false;
  }
  if (Operation *parent = op->getParentOp()) {
    if (isa<neura::FusedOp>(parent)) {
      return false;
    }
  }
  return true;
}

// Builds the inverse of kFuTypesToOperations: OperationKind -> FU class name.
const std::map<OperationKind, std::string> &kindToFuClassTable() {
  static const std::map<OperationKind, std::string> table = [] {
    std::map<OperationKind, std::string> inverse;
    for (const auto &[fu_class_name, kinds] : kFuTypesToOperations) {
      for (OperationKind kind : kinds) {
        inverse[kind] = fu_class_name;
      }
    }
    return inverse;
  }();
  return table;
}

// FU class name of an op (handles fused ops via their pattern_name attribute).
std::string fuClassOf(Operation *op) {
  if (isa<neura::FusedOp>(op)) {
    if (auto pattern_name = op->getAttrOfType<StringAttr>("pattern_name")) {
      return pattern_name.getValue().str();
    }
    return "fused";
  }
  OperationKind kind = getOperationKindFromMlirOp(op);
  auto found = kindToFuClassTable().find(kind);
  return found == kindToFuClassTable().end() ? "other" : found->second;
}

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
//===----------------------------------------------------------------------===//
IIBound calculateResMiiPerClass(Region &region, const Architecture &arch) {
  std::map<std::string, long long> work_by_class; // class -> sum of latencies.
  region.walk([&](Operation *op) {
    if (!isMaterializedForCost(op)) {
      return;
    }
    work_by_class[fuClassOf(op)] += std::max(1, getOpLatency(op));
  });

  IIBound bound;
  bound.value = 1;
  for (const auto &[fu_class, class_work] : work_by_class) {
    int fu_count = std::max(1, tilesSupportingClass(arch, fu_class));
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
  if (bound.detail.empty()) {
    bound.detail = "no materialized ops";
  }
  return bound;
}

//===----------------------------------------------------------------------===//
// RecMII — loop-carried recurrence latency / distance.
//===----------------------------------------------------------------------===//
IIBound calculateRecMiiWeighted(Region &region, const Architecture &arch) {
  (void)arch;
  IIBound bound;
  bound.value = 1;
  bound.detail = "no recurrence";
  auto recurrence_cycles = collectRecurrenceCycles(region);
  int cycle_index = 0;
  for (auto &cycle : recurrence_cycles) {
    long long cycle_latency = 0;
    int num_materialized = 0;
    for (Operation *op : cycle.operations) {
      if (is_non_materialized(op)) {
        continue;
      }
      cycle_latency += std::max(1, getOpLatency(op));
      ++num_materialized;
    }
    // Distance is 1 for a single reserve/ctrl_mov back-edge pair (this IR).
    const long long distance = 1;
    int cycle_ii = ceilDiv(cycle_latency, distance);
    if (cycle_ii > bound.value) {
      bound.value = cycle_ii;
      bound.demand = cycle_latency;
      bound.capacity = distance;
      bound.detail = "cycle#" + std::to_string(cycle_index) +
                     " ops=" + std::to_string(num_materialized) +
                     " latency=" + std::to_string(cycle_latency) +
                     " dist=" + std::to_string(distance);
    }
    ++cycle_index;
  }
  return bound;
}

//===----------------------------------------------------------------------===//
// MemMII — load/store contention on memory FUs.
//===----------------------------------------------------------------------===//
IIBound calculateMemMii(Region &region, const Architecture &arch) {
  long long num_mem_ops = 0, num_indexed_ops = 0;
  region.walk([&](Operation *op) {
    if (!isMaterializedForCost(op)) {
      return;
    }
    OperationKind kind = getOperationKindFromMlirOp(op);
    if (kind == ILoad || kind == IStore) {
      ++num_mem_ops;
    } else if (kind == ILoadIndexed || kind == IStoreIndexed) {
      ++num_indexed_ops;
    }
  });

  int mem_tiles = std::max(1, tilesSupportingClass(arch, "mem"));
  int indexed_tiles = std::max(1, tilesSupportingClass(arch, "mem_indexed"));
  int mem_ii = ceilDiv(num_mem_ops, mem_tiles);
  int indexed_ii = ceilDiv(num_indexed_ops, indexed_tiles);

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
  return bound;
}

//===----------------------------------------------------------------------===//
// RouteMII — routed move demand vs total link capacity.
//===----------------------------------------------------------------------===//
IIBound calculateRouteMii(Region &region, const Architecture &arch) {
  // Each neura.data_mov is one physical move; fanout is already expanded into
  // one data_mov per real consumer, so replicated traffic is counted here.
  long long channel_demand = 0;
  long long num_moves = 0;
  region.walk([&](neura::DataMovOp move) {
    ++num_moves;
    int link_bandwidth = 32;
    for (Link *link : arch.getAllLinks()) {
      link_bandwidth = link->getBandwidth();
      break; // links are homogeneous by default; use the common bandwidth.
    }
    if (link_bandwidth <= 0) {
      link_bandwidth = 32;
    }
    channel_demand += ceilDiv(valueBitWidth(move.getResult()), link_bandwidth);
  });

  long long total_links = static_cast<long long>(arch.getAllLinks().size());
  if (total_links <= 0) {
    total_links = 1;
  }

  IIBound bound;
  bound.value = std::max(1, ceilDiv(channel_demand, total_links));
  bound.demand = channel_demand;
  bound.capacity = total_links;
  bound.detail = "moves=" + std::to_string(num_moves) +
                 " channel_demand=" + std::to_string(channel_demand) +
                 " links=" + std::to_string(total_links);
  return bound;
}

//===----------------------------------------------------------------------===//
// RegMII — peak simultaneously-live values vs total register capacity.
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

  // Live range of each materialized value: [def_level, last_materialized_use].
  // Accumulate +1 at each value's def level and -1 at its last-use level, then
  // prefix-sum to find the peak number of simultaneously-live values.
  int max_level = 0;
  for (auto &entry : asap_level) {
    max_level = std::max(max_level, entry.second);
  }
  std::vector<long long> live_delta(max_level + 2, 0);

  region.walk([&](Operation *op) {
    if (!isMaterializedForCost(op) || op->getNumResults() == 0) {
      return;
    }
    if (!asap_level.count(op)) {
      return;
    }
    int def_level = asap_level[op];
    int last_use_level = def_level;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        // Unwrap routing users (data_mov) to the materialized consumer.
        Operation *materialized_user = user;
        if (is_non_materialized(user)) {
          for (Operation *router_user : user->getUsers()) {
            if (asap_level.count(router_user)) {
              materialized_user = router_user;
            }
          }
        }
        if (asap_level.count(materialized_user)) {
          last_use_level =
              std::max(last_use_level, asap_level[materialized_user]);
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
  if (total_registers <= 0) {
    total_registers = 1;
  }

  IIBound bound;
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
//===----------------------------------------------------------------------===//
IIBound calculateIssueMii(Region &region, const Architecture &arch) {
  long long num_ops = 0;
  region.walk([&](Operation *op) {
    if (isMaterializedForCost(op)) {
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
  if (breakdown.max_ii > 0 && max_bound_ii > breakdown.max_ii) {
    breakdown.final_ii = breakdown.max_ii;
    breakdown.clamped = true;
  } else {
    breakdown.final_ii = std::max(1, max_bound_ii);
  }
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
  os << "  final_ii=" << final_ii << " (dominant=" << dominant;
  if (clamped) {
    os << ", clamped-to-max_ii=" << max_ii;
  }
  os << ")\n";
}

} // namespace neura
} // namespace mlir
