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
  if (isa<func::FuncOp, ModuleOp>(op))
    return false;
  if (isa<neura::KernelOp>(op))
    return false;
  if (is_non_materialized(op)) // reserve / ctrl_mov / data_mov / yield
    return false;
  if (Operation *parent = op->getParentOp())
    if (isa<neura::FusedOp>(parent))
      return false;
  return true;
}

// Builds the inverse of kFuTypesToOperations: OperationKind -> FU class name.
const std::map<OperationKind, std::string> &kindToFuClass() {
  static const std::map<OperationKind, std::string> table = [] {
    std::map<OperationKind, std::string> m;
    for (const auto &[fu_name, ops] : kFuTypesToOperations)
      for (OperationKind k : ops)
        m[k] = fu_name;
    return m;
  }();
  return table;
}

// FU class name of an op (handles fused ops via their pattern_name attribute).
std::string fuClassOf(Operation *op) {
  if (isa<neura::FusedOp>(op)) {
    if (auto name = op->getAttrOfType<StringAttr>("pattern_name"))
      return name.getValue().str();
    return "fused";
  }
  OperationKind k = getOperationKindFromMlirOp(op);
  auto it = kindToFuClass().find(k);
  return it == kindToFuClass().end() ? "other" : it->second;
}

// Number of tiles that physically provide a given FU class.
int tilesSupportingClass(const Architecture &arch, const std::string &fu_class) {
  auto it = kFuTypesToOperations.find(fu_class);
  if (it == kFuTypesToOperations.end() || it->second.empty())
    return arch.getNumTiles(); // unknown class: don't over-constrain.
  OperationKind probe = it->second.front();
  int count = 0;
  for (Tile *t : arch.getAllTiles())
    if (t->canSupportOperation(probe))
      ++count;
  return count;
}

// Bit width carried by an SSA value (for routing channel demand).
int valueBits(Value v) {
  Type t = v.getType();
  if (t.isIntOrFloat())
    return static_cast<int>(t.getIntOrFloatBitWidth());
  return 32; // predicated / opaque types: conservative single-channel default.
}

int ceilDiv(long long a, long long b) {
  if (b <= 0)
    b = 1;
  return static_cast<int>((a + b - 1) / b);
}

} // namespace

namespace mlir {
namespace neura {

//===----------------------------------------------------------------------===//
// ResMII — per-FU-class, latency-weighted.
//===----------------------------------------------------------------------===//
IIBound calculateResMiiPerClass(Region &region, const Architecture &arch) {
  std::map<std::string, long long> work; // class -> sum of op latencies.
  region.walk([&](Operation *op) {
    if (!isMaterializedForCost(op))
      return;
    work[fuClassOf(op)] += std::max(1, getOpLatency(op));
  });

  IIBound b;
  b.value = 1;
  for (const auto &[fu_class, w] : work) {
    int cap = std::max(1, tilesSupportingClass(arch, fu_class));
    int v = ceilDiv(w, cap);
    if (v > b.value) {
      b.value = v;
      b.demand = w;
      b.capacity = cap;
      b.detail = "class=" + fu_class + " work=" + std::to_string(w) +
                 " fus=" + std::to_string(cap);
    }
  }
  if (b.detail.empty())
    b.detail = "no materialized ops";
  return b;
}

//===----------------------------------------------------------------------===//
// RecMII — loop-carried recurrence latency / distance.
//===----------------------------------------------------------------------===//
IIBound calculateRecMiiWeighted(Region &region, const Architecture &arch) {
  (void)arch;
  IIBound b;
  b.value = 1;
  b.detail = "no recurrence";
  auto cycles = collectRecurrenceCycles(region);
  int idx = 0;
  for (auto &cycle : cycles) {
    long long lat = 0;
    int nmat = 0;
    for (Operation *op : cycle.operations) {
      if (is_non_materialized(op))
        continue;
      lat += std::max(1, getOpLatency(op));
      ++nmat;
    }
    // Distance is 1 for a single reserve/ctrl_mov back-edge pair (this IR).
    const long long distance = 1;
    int v = ceilDiv(lat, distance);
    if (v > b.value) {
      b.value = v;
      b.demand = lat;
      b.capacity = distance;
      b.detail = "cycle#" + std::to_string(idx) + " ops=" +
                 std::to_string(nmat) + " latency=" + std::to_string(lat) +
                 " dist=" + std::to_string(distance);
    }
    ++idx;
  }
  return b;
}

//===----------------------------------------------------------------------===//
// MemMII — load/store contention on memory FUs.
//===----------------------------------------------------------------------===//
IIBound calculateMemMii(Region &region, const Architecture &arch) {
  long long mem_ops = 0, indexed_ops = 0;
  region.walk([&](Operation *op) {
    if (!isMaterializedForCost(op))
      return;
    OperationKind k = getOperationKindFromMlirOp(op);
    if (k == ILoad || k == IStore)
      ++mem_ops;
    else if (k == ILoadIndexed || k == IStoreIndexed)
      ++indexed_ops;
  });

  int mem_tiles = std::max(1, tilesSupportingClass(arch, "mem"));
  int idx_tiles = std::max(1, tilesSupportingClass(arch, "mem_indexed"));
  int v_mem = ceilDiv(mem_ops, mem_tiles);
  int v_idx = ceilDiv(indexed_ops, idx_tiles);

  IIBound b;
  if (v_idx >= v_mem) {
    b.value = std::max(1, v_idx);
    b.demand = indexed_ops;
    b.capacity = idx_tiles;
    b.detail = "indexed=" + std::to_string(indexed_ops) + " mem_indexed_fus=" +
               std::to_string(idx_tiles);
  } else {
    b.value = std::max(1, v_mem);
    b.demand = mem_ops;
    b.capacity = mem_tiles;
    b.detail = "loads+stores=" + std::to_string(mem_ops) + " mem_fus=" +
               std::to_string(mem_tiles);
  }
  return b;
}

//===----------------------------------------------------------------------===//
// RouteMII — routed move demand vs total link capacity.
//===----------------------------------------------------------------------===//
IIBound calculateRouteMii(Region &region, const Architecture &arch) {
  // Each neura.data_mov is one physical move; fanout is already expanded into
  // one data_mov per real consumer, so replicated traffic is counted here.
  long long channel_demand = 0;
  long long moves = 0;
  region.walk([&](neura::DataMovOp mov) {
    ++moves;
    int bw = 32;
    for (Link *l : arch.getAllLinks()) {
      bw = l->getBandwidth();
      break; // links are homogeneous by default; use the common bandwidth.
    }
    if (bw <= 0)
      bw = 32;
    channel_demand += ceilDiv(valueBits(mov.getResult()), bw);
  });

  long long links = static_cast<long long>(arch.getAllLinks().size());
  if (links <= 0)
    links = 1;

  IIBound b;
  b.value = std::max(1, ceilDiv(channel_demand, links));
  b.demand = channel_demand;
  b.capacity = links;
  b.detail = "moves=" + std::to_string(moves) + " channel_demand=" +
             std::to_string(channel_demand) + " links=" +
             std::to_string(links);
  return b;
}

//===----------------------------------------------------------------------===//
// RegMII — peak simultaneously-live values vs total register capacity.
//===----------------------------------------------------------------------===//
IIBound calculateRegMii(Region &region, const Architecture &arch) {
  // ASAP levels over the materialized DAG (recurrence is broken by reserve).
  // ASAP levels over the whole op graph (data_mov/reserve are their own nodes).
  // Uses the direct SSA producer — getTopologicallySortedOps guarantees
  // producers precede consumers, and the reserve op (no operands) breaks the
  // loop-carried back-edge so this is a finite DAG traversal.
  std::vector<Operation *> topo = getTopologicallySortedOps(region);
  llvm::DenseMap<Operation *, int> level;
  for (Operation *op : topo) {
    int lv = 0;
    for (Value operand : op->getOperands()) {
      Operation *prod = operand.getDefiningOp();
      if (prod && level.count(prod))
        lv = std::max(lv, level[prod] + 1);
    }
    level[op] = lv;
  }

  // Live range of each materialized value: [def_level, last_materialized_use].
  int max_level = 0;
  for (auto &kv : level)
    max_level = std::max(max_level, kv.second);
  std::vector<long long> delta(max_level + 2, 0);

  long long values_with_range = 0;
  region.walk([&](Operation *op) {
    if (!isMaterializedForCost(op) || op->getNumResults() == 0)
      return;
    if (!level.count(op))
      return;
    int def_lv = level[op];
    int last_use = def_lv;
    for (Value res : op->getResults()) {
      for (Operation *user : res.getUsers()) {
        // Unwrap routing users (data_mov) to the materialized consumer.
        Operation *muser = user;
        if (is_non_materialized(user)) {
          for (Operation *uu : user->getUsers()) {
            if (level.count(uu))
              muser = uu;
          }
        }
        if (level.count(muser))
          last_use = std::max(last_use, level[muser]);
      }
    }
    if (last_use > def_lv) {
      delta[def_lv] += 1;
      delta[last_use] -= 1;
      ++values_with_range;
    }
  });

  long long peak = 0, running = 0;
  int peak_level = 0;
  for (int i = 0; i <= max_level; ++i) {
    running += delta[i];
    if (running > peak) {
      peak = running;
      peak_level = i;
    }
  }

  long long total_regs = 0;
  for (Tile *t : arch.getAllTiles())
    total_regs += static_cast<long long>(t->getRegisters().size());
  if (total_regs <= 0)
    total_regs = 1;

  IIBound b;
  b.value = std::max(1, ceilDiv(peak, total_regs));
  b.demand = peak;
  b.capacity = total_regs;
  b.detail = "peak_live=" + std::to_string(peak) + " @level=" +
             std::to_string(peak_level) + " regs=" +
             std::to_string(total_regs);
  return b;
}

//===----------------------------------------------------------------------===//
// IssueMII — tile issue-slot occupancy (crude ceil(#ops / #tiles) baseline).
//===----------------------------------------------------------------------===//
IIBound calculateIssueMii(Region &region, const Architecture &arch) {
  long long ops = 0;
  region.walk([&](Operation *op) {
    if (isMaterializedForCost(op))
      ++ops;
  });
  int tiles = std::max(1, arch.getNumTiles());
  IIBound b;
  b.value = std::max(1, ceilDiv(ops, tiles));
  b.demand = ops;
  b.capacity = tiles;
  b.detail = "ops=" + std::to_string(ops) + " tiles=" + std::to_string(tiles);
  return b;
}

//===----------------------------------------------------------------------===//
// Combine.
//===----------------------------------------------------------------------===//
AnalyticalIIBreakdown computeAnalyticalII(Region &region,
                                          const Architecture &arch) {
  AnalyticalIIBreakdown out;
  out.res = calculateResMiiPerClass(region, arch);
  out.rec = calculateRecMiiWeighted(region, arch);
  out.mem = calculateMemMii(region, arch);
  out.route = calculateRouteMii(region, arch);
  out.reg = calculateRegMii(region, arch);
  out.issue = calculateIssueMii(region, arch);

  struct Named {
    const char *name;
    int value;
  };
  Named bounds[] = {{"res", out.res.value},     {"rec", out.rec.value},
                    {"mem", out.mem.value},     {"route", out.route.value},
                    {"reg", out.reg.value},     {"issue", out.issue.value}};
  int raw = 1;
  const char *dom = "res";
  for (const Named &n : bounds)
    if (n.value > raw) {
      raw = n.value;
      dom = n.name;
    }
  out.dominant = dom;

  out.max_ii = arch.getMaxCtrlMemItems();
  if (out.max_ii > 0 && raw > out.max_ii) {
    out.final_ii = out.max_ii;
    out.clamped = true;
  } else {
    out.final_ii = std::max(1, raw);
  }
  return out;
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
  if (clamped)
    os << ", clamped-to-max_ii=" << max_ii;
  os << ")\n";
}

} // namespace neura
} // namespace mlir
