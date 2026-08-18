//===- HardwareTemplate.cpp - Hardware Template Data Structures and Helpers
//-===//
//
// This file contains data structures and helper functions for hardware template
// merging, including pattern extraction, template creation, and cost
// calculation.
//
// This version uses Functional Unit (FU) based design where each FU executes
// exactly one operation type, and templates can have multiple FUs of the same
// type.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/Transforms/GraphMining/HardwareTemplate.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace mlir;

namespace mlir::neura {

// Initializes the operation cost model with default costs for each operation
// type.
// TODO: The cost model should be imported from the input file.
OperationCostModel::OperationCostModel() {
  costs["neura.div"] = 100;
  costs["neura.fdiv"] = 100;
  costs["neura.rem"] = 80;
  costs["neura.load"] = 50;
  costs["neura.store"] = 50;
  costs["neura.load_indexed"] = 55;
  costs["neura.store_indexed"] = 55;
  costs["neura.mul"] = 30;
  costs["neura.fmul"] = 35;
  costs["neura.gep"] = 20;
  costs["neura.add"] = 10;
  costs["neura.sub"] = 10;
  costs["neura.fadd"] = 12;
  costs["neura.fsub"] = 12;
  costs["neura.icmp"] = 15;
  costs["neura.fcmp"] = 15;
  costs["neura.and"] = 5;
  costs["neura.or"] = 5;
  costs["neura.not"] = 5;
  costs["neura.sel"] = 6;
  costs["neura.phi"] = 3;
  costs["neura.grant_predicate"] = 3;
  costs["neura.grant_once"] = 3;
  costs["neura.cast"] = 2;
  costs["neura.sext"] = 2;
  costs["neura.zext"] = 2;
  costs["neura.data_mov"] = 1;
  costs["neura.constant"] = 1;
}

// Returns the cost for a given operation, or 5.0 as default if not found.
double OperationCostModel::get(const std::string &op) const {
  auto it = costs.find(op);
  return it != costs.end() ? it->second : 5.0;
}

// Returns the cost for a single FU (same as get).
double OperationCostModel::fu_cost(const std::string &op) const {
  return get(op);
}

// Returns the total cost for a pattern by summing costs of all its operations.
double
OperationCostModel::pattern_cost(const std::vector<std::string> &ops) const {
  double sum = 0;
  for (const auto &op : ops)
    sum += get(op);
  return sum;
}

// Constructs a HardwarePattern with the given id, name, and frequency.
HardwarePattern::HardwarePattern(int64_t i, const std::string &n, int64_t f)
    : id(i), name(n), freq(f), cost(0) {}

// Constructs a FunctionalUnit with the given id and operation type.
FunctionalUnit::FunctionalUnit(int i, const std::string &op)
    : id(i), op_type(op) {}

// Constructs a HardwareTemplate with the given id and one instance.
HardwareTemplate::HardwareTemplate(int i) : id(i), instances(1) {}

// Adds a new FU with the given operation type, returns its ID.
int HardwareTemplate::add_fu(const std::string &op_type) {
  int new_id = fus.size();
  fus.emplace_back(new_id, op_type);
  return new_id;
}

// Finds an existing FU that can handle the operation and is not already used.
int HardwareTemplate::find_available_fu(const std::string &op_type,
                                        const std::set<int> &used_fus) const {
  for (const auto &fu : fus) {
    if (fu.op_type == op_type && used_fus.find(fu.id) == used_fus.end()) {
      return fu.id;
    }
  }
  return -1; // No available FU found
}

// Checks if two operations are compatible (could share hardware in extended
// designs).
bool HardwareTemplate::compatible(const std::string &a, const std::string &b) {
  if (a == b)
    return true;

  auto in_group = [](const std::string &op,
                     const std::vector<std::string> &group) -> bool {
    for (const auto &keyword : group) {
      if (op.find(keyword) != std::string::npos)
        return true;
    }
    return false;
  };

  static const std::vector<std::vector<std::string>> compatible_groups = {
      {"add", "sub"},
      {"grant_once", "grant_predicate", "grant_always"},
      {"and", "or", "not", "xor"},
      {"load", "store"}};

  for (const auto &group : compatible_groups) {
    if (in_group(a, group) && in_group(b, group)) {
      return true;
    }
  }

  return false;
}

// DFS helper for finding the best mapping with maximum FU reuse.
void HardwareTemplate::dfs_find_mapping(const HardwarePattern &pat,
                                        size_t op_idx,
                                        std::vector<int> &cur_mapping,
                                        std::set<int> &used_fus,
                                        std::vector<int> &best_mapping,
                                        int &best_reuse_count) const {
  if (op_idx >= pat.ops.size()) {
    // Count reused FUs (existing FUs that were mapped)
    int reuse_count = 0;
    for (int fu_id : cur_mapping) {
      if (fu_id >= 0 && fu_id < (int)fus.size()) {
        reuse_count++;
      }
    }
    if (reuse_count > best_reuse_count) {
      best_reuse_count = reuse_count;
      best_mapping = cur_mapping;
    }
    return;
  }

  const std::string &op = pat.ops[op_idx];

  // Try to find an existing FU that matches this operation
  for (const auto &fu : fus) {
    if (fu.op_type == op && used_fus.find(fu.id) == used_fus.end()) {
      // Check dependency constraints: all predecessors must be mapped to FUs
      // that can connect to this FU
      bool valid = true;
      if (op_idx < pat.op_preds.size()) {
        for (int pred_idx : pat.op_preds[op_idx]) {
          if (pred_idx >= 0 && pred_idx < (int)cur_mapping.size()) {
            // Predecessor is mapped, connection will be established later
            // For now, we just need to ensure the mapping is valid
          }
        }
      }

      if (valid) {
        cur_mapping.push_back(fu.id);
        used_fus.insert(fu.id);
        dfs_find_mapping(pat, op_idx + 1, cur_mapping, used_fus, best_mapping,
                         best_reuse_count);
        used_fus.erase(fu.id);
        cur_mapping.pop_back();
      }
    }
  }

  // Also try creating a "virtual" new FU (represented by -1 - op_idx to
  // distinguish) This indicates we'll need to add a new FU for this operation
  cur_mapping.push_back(-1 -
                        (int)op_idx); // Negative value indicates new FU needed
  dfs_find_mapping(pat, op_idx + 1, cur_mapping, used_fus, best_mapping,
                   best_reuse_count);
  cur_mapping.pop_back();
}

// Finds a mapping for a pattern into the existing template.
std::vector<int>
HardwareTemplate::find_mapping(const HardwarePattern &pat) const {
  std::vector<int> best_mapping;
  int best_reuse = -1;
  std::vector<int> cur_mapping;
  std::set<int> used_fus;

  dfs_find_mapping(pat, 0, cur_mapping, used_fus, best_mapping, best_reuse);

  return best_mapping;
}

// Tries to accommodate a pattern into the existing template.
bool HardwareTemplate::try_accommodate(const HardwarePattern &pat,
                                       const OperationCostModel &cm,
                                       std::vector<int> &out_mapping,
                                       double &out_cost_increase) {
  auto mapping = find_mapping(pat);
  if (mapping.empty() && !pat.ops.empty()) {
    return false;
  }

  // Calculate cost increase: count how many new FUs we need to add
  double old_cost = compute_cost(cm);

  // Convert negative indices to actual new FU IDs
  std::vector<int> final_mapping;
  int next_fu_id = fus.size();
  for (size_t i = 0; i < mapping.size(); ++i) {
    if (mapping[i] < 0) {
      final_mapping.push_back(next_fu_id++);
    } else {
      final_mapping.push_back(mapping[i]);
    }
  }

  // Calculate new cost
  double new_cost = old_cost;
  for (size_t i = 0; i < pat.ops.size(); ++i) {
    if (mapping[i] < 0) {
      new_cost += cm.fu_cost(pat.ops[i]);
    }
  }

  out_mapping = final_mapping;
  out_cost_increase = new_cost - old_cost;
  return true;
}

// Applies a mapping to the template, adding new FUs as needed.
void HardwareTemplate::apply_mapping(const HardwarePattern &pat,
                                     const std::vector<int> &m) {
  patterns.push_back(pat.id);
  mapping[pat.id] = m;

  // Add any new FUs that don't exist yet
  for (size_t i = 0; i < m.size() && i < pat.ops.size(); ++i) {
    int fu_id = m[i];
    // If this FU ID doesn't exist, we need to add FUs until it does
    while (fu_id >= (int)fus.size()) {
      // Find the operation that should go in this new FU
      for (size_t j = 0; j < m.size(); ++j) {
        if (m[j] == (int)fus.size()) {
          add_fu(pat.ops[j]);
          break;
        }
      }
      // Safety: if no matching op found, break to avoid infinite loop
      if (fu_id >= (int)fus.size() && fus.size() > 0) {
        // This shouldn't happen with correct mappings
        break;
      }
    }
  }
}

// Computes the total cost of the template based on all FUs.
double HardwareTemplate::compute_cost(const OperationCostModel &cm) const {
  double sum = 0;
  for (const auto &fu : fus) {
    sum += cm.fu_cost(fu.op_type);
  }
  return sum;
}

// Extracts operations from fused op and linearizes them via topological sort.
void extract_pattern_ops(neura::FusedOp fop, HardwarePattern &pat) {
  Region &body = fop.getBody();
  if (body.empty())
    return;
  Block &blk = body.front();

  llvm::DenseMap<Operation *, int> op_id;
  std::vector<std::string> op_names;
  std::vector<std::vector<int>> preds;

  int id = 0;
  for (Operation &op : blk.getOperations()) {
    std::string name = op.getName().getStringRef().str();
    if (name == "neura.yield")
      continue;
    op_id[&op] = id++;
    op_names.push_back(name);
    preds.push_back({});
  }

  int idx = 0;
  for (Operation &op : blk.getOperations()) {
    std::string name = op.getName().getStringRef().str();
    if (name == "neura.yield")
      continue;
    for (Value v : op.getOperands()) {
      if (Operation *def = v.getDefiningOp()) {
        if (op_id.count(def)) {
          preds[idx].push_back(op_id[def]);
        }
      }
    }
    idx++;
  }

  int n = op_names.size();
  std::vector<int> level(n, 0);
  std::vector<int> in_deg(n, 0);
  for (int i = 0; i < n; ++i)
    in_deg[i] = preds[i].size();

  std::vector<int> q;
  for (int i = 0; i < n; ++i)
    if (in_deg[i] == 0)
      q.push_back(i);

  for (size_t h = 0; h < q.size(); ++h) {
    int cur = q[h];
    for (int i = 0; i < n; ++i) {
      for (int p : preds[i]) {
        if (p == cur) {
          level[i] = std::max(level[i], level[cur] + 1);
          if (--in_deg[i] == 0)
            q.push_back(i);
        }
      }
    }
  }

  std::vector<int> order;
  for (int i = 0; i < n; ++i)
    order.push_back(i);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return level[a] < level[b]; });

  // Build mapping from old index to new index after reordering.
  std::vector<int> old_to_new(n);
  for (int new_idx = 0; new_idx < n; ++new_idx) {
    old_to_new[order[new_idx]] = new_idx;
  }

  for (int i : order) {
    pat.ops.push_back(op_names[i]);
    pat.op_levels.push_back(level[i]);

    std::vector<int> remapped_preds;
    for (int p : preds[i]) {
      remapped_preds.push_back(old_to_new[p]);
    }
    pat.op_preds.push_back(remapped_preds);
  }
}

// Extracts all patterns from module.
void extract_patterns(ModuleOp module, std::vector<HardwarePattern> &patterns,
                      OperationCostModel &cost_model) {
  module.walk([&](neura::FusedOp fop) {
    int64_t pid = fop.getPatternId();
    if (std::find_if(patterns.begin(), patterns.end(),
                     [pid](const HardwarePattern &p) { return p.id == pid; }) !=
        patterns.end())
      return;

    HardwarePattern pat(pid, fop.getPatternName().str(), fop.getFrequency());
    extract_pattern_ops(fop, pat);
    pat.cost = cost_model.pattern_cost(pat.ops);
    patterns.push_back(pat);
  });
}

// Extracts all standalone operations from module.
void extract_all_standalone_ops(ModuleOp module,
                                std::set<std::string> &all_ops) {
  module.walk([&](Operation *op) {
    if (isa<neura::FusedOp>(op))
      return;

    Operation *parent = op->getParentOp();
    while (parent) {
      if (isa<neura::FusedOp>(parent))
        return;
      parent = parent->getParentOp();
    }

    std::string op_name = op->getName().getStringRef().str();
    if (op_name.find("neura.") == 0) {
      if (op_name == "neura.yield" || op_name == "neura.fused_op")
        return;
      all_ops.insert(op_name);
    }
  });
}

// Creates hardware templates from patterns.
void create_hardware_templates(const std::vector<HardwarePattern> &patterns,
                               std::vector<HardwareTemplate> &templates,
                               OperationCostModel &cost_model) {
  auto count_distinct_ops = [](const HardwarePattern &p) -> int {
    std::set<std::string> distinct_ops;
    for (const auto &op : p.ops) {
      distinct_ops.insert(op);
    }
    return distinct_ops.size();
  };

  std::vector<int> order;
  for (size_t i = 0; i < patterns.size(); ++i)
    order.push_back(i);
  std::sort(order.begin(), order.end(),
            [&patterns, &count_distinct_ops](int a, int b) {
              int dist_a = count_distinct_ops(patterns[a]);
              int dist_b = count_distinct_ops(patterns[b]);
              if (dist_a != dist_b) {
                return dist_a > dist_b;
              }
              return patterns[a].cost > patterns[b].cost;
            });

  for (int idx : order) {
    const HardwarePattern &pat = patterns[idx];

    int best_t = -1;
    std::vector<int> best_m;
    double best_inc = 1e18;

    for (size_t t = 0; t < templates.size(); ++t) {
      HardwareTemplate temp_tmpl = templates[t];
      std::vector<int> m;
      double inc;
      if (temp_tmpl.try_accommodate(pat, cost_model, m, inc)) {
        if (inc < best_inc) {
          best_inc = inc;
          best_t = t;
          best_m = m;
        }
      }
    }

    double new_tmpl_cost = pat.cost;

    if (best_t >= 0 && best_inc <= new_tmpl_cost * 0.5) {
      templates[best_t].apply_mapping(pat, best_m);
    } else {
      // Create a new template with FUs for this pattern
      HardwareTemplate t(templates.size());
      std::vector<int> m;
      for (size_t i = 0; i < pat.ops.size(); ++i) {
        int fu_id = t.add_fu(pat.ops[i]);
        m.push_back(fu_id);
      }
      t.apply_mapping(pat, m);
      templates.push_back(t);
    }
  }

  for (auto &t : templates) {
    int64_t total_freq = 0;
    for (int64_t pid : t.patterns) {
      for (const auto &p : patterns) {
        if (p.id == pid) {
          total_freq += p.freq;
          break;
        }
      }
    }
    t.instances = std::max(1, (int)std::ceil(total_freq / 10.0));
  }
}

// Generates FU connections for all templates based on pattern mappings.
// For each pattern, creates connections based on the dependency graph
// (op_preds).
void generate_connections(const std::vector<HardwarePattern> &patterns,
                          std::vector<HardwareTemplate> &templates) {
  std::map<int64_t, const HardwarePattern *> pattern_map;
  for (const auto &p : patterns) {
    pattern_map[p.id] = &p;
  }

  for (auto &tmpl : templates) {
    tmpl.connections.clear();

    // For each pattern mapped to this template, generate connections based on
    // dependencies.
    for (const auto &[pid, fu_mapping] : tmpl.mapping) {
      auto pat_it = pattern_map.find(pid);
      if (pat_it == pattern_map.end())
        continue;
      const HardwarePattern *pat = pat_it->second;

      if (fu_mapping.size() < 2)
        continue;

      // Use op_preds to determine actual data dependencies
      if (!pat->op_preds.empty()) {
        for (size_t op_idx = 0;
             op_idx < pat->op_preds.size() && op_idx < fu_mapping.size();
             ++op_idx) {
          int to_fu = fu_mapping[op_idx];

          for (int pred_op_idx : pat->op_preds[op_idx]) {
            if (pred_op_idx >= 0 && pred_op_idx < (int)fu_mapping.size()) {
              int from_fu = fu_mapping[pred_op_idx];
              if (from_fu >= 0 && to_fu >= 0 && from_fu != to_fu) {
                tmpl.connections.insert({from_fu, to_fu});
              }
            }
          }
        }
      } else {
        // Fallback: create linear chain if no dependency info
        for (size_t i = 0; i < fu_mapping.size() - 1; ++i) {
          int from = fu_mapping[i];
          int to = fu_mapping[i + 1];
          if (from >= 0 && to >= 0 && from != to) {
            tmpl.connections.insert({from, to});
          }
        }
      }
    }
  }
}

// Checks if FU 'from' can reach FU 'to' through existing connections.
static bool
can_reach_via_connections(const std::set<std::pair<int, int>> &connections,
                          int from, int to, int num_fus) {
  if (from == to)
    return true;

  std::vector<bool> visited(num_fus, false);
  std::vector<int> queue;
  queue.push_back(from);
  visited[from] = true;

  for (size_t h = 0; h < queue.size(); ++h) {
    int cur = queue[h];
    for (const auto &conn : connections) {
      if (conn.first == cur && !visited[conn.second]) {
        if (conn.second == to)
          return true;
        visited[conn.second] = true;
        queue.push_back(conn.second);
      }
    }
  }
  return false;
}

// Generates optimized FU connections for all templates based on pattern
// dependencies. Removes redundant connections using transitive reachability.
void generate_optimized_connections(
    const std::vector<HardwarePattern> &patterns,
    std::vector<HardwareTemplate> &templates) {
  std::map<int64_t, const HardwarePattern *> pattern_map;
  for (const auto &p : patterns) {
    pattern_map[p.id] = &p;
  }

  for (auto &tmpl : templates) {
    std::set<std::pair<int, int>> required_connections;

    for (const auto &[pid, fu_mapping] : tmpl.mapping) {
      auto pat_it = pattern_map.find(pid);
      if (pat_it == pattern_map.end())
        continue;
      const HardwarePattern *pat = pat_it->second;

      if (fu_mapping.size() < 2)
        continue;

      if (!pat->op_preds.empty()) {
        for (size_t op_idx = 0;
             op_idx < pat->op_preds.size() && op_idx < fu_mapping.size();
             ++op_idx) {
          int to_fu = fu_mapping[op_idx];

          for (int pred_op_idx : pat->op_preds[op_idx]) {
            if (pred_op_idx >= 0 && pred_op_idx < (int)fu_mapping.size()) {
              int from_fu = fu_mapping[pred_op_idx];
              if (from_fu >= 0 && to_fu >= 0 && from_fu != to_fu) {
                required_connections.insert({from_fu, to_fu});
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < fu_mapping.size() - 1; ++i) {
          int from = fu_mapping[i];
          int to = fu_mapping[i + 1];
          if (from >= 0 && to >= 0 && from != to) {
            required_connections.insert({from, to});
          }
        }
      }
    }

    // Sort connections by "distance" (prefer shorter connections first for
    // transitive reduction)
    std::vector<std::pair<int, int>> sorted_connections(
        required_connections.begin(), required_connections.end());
    std::sort(sorted_connections.begin(), sorted_connections.end(),
              [](const auto &a, const auto &b) {
                return std::abs(a.second - a.first) <
                       std::abs(b.second - b.first);
              });

    // Build optimized connections - add a connection only if it's not already
    // reachable.
    tmpl.connections.clear();
    int num_fus = tmpl.fus.size();

    for (const auto &conn : sorted_connections) {
      // Check if we can already reach conn.second from conn.first via existing
      // connections.
      if (!can_reach_via_connections(tmpl.connections, conn.first, conn.second,
                                     num_fus)) {
        tmpl.connections.insert(conn);
      }
    }
  }
}

// Generates execution plans for all patterns on their assigned templates.
void generate_execution_plans(const std::vector<HardwarePattern> &patterns,
                              const std::vector<HardwareTemplate> &templates,
                              std::vector<PatternExecutionPlan> &plans) {
  std::map<int64_t, const HardwareTemplate *> pattern_to_template;

  for (const auto &t : templates) {
    for (int64_t pid : t.patterns) {
      pattern_to_template[pid] = &t;
    }
  }

  for (const auto &pat : patterns) {
    PatternExecutionPlan plan;
    plan.pattern_id = pat.id;
    plan.pattern_name = pat.name;

    auto it = pattern_to_template.find(pat.id);
    if (it == pattern_to_template.end())
      continue;

    const HardwareTemplate *tmpl = it->second;
    auto mapping_it = tmpl->mapping.find(pat.id);
    if (mapping_it == tmpl->mapping.end())
      continue;

    const std::vector<int> &fu_mapping = mapping_it->second;

    // Group operations by their topological level for parallel execution
    std::map<int, std::vector<std::pair<int, std::pair<int, std::string>>>>
        level_to_ops;

    for (size_t i = 0; i < pat.ops.size() && i < fu_mapping.size(); ++i) {
      int level = (i < pat.op_levels.size()) ? pat.op_levels[i] : (int)i;
      int fu = fu_mapping[i];
      level_to_ops[level].push_back({(int)i, {fu, pat.ops[i]}});
    }

    for (const auto &[level, ops_at_level] : level_to_ops) {
      ExecutionStage stage;
      for (const auto &[op_idx, fu_and_op] : ops_at_level) {
        stage.fus.push_back(fu_and_op.first);
        stage.ops.push_back(fu_and_op.second);
      }
      plan.stages.push_back(stage);
    }

    plans.push_back(plan);
  }
}

// Collects supported operations for each template.
void collect_supported_operations(
    const std::vector<HardwarePattern> &patterns,
    const std::vector<HardwareTemplate> &templates,
    const std::set<std::string> &all_dfg_ops,
    std::vector<TemplateSupportedOps> &supported_ops) {
  for (const auto &tmpl : templates) {
    TemplateSupportedOps ops;
    ops.template_id = tmpl.id;

    // Collect all operation types present in this template's FUs
    std::set<std::string> template_ops;
    for (const auto &fu : tmpl.fus) {
      template_ops.insert(fu.op_type);
    }

    // For each DFG op, check if this template can support it
    for (const std::string &dfg_op : all_dfg_ops) {
      bool can_support = false;

      // Direct support: template has an FU for this op
      if (template_ops.count(dfg_op)) {
        can_support = true;
      } else {
        // Check if any existing FU type is compatible
        for (const auto &existing_op : template_ops) {
          if (HardwareTemplate::compatible(existing_op, dfg_op)) {
            can_support = true;
            break;
          }
        }
      }

      if (can_support) {
        ops.single_ops.insert(dfg_op);
      }
    }

    ops.composite_ops = tmpl.patterns;

    supported_ops.push_back(ops);
  }
}

// Calculates total cost of templates.
double calculate_total_cost(const std::vector<HardwareTemplate> &templates,
                            const OperationCostModel &cost_model) {
  double total_cost = 0;
  for (const auto &t : templates) {
    total_cost += t.compute_cost(cost_model) * t.instances;
  }
  return total_cost;
}

// Escapes string for JSON output.
std::string escape_json_string(const std::string &s) {
  std::string r;
  for (char c : s) {
    if (c == '"')
      r += "\\\"";
    else if (c == '\\')
      r += "\\\\";
    else
      r += c;
  }
  return r;
}

// Writes hardware configuration to JSON file (extended version with execution
// plans and supported ops).
void write_hardware_config_json(
    const std::string &path, const std::vector<HardwarePattern> &patterns,
    const std::vector<HardwareTemplate> &templates,
    const OperationCostModel &cost_model,
    const std::vector<PatternExecutionPlan> &execution_plans,
    const std::vector<TemplateSupportedOps> &supported_ops) {
  std::error_code EC;
  llvm::raw_fd_ostream os(path, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  // Build pattern name lookup.
  std::map<int64_t, std::string> pattern_names;
  for (const auto &p : patterns)
    pattern_names[p.id] = p.name;

  os << "{\n  \"hardware_configuration\": {\n";
  os << "    \"summary\": {\n";
  os << "      \"total_templates\": " << templates.size() << "\n";
  os << "    },\n";

  os << "    \"hardware_templates\": [\n";
  for (size_t t = 0; t < templates.size(); ++t) {
    const auto &tmpl = templates[t];
    if (t)
      os << ",\n";

    os << "      {\n";
    os << "        \"template_id\": " << tmpl.id << ",\n";
    os << "        \"instance_count\": " << tmpl.instances << ",\n";

    const TemplateSupportedOps *tmpl_supported_ops = nullptr;
    for (const auto &sop : supported_ops) {
      if (sop.template_id == tmpl.id) {
        tmpl_supported_ops = &sop;
        break;
      }
    }

    if (tmpl_supported_ops) {
      os << "        \"supported_single_ops\": [";
      bool first = true;
      for (const auto &op : tmpl_supported_ops->single_ops) {
        if (!first)
          os << ", ";
        first = false;
        os << "\"" << op << "\"";
      }
      os << "],\n";

      os << "        \"supported_composite_ops\": [\n";
      for (size_t i = 0; i < tmpl_supported_ops->composite_ops.size(); ++i) {
        if (i)
          os << ",\n";
        int64_t pid = tmpl_supported_ops->composite_ops[i];
        auto name_it = pattern_names.find(pid);
        std::string pname =
            (name_it != pattern_names.end()) ? name_it->second : "";
        os << "          {\"pattern_id\": " << pid << ", \"name\": \""
           << escape_json_string(pname) << "\"}";
      }
      os << "\n        ],\n";
    }

    // Output FUs (functional units) instead of slots
    os << "        \"functional_units\": [\n";
    for (size_t f = 0; f < tmpl.fus.size(); ++f) {
      const auto &fu = tmpl.fus[f];
      if (f)
        os << ",\n";
      os << "          {\"fu_id\": " << fu.id << ", \"op_type\": \""
         << fu.op_type << "\"}";
    }
    os << "\n        ],\n";

    // Output FU connections
    os << "        \"fu_connections\": [\n";
    bool first_conn = true;
    for (const auto &conn : tmpl.connections) {
      if (!first_conn)
        os << ",\n";
      first_conn = false;
      os << "          {\"from_fu\": " << conn.first
         << ", \"to_fu\": " << conn.second << "}";
    }
    os << "\n        ],\n";

    os << "        \"pattern_execution_plans\": [\n";
    bool first_plan = true;
    for (const auto &plan : execution_plans) {
      auto mapping_it = tmpl.mapping.find(plan.pattern_id);
      if (mapping_it == tmpl.mapping.end())
        continue;

      if (!first_plan)
        os << ",\n";
      first_plan = false;

      os << "          {\n";
      os << "            \"pattern_id\": " << plan.pattern_id << ",\n";
      os << "            \"pattern_name\": \""
         << escape_json_string(plan.pattern_name) << "\",\n";
      os << "            \"fu_mapping\": [";
      const auto &m = mapping_it->second;
      for (size_t i = 0; i < m.size(); ++i) {
        if (i)
          os << ", ";
        os << m[i];
      }
      os << "],\n";
      os << "            \"execution_stages\": [\n";
      for (size_t stage_idx = 0; stage_idx < plan.stages.size(); ++stage_idx) {
        const auto &stage = plan.stages[stage_idx];
        if (stage_idx)
          os << ",\n";
        os << "              {\n";
        os << "                \"stage\": " << stage_idx << ",\n";
        os << "                \"parallel_fus\": [";
        for (size_t i = 0; i < stage.fus.size(); ++i) {
          if (i)
            os << ", ";
          os << stage.fus[i];
        }
        os << "],\n";
        os << "                \"parallel_ops\": [";
        for (size_t i = 0; i < stage.ops.size(); ++i) {
          if (i)
            os << ", ";
          os << "\"" << stage.ops[i] << "\"";
        }
        os << "]\n";
        os << "              }";
      }
      os << "\n            ]\n";
      os << "          }";
    }
    os << "\n        ]\n";
    os << "      }";
  }
  os << "\n    ]\n";

  os << "  }\n}\n";
}

// Legacy version for backward compatibility.
void write_hardware_config_json(const std::string &path,
                                const std::vector<HardwarePattern> &patterns,
                                const std::vector<HardwareTemplate> &templates,
                                const OperationCostModel &cost_model) {
  std::vector<PatternExecutionPlan> plans;
  generate_execution_plans(patterns, templates, plans);

  std::set<std::string> all_dfg_ops;
  for (const auto &p : patterns) {
    for (const auto &op : p.ops) {
      all_dfg_ops.insert(op);
    }
  }

  std::vector<TemplateSupportedOps> supported_ops;
  collect_supported_operations(patterns, templates, all_dfg_ops, supported_ops);

  write_hardware_config_json(path, patterns, templates, cost_model, plans,
                             supported_ops);
}

} // namespace mlir::neura
