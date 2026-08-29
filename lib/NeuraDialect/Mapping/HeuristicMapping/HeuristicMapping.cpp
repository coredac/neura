#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace neura {
bool HeuristicMapping::map(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    std::set<Operation *> &critical_ops, const Architecture &architecture,
    MappingState &mapping_state) {
  // Starts the backtracking mapping process.
  return mapWithBacktrack(sorted_ops_with_levels, critical_ops, architecture,
                          mapping_state);
}

bool HeuristicMapping::mapWithBacktrack(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    std::set<Operation *> &critical_ops, const Architecture &architecture,
    MappingState &mapping_state) {
  llvm::outs() << "---------------------------------------------------------\n";
  llvm::outs() << "[HeuristicMapping] Starting mapping with "
               << sorted_ops_with_levels.size() << " operations.\n";
  llvm::outs() << "Configuration: MAX Backtrack Depth = "
               << this->max_backtrack_depth
               << ", MAX Candidate Locations = " << this->max_location_to_try
               << "\n";

  std::vector<std::pair<Operation *, int>> materialized_ops;
  for (auto [op, level] : sorted_ops_with_levels) {
    if (occupiesFU(op)) {
      materialized_ops.emplace_back(op, level);
    }
  }

  llvm::outs() << "[HeuristicMapping] Filtered "
               << sorted_ops_with_levels.size() - materialized_ops.size()
               << " non-materialized operations, " << materialized_ops.size()
               << " operations require physical mapping." << "\n";

  llvm::outs() << "[HeuristicMapping] Materialized operations list:\n";
  for (size_t i = 0; i < materialized_ops.size(); ++i) {
    llvm::outs() << i << " " << *materialized_ops[i].first
                 << " (level: " << materialized_ops[i].second << ")\n";
  }

  // Stores the mapping state snapshots for backtracking.
  std::vector<MappingStateSnapshot> snapshots;

  // For each operation, stores the candidate locations.
  std::vector<int> candidate_history;

  // Stores the mapping history for each operation.
  std::vector<int> operation_index_history;

  // Starts from the initial mapping state.
  snapshots.push_back(MappingStateSnapshot(mapping_state));
  candidate_history.push_back(0);
  operation_index_history.push_back(0);

  // Tracks the maximum depth of operations reached during mapping.
  int max_op_reached = 0;

  while (!operation_index_history.empty()) {
    // Gets the current operation index to map.
    int current_op_index = operation_index_history.back();

    if (current_op_index >= static_cast<int>(materialized_ops.size())) {
      // All operations have been mapped successfully.
      llvm::outs() << "[HeuristicMapping] Successfully mapped all "
                   << materialized_ops.size() << " operations.\n";
      return true;
    }

    max_op_reached =
        std::max(max_op_reached,
                 current_op_index); // Updates the max operation reached.

    if (max_op_reached - current_op_index > this->max_backtrack_depth) {
      llvm::outs() << "[HeuristicMapping] Max backtrack depth exceeded: "
                   << (max_op_reached - current_op_index) << " > "
                   << this->max_backtrack_depth << ".\n";
      return false; // Backtrack failed, max depth exceeded.
    }

    Operation *current_op = materialized_ops[current_op_index].first;
    std::vector<MappingLoc> candidate_locs;
    candidate_locs = calculateAward(current_op, critical_ops,
                                    materialized_ops[current_op_index].second,
                                    architecture, mapping_state);

    if (candidate_locs.empty()) {
      llvm::outs() << "[HeuristicMapping] No candidate locations found "
                   << "for operation: " << *current_op << "\n";
      // No candidate locations available, backtrack to the previous operation.
      snapshots.pop_back(); // Restore the previous mapping state.
      candidate_history.pop_back();
      operation_index_history.pop_back();

      if (snapshots.empty()) {
        llvm::outs() << "[HeuristicMapping] No more snapshots to restore, "
                     << "mapping failed.\n";
        return false; // No more snapshots to restore, mapping failed.
      }

      snapshots.back().restore(mapping_state);
      candidate_history.back()++;
      llvm::outs() << "[HeuristicMapping] Backtracking to operation "
                   << operation_index_history.back() << "(depth = "
                   << (max_op_reached - operation_index_history.back())
                   << ")\n";
      continue; // Backtrack to the previous operation.
    }

    llvm::outs() << "[HeuristicMapping] Found " << candidate_locs.size()
                 << " candidate locations for operation: " << *current_op
                 << "\n";
    // Limits the number of locations to try.
    if (candidate_locs.size() >
        static_cast<size_t>(this->max_location_to_try)) {
      candidate_locs.resize(static_cast<size_t>(this->max_location_to_try));
    }

    int &current_candidate_index = candidate_history.back();

    if (current_candidate_index >= static_cast<int>(candidate_locs.size())) {
      // Needs to backtrack since all candidate locations have been tried.
      llvm::outs() << "[HeuristicMapping] All " << candidate_locs.size()
                   << " locations for " << current_op_index
                   << " tried, backtracking...\n";

      // Removes the last mapping state snapshot and candidate history.
      snapshots.pop_back();
      candidate_history.pop_back();
      operation_index_history.pop_back();

      // If no more operation indices to backtrack, mapping failed.
      if (operation_index_history.empty()) {
        llvm::outs() << "[HeuristicMapping] FAILURE: No more operations "
                        "available for backtracking.\n";
        return false;
      }

      // Restores the previous state
      snapshots.back().restore(mapping_state);

      // Increments the candidate location index for the previous decision point
      candidate_history.back()++;

      llvm::outs() << "[HeuristicMapping] Backtracking to operation "
                   << operation_index_history.back() << " (depth = "
                   << (max_op_reached - operation_index_history.back())
                   << ").\n";

      continue;
    }

    MappingLoc candidate_loc = candidate_locs[current_candidate_index];

    llvm::outs() << "[HeuristicMapping] Trying candidate "
                 << (current_candidate_index + 1) << "/"
                 << candidate_locs.size() << " at "
                 << candidate_loc.resource->getType() << "#"
                 << candidate_loc.resource->getId()
                 << " @t=" << candidate_loc.time_step << "\n";

    // Saves the current mapping state snapshot before attempting to map.
    MappingStateSnapshot snapshot(mapping_state);

    // Attempts to place and route the operation at the candidate location.
    if (placeAndRoute(current_op, candidate_loc, mapping_state)) {
      llvm::outs() << "[HeuristicMapping] Successfully mapped operation "
                   << *current_op << "\n";

      // Adds a new decision point for the next operation.
      snapshots.push_back(MappingStateSnapshot(mapping_state));
      candidate_history.push_back(0);
      operation_index_history.push_back(current_op_index + 1);
    } else {
      // Mapping failed, restores state and tries the next candidate.
      llvm::outs() << "[HeuristicMapping] Failed to map operation "
                   << *current_op << " to candidate location "
                   << (current_candidate_index + 1) << "/"
                   << candidate_locs.size() << "\n";

      snapshot.restore(mapping_state);
      current_candidate_index++;
    }
  }

  // If we reach here, it means we have exhausted all possibilities.
  llvm::errs() << "[HeuristicMapping] FAILURE: Exhausted all possibilities\n";
  return false;
}

} // namespace neura
} // namespace mlir
