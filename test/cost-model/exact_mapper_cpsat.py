#!/usr/bin/env python3
"""Budgeted CGRA modulo mapper (OR-Tools CP-SAT).

Unlike the lower-bound oracle, this models the full mapping and produces a
concrete assignment: placement + schedule + the routing path of every value +
register/link occupancy — all on the real link graph (works for any topology
including multi-CGRA), with modulo resource limits. The first II for which it
finds a witness is reported; because each solve has a deterministic search
budget, that result is not a proof of the minimum II. The emitted mapping is
directly usable by a backend.

One joint CP-SAT model is solved per II: it chooses placement, issue time, and
the time-expanded routing path together.  A route/resource conflict therefore
feeds back into placement and scheduling at the same II instead of causing a
false II+1 retry.  ``schedule`` and ``route`` remain below as independently
testable reference helpers.
If a witness cannot be found within budget, the driver tries II+1. The legacy
``--fallback`` flag remains accepted for callers that already pass it; retries
are now the normal ladder behavior. The routes are the emitted mapping.

Input dfg.json (from --dump-dfg-json): ops[{class,latency}], edges[{s,d,w}]
(w=1 marks a loop-carried edge), and arch{tiles[{id,x,y,regs}], links[[s,d,lat]],
fu_class_tiles{class:[tile ids]}, num_tiles, ctrl_mem_items}.

Usage: exact_mapper_cpsat.py dfg.json [--max-ii N]
       [--deterministic-time T] [-v] [--emit out.json]
"""
import argparse
import collections
import json
import sys

from ortools.sat.python import cp_model


INFEASIBLE = "infeasible"
BUDGET_EXHAUSTED = "budget_exhausted"
SOLVER_ERROR = "solver_error"


def configure_solver(solver, deterministic_time):
    """Use a CP-SAT solver-work budget for every solve."""
    if deterministic_time <= 0:
        raise ValueError("deterministic_time must be positive")
    # Deterministic time avoids CPU-speed-dependent cutoffs; pin the OR-Tools
    # version and model construction order as well when reproducing a run.
    solver.parameters.max_deterministic_time = deterministic_time
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0


def solver_outcome(status):
    """Map a CP-SAT status to the driver's small status vocabulary."""
    if status == cp_model.INFEASIBLE:
        return INFEASIBLE
    if status == cp_model.UNKNOWN:
        return BUDGET_EXHAUSTED
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "feasible"
    return SOLVER_ERROR


def shortest_hops(arch):
    """All-pairs shortest path (in link latency) over the real directed link
    graph. Returns dist[(a,b)] in cycles; missing pairs -> None (unreachable).

    Example (4x4 mesh, id = y*4+x, latency-1 links): dist[(0,0)] = 0,
    dist[(0,1)] = 1 (adjacent), dist[(0,5)] = 2 (0->1->5 or 0->4->5),
    dist[(0,15)] = 6 (opposite corners)."""
    tile_ids = [t["id"] for t in arch["tiles"]]
    INF = float("inf")
    dist = {(a, b): (0 if a == b else INF)
            for a in tile_ids for b in tile_ids}
    for src, dst, latency in arch["links"]:
        dist[(src, dst)] = min(dist[(src, dst)], latency)
    # Floyd-Warshall: relax every path through an intermediate tile `via`.
    for via in tile_ids:
        for src in tile_ids:
            dist_src_via = dist[(src, via)]
            if dist_src_via == INF:
                continue
            for dst in tile_ids:
                relaxed = dist_src_via + dist[(via, dst)]
                if relaxed < dist[(src, dst)]:
                    dist[(src, dst)] = relaxed
    return dist


def unprovidable_classes(data):
    """Return architecture-defined classes with zero capacity.

    Every emitted operation class must be present in ``fu_class_tiles``.  A
    missing class is a malformed DFG/architecture contract, not permission to
    silently place it on every tile.  Empty classes remain a genuine
    infeasibility for this shape."""
    fu_class_tiles = data["arch"]["fu_class_tiles"]
    classes = {op["class"] for op in data["ops"]}
    missing = sorted(classes - fu_class_tiles.keys())
    assert not missing, f"architecture has no FU-class definition for {missing}"
    return sorted(cls for cls in classes if not fu_class_tiles[cls])


def compact_schedule_horizon(ops, ii, max_hop):
    """Finite absolute-time window for one modulo-schedule witness.

    The model uses absolute time only to materialize an initial pipeline fill;
    steady-state resource conflicts are modulo-II constraints.  The former
    bound double-counted every operation *and* its latency, which expanded each
    time-layered routing graph far beyond a compact witness needs.  This bound
    uses three II windows for recurrence fill, plus enough room for one
    operation and one architecture-diameter transfer.  This deliberately asks
    for a compact witness; the driver treats failure to find one exactly like
    any other bounded attempt and tries II+1.  It is deterministic and
    independent of host performance.
    """
    return max(3 * ii,
               max((max(1, op["latency"]) for op in ops), default=1) + max_hop)


def schedule(data, ii, deterministic_time, hops, minimize_routing=False):
    """Stage 1: decide place[i] (tile) and issue_time[i] for every op at this II.
    Returns {i:(tile,time)} if feasible, None if infeasible, or
    ``BUDGET_EXHAUSTED`` when the deterministic search budget is consumed.
    ``minimize_routing`` biases toward a router-friendly witness (see below).
    """
    ops, edges, arch = data["ops"], data["edges"], data["arch"]
    tile_ids = [t["id"] for t in arch["tiles"]]
    fu_class_tiles = arch["fu_class_tiles"]
    num_ops = len(ops)
    if unprovidable_classes(data):
        return None                             # no tile can run some op: infeasible
    # Upper bound on any op's issue time. A modulo schedule fits within one II
    # window per op plus slack for the dependence chain and the pipeline depth of
    # multi-cycle ops; include the total op latency so a long chain of latency>1
    # ops can't overflow the domain and get mislabeled INFEASIBLE. Still a loose
    # ceiling that just keeps CP-SAT's integer domains finite.
    max_hop = max((h for h in hops.values() if h != float("inf")), default=0)
    max_time = compact_schedule_horizon(ops, ii, int(max_hop))
    model = cp_model.CpModel()
    place = []                                  # place[i]      = tile of op i
    issue_time = []                             # issue_time[i] = cycle op i fires
    for i, op in enumerate(ops):
        # The schema assertion in unprovidable_classes guarantees that every
        # operation class has an explicit FU-class entry.
        valid_tiles = fu_class_tiles[op["class"]]
        place_var = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(valid_tiles), f"place{i}")
        place.append(place_var)
        issue_time.append(model.NewIntVar(0, max_time, f"time{i}"))
    # Match MappingState's pipeline-aware occupancy. SINGLE is exclusive;
    # START conflicts only with another START; END conflicts only with another
    # END; and IN can share with any non-SINGLE stage. Do not put all stages in
    # one AllDifferent: a pipelined START/END overlap is legal, including an
    # operation's own modulo overlap when latency exceeds II.
    single_cells = []
    start_cells = []
    end_cells = []
    pipeline_cells = []
    for i in range(num_ops):
        residue = model.NewIntVar(0, ii - 1, f"residue{i}")
        iteration = model.NewIntVar(0, max_time, f"iter{i}")
        model.Add(issue_time[i] == iteration * ii + residue)  # residue = time % II
        latency_i = max(1, ops[i]["latency"])
        for k in range(latency_i):
            # occ_residue = (residue + k) mod II (the k-th pipeline stage's slot).
            occ_residue = model.NewIntVar(0, ii - 1, f"occ{i}_{k}")
            model.AddModuloEquality(occ_residue, residue + k, ii)
            cell = model.NewIntVar(0, (max(tile_ids) + 1) * ii, f"cell{i}_{k}")
            model.Add(cell == place[i] * ii + occ_residue)
            if latency_i == 1:
                single_cells.append(cell)
            elif k == 0:
                start_cells.append(cell)
                pipeline_cells.append(cell)
            elif k == latency_i - 1:
                end_cells.append(cell)
                pipeline_cells.append(cell)
            else:
                pipeline_cells.append(cell)
    if single_cells:
        model.AddAllDifferent(single_cells)
    if start_cells:
        model.AddAllDifferent(start_cells)
    if end_cells:
        model.AddAllDifferent(end_cells)
    for single_cell in single_cells:
        for pipeline_cell in pipeline_cells:
            model.Add(single_cell != pipeline_cell)
    # Precedence/recurrence. hop is the place-dependent travel latency between the
    # producer and consumer tiles, pinned to the shortest-path table so the timing
    # follows the selected shortest path. is_loop_carried (w=1) relaxes the bound
    # by one II (next iteration).
    # Example: producer src on tile 0 issued at time 3 with latency 1, consumer
    # dst on tile 5 (2 hops away) -> time[dst] >= 3 + 1 + 2 = 6. If it were a
    # loop-carried edge (w=1, II=4) the value is consumed next iteration:
    # time[dst] >= 3 + 1 + 2 - 4 = 2.
    hop_vars = []
    for edge in edges:
        src, dst, is_loop_carried = edge["s"], edge["d"], edge["w"]
        src_latency = ops[src]["latency"]
        hop = model.NewIntVar(0, int(max_hop), "")
        # hop == hops[(place[src], place[dst])] via a table constraint listing
        # every reachable (src_tile, dst_tile, hop_count) triple.
        allowed_hops = [[src_tile, dst_tile, int(hops[(src_tile, dst_tile)])]
                        for src_tile in tile_ids for dst_tile in tile_ids
                        if hops[(src_tile, dst_tile)] != float("inf")]
        model.AddAllowedAssignments([place[src], place[dst], hop], allowed_hops)
        model.Add(issue_time[dst] >=
                  issue_time[src] + src_latency + hop - is_loop_carried * ii)
        hop_vars.append(hop)
    # Optional: make the schedule reproducible by the backend's greedy per-net
    # router. That router struggles with long register-hold spans and long
    # paths, so prefer a COMPACT schedule (small makespan => short holds) first
    # and LOCAL placements (few hops) second. Neither changes which II is
    # feasible; both just pick a router-friendly witness. Off for the fast sweep.
    if minimize_routing:
        makespan = model.NewIntVar(0, max_time, "makespan")
        for i in range(num_ops):
            model.Add(makespan >= issue_time[i])
        model.Minimize(makespan * (int(max_hop) * len(edges) + 1) + sum(hop_vars))
    solver = cp_model.CpSolver()
    configure_solver(solver, deterministic_time)
    status = solver.Solve(model)
    outcome = solver_outcome(status)
    if outcome == "feasible":
        return {i: (solver.Value(place[i]), solver.Value(issue_time[i]))
                for i in range(num_ops)}
    return None if outcome == INFEASIBLE else outcome


def route(data, sched, ii, deterministic_time, want_routes=False):
    """Stage 2: with placement+times fixed, find a modulo-feasible routing of
    every value. Returns (feasible, routes, status). routes is None unless
    want_routes; otherwise routes[(s,d)] is the concrete [(tile,cycle),...]
    path per edge."""
    ops, edges, arch = data["ops"], data["edges"], data["arch"]
    tile_ids = [t["id"] for t in arch["tiles"]]
    num_regs = {t["id"]: t["regs"] for t in arch["tiles"]}
    out_links = collections.defaultdict(list)   # tile -> [(dst_tile, latency)]
    for src, dst, latency in arch["links"]:
        out_links[src].append((dst, latency))
    # A "net" is one producer feeding its consumers. produce_cycle is the cycle
    # the producer's result is ready; deadline is when a consumer must have it (a
    # loop-carried consumer needs it one II later).
    # Example: producer 7 on tile 2 at t=2, latency 1 -> produce_cycle[7]=3. A
    # normal consumer on tile 9 at t=6 -> deadline 6; a loop-carried one (w=1,
    # II=4) at t=6 -> deadline 10. So nets[7] = [(9, 6), ...].
    nets = collections.defaultdict(list)        # producer -> [(cons_tile, deadline)]
    produce_cycle = {}
    for i, (tile, fire_time) in sched.items():
        produce_cycle[i] = fire_time + ops[i]["latency"]
    for edge in edges:
        src, dst, omega = edge["s"], edge["d"], edge["w"]
        cons_tile, cons_time = sched[dst]
        # A distance-omega loop-carried edge is consumed omega iterations later;
        # use omega*ii (not a boolean) so this matches schedule()'s -w*ii bound
        # for any omega, not just 0/1.
        deadline = cons_time + omega * ii
        nets[src].append((cons_tile, deadline))
    model = cp_model.CpModel()
    # Each net is routed as a presence flow on a time-expanded graph:
    # present[(tile,cycle)] means "the value sits on this tile at cycle". It
    # enters at the producer and must reach every consumer node; between them it
    # either holds (a register on the same tile, next cycle) or moves (a link
    # into the tile). link_use / reg_use collect the booleans that share a modulo
    # slot so we can cap them.
    # Each backend register file exposes ONE write and ONE read port, so a tile
    # can write (or read) at most #regfiles values in one modulo slot -- a real
    # limit the storage-count cap below misses, which is what makes a register-
    # routing hub fail to import. The regfile count is read straight from the arch
    # (emitted by --dump-dfg-json); older dumps without it fall back to regs/8
    # (Architecture.cpp k_num_regs_per_regfile). We charge write/read ports at
    # register-run boundaries; a bare link move does not use a register port.
    num_regfiles = {t["id"]: t.get("regfiles", t["regs"] // 8)
                    for t in arch["tiles"]}
    link_use = collections.defaultdict(list)  # (link_key, residue) -> [bool]
    reg_use = collections.defaultdict(list)   # (tile, residue)     -> [bool]
    # A register slot is owned by one producer net at one absolute cycle. This
    # lets a direct local consumer share the same slot with a longer route from
    # that producer without merging distinct live slots at the same residue.
    register_slot_use = collections.defaultdict(list)
    # Port candidates are consolidated per producer/tile/absolute slot below,
    # so fan-out of one value can reuse its register address and port event.
    write_port_candidates = collections.defaultdict(list)
    read_port_candidates = collections.defaultdict(list)
    write_port_use = collections.defaultdict(list)  # (tile, residue) -> [bool]
    read_port_use = collections.defaultdict(list)   # (tile, residue) -> [bool]
    present_of = {}                           # producer -> {(tile,cycle): boolvar}
    # producer -> {(child_tile,child_cyc): [(kind, parent_node, boolvar)]}. Used
    # after solving to trace each consumer back to the producer (concrete route).
    arcs_of = collections.defaultdict(lambda: collections.defaultdict(list))
    for producer, consumers in nets.items():
        src_tile = sched[producer][0]
        start_cycle = produce_cycle[producer]
        last_cycle = max(deadline for _, deadline in consumers)
        if last_cycle < start_cycle:
            last_cycle = start_cycle
        cycles = list(range(start_cycle, last_cycle + 1))
        present = {(tile, cycle): model.NewBoolVar(f"pr{producer}_{tile}_{cycle}")
                   for tile in tile_ids for cycle in cycles}
        present_of[producer] = present
        hold_vars = {}
        model.Add(present[(src_tile, start_cycle)] == 1)   # starts at producer
        for (cons_tile, deadline) in consumers:
            model.Add(present[(cons_tile, deadline)] == 1)  # present at consumer
        # Flow: every present node except the source has an incoming arc.
        for tile in tile_ids:
            for cycle in cycles:
                if (tile, cycle) == (src_tile, start_cycle):
                    continue
                incoming = []
                if cycle - 1 in cycles:
                    # hold on same tile: present here now if it was here last
                    # cycle (e.g. present[(tile 3, cycle 5)] can be justified by
                    # holding a register from present[(tile 3, cycle 4)], or by a
                    # link move below).
                    hold_var = model.NewBoolVar(f"hold{producer}_{tile}_{cycle}")
                    model.Add(present[(tile, cycle - 1)] == 1).OnlyEnforceIf(hold_var)
                    model.Add(present[(tile, cycle)] == 1).OnlyEnforceIf(hold_var)
                    incoming.append(hold_var)
                    hold_vars[(tile, cycle)] = hold_var
                    arcs_of[producer][(tile, cycle)].append(
                        ("hold", (tile, cycle - 1), hold_var))
                    # link moves from neighbors into `tile` (arrive after latency)
                    for neighbor in tile_ids:
                        for (dst_tile, latency) in out_links[neighbor]:
                            if (dst_tile == tile and (cycle - latency) in cycles
                                    and latency >= 1):
                                move_var = model.NewBoolVar(
                                    f"move{producer}_{neighbor}_{tile}_{cycle}")
                                model.Add(
                                    present[(neighbor, cycle - latency)] == 1
                                ).OnlyEnforceIf(move_var)
                                model.Add(present[(tile, cycle)] == 1).OnlyEnforceIf(
                                    move_var)
                                incoming.append(move_var)
                                link_use[((neighbor, dst_tile),
                                          (cycle - latency) % ii)].append(move_var)
                                arcs_of[producer][(tile, cycle)].append(
                                    ("move", (neighbor, cycle - latency), move_var))
                # present => one incoming arc chosen. Besides making
                # route reconstruction unambiguous, this prevents a net from
                # charging multiple holds for the same occupied node.
                model.Add(sum(incoming) == 1).OnlyEnforceIf(present[(tile, cycle)])
                if not incoming:
                    model.Add(present[(tile, cycle)] == 0)
        # A register run starts at the first selected hold and ends at the last
        # selected hold. Only those two boundary slots use regfile ports; the
        # intermediate holds consume storage but no port. H_(tile, cycle) is
        # the register slot at `cycle - 1`, because it holds into `cycle`.
        for (tile, cycle), hold_var in hold_vars.items():
            absolute_slot = cycle - 1
            register_slot_use[(producer, tile, absolute_slot)].append(hold_var)
            previous_hold = hold_vars.get((tile, cycle - 1))
            next_hold = hold_vars.get((tile, cycle + 1))
            write_boundary = model.NewBoolVar(
                f"write{producer}_{tile}_{absolute_slot}")
            read_boundary = model.NewBoolVar(
                f"read{producer}_{tile}_{absolute_slot}")
            if previous_hold is None:
                model.Add(write_boundary == hold_var)
            else:
                model.Add(write_boundary <= hold_var)
                model.Add(write_boundary + previous_hold <= 1)
                model.Add(write_boundary >= hold_var - previous_hold)
            if next_hold is None:
                model.Add(read_boundary == hold_var)
            else:
                model.Add(read_boundary <= hold_var)
                model.Add(read_boundary + next_hold <= 1)
                model.Add(read_boundary >= hold_var - next_hold)
            port_key = (producer, tile, absolute_slot)
            write_port_candidates[port_key].append(write_boundary)
            read_port_candidates[port_key].append(read_boundary)
        # A one-node route is a legal backend representation for a same-tile
        # value transfer, but it still consumes one register slot. The slot is
        # also charged with one write and one read at this cycle; with zero
        # registers (or zero register files) the model therefore rejects it.
        local_transfer = any(cons_tile == src_tile and deadline == start_cycle
                             for cons_tile, deadline in consumers)
        if local_transfer:
            local_slot = model.NewBoolVar(
                f"local{producer}_{src_tile}_{start_cycle}")
            model.Add(local_slot == 1)
            absolute_slot = start_cycle
            register_slot_use[(producer, src_tile, absolute_slot)].append(
                local_slot)
            port_key = (producer, src_tile, absolute_slot)
            write_port_candidates[port_key].append(local_slot)
            read_port_candidates[port_key].append(local_slot)
    # Consolidate hold arcs and direct local transfers of one producer net into
    # one physical register slot per absolute cycle.
    for (producer, tile, absolute_slot), uses in register_slot_use.items():
        slot = model.NewBoolVar(f"regslot{producer}_{tile}_{absolute_slot}")
        for use in uses:
            model.Add(slot >= use)
        model.Add(slot <= sum(uses))
        reg_use[(tile, absolute_slot % ii)].append(slot)
    # Merge boundary events belonging to the same producer net and absolute
    # slot. A direct local edge and a longer fan-out route may use the same
    # register and therefore the same physical port event.
    for (producer, tile, absolute_slot), uses in write_port_candidates.items():
        boundary = model.NewBoolVar(
            f"writeport{producer}_{tile}_{absolute_slot}")
        for use in uses:
            model.Add(boundary >= use)
        model.Add(boundary <= sum(uses))
        write_port_use[(tile, absolute_slot % ii)].append(boundary)
    for (producer, tile, absolute_slot), uses in read_port_candidates.items():
        boundary = model.NewBoolVar(
            f"readport{producer}_{tile}_{absolute_slot}")
        for use in uses:
            model.Add(boundary >= use)
        model.Add(boundary <= sum(uses))
        read_port_use[(tile, absolute_slot % ii)].append(boundary)
    # Modulo resources: a link carries one net-move per residue; a tile holds no
    # more values per residue than it has registers.
    for key, bool_vars in link_use.items():
        model.Add(sum(bool_vars) <= 1)          # one net-move per (link, residue)
    for (tile, residue), bool_vars in reg_use.items():
        model.Add(sum(bool_vars) <= num_regs[tile])  # storage capacity
    # Regfile port limits (see num_regfiles above): register-run boundaries per
    # modulo slot cannot exceed the tile's regfile count. Bare link moves do not
    # use a register port.
    for (tile, residue), bool_vars in write_port_use.items():
        model.Add(sum(bool_vars) <= num_regfiles[tile])
    for (tile, residue), bool_vars in read_port_use.items():
        model.Add(sum(bool_vars) <= num_regfiles[tile])
    solver = cp_model.CpSolver()
    configure_solver(solver, deterministic_time)
    status = solver.Solve(model)
    outcome = solver_outcome(status)
    feasible = outcome == "feasible"
    if not want_routes or not feasible:
        return (feasible, None, outcome)

    # Reconstruct the concrete route of every edge: trace each consumer node
    # back to the producer through the chosen arcs. routes[(s,d)] = ordered list
    # of (tile, cycle) the value occupies, producer-tile first, consumer last.
    # e.g. [(2,3),(2,4),(3,5)] = produced on tile 2 at cycle 3, held in a tile-2
    # register through cycle 4, then a link move arriving on tile 3 at cycle 5.
    def trace(producer, start_node):
        src_tile = sched[producer][0]
        start_cycle = produce_cycle[producer]
        node = start_node
        reversed_path = [node]
        guard = 0
        while node != (src_tile, start_cycle) and guard < 100000:
            guard += 1
            # pick any incoming arc that is set and whose parent is also present
            parent = None
            for kind, parent_node, var in arcs_of[producer].get(node, []):
                if solver.Value(var) and solver.Value(present_of[producer][parent_node]):
                    parent = parent_node
                    break
            if parent is None:
                return None
            reversed_path.append(parent)
            node = parent
        return list(reversed(reversed_path))

    routes = {}
    for edge in edges:
        src, dst, omega = edge["s"], edge["d"], edge["w"]
        deadline = sched[dst][1] + omega * ii
        path = trace(src, (sched[dst][0], deadline))
        if path is not None:
            routes[(src, dst)] = path
    return (feasible, routes, outcome)


def joint_solve(data, ii, deterministic_time, want_routes=False):
    """Solve placement, modulo schedule, and routes in one CP-SAT model.

    ``schedule`` + ``route`` remain as small independently-testable helpers,
    but the command-line mapper calls this routine.  In particular, a route
    resource conflict can make CP-SAT choose a *different* placement/schedule
    at the same II instead of incorrectly advancing to II+1 after only one
    stage-1 witness fails to route.
    """
    ops, edges, arch = data["ops"], data["edges"], data["arch"]
    tile_ids = [t["id"] for t in arch["tiles"]]
    if unprovidable_classes(data):
        return (False, None, None, INFEASIBLE)
    fu_class_tiles = arch["fu_class_tiles"]
    num_ops = len(ops)
    hops = shortest_hops(arch)
    max_hop = max((h for h in hops.values() if h != float("inf")), default=0)
    max_time = compact_schedule_horizon(ops, ii, int(max_hop))
    max_latency = max((max(1, op["latency"]) for op in ops), default=1)
    max_omega = max((edge["w"] for edge in edges), default=0)
    horizon = max_time + max_latency + max_omega * ii
    model = cp_model.CpModel()

    place, issue_time, residue = [], [], []
    for i, op in enumerate(ops):
        valid_tiles = fu_class_tiles[op["class"]]
        place.append(model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(valid_tiles), f"place{i}"))
        issue_time.append(model.NewIntVar(0, max_time, f"time{i}"))
        residue_i = model.NewIntVar(0, ii - 1, f"residue{i}")
        iteration = model.NewIntVar(0, max_time, f"iter{i}")
        model.Add(issue_time[-1] == iteration * ii + residue_i)
        residue.append(residue_i)

    # Same pipeline-aware FU occupancy rules as MappingState and schedule().
    single_cells, start_cells, end_cells, pipeline_cells = [], [], [], []
    for i, op in enumerate(ops):
        latency_i = max(1, op["latency"])
        for k in range(latency_i):
            occ = model.NewIntVar(0, ii - 1, f"occ{i}_{k}")
            model.AddModuloEquality(occ, residue[i] + k, ii)
            cell = model.NewIntVar(0, (max(tile_ids) + 1) * ii, f"cell{i}_{k}")
            model.Add(cell == place[i] * ii + occ)
            if latency_i == 1:
                single_cells.append(cell)
            elif k == 0:
                start_cells.append(cell); pipeline_cells.append(cell)
            elif k == latency_i - 1:
                end_cells.append(cell); pipeline_cells.append(cell)
            else:
                pipeline_cells.append(cell)
    if single_cells: model.AddAllDifferent(single_cells)
    if start_cells: model.AddAllDifferent(start_cells)
    if end_cells: model.AddAllDifferent(end_cells)
    for single in single_cells:
        for pipelined in pipeline_cells:
            model.Add(single != pipelined)

    # The shortest-path constraint is redundant with the time-expanded route
    # below, but is a cheap propagation aid and preserves the schedule helper's
    # latency semantics.
    allowed_hops = [[a, b, int(hops[(a, b)])] for a in tile_ids for b in tile_ids
                    if hops[(a, b)] != float("inf")]
    for edge in edges:
        src, dst, omega = edge["s"], edge["d"], edge["w"]
        hop = model.NewIntVar(0, int(max_hop), f"hop{src}_{dst}")
        model.AddAllowedAssignments([place[src], place[dst], hop], allowed_hops)
        model.Add(issue_time[dst] >= issue_time[src] +
                  max(1, ops[src]["latency"]) + hop - omega * ii)

    # Reified (op, tile, issue-cycle) membership.  These literals connect the
    # placement/schedule variables to each time-expanded route graph.
    at_cache = {}
    def at(op, tile, cycle):
        if cycle < 0 or cycle > max_time:
            return None
        key = (op, tile, cycle)
        if key in at_cache:
            return at_cache[key]
        tile_eq = model.NewBoolVar(f"at_tile{op}_{tile}_{cycle}")
        time_eq = model.NewBoolVar(f"at_time{op}_{tile}_{cycle}")
        value = model.NewBoolVar(f"at{op}_{tile}_{cycle}")
        model.Add(place[op] == tile).OnlyEnforceIf(tile_eq)
        model.Add(place[op] != tile).OnlyEnforceIf(tile_eq.Not())
        model.Add(issue_time[op] == cycle).OnlyEnforceIf(time_eq)
        model.Add(issue_time[op] != cycle).OnlyEnforceIf(time_eq.Not())
        model.AddBoolAnd([tile_eq, time_eq]).OnlyEnforceIf(value)
        model.AddBoolOr([tile_eq.Not(), time_eq.Not(), value])
        at_cache[key] = value
        return value

    out_links = collections.defaultdict(list)
    for src, dst, latency in arch["links"]:
        out_links[src].append((dst, latency))
    regs = {t["id"]: t["regs"] for t in arch["tiles"]}
    regfiles = {t["id"]: t.get("regfiles", t["regs"] // 8)
                for t in arch["tiles"]}
    link_use = collections.defaultdict(list)
    register_use = collections.defaultdict(list)
    write_port_use = collections.defaultdict(list)
    read_port_use = collections.defaultdict(list)
    route_presence = []
    # One time-expanded route graph per producer preserves the mapper's fanout
    # semantics: a producer can broadcast one physical value through a shared
    # tree to several consumers.  The graph is still coupled to the variable
    # placements and issue times below, so this remains one joint CP-SAT model.
    # Building a graph per *edge* both double-counts a broadcast's resources
    # and makes the model unnecessarily large on real kernels.
    consumers_by_producer = collections.defaultdict(list)
    for edge in edges:
        consumers_by_producer[edge["s"]].append(edge)
    route_models = []
    for producer, producer_edges in consumers_by_producer.items():
        present = {(tile, cycle): model.NewBoolVar(f"p{producer}_{tile}_{cycle}")
                   for tile in tile_ids for cycle in range(horizon + 1)}
        route_presence.extend(present.values())
        arcs = collections.defaultdict(list)
        holds = {}
        origins, destinations = {}, collections.defaultdict(list)
        for tile in tile_ids:
            for cycle in range(horizon + 1):
                origin = at(producer, tile,
                            cycle - max(1, ops[producer]["latency"]))
                if origin is not None:
                    origins[(tile, cycle)] = origin
                    model.Add(present[(tile, cycle)] == 1).OnlyEnforceIf(origin)
                for edge in producer_edges:
                    target = at(edge["d"], tile, cycle - edge["w"] * ii)
                    if target is not None:
                        destinations[(tile, cycle)].append(target)
                        model.Add(present[(tile, cycle)] == 1).OnlyEnforceIf(target)
                incoming = []
                if cycle:
                    hold = model.NewBoolVar(f"hold{producer}_{tile}_{cycle}")
                    model.AddBoolAnd([present[(tile, cycle - 1)],
                                      present[(tile, cycle)]]).OnlyEnforceIf(hold)
                    incoming.append(hold)
                    holds[(tile, cycle)] = hold
                    arcs[(tile, cycle)].append(((tile, cycle - 1), hold))
                for parent in tile_ids:
                    for child, latency in out_links[parent]:
                        if child != tile or cycle < latency:
                            continue
                        move = model.NewBoolVar(
                            f"move{producer}_{parent}_{tile}_{cycle}")
                        model.AddBoolAnd([present[(parent, cycle - latency)],
                                          present[(tile, cycle)]]).OnlyEnforceIf(move)
                        incoming.append(move)
                        arcs[(tile, cycle)].append(((parent, cycle - latency), move))
                        link_use[((parent, tile), (cycle - latency) % ii)].append(move)
                # A present node must be the actual production point or have a
                # unique incoming hold/link.  Multiple outgoing arcs remain
                # legal: they are precisely a producer's shared fanout tree.
                source = origins.get((tile, cycle))
                model.Add(sum(incoming) + (source if source is not None else 0)
                          >= present[(tile, cycle)])
                if incoming:
                    model.Add(sum(incoming) <= 1)
        # A register run has ports only at its two boundaries.  Interior holds
        # consume storage but neither a new write nor a new read port.  All
        # uses are consolidated by producer/tile/cycle below so fanout shares
        # one physical register slot and one port event.
        register_slot_use = collections.defaultdict(list)
        write_port_candidates = collections.defaultdict(list)
        read_port_candidates = collections.defaultdict(list)
        for (tile, cycle), hold in holds.items():
            previous = holds.get((tile, cycle - 1))
            following = holds.get((tile, cycle + 1))
            write = model.NewBoolVar(f"write{producer}_{tile}_{cycle - 1}")
            read = model.NewBoolVar(f"read{producer}_{tile}_{cycle - 1}")
            if previous is None:
                model.Add(write == hold)
            else:
                model.AddBoolAnd([hold, previous.Not()]).OnlyEnforceIf(write)
                model.AddBoolOr([hold.Not(), previous, write])
            if following is None:
                model.Add(read == hold)
            else:
                model.AddBoolAnd([hold, following.Not()]).OnlyEnforceIf(read)
                model.AddBoolOr([hold.Not(), following, read])
            slot_key = (producer, tile, cycle - 1)
            register_slot_use[slot_key].append(hold)
            write_port_candidates[slot_key].append(write)
            read_port_candidates[slot_key].append(read)
        # Same-tile instantaneous transfers need a real local register for the
        # backend representation. It is the conjunction of the producer-ready
        # literal and any of this net's consumer-ready literals at that node.
        for node, origin in origins.items():
            targets = destinations.get(node)
            if not targets:
                continue
            target_any = model.NewBoolVar(f"target{producer}_{node[0]}_{node[1]}")
            for target in targets:
                model.AddImplication(target, target_any)
            model.AddBoolOr([target_any.Not(), *targets])
            local = model.NewBoolVar(f"local{producer}_{node[0]}_{node[1]}")
            model.AddBoolAnd([origin, target_any]).OnlyEnforceIf(local)
            model.AddBoolOr([origin.Not(), target_any.Not(), local])
            slot_key = (producer, node[0], node[1])
            register_slot_use[slot_key].append(local)
            write_port_candidates[slot_key].append(local)
            read_port_candidates[slot_key].append(local)
        for (_, tile, cycle), uses in register_slot_use.items():
            if len(uses) == 1:
                slot = uses[0]
            else:
                slot = model.NewBoolVar(f"regslot{producer}_{tile}_{cycle}")
                model.AddMaxEquality(slot, uses)
            register_use[(tile, cycle % ii)].append(slot)
        for (_, tile, cycle), uses in write_port_candidates.items():
            if len(uses) == 1:
                boundary = uses[0]
            else:
                boundary = model.NewBoolVar(f"writeport{producer}_{tile}_{cycle}")
                model.AddMaxEquality(boundary, uses)
            write_port_use[(tile, cycle % ii)].append(boundary)
        for (_, tile, cycle), uses in read_port_candidates.items():
            if len(uses) == 1:
                boundary = uses[0]
            else:
                boundary = model.NewBoolVar(f"readport{producer}_{tile}_{cycle}")
                model.AddMaxEquality(boundary, uses)
            read_port_use[(tile, cycle % ii)].append(boundary)
        route_models.append((producer, producer_edges, arcs))
    for (_, residue_key), uses in link_use.items():
        model.Add(sum(uses) <= 1)
    for (tile, residue_key), uses in register_use.items():
        model.Add(sum(uses) <= regs[tile])
    for (tile, residue_key), uses in write_port_use.items():
        model.Add(sum(uses) <= regfiles[tile])
    for (tile, residue_key), uses in read_port_use.items():
        model.Add(sum(uses) <= regfiles[tile])

    # The constraint model accepts any absolute-time translation of a modulo
    # schedule. Select the compact representative first, then the route with
    # the fewest occupied time-expanded nodes. Besides making output stable,
    # this prevents an arbitrary long register hold from bloating the replayed
    # witness while leaving II feasibility unchanged.
    makespan = model.NewIntVar(0, max_time, "makespan")
    for time in issue_time:
        model.Add(makespan >= time)
    route_weight = len(route_presence) + 1
    model.Minimize(makespan * route_weight + sum(route_presence))

    solver = cp_model.CpSolver()
    configure_solver(solver, deterministic_time)
    status = solver.Solve(model)
    outcome = solver_outcome(status)
    if outcome != "feasible":
        return (False, None, None, outcome)
    sched = {i: (solver.Value(place[i]), solver.Value(issue_time[i]))
             for i in range(num_ops)}
    if not want_routes:
        return (True, sched, None, outcome)

    def trace(source, start, arcs):
        node, reverse, guard = start, [start], 0
        while node != source and guard < horizon + 2:
            guard += 1
            candidates = [(parent, var) for parent, var in arcs.get(node, [])
                          if solver.Value(var)]
            if not candidates:
                break
            node = candidates[0][0]
            reverse.append(node)
        return list(reversed(reverse))

    routes = {}
    for producer, producer_edges, arcs in route_models:
        source = (sched[producer][0],
                  sched[producer][1] + max(1, ops[producer]["latency"]))
        for edge in producer_edges:
            dst, omega = edge["d"], edge["w"]
            target = (sched[dst][0], sched[dst][1] + omega * ii)
            path = trace(source, target, arcs)
            assert path[0] == source, (
                f"joint route {producer}->{dst} begins at {path[0]}, "
                f"not source {source}")
            # A direct local route has no incoming arc, hence a one-node path.
            routes[(producer, dst)] = path
    return (True, sched, routes, outcome)


def emit_mapping(data, sched, ii, path, routes=None):
    """Writes the concrete real mapping (op -> tile placement + modulo schedule)
    as JSON. index_per_ii = time_step % II and invalid_iterations = time_step //
    II follow the backend's convention, so this is directly comparable to the
    heuristic mapper's per-op placement. Example (II=6): time_step 4 ->
    index_per_ii 4, invalid_iterations 0; time_step 11 -> index_per_ii 5,
    invalid_iterations 1 (op runs in the second, still-filling iteration)."""
    tile_xy = {t["id"]: (t.get("x"), t.get("y")) for t in data["arch"]["tiles"]}
    placements = []
    for i, op in enumerate(data["ops"]):
        tile, time_step = sched[i]
        x, y = tile_xy.get(tile, (None, None))
        placements.append({
            "id": i, "class": op["class"], "tile": tile, "x": x, "y": y,
            "time_step": time_step,
            "index_per_ii": time_step % ii, "invalid_iterations": time_step // ii,
        })
    out = {"compiled_ii": ii, "num_tiles": data["arch"]["num_tiles"],
           "placements": placements}
    if routes is not None:
        # Concrete per-edge route: producer -> consumer as an ordered list of
        # [tile, cycle] hops (consecutive different tiles => a link move; same
        # tile across cycles => a register hold). The backend importer replays
        # selected path instead of greedily re-routing, so it reproduces the
        # solver's joint routing even on large kernels.
        out["routes"] = [{"s": src, "d": dst,
                          "path": [[tile, cycle] for (tile, cycle) in routes[(src, dst)]]}
                         for (src, dst) in sorted(routes)]
    json.dump(out, open(path, "w"), indent=1)
    num_routes = len(routes) if routes is not None else 0
    print(f"[emit] wrote real mapping ({len(placements)} ops, {num_routes} routes, "
          f"II={ii}) -> {path}", file=sys.stderr)


def main():
    # Driver: climb II from min to max and report the first concrete witness
    # found under the deterministic solve budget. --emit writes that mapping;
    # --fallback is retained as a compatibility flag; bounded retries are the
    # normal ladder behavior.
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--min-ii", type=int, default=1)
    ap.add_argument("--max-ii", type=int, default=0)
    ap.add_argument("--deterministic-time", type=float, default=None,
                    help="CP-SAT deterministic work budget per II (default: 12).")
    ap.add_argument("--seconds", type=float, default=None,
                    help="Deprecated alias for --deterministic-time.")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--emit", default=None,
                    help="Write the concrete placement+schedule mapping to JSON.")
    ap.add_argument("--fallback", action="store_true",
                    help="Deprecated compatibility flag; the II ladder always "
                         "retries an infeasible or budget-exhausted stage.")
    a = ap.parse_args()
    if a.deterministic_time is not None and a.seconds is not None:
        ap.error("use only one of --deterministic-time and deprecated --seconds")
    deterministic_time = (a.deterministic_time if a.deterministic_time is not None
                          else a.seconds if a.seconds is not None else 12.0)
    if a.seconds is not None:
        print("[deprecated] --seconds is an alias for --deterministic-time; "
              "the value is a deterministic work budget", file=sys.stderr)
    if deterministic_time <= 0:
        ap.error("deterministic budgets must be positive")
    data = json.load(open(a.json))
    hops = shortest_hops(data["arch"])
    max_ii = a.max_ii or data["arch"]["ctrl_mem_items"]
    # A joint witness is replayed directly by the backend, so there is no
    # second greedy router to bias.  Keep the legacy option accepted for CLI
    # compatibility, but use one deterministic work budget for the one solve.
    solve_budget = deterministic_time
    # An op whose FU class no tile provides cannot be placed at ANY II, so say so
    # once instead of proving every II infeasible. Reported with the same
    # "> max_ii" verdict the II sweep uses when nothing fits this shape, which is
    # what the C++ caller already reads as "not mappable at this shape".
    missing = unprovidable_classes(data)
    if missing:
        print(f"[infeasible] no tile provides fu class(es): {', '.join(missing)}",
              file=sys.stderr)
        print(f"NO_MAPPED_II_WITHIN_BUDGET (through II={max_ii})")
        return
    for ii in range(a.min_ii, max_ii + 1):
        ok, sched, routes, route_status = joint_solve(
            data, ii, solve_budget, want_routes=bool(a.emit))
        if a.v:
            print(f"  II={ii}: joint placement/schedule/routing="
                  f"{route_status.upper()}",
                  file=sys.stderr)
        if route_status == SOLVER_ERROR:
            print(f"  II={ii}: joint solver error; aborting", file=sys.stderr)
            return 2
        if ok:
            print(f"MAPPED_II = {ii} (placement+schedule+routing witness)")
            if a.emit:
                emit_mapping(data, sched, ii, a.emit, routes)
            return
        # This schedule did not route (infeasible or budget exhausted); try II+1.
    print(f"NO_MAPPED_II_WITHIN_BUDGET (through II={max_ii})")


if __name__ == "__main__":
    sys.exit(main() or 0)
