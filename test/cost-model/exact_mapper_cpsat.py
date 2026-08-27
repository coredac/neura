#!/usr/bin/env python3
"""Budgeted CGRA modulo mapper (OR-Tools CP-SAT).

Unlike the lower-bound oracle, this models the full mapping and produces a
concrete assignment: placement + schedule + the routing path of every value +
register/link occupancy — all on the real link graph (works for any topology
including multi-CGRA), with modulo resource limits. The first II for which it
finds a witness is reported; because each solve has a deterministic search
budget, that result is not a proof of the minimum II. The emitted mapping is
directly usable by a backend.

Two-stage per II (iterated upward from a lower bound):
  1. schedule(): CP-SAT places ops on FU-supporting tiles and assigns issue
     times, enforcing the FU modulo resource and precedence/recurrence with the
     shortest-path hop latency over the link graph. A same-tile dependence is
     represented by a one-slot local register transfer during routing.
  2. route(): with placement+times fixed, CP-SAT routes every net (producer ->
     all its consumers) as presence-flow over the time-expanded modulo graph,
     enforcing per-(link,residue) <=1 net and per-(tile,residue) <= #registers.
     Fan-out shares links (same net is free to reuse a (link,residue)).
If a witness cannot be found within budget, the driver tries II+1. The legacy
``--fallback`` flag remains accepted for callers that already pass it; retries
are now the normal ladder behavior. The routes are the emitted mapping.

Input dfg.json (from --dump-dfg-json): ops[{class,latency}], edges[{s,d,w}]
(w=1 marks a loop-carried edge), and arch{tiles[{id,x,y,regs}], links[[s,d,lat]],
fu_class_tiles{class:[tile ids]}, num_tiles, ctrl_mem_items}.

Usage: exact_mapper_cpsat.py dfg.json [--max-ii N]
       [--deterministic-time T] [-v] [--emit out.json]

Install the pinned solver version from test/cost-model/requirements.txt when
comparing results across servers.
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
    """FU classes this kernel uses that the architecture describes but that NO
    tile provides, i.e. capacity zero -- the kernel cannot be placed here at any
    II. On a pruned (L/T-shaped) architecture this is a real outcome.

    A class that is ABSENT from fu_class_tiles is a different case: nothing is
    known about which FU runs it, so it is unconstrained and every tile is a
    candidate (see the .get(cls, tile_ids) default below). Note that
    `fu_class_tiles.get(cls) or tile_ids` collapses these two cases --
    an empty list is falsy, so a class no tile provides silently became ALL
    tiles, and the mapper reported capacity num_tiles where the analytical cost
    model reported infeasible."""
    fu_class_tiles = data["arch"]["fu_class_tiles"]
    return sorted({op["class"] for op in data["ops"]
                   if op["class"] in fu_class_tiles
                   and not fu_class_tiles[op["class"]]})


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
    total_latency = sum(max(1, op["latency"]) for op in ops)
    max_time = 2 * num_ops + 2 * ii + total_latency + 8
    model = cp_model.CpModel()
    place = []                                  # place[i]      = tile of op i
    issue_time = []                             # issue_time[i] = cycle op i fires
    for i, op in enumerate(ops):
        # Restrict placement to tiles whose FU supports this op's class. A class
        # the arch does not list at all is unconstrained -> any tile; a class it
        # lists as [] is provided by no tile and was rejected above, so the
        # default here must be `.get(cls, tile_ids)`, never `.get(cls) or
        # tile_ids` (the two differ specifically on the empty list).
        valid_tiles = fu_class_tiles.get(op["class"], tile_ids)
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
    max_hop = max((h for h in hops.values() if h != float("inf")), default=0)
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
                    help="CP-SAT deterministic work budget per stage (default: 30).")
    ap.add_argument("--seconds", type=float, default=None,
                    help="Deprecated alias for --deterministic-time.")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--emit", default=None,
                    help="Write the concrete placement+schedule mapping to JSON.")
    ap.add_argument("--fallback", action="store_true",
                    help="Deprecated compatibility flag; the II ladder always "
                         "retries an infeasible or budget-exhausted stage.")
    ap.add_argument("--minimize-routing", dest="minimize_routing",
                    action="store_true",
                    default=None,
                    help="Bias stage 1 toward a router-friendly placement. "
                         "Enabled by default when emitting a witness.")
    ap.add_argument("--no-minimize-routing", dest="minimize_routing",
                    action="store_false",
                    help="Skip the router-friendly bias. Faster, but one witness "
                         "may be harder for the backend router to reproduce.")
    ap.add_argument("--emit-deterministic-time", type=float, default=None,
                    help="Per-stage deterministic budget while emitting (default: 12).")
    ap.add_argument("--emit-cap", type=float, default=None,
                    help="Deprecated alias for --emit-deterministic-time.")
    a = ap.parse_args()
    if a.deterministic_time is not None and a.seconds is not None:
        ap.error("use only one of --deterministic-time and deprecated --seconds")
    if a.emit_deterministic_time is not None and a.emit_cap is not None:
        ap.error("use only one of --emit-deterministic-time and deprecated --emit-cap")
    deterministic_time = (a.deterministic_time if a.deterministic_time is not None
                          else a.seconds if a.seconds is not None else 30.0)
    emit_deterministic_time = (a.emit_deterministic_time
                               if a.emit_deterministic_time is not None else
                               a.emit_cap if a.emit_cap is not None else 12.0)
    if a.seconds is not None:
        print("[deprecated] --seconds is an alias for --deterministic-time; "
              "the value is a deterministic work budget", file=sys.stderr)
    if a.emit_cap is not None:
        print("[deprecated] --emit-cap is an alias for "
              "--emit-deterministic-time", file=sys.stderr)
    if deterministic_time <= 0 or emit_deterministic_time <= 0:
        ap.error("deterministic budgets must be positive")
    data = json.load(open(a.json))
    hops = shortest_hops(data["arch"])
    max_ii = a.max_ii or data["arch"]["ctrl_mem_items"]
    # When emitting a mapping for the backend, prefer low-hop (router-friendly)
    # placements so the greedy per-net router can reproduce them. A separate
    # deterministic cap keeps witness generation bounded.
    minimize_routing = (bool(a.emit) if a.minimize_routing is None
                        else a.minimize_routing)
    solve_budget = (min(deterministic_time, emit_deterministic_time)
                    if a.emit else deterministic_time)
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
        sched = schedule(data, ii, solve_budget, hops, minimize_routing)
        if sched is None:
            if a.v: print(f"  II={ii}: schedule infeasible", file=sys.stderr)
            continue                            # Try the next II.
        if sched == BUDGET_EXHAUSTED:
            # The solver could not settle this II within the deterministic
            # budget. This is deliberately treated as a reason to try II+1,
            # not as evidence about the minimum II.
            if a.v: print(f"  II={ii}: schedule budget exhausted, backing "
                          f"off to II+1", file=sys.stderr)
            continue
        if sched == SOLVER_ERROR:
            print(f"  II={ii}: schedule solver error; aborting", file=sys.stderr)
            return 2
        ok, routes, route_status = route(data, sched, ii, solve_budget,
                                         want_routes=bool(a.emit))
        if a.v:
            print(f"  II={ii}: schedule OK, routing={route_status.upper()}",
                  file=sys.stderr)
        if route_status == SOLVER_ERROR:
            print(f"  II={ii}: routing solver error; aborting", file=sys.stderr)
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
