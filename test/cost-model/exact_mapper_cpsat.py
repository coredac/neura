#!/usr/bin/env python3
"""Complete exact CGRA modulo mapper (OR-Tools CP-SAT).

Unlike exact_oracle_cpsat.py (a relaxation lower bound), this models the FULL
mapping and produces a concrete, valid assignment: placement + schedule + the
routing path of every value + register/link occupancy — all on the REAL link
graph (works for any topology incl. multi-CGRA), with modulo resource limits.
The minimum II for which it succeeds is the true contention-aware optimum, and
the emitted mapping is directly usable by a backend.

Two-stage per II (iterated upward from a lower bound):
  1. schedule(): CP-SAT places ops on FU-supporting tiles and assigns issue
     times, enforcing the FU modulo resource (<=1 op per tile per residue) and
     precedence/recurrence with the true shortest-path hop latency over the link
     graph.
  2. route(): with placement+times fixed, CP-SAT routes every net (producer ->
     all its consumers) as presence-flow over the time-expanded modulo graph,
     enforcing per-(link,residue) <=1 net and per-(tile,residue) <= #registers.
     Fan-out shares links (same net is free to reuse a (link,residue)).
First II with both stages feasible = true optimal; the routes are the mapping.

Input dfg.json (from --dump-dfg-json): ops[{class,latency}], edges[{s,d,w}]
(w=1 marks a loop-carried edge), and arch{tiles[{id,x,y,regs}], links[[s,d,lat]],
fu_class_tiles{class:[tile ids]}, num_tiles, ctrl_mem_items}.

Usage: exact_mapper_cpsat.py dfg.json [--max-ii N] [--seconds S] [-v] [--emit out.json]
"""
import argparse
import collections
import json
import sys

from ortools.sat.python import cp_model


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
    `fu_class_tiles.get(cls) or tile_ids` collapses exactly these two cases --
    an empty list is falsy, so a class no tile provides silently became ALL
    tiles, and the mapper reported capacity num_tiles where the analytical cost
    model reported infeasible."""
    fu_class_tiles = data["arch"]["fu_class_tiles"]
    return sorted({op["class"] for op in data["ops"]
                   if op["class"] in fu_class_tiles
                   and not fu_class_tiles[op["class"]]})


def schedule(data, ii, seconds, hops, minimize_routing=False):
    """Stage 1: decide place[i] (tile) and issue_time[i] for every op at this II.
    Returns {i:(tile,time)} if feasible, None if infeasible, "unknown" on
    timeout. minimize_routing biases toward a router-friendly witness (see below).
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
        # tile_ids` (the two differ exactly on the empty list).
        valid_tiles = fu_class_tiles.get(op["class"], tile_ids)
        place_var = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(valid_tiles), f"place{i}")
        place.append(place_var)
        issue_time.append(model.NewIntVar(0, max_time, f"time{i}"))
    # FU modulo resource: a latency-L op occupies its tile for the WHOLE pipeline
    # window [t, t+L) -- matching the backend, which marks START/IN/END pipe
    # stages and rejects any overlap (MappingState bindMultiCycleOp). Model it as
    # a single AllDifferent over one cell per occupied (tile, residue): for op i
    # and pipeline offset k in [0,L_i), occupied residue = (time+k) mod II and
    # cell = place*II + that residue. Equal cells => two ops (or two stages) share
    # a tile at congruent times, exactly the conflict to ban. The L=1 case reduces
    # to the old "one cell per op". If L_i > II the op's own stages collide on a
    # residue -> AllDifferent is infeasible -> that II is correctly rejected (the
    # FU cannot be re-issued every II). Example (II=4): op on tile 1 at t=5,
    # latency 2 occupies residues 5%4=1 and 6%4=2 -> cells 1*4+1=5 and 1*4+2=6.
    cells = []
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
            cells.append(cell)
    model.AddAllDifferent(cells)
    # Precedence/recurrence. hop is the place-dependent travel latency between the
    # producer and consumer tiles, pinned to the shortest-path table so the timing
    # is exact. is_loop_carried (w=1) relaxes the bound by one II (next iteration).
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
        allowed_hops = [[a, b, int(hops[(a, b)])]
                        for a in tile_ids for b in tile_ids
                        if hops[(a, b)] != float("inf")]
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
    solver.parameters.max_time_in_seconds = seconds
    # Deterministic: single worker + fixed seed (reproducible for tests).
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {i: (solver.Value(place[i]), solver.Value(issue_time[i]))
                for i in range(num_ops)}
    return None if status == cp_model.INFEASIBLE else "unknown"


def route(data, sched, ii, seconds, want_routes=False):
    """Stage 2: with placement+times fixed, find a modulo-feasible routing of
    every value. Returns (feasible, routes). routes is None unless want_routes;
    otherwise routes[(s,d)] is the concrete [(tile,cycle),...] path per edge."""
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
    # (Architecture.cpp k_num_regs_per_regfile). We enforce it as a necessary
    # condition: every value MOVING INTO a tile takes a write port, every value
    # MOVING OUT takes a read port. (Conservative: an arrival consumed the same
    # cycle need not persist, so this may slightly over-constrain, but it never
    # lets the oracle claim a mapping the backend cannot realize.)
    num_regfiles = {t["id"]: max(1, t.get("regfiles", t["regs"] // 8))
                    for t in arch["tiles"]}
    link_use = collections.defaultdict(list)  # (link_key, residue) -> [bool]
    reg_use = collections.defaultdict(list)   # (tile, residue)     -> [bool]
    write_port_use = collections.defaultdict(list)  # (tile, residue) -> [bool] in
    read_port_use = collections.defaultdict(list)   # (tile, residue) -> [bool] out
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
                    incoming.append(hold_var)
                    reg_use[(tile, (cycle - 1) % ii)].append(hold_var)
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
                                incoming.append(move_var)
                                link_use[((neighbor, dst_tile),
                                          (cycle - latency) % ii)].append(move_var)
                                # A move consumes a write port on the destination
                                # (arrival residue) and a read port on the source
                                # (departure residue).
                                write_port_use[(tile, cycle % ii)].append(move_var)
                                read_port_use[(neighbor,
                                               (cycle - latency) % ii)].append(move_var)
                                arcs_of[producer][(tile, cycle)].append(
                                    ("move", (neighbor, cycle - latency), move_var))
                # present => at least one incoming arc chosen
                model.Add(sum(incoming) >= 1).OnlyEnforceIf(present[(tile, cycle)])
                if not incoming:
                    model.Add(present[(tile, cycle)] == 0)
    # Modulo resources: a link carries one net-move per residue; a tile holds no
    # more values per residue than it has registers.
    for key, bool_vars in link_use.items():
        model.Add(sum(bool_vars) <= 1)          # one net-move per (link, residue)
    for (tile, residue), bool_vars in reg_use.items():
        model.Add(sum(bool_vars) <= max(1, num_regs[tile]))  # storage capacity
    # Regfile port limits (see num_regfiles above): writes-in / reads-out per
    # modulo slot cannot exceed the tile's regfile count.
    for (tile, residue), bool_vars in write_port_use.items():
        model.Add(sum(bool_vars) <= num_regfiles[tile])
    for (tile, residue), bool_vars in read_port_use.items():
        model.Add(sum(bool_vars) <= num_regfiles[tile])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    # Deterministic: single worker + fixed seed (reproducible for tests).
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if not want_routes or not feasible:
        return (feasible, None)

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
    return (feasible, routes)


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
        # exactly this path instead of greedily re-routing, so it reproduces the
        # solver's joint routing even on large kernels.
        out["routes"] = [{"s": src, "d": dst,
                          "path": [[tile, cycle] for (tile, cycle) in routes[(src, dst)]]}
                         for (src, dst) in sorted(routes)]
    json.dump(out, open(path, "w"), indent=1)
    num_routes = len(routes) if routes is not None else 0
    print(f"[emit] wrote real mapping ({len(placements)} ops, {num_routes} routes, "
          f"II={ii}) -> {path}", file=sys.stderr)


def main():
    # Driver: climb II from min to max; the first II whose schedule AND routing
    # are both feasible is the answer. --emit writes that mapping; --fallback and
    # --emit-cap trade a possibly-larger II for a bounded solve time (see below).
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--min-ii", type=int, default=1)
    ap.add_argument("--max-ii", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--emit", default=None,
                    help="Write the concrete placement+schedule mapping to JSON.")
    ap.add_argument("--fallback", action="store_true",
                    help="Unified mode: if the schedule times out (or does not "
                         "route) within the solve budget at this II, back off "
                         "to II+1 and retry, instead of giving up. Yields the "
                         "smallest II for which a real, routable mapping is "
                         "actually found within budget (>= the true minimum).")
    ap.add_argument("--minimize-routing", dest="mr", action="store_true",
                    default=None,
                    help="Bias stage 1 toward a router-friendly placement (mr). "
                         "Without it stage 1 ignores routability, so at a tight "
                         "II it hands stage 2 a placement that cannot be routed "
                         "and the II is rejected -- even though another "
                         "placement at that II would route. Measured: "
                         "relu_tiled_20 on the graph-isomorphic shapes 8x2 and "
                         "2x8 (same tiles, links and mean hops) comes back 3 and "
                         "2, reproducibly and with no timeouts, purely from "
                         "which placement stage 1 happened to return. On by "
                         "default; --no-minimize-routing trades that accuracy "
                         "for speed.")
    ap.add_argument("--no-minimize-routing", dest="mr", action="store_false",
                    help="Skip the router-friendly bias. Faster, and reports an "
                         "II that may be above the optimum for the reason above.")
    ap.add_argument("--emit-cap", type=float, default=12.0,
                    help="When emitting, cap each per-II solve at this many "
                         "seconds. We only need a compact, ROUTABLE witness, "
                         "not a proven-optimal makespan, and the compact "
                         "solution is found early -- the long tail is just the "
                         "solver proving minimality. Keeps emit fast.")
    a = ap.parse_args()
    data = json.load(open(a.json))
    hops = shortest_hops(data["arch"])
    max_ii = a.max_ii or data["arch"]["ctrl_mem_items"]
    # When emitting a mapping for the backend, prefer low-hop (router-friendly)
    # placements so the greedy per-net router can reproduce them. We do NOT need
    # to prove the makespan optimal (the compact witness appears early), so cap
    # the per-solve time to avoid the optimality-proving tail.
    # Unset leaves the historical behaviour exactly as it was -- on for --emit,
    # off otherwise -- so every existing caller and every committed
    # *.exact-mapping.json is reproduced bit for bit. Callers that use this as a
    # VERIFIER should pass --minimize-routing: for them the bias is not a
    # convenience for the backend router, it is what stops a tight II being
    # rejected on the strength of one unroutable placement.
    minimize_routing = bool(a.emit) if a.mr is None else a.mr
    solve_seconds = min(a.seconds, a.emit_cap) if a.emit else a.seconds
    # An op whose FU class no tile provides cannot be placed at ANY II, so say so
    # once instead of proving every II infeasible. Reported with the same
    # "> max_ii" verdict the II sweep uses when nothing fits this shape, which is
    # what the C++ caller already reads as "not mappable at this shape".
    missing = unprovidable_classes(data)
    if missing:
        print(f"[infeasible] no tile provides fu class(es): {', '.join(missing)}",
              file=sys.stderr)
        print(f"{'MAPPED_II' if a.fallback else 'TRUE_MIN_II'} > {max_ii}")
        return
    for ii in range(a.min_ii, max_ii + 1):
        sched = schedule(data, ii, solve_seconds, hops, minimize_routing)
        if sched is None:
            if a.v: print(f"  II={ii}: schedule infeasible", file=sys.stderr)
            continue                            # II provably too small; try II+1
        if sched == "unknown":
            # Solver could not settle this II within the time budget.
            if a.fallback:
                if a.v: print(f"  II={ii}: schedule timeout, backing off to "
                              f"II+1", file=sys.stderr)
                continue
            print(f"TRUE_MIN_II >= {ii} (schedule timeout)"); return
        ok, routes = route(data, sched, ii, solve_seconds,
                           want_routes=bool(a.emit))
        if a.v: print(f"  II={ii}: schedule OK, routing={'OK' if ok else 'FAIL'}",
                      file=sys.stderr)
        if ok:
            # In fallback mode this II may be above the proven minimum, so the
            # tag reflects "found within budget" rather than "proven optimal".
            tag = ("MAPPED_II" if a.fallback else "TRUE_MIN_II")
            print(f"{tag} = {ii} (placement+schedule+routing all feasible)")
            if a.emit:
                emit_mapping(data, sched, ii, a.emit, routes)
            return
        # else: this schedule did not route; try II+1 (conservative).
    print(f"{'MAPPED_II' if a.fallback else 'TRUE_MIN_II'} > {max_ii}")


if __name__ == "__main__":
    main()
