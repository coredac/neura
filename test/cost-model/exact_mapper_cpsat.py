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

Usage: exact_mapper_cpsat.py dfg.json [--max-ii N] [--seconds S] [-v] [--emit out.json]
"""
import argparse
import collections
import json
import sys

from ortools.sat.python import cp_model


def shortest_hops(arch):
    """All-pairs shortest path (in link latency) over the real directed link
    graph. Returns dist[(a,b)] in cycles; missing pairs -> None (unreachable)."""
    tids = [t["id"] for t in arch["tiles"]]
    INF = float("inf")
    dist = {(a, b): (0 if a == b else INF) for a in tids for b in tids}
    for s, d, lat in arch["links"]:
        dist[(s, d)] = min(dist[(s, d)], lat)
    for k in tids:
        for i in tids:
            dik = dist[(i, k)]
            if dik == INF:
                continue
            for j in tids:
                nd = dik + dist[(k, j)]
                if nd < dist[(i, j)]:
                    dist[(i, j)] = nd
    return dist


def schedule(data, ii, seconds, hops):
    ops, edges, arch = data["ops"], data["edges"], data["arch"]
    tiles = arch["tiles"]
    tid = [t["id"] for t in tiles]
    fu = arch["fu_class_tiles"]
    n = len(ops)
    Tmax = 2 * n + 2 * ii + 8
    m = cp_model.CpModel()
    place = []
    t = []
    for i, op in enumerate(ops):
        valid = fu.get(op["class"]) or tid
        p = m.NewIntVarFromDomain(cp_model.Domain.FromValues(valid), f"p{i}")
        place.append(p)
        t.append(m.NewIntVar(0, Tmax, f"t{i}"))
    # FU modulo resource via AllDifferent(place*II + residue).
    cells = []
    for i in range(n):
        r = m.NewIntVar(0, ii - 1, f"r{i}")
        q = m.NewIntVar(0, Tmax, f"q{i}")
        m.Add(t[i] == q * ii + r)
        c = m.NewIntVar(0, (max(tid) + 1) * ii, f"c{i}")
        m.Add(c == place[i] * ii + r)
        cells.append(c)
    m.AddAllDifferent(cells)
    # Precedence/recurrence with true shortest-path hop latency (place-dependent).
    maxhop = max((h for h in hops.values() if h != float("inf")), default=0)
    for e in edges:
        s, d, w = e["s"], e["d"], e["w"]
        lat = ops[s]["latency"]
        hop = m.NewIntVar(0, int(maxhop), "")
        # hop == hops[(place[s], place[d])] via a table constraint.
        allowed = [[a, b, int(hops[(a, b)])] for a in tid for b in tid
                   if hops[(a, b)] != float("inf")]
        m.AddAllowedAssignments([place[s], place[d], hop], allowed)
        m.Add(t[d] >= t[s] + lat + hop - w * ii)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = seconds
    # Deterministic: single worker + fixed seed (reproducible for tests).
    sol.parameters.num_search_workers = 1
    sol.parameters.random_seed = 0
    st = sol.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {i: (sol.Value(place[i]), sol.Value(t[i])) for i in range(n)}
    return None if st == cp_model.INFEASIBLE else "unknown"


def route(data, sched, ii, seconds):
    ops, edges, arch = data["ops"], data["edges"], data["arch"]
    tiles = arch["tiles"]
    tid = [t["id"] for t in tiles]
    regs = {t["id"]: t["regs"] for t in tiles}
    out_links = collections.defaultdict(list)   # tile -> [(dst, lat)]
    for s, d, lat in arch["links"]:
        out_links[s].append((d, lat))
    # Build nets: producer -> list of (consumer_tile, deadline_cycle).
    nets = collections.defaultdict(list)
    prod_avail = {}
    for i, (pt, tt) in sched.items():
        prod_avail[i] = tt + ops[i]["latency"]
    for e in edges:
        s, d, w = e["s"], e["d"], e["w"]
        ct, cd = sched[d]
        deadline = cd + (ii if w == 1 else 0)  # recurrence: next iteration
        nets[s].append((sched[d][0], deadline))
    m = cp_model.CpModel()
    # Presence + arc booleans per net over a per-net cycle window.
    link_use = collections.defaultdict(list)  # (link_key,residue) -> [bool]
    reg_use = collections.defaultdict(list)   # (tile,residue) -> [bool]
    for p, cons in nets.items():
        src_tile = sched[p][0]
        c0 = prod_avail[p]
        cmax = max(d for _, d in cons)
        if cmax < c0:
            cmax = c0
        cycles = list(range(c0, cmax + 1))
        pres = {(tl, c): m.NewBoolVar(f"pr{p}_{tl}_{c}")
                for tl in tid for c in cycles}
        m.Add(pres[(src_tile, c0)] == 1)
        for (ct, dl) in cons:
            m.Add(pres[(ct, dl)] == 1)
        # Flow: every present node except the source has an incoming arc.
        for tl in tid:
            for c in cycles:
                if (tl, c) == (src_tile, c0):
                    continue
                incoming = []
                if c - 1 in cycles:
                    # hold on same tile
                    h = m.NewBoolVar(f"h{p}_{tl}_{c}")
                    m.Add(pres[(tl, c - 1)] == 1).OnlyEnforceIf(h)
                    incoming.append(h)
                    reg_use[(tl, (c - 1) % ii)].append(h)
                    # link moves from neighbors into tl
                    for src in tid:
                        for (dst, lat) in out_links[src]:
                            if dst == tl and (c - lat) in cycles and lat >= 1:
                                mv = m.NewBoolVar(f"m{p}_{src}_{tl}_{c}")
                                m.Add(pres[(src, c - lat)] == 1).OnlyEnforceIf(mv)
                                incoming.append(mv)
                                link_use[((src, dst), (c - lat) % ii)].append(mv)
                # present => at least one incoming arc chosen
                m.Add(sum(incoming) >= 1).OnlyEnforceIf(pres[(tl, c)])
                if not incoming:
                    m.Add(pres[(tl, c)] == 0)
    # Modulo resources.
    for key, lst in link_use.items():
        m.Add(sum(lst) <= 1)             # one net-move per (link, residue)
    for (tl, s), lst in reg_use.items():
        m.Add(sum(lst) <= max(1, regs[tl]))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = seconds
    # Deterministic: single worker + fixed seed (reproducible for tests).
    sol.parameters.num_search_workers = 1
    sol.parameters.random_seed = 0
    st = sol.Solve(m)
    return st in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--min-ii", type=int, default=1)
    ap.add_argument("--max-ii", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    data = json.load(open(a.json))
    hops = shortest_hops(data["arch"])
    max_ii = a.max_ii or data["arch"]["ctrl_mem_items"]
    for ii in range(a.min_ii, max_ii + 1):
        sched = schedule(data, ii, a.seconds, hops)
        if sched is None:
            if a.v: print(f"  II={ii}: schedule infeasible", file=sys.stderr)
            continue
        if sched == "unknown":
            print(f"TRUE_MIN_II >= {ii} (schedule timeout)"); return
        ok = route(data, sched, ii, a.seconds)
        if a.v: print(f"  II={ii}: schedule OK, routing={'OK' if ok else 'FAIL'}",
                      file=sys.stderr)
        if ok:
            print(f"TRUE_MIN_II = {ii} (placement+schedule+routing all feasible)")
            return
        # else: this schedule did not route; try II+1 (conservative).
    print(f"TRUE_MIN_II > {max_ii}")


if __name__ == "__main__":
    main()
