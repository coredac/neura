#!/usr/bin/env python3
"""Exact modulo-scheduling oracle (OR-Tools CP-SAT) for a single Neura task.

Reads the DFG+arch JSON from `mlir-neura-opt --dump-dfg-json` and computes the
MINIMUM II for which an exact modulo schedule exists. CP-SAT is far faster than
z3 on this scheduling problem, so it pins the optimum on kernels z3 could not.

Model (Phase 1 — every constraint is a necessary condition of any real mapping):
  - place[i] in {tiles whose FU set supports op i's class};
  - FU modulo resource: AllDifferent(place[i]*II + residue[i]) — at most one op
    per (tile, residue-mod-II) cell;
  - precedence/recurrence: t[d] >= t[s] + lat[s] + hop(s,d) - w*II, where
    hop = |x[s]-x[d]| + |y[s]-y[d]| (Manhattan, link latency 1), w = 0/1.

Link contention is added in Phase 2 (--with-routing). Phase 1 omits it, so its
optimum is a rigorous LOWER BOUND on the true mapper II (and equals it whenever
contention does not bind — verified true for fir/relu/histogram).

For each candidate II from a computed lower bound upward, a feasibility model is
solved; the first feasible II is the exact minimum for the modeled constraints.

Usage: exact_oracle_cpsat.py dfg.json [--max-ii N] [--per-ii-seconds S] [-v]
"""
import argparse
import json
import math
import sys

from ortools.sat.python import cp_model


def cheap_lower_bound(data):
    """max(resource bound, recurrence op-count bound) to skip trivial IIs."""
    ops = data["ops"]
    fu_tiles = data["arch"]["fu_class_tiles"]
    ntiles = data["arch"]["num_tiles"]
    # Resource: per FU class, ceil(#ops / #supporting tiles).
    from collections import Counter
    cnt = Counter(o["class"] for o in ops)
    res = 1
    for cls, c in cnt.items():
        cap = max(1, len(fu_tiles.get(cls) or list(range(ntiles))))
        res = max(res, -(-c // cap))
    return max(1, res)


def build_and_check(data, ii, seconds, verbose):
    ops = data["ops"]
    edges = data["edges"]
    arch = data["arch"]
    tiles = arch["tiles"]
    tile_ids = [t["id"] for t in tiles]
    idx_of = {t["id"]: k for k, t in enumerate(tiles)}
    xs = [t["x"] for t in tiles]
    ys = [t["y"] for t in tiles]
    fu_tiles = arch["fu_class_tiles"]
    n = len(ops)
    diameter = (max(xs) - min(xs)) + (max(ys) - min(ys))

    # Time horizon: longest forward (intra-iteration) dependence chain, each
    # node charged (latency + mesh diameter) as a safe per-hop upper bound, plus
    # one II. Tighter than 2n when the DAG is shallow -> much smaller domains.
    succ = {}
    indeg = [0] * n
    for e in edges:
        if e["w"] == 0:
            succ.setdefault(e["s"], []).append(e["d"])
            indeg[e["d"]] += 1
    # longest path (nodes) via topological DP
    import collections
    dq = collections.deque([i for i in range(n) if indeg[i] == 0])
    depth = [1] * n
    seen = 0
    ind = indeg[:]
    while dq:
        u = dq.popleft(); seen += 1
        for v in succ.get(u, []):
            depth[v] = max(depth[v], depth[u] + 1)
            ind[v] -= 1
            if ind[v] == 0:
                dq.append(v)
    longest = max(depth) if depth else 1
    if seen < n:  # cycle fallback (shouldn't happen: forward edges are a DAG)
        longest = n
    max_lat = max((o["latency"] for o in ops), default=1)
    Tmax = min(2 * n + 2 * ii + 8,
               longest * (max_lat + diameter) + 2 * ii + 4)

    m = cp_model.CpModel()
    place = []   # index into tiles[] (0..len-1)
    x = []
    y = []
    t = []
    cell = []
    for i, op in enumerate(ops):
        valid_ids = fu_tiles.get(op["class"]) or tile_ids
        valid_idx = [idx_of[tid] for tid in valid_ids]
        p = m.NewIntVarFromDomain(cp_model.Domain.FromValues(valid_idx), f"p{i}")
        xi = m.NewIntVar(0, max(xs), f"x{i}")
        yi = m.NewIntVar(0, max(ys), f"y{i}")
        m.AddElement(p, xs, xi)
        m.AddElement(p, ys, yi)
        ti = m.NewIntVar(0, Tmax, f"t{i}")
        ri = m.NewIntVar(0, ii - 1, f"r{i}")
        m.AddModuloEquality(ri, ti, ii)
        ci = m.NewIntVar(0, len(tiles) * ii, f"c{i}")
        m.Add(ci == p * ii + ri)
        place.append(p); x.append(xi); y.append(yi); t.append(ti)
        cell.append(ci)

    # FU modulo resource: <=1 op per (tile, residue) cell.
    m.AddAllDifferent(cell)

    # Precedence / recurrence with Manhattan hop latency.
    for e in edges:
        s, d, w = e["s"], e["d"], e["w"]
        lat = ops[s]["latency"]
        dx = m.NewIntVar(-max(xs), max(xs), "")
        dy = m.NewIntVar(-max(ys), max(ys), "")
        adx = m.NewIntVar(0, max(xs), "")
        ady = m.NewIntVar(0, max(ys), "")
        m.Add(dx == x[s] - x[d]); m.AddAbsEquality(adx, dx)
        m.Add(dy == y[s] - y[d]); m.AddAbsEquality(ady, dy)
        m.Add(t[d] >= t[s] + lat + adx + ady - w * ii)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 16
    res = solver.Solve(m)
    if verbose:
        name = {cp_model.OPTIMAL: "FEASIBLE", cp_model.FEASIBLE: "FEASIBLE",
                cp_model.INFEASIBLE: "infeasible",
                cp_model.UNKNOWN: "unknown(timeout)"}.get(res, str(res))
        print(f"  II={ii}: {name}", file=sys.stderr)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--max-ii", type=int, default=0)
    ap.add_argument("--min-ii", type=int, default=0)
    ap.add_argument("--per-ii-seconds", type=float, default=30.0)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    data = json.load(open(a.json))
    max_ii = a.max_ii or data["arch"]["ctrl_mem_items"]
    lo = a.min_ii or cheap_lower_bound(data)
    print(f"[cpsat-oracle] {len(data['ops'])} ops, {len(data['edges'])} edges, "
          f"{data['arch']['num_tiles']} tiles, search II in [{lo},{max_ii}]",
          file=sys.stderr)
    for ii in range(lo, max_ii + 1):
        r = build_and_check(data, ii, a.per_ii_seconds, a.v)
        if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"EXACT_MIN_II = {ii}")
            return
        if r == cp_model.UNKNOWN:
            print(f"EXACT_MIN_II >= {ii} (CP-SAT timeout at II={ii})")
            return
    print(f"EXACT_MIN_II > {max_ii}")


if __name__ == "__main__":
    main()
