#!/usr/bin/env python3
"""Exact modulo-scheduling oracle (z3) for a single Neura task.

Reads the DFG+arch JSON produced by `mlir-neura-opt --dump-dfg-json` and finds
the minimum II for which an exact modulo schedule exists under these *necessary*
constraints (every one is a hard requirement of any real CGRA mapping):

  - placement: each op sits on a tile whose FU set supports its class;
  - FU modulo resource: at most one op per tile per residue class (mod II);
  - precedence: for edge s->d, time[d] >= time[s] + lat[s] + hop(s,d) - w*II,
    where hop(s,d) = Manhattan distance between the two tiles (link latency 1)
    and w is the loop-carried iteration distance (0 forward, 1 recurrence);
  - recurrence is handled by the w*II term (modulo constraint).

Link *contention* (two moves sharing a link in the same slot) is NOT modelled,
so the result is a rigorous LOWER BOUND on the true mapper II — but it is far
tighter than the aggregate analytical bounds because it enforces real
schedulability + placement + hop latency + recurrence. Combined with the
heuristic mapper (an upper bound), it brackets the true II; where the two meet,
the true II is pinned exactly.

Usage: exact_oracle.py dfg.json [--max-ii N] [--per-ii-timeout-ms MS] [-v]
"""
import argparse
import json
import math
import sys

from z3 import Solver, Int, Or, And, Not, Abs, sat, unsat


def solve_ii(data, ii, timeout_ms, verbose):
    ops = data["ops"]
    edges = data["edges"]
    arch = data["arch"]
    tiles = arch["tiles"]
    coord = {t["id"]: (t["x"], t["y"]) for t in tiles}
    fu_tiles = arch["fu_class_tiles"]
    all_ids = [t["id"] for t in tiles]

    n = len(ops)
    # Absolute time horizon: DAG depth + hop slack (times need not exceed this).
    Tmax = 2 * n + 2 * ii + 4

    s = Solver()
    s.set("timeout", timeout_ms)

    x = [Int(f"x{i}") for i in range(n)]
    y = [Int(f"y{i}") for i in range(n)]
    t = [Int(f"t{i}") for i in range(n)]

    for i, op in enumerate(ops):
        cls = op["class"]
        valid = fu_tiles.get(cls) or all_ids  # unknown class -> any tile
        # (x[i],y[i]) must equal some valid tile's coordinates.
        s.add(Or([And(x[i] == coord[tid][0], y[i] == coord[tid][1])
                  for tid in valid]))
        s.add(t[i] >= 0, t[i] <= Tmax)

    # FU modulo resource: at most one op per tile per residue class (mod II).
    # Explicit bounded residue r[i] gives z3's LIA solver useful structure
    # (proves the infeasibility lower bounds faster than a derived mod term).
    r = [Int(f"r{i}") for i in range(n)]
    q = [Int(f"q{i}") for i in range(n)]
    for i in range(n):
        s.add(q[i] >= 0, r[i] >= 0, r[i] < ii, t[i] == q[i] * ii + r[i])
    for i in range(n):
        for j in range(i + 1, n):
            s.add(Not(And(x[i] == x[j], y[i] == y[j], r[i] == r[j])))

    # Precedence / recurrence with Manhattan hop latency.
    for e in edges:
        si, di, w = e["s"], e["d"], e["w"]
        lat = ops[si]["latency"]
        hop = Abs(x[si] - x[di]) + Abs(y[si] - y[di])
        s.add(t[di] >= t[si] + lat + hop - w * ii)

    res = s.check()
    if verbose:
        print(f"  II={ii}: {res}", file=sys.stderr)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--max-ii", type=int, default=0,
                    help="0 = use arch ctrl_mem_items")
    ap.add_argument("--min-ii", type=int, default=1)
    ap.add_argument("--per-ii-timeout-ms", type=int, default=20000)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()

    with open(a.json) as f:
        data = json.load(f)
    max_ii = a.max_ii or data["arch"]["ctrl_mem_items"]

    # A cheap starting lower bound: recurrence + resource, to skip tiny IIs.
    n = len(data["ops"])
    print(f"[exact-oracle] {n} ops, {len(data['edges'])} edges, "
          f"{data['arch']['num_tiles']} tiles, searching II in "
          f"[{a.min_ii}, {max_ii}]", file=sys.stderr)

    for ii in range(a.min_ii, max_ii + 1):
        res = solve_ii(data, ii, a.per_ii_timeout_ms, a.v)
        if res == sat:
            print(f"EXACT_MIN_II(lower-bound) = {ii}")
            return
        if res != unsat:  # unknown (timeout)
            print(f"EXACT_MIN_II = >= {ii} (z3 timeout at II={ii}; "
                  f"inconclusive above)")
            return
    print(f"EXACT_MIN_II = >{max_ii} (infeasible up to ctrl_mem_items)")


if __name__ == "__main__":
    main()
