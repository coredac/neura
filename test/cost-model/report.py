#!/usr/bin/env python3
"""Analyze cost-model validation results (analytical vs mapper oracle).

Reads the harness CSV and reports, over all rows with a valid oracle II:
  - N, exact-match rate, lower-bound-holds rate (predicted <= oracle),
  - MAE and max |predicted - oracle| (overall and within II>=5, II>=10 buckets),
  - dominant-bound histogram,
  - per-row table sorted by oracle II.

Usage: report.py [results.csv]
"""
import csv
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/work/shared/users/sg2682/project/cost-model-work/validation/results.csv"


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        r["oracle"] = to_int(r.get("oracle_ii"))
        r["pred"] = to_int(r.get("analytical_ii"))
        rows.append(r)

valid = [r for r in rows if r["oracle"] is not None and r["pred"] is not None]


def stats(subset, label):
    if not subset:
        print(f"  [{label}] no rows")
        return
    n = len(subset)
    exact = sum(1 for r in subset if r["pred"] == r["oracle"])
    lb = sum(1 for r in subset if r["pred"] <= r["oracle"])
    errs = [abs(r["pred"] - r["oracle"]) for r in subset]
    signed = [r["pred"] - r["oracle"] for r in subset]
    mae = sum(errs) / n
    print(f"  [{label}] N={n}  exact={exact}/{n} ({100*exact/n:.0f}%)  "
          f"lb_holds={lb}/{n} ({100*lb/n:.0f}%)  "
          f"MAE={mae:.2f}  max_abs_err={max(errs)}  "
          f"mean_signed={sum(signed)/n:+.2f}")


print(f"=== cost-model validation report ({path}) ===")
print(f"total rows={len(rows)}  with-oracle={len(valid)}  "
      f"no-oracle(NA/timeout)={len(rows)-len(valid)}")
print()
print("Accuracy (analytical vs mapper oracle):")
stats(valid, "all")
stats([r for r in valid if r["oracle"] >= 5], "oracle_ii>=5")
stats([r for r in valid if r["oracle"] >= 10], "oracle_ii>=10")
print()

dom = Counter(r.get("dominant", "?") for r in valid)
print("Dominant bound histogram:", dict(dom))
print()

print("Per-case (sorted by oracle II):")
hdr = ["kernel", "shape", "oracle", "pred", "res", "rec", "mem", "route",
       "reg", "issue", "dom", "err"]
print("  " + "  ".join(f"{h:>7}" for h in hdr))
for r in sorted(valid, key=lambda r: (-(r["oracle"] or 0), r["kernel"])):
    err = r["pred"] - r["oracle"]
    vals = [r["kernel"][:7], r["shape"], r["oracle"], r["pred"], r["res"],
            r["rec"], r["mem"], r["route"], r["reg"], r["issue"],
            r["dominant"], f"{err:+d}"]
    print("  " + "  ".join(f"{str(v):>7}" for v in vals))

# Cases where the lower bound is violated (should be none if model is sound).
bad = [r for r in valid if r["pred"] > r["oracle"]]
if bad:
    print("\n!!! LOWER-BOUND VIOLATIONS (predicted > oracle) — investigate:")
    for r in bad:
        print(f"    {r['kernel']} {r['shape']}: pred={r['pred']} > "
              f"oracle={r['oracle']} (dom={r['dominant']})")
else:
    print("\nAll predictions are valid lower bounds (predicted <= oracle).")
