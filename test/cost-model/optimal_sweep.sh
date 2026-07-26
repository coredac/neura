#!/usr/bin/env bash
# For each kernel: dump the pre-map DFG+arch (--dump-dfg-json) and compute the
# exact minimum II with the CP-SAT oracle. Emits a CSV of exact optimal II.
set -u
NEURA=/work/shared/users/sg2682/project/neura
OPT=$NEURA/build/tools/mlir-neura-opt/mlir-neura-opt
VEXO=/work/shared/users/sg2682/project/pim-compilation/.venv-exo/bin/python
ORA=$NEURA/test/cost-model/exact_oracle_cpsat.py
W=/work/shared/users/sg2682/project/cost-model-work
CM=$W/cmwork; mkdir -p "$CM"
SECS="${PER_II_SECONDS:-40}"
CSV=$CM/optimal_ii.csv
echo "kernel,ops,edges,exact_min_ii" > "$CSV"

# name|premap.mlir|arch.yaml
CASES=(
  "fir|$W/validation/fir/premap.mlir|$NEURA/test/arch_spec/architecture.yaml"
  "relu|$W/validation/relu/premap.mlir|$NEURA/test/arch_spec/architecture.yaml"
  "histogram|$W/validation/histogram/premap.mlir|$NEURA/test/arch_spec/architecture.yaml"
  "bicg|$W/validation/bicg/premap.mlir|$NEURA/test/arch_spec/architecture.yaml"
  "gemm|$W/validation/gemm/premap.mlir|$NEURA/test/arch_spec/architecture.yaml"
  "fft|$W/validation/fft/premap.mlir|$NEURA/test/arch_spec/architecture.yaml"
  "spmv|$W/validation/spmv/premap.mlir|$NEURA/test/arch_spec/architecture.yaml"
  "mul3|$W/diag/mul3p.mlir|$NEURA/test/cost-model/arch/scarce_mul.yaml"
)

for entry in "${CASES[@]}"; do
  IFS='|' read -r name premap arch <<< "$entry"
  [ -f "$premap" ] || { echo "$name: no premap"; continue; }
  $OPT "$premap" --dump-dfg-json --architecture-spec="$arch" -o /dev/null 2>/dev/null > "$CM/$name.json"
  read no ne < <(python3 -c "import json;d=json.load(open('$CM/$name.json'));print(len(d['ops']),len(d['edges']))")
  r=$(timeout $((SECS*8)) $VEXO $ORA "$CM/$name.json" --per-ii-seconds "$SECS" 2>/dev/null | grep EXACT | sed 's/.*= *//;s/EXACT_MIN_II *//')
  echo "$name,$no,$ne,$r" | tee -a "$CSV"
done
echo "=== optimal sweep done: $CSV ==="
