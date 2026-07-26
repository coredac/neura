#!/usr/bin/env bash
# Validates the analytical model on CONSTRUCTED kernels that each saturate a
# specific resource, so bounds other than the recurrence are exercised. Each is
# small enough for the heuristic mapper to converge, giving a clean oracle.
set -u
NEURA=/work/shared/users/sg2682/project/neura
LLVMBIN=/work/shared/users/sg2682/project/llvm-project/build/bin
OPT=$NEURA/build/tools/mlir-neura-opt/mlir-neura-opt
KDIR=$NEURA/test/cost-model/kernels
W=/work/shared/users/sg2682/project/cost-model-work/constructed
mkdir -p "$W"
LOWER_PASSES="--assign-accelerator --lower-llvm-to-neura --promote-input-arg-to-const --fold-constant --canonicalize-return --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov"

# name|source.c|arch.yaml|expected-dominant-bound
CASES=(
  "mul|mul_kernel.c|$NEURA/test/cost-model/arch/scarce_mul.yaml|res"
  "mem|mem_kernel.c|$NEURA/test/arch_spec/architecture.yaml|mem-or-res"
  "chain|chain_kernel.c|$NEURA/test/arch_spec/architecture.yaml|rec"
)
grepnum() { grep -oE "$2 = [0-9]+" "$1" 2>/dev/null | head -1 | grep -oE "[0-9]+"; }

echo "=== Constructed-case validation ==="
for entry in "${CASES[@]}"; do
  IFS='|' read -r name src arch expect <<< "$entry"
  d=$W/$name; mkdir -p "$d"
  $LLVMBIN/clang -S -emit-llvm -O1 -fno-vectorize -fno-unroll-loops -std=c11 \
      -o "$d/full.ll" "$KDIR/$src" 2>"$d/clang.err" || { echo "$name clang FAIL"; continue; }
  $LLVMBIN/llvm-extract --rfunc=".*kernel.*" "$d/full.ll" -o "$d/only.ll" 2>>"$d/clang.err" || { echo "$name extract FAIL"; continue; }
  $LLVMBIN/mlir-translate --import-llvm "$d/only.ll" -o "$d/k.mlir" 2>>"$d/clang.err" || { echo "$name translate FAIL"; continue; }
  $OPT "$d/k.mlir" $LOWER_PASSES --architecture-spec=$arch -o "$d/premap.mlir" 2>"$d/lower.err" || { echo "$name lower FAIL"; continue; }

  $OPT "$d/premap.mlir" --cost-model-analytical --architecture-spec=$arch -o "$d/ana.mlir" 2>"$d/ana.err"
  ana=$(grepnum "$d/ana.mlir" analytical_ii); res=$(grepnum "$d/ana.mlir" res_mii)
  rec=$(grepnum "$d/ana.mlir" rec_mii); mem=$(grepnum "$d/ana.mlir" mem_mii)
  issue=$(grepnum "$d/ana.mlir" issue_mii)
  dom=$(grep -oE 'dominant = "[a-z]+"' "$d/ana.mlir" | head -1 | grep -oE '"[a-z]+"' | tr -d '"')

  timeout 200 $OPT "$d/premap.mlir" --map-to-accelerator="mapping-strategy=heuristic dump-mapping-table=false" \
      --architecture-spec=$arch -o "$d/map.mlir" >/dev/null 2>"$d/map.err"
  oracle=$(grepnum "$d/map.mlir" compiled_ii); [ -z "$oracle" ] && oracle=NA

  echo "[$name] arch=$(basename $arch) expect=$expect | oracle=$oracle analytical=$ana (res=$res rec=$rec mem=$mem issue=$issue dom=$dom)"
done
