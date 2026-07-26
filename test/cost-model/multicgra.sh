#!/usr/bin/env bash
# Predicts per-task II on the multi-CGRA taskflow tests with the analytical cost
# model, and (where the mapper converges) compares against the compiled oracle.
set -u
NEURA=/work/shared/users/sg2682/project/neura
OPT=$NEURA/build/tools/mlir-neura-opt/mlir-neura-opt
W=/work/shared/users/sg2682/project/cost-model-work/multicgra; mkdir -p "$W"

PRE="--affine-loop-tree-serialization --convert-affine-to-taskflow --memory-access-streaming-fusion"
MID="--affine-loop-tree-serialization --affine-loop-perfection --construct-hyperblock-from-task --classify-task-and-counter --convert-taskflow-to-neura --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --assign-accelerator --lower-memref-to-neura --lower-arith-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --promote-input-arg-to-const --fold-constant --canonicalize-return --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant"

# name|src|arch
CASES=(
  "multi-nested|test/multi-cgra/taskflow/multi-nested/multi-nested.mlir|test/arch_spec/architecture_with_counter.yaml"
  "parallel-nested|test/multi-cgra/taskflow/parallel-nested/parallel-nested.mlir|test/arch_spec/architecture_with_counter.yaml"
  "irregular-loop|test/multi-cgra/taskflow/irregular-loop/irregular-loop.mlir|test/arch_spec/architecture.yaml"
  "resnet|test/multi-cgra/taskflow/resnet/simple_resnet_tosa.mlir|test/arch_spec/architecture_4x4.yaml"
)

per_task_ii() { grep -oE "compiled_ii = [0-9]+ : i32" "$1" 2>/dev/null | grep -oE "[0-9]+" | tr '\n' ' '; }

for entry in "${CASES[@]}"; do
  IFS='|' read -r name src arch <<< "$entry"
  d=$W/$name; mkdir -p "$d"; A=$NEURA/$arch
  $OPT $NEURA/$src $PRE -o "$d/stream.mlir" 2>"$d/pre.err" || { echo "$name PRE FAIL"; continue; }
  # Analytical (fast, no mapper).
  $OPT "$d/stream.mlir" $MID "--resource-aware-task-optimization=estimation-mode=cost-model-analytical" \
      --architecture-spec=$A -o "$d/ana.mlir" 2>"$d/ana.err"
  ana=$(per_task_ii "$d/ana.mlir")
  # Oracle (mapper per task; may be slow -> timeout).
  timeout "${MAP_TIMEOUT:-400}" $OPT "$d/stream.mlir" $MID "--resource-aware-task-optimization=estimation-mode=compiled" \
      --architecture-spec=$A -o "$d/map.mlir" >/dev/null 2>"$d/map.err"
  rc=$?
  oracle=$(per_task_ii "$d/map.mlir"); [ $rc -eq 124 ] && oracle="(timeout)"
  echo "[$name] tasks: analytical_ii=[ $ana] | oracle_ii=[ ${oracle:-NA}]"
done
echo "=== multicgra done ==="
