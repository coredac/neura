#!/usr/bin/env bash
# Validation harness: for each kernel + CGRA shape, lower to pre-map Neura IR,
# then run BOTH the analytical model (--cost-model-analytical) and the mapper
# oracle (--map-to-accelerator) and record predicted vs oracle II + every bound.
# The mapper is used ONLY as an offline oracle; the model never calls it.
set -u
NEURA=/work/shared/users/sg2682/project/neura
LLVMBIN=/work/shared/users/sg2682/project/llvm-project/build/bin
OPT=$NEURA/build/tools/mlir-neura-opt/mlir-neura-opt
ARCH=${ARCH_OVERRIDE:-$NEURA/test/arch_spec/architecture.yaml}
KROOT=$NEURA/test/benchmark/CGRA-Bench/kernels
W=/work/shared/users/sg2682/project/cost-model-work/validation
mkdir -p "$W"
CSV=$W/results.csv
HDR="kernel,shape,oracle_ii,analytical_ii,res,rec,mem,route,reg,issue,dominant,mapper_res,mapper_rec,lb_holds,abs_err"
# Fresh file unless doing an ONLY top-up run (then append to keep prior rows).
if [ -z "${ONLY:-}" ] || [ ! -f "$CSV" ]; then echo "$HDR" > "$CSV"; fi

LOWER_PASSES="--assign-accelerator --lower-llvm-to-neura --promote-input-arg-to-const --fold-constant --canonicalize-return --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov"

# kernel|source (rel to KROOT)|extra clang flags. Compiler (clang/clang++) and
# -std are auto-selected from the .c/.cpp extension.
KERNELS=(
  "gemm|gemm/gemm_int.c|"
  "bicg|bicg/bicg_int.c|-I $KROOT/bicg -DSMALL_DATASET"
  "fft|fft/fft_int.c|"
  "fir|fir/fir_int.cpp|"
  "histogram|histogram/histogram_int.cpp|"
  "relu|relu/relu_int.cpp|"
  "spmv|spmv/spmv.c|"
)
# shape name|x-tiles|y-tiles  (0 0 = global singleton = 4x4 single CGRA)
# Multi-shape can be enabled via SHAPES_OVERRIDE; default is single CGRA because
# the heuristic mapper backtracks pathologically on larger arrays (minutes/run).
SHAPES_DEFAULT="4x4|0|0"
IFS=';' read -r -a SHAPES <<< "${SHAPES_OVERRIDE:-$SHAPES_DEFAULT}"

grepnum() { grep -oE "$2 = [0-9]+" "$1" 2>/dev/null | head -1 | grep -oE "[0-9]+"; }

for entry in "${KERNELS[@]}"; do
  IFS='|' read -r k src flags <<< "$entry"
  # Optional ONLY="k1 k2" filter to run a subset (top-up runs).
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $k "* ]]; then continue; fi
  d=$W/$k; mkdir -p "$d"
  echo "### lowering $k"
  case "$src" in
    *.cpp) CC="$LLVMBIN/clang++"; STD="-std=c++17";;
    *)     CC="$LLVMBIN/clang";   STD="-std=c11";;
  esac
  $CC -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops $STD $flags \
      -o "$d/full.ll" "$KROOT/$src" 2>"$d/clang.err" || { echo "  clang FAIL"; continue; }
  $LLVMBIN/llvm-extract --rfunc=".*kernel.*" "$d/full.ll" -o "$d/only.ll" 2>>"$d/clang.err" || { echo "  extract FAIL"; continue; }
  $LLVMBIN/mlir-translate --import-llvm "$d/only.ll" -o "$d/k.mlir" 2>>"$d/clang.err" || { echo "  translate FAIL"; continue; }
  $OPT "$d/k.mlir" $LOWER_PASSES --architecture-spec=$ARCH -o "$d/premap.mlir" 2>"$d/lower.err" || { echo "  lower FAIL"; continue; }

  for s in "${SHAPES[@]}"; do
    IFS='|' read -r shape x y <<< "$s"
    topt=""; [ "$x" -gt 0 ] && topt="x-tiles=$x y-tiles=$y"
    # Analytical model (never fails).
    $OPT "$d/premap.mlir" --cost-model-analytical="$topt" --architecture-spec=$ARCH \
        -o "$d/ana_$shape.mlir" 2>"$d/ana_$shape.err"
    ana=$(grepnum "$d/ana_$shape.mlir" "analytical_ii")
    res=$(grepnum "$d/ana_$shape.mlir" "res_mii")
    rec=$(grepnum "$d/ana_$shape.mlir" "rec_mii")
    mem=$(grepnum "$d/ana_$shape.mlir" "mem_mii")
    route=$(grepnum "$d/ana_$shape.mlir" "route_mii")
    reg=$(grepnum "$d/ana_$shape.mlir" "reg_mii")
    issue=$(grepnum "$d/ana_$shape.mlir" "issue_mii")
    dom=$(grep -oE 'dominant = "[a-z]+"' "$d/ana_$shape.mlir" | head -1 | grep -oE '"[a-z]+"' | tr -d '"')

    # Mapper oracle (may fail / timeout -> record NA). Debug dump suppressed;
    # the mapper's [DEBUG] chatter on stdout is discarded.
    timeout "${MAP_TIMEOUT:-150}" $OPT "$d/premap.mlir" \
        --map-to-accelerator="mapping-strategy=heuristic dump-mapping-table=false $topt" \
        --architecture-spec=$ARCH -o "$d/map_$shape.mlir" \
        >/dev/null 2>"$d/map_$shape.err"
    oracle=$(grepnum "$d/map_$shape.mlir" "compiled_ii")
    m_res=$(grepnum "$d/map_$shape.err" "res_mii"); [ -z "$m_res" ] && m_res=$(grep -oE "Total operations: [0-9]+" "$d/map_$shape.err" | head -1 | grep -oE "[0-9]+")
    m_rec=$(grep -oE "Calculated Recurrence MII: [0-9]+" "$d/map_$shape.err" | tail -1 | grep -oE "[0-9]+")

    [ -z "$ana" ] && ana=NA
    [ -z "$oracle" ] && oracle=NA
    lb=NA; err=NA
    if [ "$ana" != NA ] && [ "$oracle" != NA ]; then
      if [ "$ana" -le "$oracle" ]; then lb=1; else lb=0; fi
      err=$(( ana > oracle ? ana - oracle : oracle - ana ))
    fi
    echo "$k,$shape,$oracle,$ana,$res,$rec,$mem,$route,$reg,$issue,$dom,$m_res,$m_rec,$lb,$err" >> "$CSV"
    echo "  [$k $shape] oracle=$oracle analytical=$ana (res=$res rec=$rec mem=$mem route=$route reg=$reg issue=$issue dom=$dom) lb_holds=$lb"
  done
done
echo "=== CSV written: $CSV ==="
