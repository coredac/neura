// Compiles a GEMM kernel to LLVM IR, imports to MLIR, then lowers via Neura.
// RUN: clang -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c11 \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/gemm/gemm_int.c
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir
//
// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML-GEP2D

// The GEP for A[i][k] (dfg_id 96) steps over rows of NK = 1200 ints. Tile
// order in the YAML depends on placement, so the check anchors on the id.
// YAML-GEP2D:      - opcode: "GEP_2D"
// YAML-GEP2D:      {{^ +}}id: 96{{$}}
// YAML-GEP2D-NEXT:   time_step: {{[0-9]+}}
// YAML-GEP2D-NEXT:   invalid_iterations: {{[0-9]+}}
// YAML-GEP2D-NEXT:   stride: 1200
//
// MAPPING:   func.func @kernel_int(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 17 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 13 : i32, res_mii = 6 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %1 = "neura.constant"() <{value = "%arg0"}> {dfg_id = 1 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"() <{constant_value = "%arg1"}> {dfg_id = 2 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.constant"() <{value = "%arg1"}> {dfg_id = 3 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %4 = "neura.grant_once"() <{constant_value = "%arg2"}> {dfg_id = 4 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %5 = "neura.constant"() <{value = "%arg2"}> {dfg_id = 5 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %6 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 6 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%1) {dfg_id = 28 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.icmp"(%7) <{cmpType = "sgt"}> {dfg_id = 34 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %9 = "neura.data_mov"(%5) {dfg_id = 32 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %10 = "neura.icmp"(%9) <{cmpType = "sgt"}> {dfg_id = 36 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%8) {dfg_id = 37 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%10) {dfg_id = 39 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %13 = "neura.and"(%11, %12) {dfg_id = 40 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%3) {dfg_id = 30 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %15 = "neura.icmp"(%14) <{cmpType = "sgt"}> {dfg_id = 35 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%13) {dfg_id = 41 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%15) {dfg_id = 38 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %18 = "neura.and"(%16, %17) {dfg_id = 42 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%18) {dfg_id = 44 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %20 = "neura.grant_once"(%19) {dfg_id = 46 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %21 = "neura.data_mov"(%18) {dfg_id = 43 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 1 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 1 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 1 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 1 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 1 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 1 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 1 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 1 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 1 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 1 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 1 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %22 = "neura.not"(%21) {dfg_id = 45 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 15 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%22) {dfg_id = 47 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %24 = "neura.grant_once"(%23) {dfg_id = 53 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%0) {dfg_id = 27 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%20) {dfg_id = 52 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 8 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}, {id = 8 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}, {id = 8 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 7 : i32}, {id = 8 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %27 = neura.grant_predicate %25, %26 {dfg_id = 58 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%4) {dfg_id = 31 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%20) {dfg_id = 51 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 14 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 20 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = neura.grant_predicate %28, %29 {dfg_id = 57 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%2) {dfg_id = 29 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%20) {dfg_id = 50 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %33 = neura.grant_predicate %31, %32 {dfg_id = 56 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%6) {dfg_id = 33 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%20) {dfg_id = 49 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %36 = neura.grant_predicate %34, %35 {dfg_id = 55 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%20) {dfg_id = 48 : i32, mapping_locs = [{id = 4010 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 4 : i32}, {id = 4010 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 5 : i32}, {id = 4010 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 6 : i32}, {id = 4010 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 7 : i32}, {id = 4010 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 8 : i32}, {id = 4010 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 9 : i32}, {id = 4010 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 10 : i32}, {id = 4010 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 11 : i32}, {id = 4010 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 12 : i32}, {id = 4010 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 13 : i32}, {id = 4010 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 14 : i32}, {id = 4010 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %38 = "neura.not"(%37) {dfg_id = 54 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%24) {dfg_id = 59 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%38) {dfg_id = 60 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %41 = neura.grant_predicate %39, %40 {dfg_id = 66 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 17 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%27) {dfg_id = 65 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %43 = neura.zext %42 {dfg_id = 71 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %44 = "neura.data_mov"(%30) {dfg_id = 64 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %45 = neura.zext %44 {dfg_id = 70 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %46 = "neura.data_mov"(%33) {dfg_id = 63 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %47 = neura.zext %46 {dfg_id = 69 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %48 = neura.reserve {dfg_id = 7 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %49 = "neura.data_mov"(%43) {dfg_id = 78 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %50 = neura.phi_start %49, %48 {dfg_id = 84 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %51 = neura.reserve {dfg_id = 8 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %52 = "neura.data_mov"(%45) {dfg_id = 77 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %53 = neura.phi_start %52, %51 {dfg_id = 83 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %54 = neura.reserve {dfg_id = 9 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %55 = "neura.data_mov"(%47) {dfg_id = 76 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %56 = neura.phi_start %55, %54 {dfg_id = 82 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %57 = neura.reserve {dfg_id = 10 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %58 = "neura.data_mov"(%36) {dfg_id = 62 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %59 = neura.phi_start %58, %57 {dfg_id = 68 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %60 = neura.reserve {dfg_id = 11 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %61 = "neura.data_mov"(%36) {dfg_id = 61 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 14 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 192 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 192 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 192 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %62 = neura.phi_start %61, %60 {dfg_id = 67 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %63 = neura.reserve {dfg_id = 12 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %64 = "neura.data_mov"(%50) {dfg_id = 93 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %65 = neura.phi_start %64, %63 {dfg_id = 101 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %66 = neura.reserve {dfg_id = 13 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %67 = "neura.data_mov"(%53) {dfg_id = 92 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %68 = neura.phi_start %67, %66 {dfg_id = 100 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %69 = neura.reserve {dfg_id = 14 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %70 = "neura.data_mov"(%56) {dfg_id = 91 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %71 = neura.phi_start %70, %69 {dfg_id = 99 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %72 = neura.reserve {dfg_id = 15 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %73 = "neura.data_mov"(%59) {dfg_id = 75 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %74 = neura.phi_start %73, %72 {dfg_id = 81 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %75 = neura.reserve {dfg_id = 16 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %76 = "neura.data_mov"(%62) {dfg_id = 73 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %77 = neura.phi_start %76, %75 {dfg_id = 79 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %78 = neura.reserve {dfg_id = 17 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %79 = "neura.data_mov"(%59) {dfg_id = 74 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 4000 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 4000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %80 = neura.phi_start %79, %78 {dfg_id = 80 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %81 = "neura.data_mov"(%77) {dfg_id = 86 : i32, mapping_locs = [{id = 208 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 11 : i32}, {id = 208 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 12 : i32}, {id = 208 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %82 = "neura.data_mov"(%80) {dfg_id = 88 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 176 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 10 : i32}, {id = 14 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 201 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 12 : i32}, {id = 201 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %83 = "neura.gep"(%81, %82) <{operandSegmentSizes = array<i32: 0, 2>, strides = [1200, 1]}> {dfg_id = 96 : i32, lhs_value = "%arg4", mapping_locs = [{id = 6 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 14 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %84 = neura.reserve {dfg_id = 18 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %85 = "neura.data_mov"(%74) {dfg_id = 90 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 193 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 193 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 193 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 193 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 193 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 193 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 193 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 193 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %86 = neura.phi_start %85, %84 {dfg_id = 98 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %87 = neura.reserve {dfg_id = 19 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %88 = "neura.data_mov"(%65) {dfg_id = 116 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %89 = neura.phi_start %88, %87 {dfg_id = 123 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %90 = neura.reserve {dfg_id = 20 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %91 = "neura.data_mov"(%68) {dfg_id = 115 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %92 = neura.phi_start %91, %90 {dfg_id = 122 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %93 = neura.reserve {dfg_id = 21 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %94 = "neura.data_mov"(%71) {dfg_id = 114 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %95 = neura.phi_start %94, %93 {dfg_id = 121 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %96 = neura.reserve {dfg_id = 22 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %97 = "neura.data_mov"(%77) {dfg_id = 85 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %98 = neura.phi_start %97, %96 {dfg_id = 94 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %99 = neura.reserve {dfg_id = 23 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %100 = "neura.data_mov"(%80) {dfg_id = 87 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>

// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 17
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 66
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 184
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"

// ASM:      # Compiled II: 17
// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [EAST, RED] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [$8] -> [NORTH, RED] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [NORTH, RED] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [NORTH, RED] (t=4, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$8] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [arg0] -> [$0] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$8] -> [$0] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   ZEXT, [$0] -> [NORTH, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$8] (t=13, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=13)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$2] (t=14, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=14)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$1] -> [$0] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [$0] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)