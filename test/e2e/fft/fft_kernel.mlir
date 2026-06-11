// Compiles the original C FFT kernel to LLVM IR, imports to MLIR, then lowers via Neura.
// RUN: clang -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c11 \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/fft/fft_int.c
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

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

// MAPPING:       func.func @kernel(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 20 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 16 : i32, res_mii = 10 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i32}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = 128 : i32}> {dfg_id = 1 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"() <{constant_value = 1 : i32}> {dfg_id = 2 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 3 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = neura.reserve {dfg_id = 4 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %5 = "neura.data_mov"(%3) {dfg_id = 39 : i32, mapping_locs = [{id = 3000 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = neura.phi_start %5, %4 {dfg_id = 44 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = neura.reserve {dfg_id = 5 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.data_mov"(%0) {dfg_id = 35 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 320 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %9 = neura.phi_start %8, %7 {dfg_id = 40 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %10 = neura.reserve {dfg_id = 6 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%2) {dfg_id = 38 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 160 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %12 = neura.phi_start %11, %10 {dfg_id = 43 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %13 = neura.reserve {dfg_id = 7 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%1) {dfg_id = 37 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %15 = neura.phi_start %14, %13 {dfg_id = 42 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %16 = neura.reserve {dfg_id = 8 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%0) {dfg_id = 36 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %18 = neura.phi_start %17, %16 {dfg_id = 41 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%15) {dfg_id = 51 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %20 = "neura.icmp"(%19) <{cmpType = "sgt"}> {dfg_id = 56 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %21 = "neura.data_mov"(%18) {dfg_id = 48 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%20) {dfg_id = 62 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 32 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 360 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}, {id = 360 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}, {id = 360 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %23 = neura.grant_predicate %21, %22 {dfg_id = 68 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %24 = "neura.data_mov"(%15) {dfg_id = 50 : i32, mapping_locs = [{id = 296 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 1 : i32}, {id = 296 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 2 : i32}, {id = 296 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%20) {dfg_id = 61 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 288 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %26 = neura.grant_predicate %24, %25 {dfg_id = 67 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %27 = "neura.data_mov"(%12) {dfg_id = 53 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%20) {dfg_id = 60 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 168 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}, {id = 168 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}, {id = 168 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}, {id = 168 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %29 = neura.grant_predicate %27, %28 {dfg_id = 66 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %30 = "neura.data_mov"(%6) {dfg_id = 55 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 7 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%20) {dfg_id = 59 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 33 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %32 = neura.grant_predicate %30, %31 {dfg_id = 65 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %33 = "neura.data_mov"(%9) {dfg_id = 46 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 320 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%20) {dfg_id = 58 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 328 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}, {id = 328 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}, {id = 328 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}, {id = 328 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}, {id = 328 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 7 : i32}, {id = 328 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %35 = neura.grant_predicate %33, %34 {dfg_id = 64 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %36 = "neura.data_mov"(%20) {dfg_id = 57 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %37 = "neura.not"(%36) {dfg_id = 63 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %38 = "neura.data_mov"(%12) {dfg_id = 52 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 30 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 418 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 418 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 418 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 418 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 418 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 418 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%37) {dfg_id = 73 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 425 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 4 : i32}, {id = 425 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 5 : i32}, {id = 425 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 6 : i32}, {id = 425 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 7 : i32}, {id = 425 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 8 : i32}, {id = 425 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 9 : i32}, {id = 425 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 10 : i32}, {id = 425 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 11 : i32}, {id = 425 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 12 : i32}, {id = 425 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %40 = neura.grant_predicate %38, %39 {dfg_id = 88 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 14 : i32, x = 1 : i32, y = 3 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %41 = "neura.data_mov"(%15) {dfg_id = 49 : i32, mapping_locs = [{id = 296 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 1 : i32}, {id = 296 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 2 : i32}, {id = 296 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}, {id = 296 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%37) {dfg_id = 72 : i32, mapping_locs = [{id = 289 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 289 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %43 = neura.grant_predicate %41, %42 {dfg_id = 87 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %44 = "neura.data_mov"(%18) {dfg_id = 47 : i32, mapping_locs = [{id = 362 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 6 : i32}, {id = 362 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 7 : i32}, {id = 362 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 8 : i32}, {id = 362 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 9 : i32}, {id = 362 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 10 : i32}, {id = 362 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 11 : i32}, {id = 362 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %45 = "neura.data_mov"(%37) {dfg_id = 71 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 32 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 368 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 5 : i32}, {id = 368 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 6 : i32}, {id = 368 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 7 : i32}, {id = 368 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 8 : i32}, {id = 368 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 9 : i32}, {id = 368 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 10 : i32}, {id = 368 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 11 : i32}, {id = 368 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %46 = neura.grant_predicate %44, %45 {dfg_id = 86 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %47 = "neura.data_mov"(%9) {dfg_id = 45 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 320 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 320 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %48 = "neura.data_mov"(%37) {dfg_id = 70 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 329 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 4 : i32}, {id = 329 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 5 : i32}, {id = 329 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 6 : i32}, {id = 329 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 7 : i32}, {id = 329 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 8 : i32}, {id = 329 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %49 = neura.grant_predicate %47, %48 {dfg_id = 85 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %50 = "neura.data_mov"(%6) {dfg_id = 54 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 1000 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 1000 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 1000 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 1000 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 1000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 1000 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 1000 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 1000 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 1000 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 1000 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 1000 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %51 = "neura.data_mov"(%37) {dfg_id = 69 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 15 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 1008 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}, {id = 1008 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}, {id = 1008 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 7 : i32}, {id = 1008 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}, {id = 1008 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}, {id = 1008 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 10 : i32}, {id = 1008 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 11 : i32}, {id = 1008 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 12 : i32}, {id = 1008 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 13 : i32}, {id = 1008 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %52 = neura.grant_predicate %50, %51 {dfg_id = 84 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 15 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %53 = "neura.data_mov"(%23) {dfg_id = 83 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %54 = neura.sext %53 {dfg_id = 98 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %55 = "neura.data_mov"(%26) {dfg_id = 81 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %56 = neura.zext %55 {dfg_id = 96 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %57 = "neura.data_mov"(%29) {dfg_id = 78 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %58 = neura.zext %57 {dfg_id = 93 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %59 = "neura.data_mov"(%26) {dfg_id = 80 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %60 = neura.zext %59 {dfg_id = 95 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %61 = neura.reserve {dfg_id = 9 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %62 = "neura.data_mov"(%35) {dfg_id = 74 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %63 = "neura.phi"(%61, %62) {dfg_id = 89 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %64 = neura.reserve {dfg_id = 10 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %65 = "neura.data_mov"(%23) {dfg_id = 82 : i32, mapping_locs = [{id = 360 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 7 : i32}, {id = 360 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}, {id = 360 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %66 = "neura.phi"(%64, %65) {dfg_id = 97 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %67 = neura.reserve {dfg_id = 11 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %68 = "neura.data_mov"(%29) {dfg_id = 77 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 4000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 4000 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 4000 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %69 = "neura.phi"(%67, %68) {dfg_id = 92 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %70 = neura.reserve {dfg_id = 12 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %71 = "neura.data_mov"(%58) {dfg_id = 111 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %72 = "neura.phi"(%70, %71) {dfg_id = 124 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %73 = neura.reserve {dfg_id = 13 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %74 = "neura.data_mov"(%60) {dfg_id = 114 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %75 = "neura.phi"(%73, %74) {dfg_id = 126 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %76 = neura.reserve {dfg_id = 14 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %77 = "neura.data_mov"(%32) {dfg_id = 76 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %78 = "neura.phi"(%76, %77) {dfg_id = 91 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %79 = neura.reserve {dfg_id = 15 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %80 = "neura.data_mov"(%26) {dfg_id = 79 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 416 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 416 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %81 = "neura.phi"(%79, %80) {dfg_id = 94 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %82 = neura.reserve {dfg_id = 16 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %83 = "neura.data_mov"(%56) {dfg_id = 115 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %84 = "neura.phi"(%82, %83) {dfg_id = 127 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %85 = neura.reserve {dfg_id = 17 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %86 = "neura.data_mov"(%54) {dfg_id = 117 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %87 = "neura.phi"(%85, %86) {dfg_id = 129 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %88 = neura.reserve {dfg_id = 18 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %89 = "neura.data_mov"(%32) {dfg_id = 75 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %90 = "neura.phi"(%88, %89) {dfg_id = 90 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %91 = "neura.data_mov"(%90) {dfg_id = 107 : i32, mapping_locs = [{id = 18 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 224 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 224 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 224 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 224 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %92 = "neura.data_mov"(%87) {dfg_id = 151 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %93 = "neura.add"(%91, %92) {dfg_id = 158 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %94 = "neura.data_mov"(%93) {dfg_id = 173 : i32, mapping_locs = [{id = 22 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %95 = "neura.gep"(%94) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 178 : i32, lhs_value = "%arg2", mapping_locs = [{id = 3 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>

// YAML:      array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 20
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2830001
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "1"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 54
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 69
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 14
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2830000
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 84
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 18
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   id: 337
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 19
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 417
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"

// ASM: # Compiled II: 20
// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM:      PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$8] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [WEST, RED] (t=14, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=14)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$8] -> [$0] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$0], [NORTH, RED] -> [$0] (t=18, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=18)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [EAST, RED] (t=19, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=19)