// Compiles a small int GEMV kernel to LLVM IR, imports to MLIR, then lowers via Neura.
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c++17 \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/gemv/gemv_int.cpp
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
//
// MAPPING: module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i8 = dense<8> : vector<2xi64>>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
// MAPPING-NEXT:   func.func @kernel_gemv_int(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 11 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 3 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0 : i32}> {dfg_id = 1 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = neura.reserve {dfg_id = 2 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.data_mov"(%1) {dfg_id = 14 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %4 = neura.phi_start %3, %2 {dfg_id = 17 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %5 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%0) {dfg_id = 12 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 1000 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = neura.phi_start %6, %5 {dfg_id = 15 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = neura.reserve {dfg_id = 4 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = "neura.data_mov"(%0) {dfg_id = 13 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %10 = neura.phi_start %9, %8 {dfg_id = 16 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%10) {dfg_id = 21 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %12 = "neura.shl"(%11) {dfg_id = 27 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}], rhs_value = 4 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %13 = "neura.data_mov"(%12) {dfg_id = 37 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %14 = "neura.gep"(%13) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 43 : i32, lhs_value = "%arg0", mapping_locs = [{id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %15 = neura.reserve {dfg_id = 5 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%4) {dfg_id = 23 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 320 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %17 = "neura.phi"(%15, %16) {dfg_id = 29 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %18 = neura.reserve {dfg_id = 6 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%7) {dfg_id = 19 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 184 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 24 : i32, resource = "register", time_step = 3 : i32}, {id = 184 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 24 : i32, resource = "register", time_step = 4 : i32}, {id = 184 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 24 : i32, resource = "register", time_step = 5 : i32}, {id = 184 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 24 : i32, resource = "register", time_step = 6 : i32}, {id = 184 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 24 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %20 = "neura.phi"(%18, %19) {dfg_id = 25 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %21 = neura.reserve {dfg_id = 7 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%10) {dfg_id = 20 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 4000 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 4000 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 4000 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %23 = "neura.phi"(%21, %22) {dfg_id = 26 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %24 = neura.reserve {dfg_id = 8 : i32} : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%14) {dfg_id = 47 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %26 = "neura.phi"(%24, %25) {dfg_id = 50 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %27 = neura.reserve {dfg_id = 9 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%4) {dfg_id = 22 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %29 = "neura.phi"(%27, %28) {dfg_id = 28 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %30 = neura.reserve {dfg_id = 10 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%7) {dfg_id = 18 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %32 = "neura.phi"(%30, %31) {dfg_id = 24 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %33 = "neura.data_mov"(%26) {dfg_id = 58 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 160 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%32) {dfg_id = 32 : i32, mapping_locs = [{id = 168 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}, {id = 168 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}, {id = 168 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %35 = "neura.gep"(%33, %34) <{operandSegmentSizes = array<i32: 1, 1>}> {dfg_id = 63 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %36 = "neura.data_mov"(%35) {dfg_id = 74 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %37 = "neura.load"(%36) {dfg_id = 82 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %38 = "neura.data_mov"(%32) {dfg_id = 31 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 192 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %39 = "neura.gep"(%38) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 42 : i32, lhs_value = "%arg1", mapping_locs = [{id = 6 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%39) {dfg_id = 46 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %41 = "neura.load"(%40) {dfg_id = 49 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%41) {dfg_id = 56 : i32, mapping_locs = [{id = 2000 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 2000 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %43 = "neura.data_mov"(%37) {dfg_id = 91 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %44 = "neura.mul"(%42, %43) {dfg_id = 93 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %45 = "neura.data_mov"(%44) {dfg_id = 97 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %46 = "neura.data_mov"(%29) {dfg_id = 38 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 192 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %47 = "neura.add"(%45, %46) {dfg_id = 100 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %48 = "neura.data_mov"(%32) {dfg_id = 30 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %49 = "neura.add"(%48) {dfg_id = 41 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %50 = "neura.data_mov"(%49) {dfg_id = 45 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %51 = "neura.icmp"(%50) <{cmpType = "eq"}> {dfg_id = 48 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 4 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %52 = "neura.data_mov"(%51) {dfg_id = 55 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %53 = "neura.not"(%52) {dfg_id = 62 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %54 = "neura.data_mov"(%49) {dfg_id = 44 : i32, mapping_locs = [{id = 176 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 4 : i32}, {id = 176 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 5 : i32}, {id = 176 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %55 = "neura.data_mov"(%53) {dfg_id = 73 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %56 = neura.grant_predicate %54, %55 {dfg_id = 81 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %56 -> %30 {dfg_id = 90 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 160 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 160 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 160 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 160 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %57 = "neura.data_mov"(%47) {dfg_id = 106 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 192 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %58 = "neura.data_mov"(%53) {dfg_id = 72 : i32, mapping_locs = [{id = 208 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 6 : i32}, {id = 208 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 7 : i32}, {id = 208 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 8 : i32}, {id = 208 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 9 : i32}, {id = 208 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %59 = neura.grant_predicate %57, %58 {dfg_id = 112 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %59 -> %27 {dfg_id = 117 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 11 : i32}, {id = 329 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 12 : i32}, {id = 329 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 13 : i32}, {id = 329 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 14 : i32}, {id = 329 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 15 : i32}, {id = 329 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 16 : i32}, {id = 329 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 17 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING-NEXT:     %60 = "neura.data_mov"(%26) {dfg_id = 57 : i32, mapping_locs = [{id = 4002 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 4002 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 4002 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 4002 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}, {id = 4002 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 4002 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 4002 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %61 = "neura.data_mov"(%53) {dfg_id = 71 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 13 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 4016 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 8 : i32}, {id = 4016 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 9 : i32}, {id = 4016 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %62 = neura.grant_predicate %60, %61 {dfg_id = 80 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     neura.ctrl_mov %62 -> %24 {dfg_id = 89 : i32, mapping_locs = [{id = 4002 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 4002 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 4002 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 4002 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %63 = "neura.data_mov"(%23) {dfg_id = 36 : i32, mapping_locs = [{id = 4008 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}, {id = 4008 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}, {id = 4008 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 7 : i32}, {id = 4008 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}, {id = 4008 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %64 = "neura.data_mov"(%53) {dfg_id = 70 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 13 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 4009 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 8 : i32}, {id = 4009 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %65 = neura.grant_predicate %63, %64 {dfg_id = 79 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %65 -> %21 {dfg_id = 88 : i32, mapping_locs = [{id = 4001 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 4001 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 4001 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 4001 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 4001 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 4001 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %66 = "neura.data_mov"(%20) {dfg_id = 34 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 161 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 161 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %67 = "neura.data_mov"(%53) {dfg_id = 69 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 176 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 7 : i32}, {id = 176 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 8 : i32}, {id = 176 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 9 : i32}, {id = 176 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %68 = neura.grant_predicate %66, %67 {dfg_id = 78 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %68 -> %18 {dfg_id = 87 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 161 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 161 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 161 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 161 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 161 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 161 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 161 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %69 = "neura.data_mov"(%17) {dfg_id = 40 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 320 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %70 = "neura.data_mov"(%53) {dfg_id = 68 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 321 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 321 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 321 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %71 = neura.grant_predicate %69, %70 {dfg_id = 77 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %71 -> %15 {dfg_id = 86 : i32, mapping_locs = [{id = 328 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 10 : i32}, {id = 328 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 11 : i32}, {id = 328 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 12 : i32}, {id = 328 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 13 : i32}, {id = 328 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 14 : i32}, {id = 328 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 15 : i32}, {id = 328 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 16 : i32}, {id = 328 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 17 : i32}, {id = 328 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 18 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING-NEXT:     %72 = "neura.data_mov"(%23) {dfg_id = 35 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %73 = "neura.data_mov"(%51) {dfg_id = 54 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %74 = neura.grant_predicate %72, %73 {dfg_id = 61 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %75 = "neura.data_mov"(%47) {dfg_id = 105 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %76 = "neura.data_mov"(%51) {dfg_id = 53 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 200 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}, {id = 200 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 7 : i32}, {id = 200 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}, {id = 200 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %77 = neura.grant_predicate %75, %76 {dfg_id = 111 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %78 = "neura.data_mov"(%20) {dfg_id = 33 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 161 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %79 = "neura.data_mov"(%51) {dfg_id = 52 : i32, mapping_locs = [{id = 169 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 5 : i32}, {id = 169 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 6 : i32}, {id = 169 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 7 : i32}, {id = 169 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 8 : i32}, {id = 169 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %80 = neura.grant_predicate %78, %79 {dfg_id = 60 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %81 = "neura.data_mov"(%17) {dfg_id = 39 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %82 = "neura.data_mov"(%51) {dfg_id = 51 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 288 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 288 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 288 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %83 = neura.grant_predicate %81, %82 {dfg_id = 59 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %84 = "neura.data_mov"(%74) {dfg_id = 67 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 0 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 0 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %85 = "neura.gep"(%84) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 76 : i32, lhs_value = "%arg2", mapping_locs = [{id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %86 = "neura.data_mov"(%77) {dfg_id = 116 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %87 = "neura.data_mov"(%85) {dfg_id = 85 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 3 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%86, %87) {dfg_id = 118 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %88 = "neura.data_mov"(%74) {dfg_id = 66 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %89 = "neura.add"(%88) {dfg_id = 75 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %90 = "neura.data_mov"(%89) {dfg_id = 84 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %91 = "neura.icmp"(%90) <{cmpType = "eq"}> {dfg_id = 92 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 4 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %92 = "neura.data_mov"(%91) {dfg_id = 96 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %93 = "neura.not"(%92) {dfg_id = 99 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %94 = "neura.data_mov"(%89) {dfg_id = 83 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 15 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 1000 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %95 = "neura.data_mov"(%93) {dfg_id = 104 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %96 = neura.grant_predicate %94, %95 {dfg_id = 110 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %96 -> %8 {dfg_id = 115 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 11 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %97 = "neura.data_mov"(%80) {dfg_id = 65 : i32, mapping_locs = [{id = 168 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 10 : i32}, {id = 168 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %98 = "neura.data_mov"(%93) {dfg_id = 103 : i32, mapping_locs = [{id = 170 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 9 : i32}, {id = 170 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 10 : i32}, {id = 170 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %99 = neura.grant_predicate %97, %98 {dfg_id = 109 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %99 -> %5 {dfg_id = 114 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 12 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %100 = "neura.data_mov"(%83) {dfg_id = 64 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %101 = "neura.data_mov"(%93) {dfg_id = 102 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %102 = neura.grant_predicate %100, %101 {dfg_id = 108 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %102 -> %2 {dfg_id = 113 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 321 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 321 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 321 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 321 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 321 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING-NEXT:     %103 = "neura.data_mov"(%91) {dfg_id = 94 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %104 = "neura.data_mov"(%91) {dfg_id = 95 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %105 = neura.grant_predicate %103, %104 {dfg_id = 98 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING-NEXT:     %106 = "neura.data_mov"(%105) {dfg_id = 101 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     neura.return_void %106 : !neura.data<i1, i1> {dfg_id = 107 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 2 : i32}]}
// MAPPING-NEXT:     neura.yield {dfg_id = 11 : i32}
// MAPPING-NEXT:   }
// MAPPING-NEXT: }
// MAPPING-EMPTY:

// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 11
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 0
// YAML-NEXT:                   time_step: 0
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 115
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 16
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "SHL"
// YAML-NEXT:                   id: 27
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 43
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 67
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 76
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "1"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 12
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 15
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD"
// YAML-NEXT:                   id: 82
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 83
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 110
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 850000
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "2"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "STORE"
// YAML-NEXT:                   id: 118
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD"
// YAML-NEXT:                   id: 49
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "MUL"
// YAML-NEXT:                   id: 93
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "4"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 80
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 20
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   id: 50
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   id: 26
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 61
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 75
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_EQ"
// YAML-NEXT:                   id: 92
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 71
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 70
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$9"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 98
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 79
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$9"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "5"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 78
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 109
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$10"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   id: 24
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 19
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$24"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 41
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_EQ"
// YAML-NEXT:                   id: 48
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$9"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 58
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 63
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 81
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 710000
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 69
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   id: 25
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$24"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 830000
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 99
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$10"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 60
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$9"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "6"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 112
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 31
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 42
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 62
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 53
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 38
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 100
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 111
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "8"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "RETURN_VOID"
// YAML-NEXT:                   id: 107
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "9"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 51
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 59
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 108
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "10"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 113
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 117
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$9"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 1
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 17
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   id: 28
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$9"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 68
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   id: 29
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 77
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"

// ASM: # Compiled II: 11
// ASM-EMPTY:
// ASM-NEXT: PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [EAST, RED], [$0] (t=0, inv_iters=0)
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [$8] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$8] -> [$0], [NORTH, RED] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   SHL, [$0], [#4] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg0], [$0] -> [NORTH, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg2], [$0] -> [EAST, RED] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-EMPTY:
// ASM-NEXT: PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [NORTH, RED] -> [NORTH, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [NORTH, RED] -> [EAST, RED] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [WEST, RED] (t=10, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-EMPTY:
// ASM-NEXT: PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   STORE, [NORTH, RED], [WEST, RED] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [NORTH, RED] -> [$0] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   MUL, [$0], [WEST, RED] -> [NORTH, RED] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-EMPTY:
// ASM-NEXT: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [$16] -> [$2] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$2], [SOUTH, RED] -> [EAST, RED], [$2] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$1], [$0] -> [$8], [$0] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [EAST, RED] -> [SOUTH, RED], [$0] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [$0], [EAST, RED] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [$0], [#4] -> [EAST, RED], [$0] (t=8, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$16] (t=8, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$9] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$0] -> [NORTH, RED] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$8], [$9] -> [$1] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-EMPTY:
// ASM-NEXT: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$1], [$16] -> [$1] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$8], [$10] -> [SOUTH, RED] (t=12, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$0], [SOUTH, RED] -> [$8], [EAST, RED], [$0] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$24] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [$0], [$16] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [$0], [#4] -> [EAST, RED], [WEST, RED], [$9], [NORTH, RED] (t=5, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [$0], [$8] -> [SOUTH, RED] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$16], [EAST, RED] -> [$0] (t=7, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [WEST, RED] (t=7, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$16] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$1], [$24] -> [$1] (t=8, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [WEST, RED] -> [SOUTH, RED], [$10], [NORTH, RED] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$1], [$9] -> [$8] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-EMPTY:
// ASM-NEXT: PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$16] -> [NORTH, RED] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg1], [$0] -> [SOUTH, RED] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [WEST, RED] -> [WEST, RED], [$16], [NORTH, RED] (t=6, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$8] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [SOUTH, RED], [$0] -> [$0] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$8] -> [SOUTH, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-EMPTY:
// ASM-NEXT: PE(0,2):
// ASM-NEXT: {
// ASM-NEXT:   RETURN_VOID, [SOUTH, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-EMPTY:
// ASM-NEXT: PE(1,2):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$0] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [EAST, RED], [$0] -> [$0] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [SOUTH, RED] -> [EAST, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-EMPTY:
// ASM-NEXT: PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [WEST, RED] -> [$1] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [SOUTH, RED] -> [$9] (t=12, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [$0] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$1] -> [$0] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$9], [$0] -> [SOUTH, RED] (t=7, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$1] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$8], [$0] -> [$0], [WEST, RED] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [$8] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-EMPTY:
