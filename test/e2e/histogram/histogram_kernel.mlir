// Compiles the original C kernel to mlir, then lowers it via Neura.
// TODO: Got error when using -O3 -fno-vectorize -fno-slp-vectorize -mllvm -force-vector-width=1 
// Issue: https://github.com/coredac/dataflow/issues/164
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/histogram/histogram_int.cpp
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

// The analytical cost model predicts the II lower bound directly from the DFG
// (no mapper run).
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
// RUN:   --cost-model-analytical -o %t-cost.mlir
// RUN: FileCheck %s --input-file=%t-cost.mlir --check-prefix=CM

// The heuristic mapper baseline settles at II=6 and generates code.
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
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM
// The analytical CP-SAT mapper independently finds a complete placement,
// schedule, and routing witness at II=5.
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
// RUN:   --map-to-accelerator="mapping-strategy=analytical python-executable=%neura_python cpsat-deterministic-work=5000" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   -o %t-analytical.mlir
// RUN: FileCheck %s --input-file=%t-analytical.mlir --check-prefix=ANALYTICAL
// REQUIRES: neura-ortools

// CM: analytical_ii = 5 : i32

// MAPPING: module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i8 = dense<8> : vector<2xi64>>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
// MAPPING-NEXT:   func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 6 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = neura.phi_start %2, %1 {dfg_id = 4 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = "neura.data_mov"(%3) {dfg_id = 6 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 8 : i32, lhs_value = "%arg0", mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%5) {dfg_id = 11 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %7 = "neura.load"(%6) {dfg_id = 13 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.data_mov"(%7) {dfg_id = 17 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %9 = "neura.mul"(%8) {dfg_id = 20 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = 5 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%9) {dfg_id = 23 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %11 = "neura.add"(%10) {dfg_id = 26 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 0 : i32}], rhs_value = -5 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%11) {dfg_id = 28 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %13 = "neura.div"(%12) {dfg_id = 29 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 18 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%13) {dfg_id = 30 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %15 = neura.sext %14 {dfg_id = 31 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%15) {dfg_id = 32 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 33 : i32, lhs_value = "%arg1", mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%17) {dfg_id = 35 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %19 = "neura.load"(%18) {dfg_id = 36 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%19) {dfg_id = 37 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %21 = "neura.add"(%20) {dfg_id = 38 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%21) {dfg_id = 39 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%17) {dfg_id = 34 : i32, mapping_locs = [{id = 4008 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}, {id = 4008 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}, {id = 4008 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%22, %23) {dfg_id = 40 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %24 = "neura.data_mov"(%3) {dfg_id = 5 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %25 = "neura.add"(%24) {dfg_id = 7 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%25) {dfg_id = 10 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %27 = "neura.icmp"(%26) <{cmpType = "eq"}> {dfg_id = 12 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 20 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%27) {dfg_id = 16 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %29 = "neura.not"(%28) {dfg_id = 19 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = "neura.data_mov"(%25) {dfg_id = 9 : i32, mapping_locs = [{id = 168 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 2 : i32}, {id = 168 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}, {id = 168 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%29) {dfg_id = 22 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %32 = neura.grant_predicate %30, %31 {dfg_id = 25 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %32 -> %1 {dfg_id = 27 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 1000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %33 = "neura.data_mov"(%27) {dfg_id = 14 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 288 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 288 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 288 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 288 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 288 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 288 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%27) {dfg_id = 15 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 288 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 288 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 288 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 288 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 288 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 288 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %35 = neura.grant_predicate %33, %34 {dfg_id = 18 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING-NEXT:     %36 = "neura.data_mov"(%35) {dfg_id = 21 : i32, mapping_locs = [{id = 296 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     neura.return_void %36 : !neura.data<i1, i1> {dfg_id = 24 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 2 : i32}]}
// MAPPING-NEXT:     neura.yield {dfg_id = 2 : i32}
// MAPPING-NEXT:   }
// MAPPING-NEXT: }
// MAPPING-EMPTY:

// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 6
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
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 26
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#-5"
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
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 27
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:

// ASM: # Compiled II: 6
// ASM-EMPTY:
// ASM-NEXT: PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [EAST, RED] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [EAST, RED], [#-5] -> [NORTH, RED] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-EMPTY:
// ASM-NEXT: PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [NORTH, RED] -> [$0] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [WEST, RED], [$0] -> [$0], [NORTH, RED] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg0], [$0] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [$0] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   MUL, [$0], [#5] -> [WEST, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   DIV, [SOUTH, RED], [#18] -> [$0] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   SEXT, [$0] -> [$0] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg1], [$0] -> [$0], [$8] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [$0] -> [$0] (t=9, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [$0] (t=10, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   STORE, [$0], [$8] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-EMPTY:
// ASM-NEXT: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   ADD, [SOUTH, RED], [#1] -> [$0], [$8] (t=2, inv_iters=0)

// ANALYTICAL: module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i8 = dense<8> : vector<2xi64>>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
// ANALYTICAL-NEXT:   func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "exact-cpsat", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// ANALYTICAL-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %3 = neura.phi_start %2, %1 {dfg_id = 4 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %4 = "neura.data_mov"(%3) {dfg_id = 6 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 8 : i32, lhs_value = "%arg0", mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// ANALYTICAL-NEXT:     %6 = "neura.data_mov"(%5) {dfg_id = 11 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// ANALYTICAL-NEXT:     %7 = "neura.load"(%6) {dfg_id = 13 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %8 = "neura.data_mov"(%7) {dfg_id = 17 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %9 = "neura.mul"(%8) {dfg_id = 20 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}], rhs_value = 5 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %10 = "neura.data_mov"(%9) {dfg_id = 23 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %11 = "neura.add"(%10) {dfg_id = 26 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 2 : i32}], rhs_value = -5 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %12 = "neura.data_mov"(%11) {dfg_id = 28 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %13 = "neura.div"(%12) {dfg_id = 29 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 2 : i32}], rhs_value = 18 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %14 = "neura.data_mov"(%13) {dfg_id = 30 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %15 = neura.sext %14 {dfg_id = 31 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %16 = "neura.data_mov"(%15) {dfg_id = 32 : i32, mapping_locs = [{id = 26 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 33 : i32, lhs_value = "%arg1", mapping_locs = [{id = 12 : i32, index_per_ii = 2 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// ANALYTICAL-NEXT:     %18 = "neura.data_mov"(%17) {dfg_id = 35 : i32, mapping_locs = [{id = 12000 : i32, index_per_ii = 3 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// ANALYTICAL-NEXT:     %19 = "neura.load"(%18) {dfg_id = 36 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 3 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 13 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %20 = "neura.data_mov"(%19) {dfg_id = 37 : i32, mapping_locs = [{id = 12000 : i32, index_per_ii = 4 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %21 = "neura.add"(%20) {dfg_id = 38 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 4 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 14 : i32, x = 0 : i32, y = 3 : i32}], rhs_value = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %22 = "neura.data_mov"(%21) {dfg_id = 39 : i32, mapping_locs = [{id = 12000 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// ANALYTICAL-NEXT:     %23 = "neura.data_mov"(%17) {dfg_id = 34 : i32, mapping_locs = [{id = 39 : i32, index_per_ii = 3 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 13 : i32}, {id = 26 : i32, index_per_ii = 4 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 14 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// ANALYTICAL-NEXT:     "neura.store"(%22, %23) {dfg_id = 40 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 15 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// ANALYTICAL-NEXT:     %24 = "neura.data_mov"(%3) {dfg_id = 5 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %25 = "neura.add"(%24) {dfg_id = 7 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %26 = "neura.data_mov"(%25) {dfg_id = 10 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %27 = "neura.icmp"(%26) <{cmpType = "eq"}> {dfg_id = 12 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 0 : i32}], rhs_value = 20 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// ANALYTICAL-NEXT:     %28 = "neura.data_mov"(%27) {dfg_id = 16 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// ANALYTICAL-NEXT:     %29 = "neura.not"(%28) {dfg_id = 19 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// ANALYTICAL-NEXT:     %30 = "neura.data_mov"(%25) {dfg_id = 9 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %31 = "neura.data_mov"(%29) {dfg_id = 22 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// ANALYTICAL-NEXT:     %32 = neura.grant_predicate %30, %31 {dfg_id = 25 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     neura.ctrl_mov %32 -> %1 {dfg_id = 27 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// ANALYTICAL-NEXT:     %33 = "neura.data_mov"(%27) {dfg_id = 14 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 3 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}, {id = 6 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}, {id = 9 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}, {id = 23 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}, {id = 37 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// ANALYTICAL-NEXT:     %34 = "neura.data_mov"(%27) {dfg_id = 15 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 3 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}, {id = 6 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}, {id = 9 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}, {id = 23 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}, {id = 37 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// ANALYTICAL-NEXT:     %35 = neura.grant_predicate %33, %34 {dfg_id = 18 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 11 : i32, x = 3 : i32, y = 3 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// ANALYTICAL-NEXT:     %36 = "neura.data_mov"(%35) {dfg_id = 21 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 2 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 12 : i32}, {id = 45 : i32, index_per_ii = 3 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// ANALYTICAL-NEXT:     neura.return_void %36 : !neura.data<i1, i1> {dfg_id = 24 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 14 : i32, x = 2 : i32, y = 2 : i32}]}
// ANALYTICAL-NEXT:     neura.yield {dfg_id = 2 : i32}
// ANALYTICAL-NEXT:   }
// ANALYTICAL-NEXT: }
// ANALYTICAL-EMPTY:
