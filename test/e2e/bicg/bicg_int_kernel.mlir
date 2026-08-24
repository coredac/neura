// Compile the int BiCG kernel to LLVM IR.
// Use -I %S so local headers are visible if needed.
// RUN: clang -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c11 \
// RUN:   -I %S/../../benchmark/CGRA-Bench/kernels/bicg -DSMALL_DATASET \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/bicg/bicg_int.c
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

// The analytical cost model predicts the II lower bound directly from the DFG
// (no mapper run). For BiCG the recurrence bound dominates: II >= 9.
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

// The heuristic mapper settles at II=10 here; the exact CP-SAT mapper proves
// II=9 is achievable and emits that placement (bicg_int.exact-mapping.json).
// We adopt the smaller II=9 and generate code from it.
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
// RUN:   --map-to-accelerator="import-mapping=%S/bicg_int.exact-mapping.json" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM

// CM: analytical_ii = 9 : i32

// MAPPING:   func.func @kernel_bicg_int(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 9 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "exact-cpsat", rec_mii = 9 : i32, res_mii = 4 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.constant"() <{value = "%arg3"}> {dfg_id = 0 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %1 = "neura.constant"() <{value = 0 : i8}> {dfg_id = 1 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i8, i1>
// MAPPING-NEXT:     %2 = "neura.constant"() <{value = 32 : i64}> {dfg_id = 2 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 3 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = "neura.data_mov"(%0) {dfg_id = 13 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 320 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %5 = "neura.data_mov"(%1) {dfg_id = 14 : i32, mapping_locs = [{id = 328 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}, {id = 328 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}, {id = 328 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i8, i1>) -> !neura.data<i8, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%2) {dfg_id = 15 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     "neura.memset"(%4, %5, %6) <{is_volatile = false}> {dfg_id = 18 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i8, i1>, !neura.data<i64, i1>) -> ()
// MAPPING-NEXT:     %7 = neura.reserve {dfg_id = 4 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = "neura.data_mov"(%3) {dfg_id = 16 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = neura.phi_start %8, %7 {dfg_id = 19 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %10 = neura.reserve {dfg_id = 5 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%3) {dfg_id = 17 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}, {id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %12 = neura.phi_start %11, %10 {dfg_id = 20 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %13 = "neura.data_mov"(%12) {dfg_id = 26 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 12 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %14 = "neura.gep"(%13) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 32 : i32, lhs_value = "%arg4", mapping_locs = [{id = 8 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %15 = "neura.data_mov"(%14) {dfg_id = 44 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 8000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 8000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%15) {dfg_id = 51 : i32, lhs_value = 0 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %16 = "neura.data_mov"(%12) {dfg_id = 25 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 31 : i32, lhs_value = "%arg2", mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%12) {dfg_id = 24 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %19 = "neura.shl"(%18) {dfg_id = 30 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 5 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%19) {dfg_id = 41 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %21 = "neura.gep"(%20) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 48 : i32, lhs_value = "%arg0", mapping_locs = [{id = 4 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %22 = neura.reserve {dfg_id = 6 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%9) {dfg_id = 22 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %24 = "neura.phi"(%22, %23) {dfg_id = 28 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %25 = neura.reserve {dfg_id = 7 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%12) {dfg_id = 23 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %27 = "neura.phi"(%25, %26) {dfg_id = 29 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %28 = neura.reserve {dfg_id = 8 : i32} : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%14) {dfg_id = 43 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %30 = "neura.phi"(%28, %29) {dfg_id = 50 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %31 = neura.reserve {dfg_id = 9 : i32} : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%17) {dfg_id = 42 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %33 = "neura.phi"(%31, %32) {dfg_id = 49 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %34 = neura.reserve {dfg_id = 10 : i32} : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%21) {dfg_id = 57 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %36 = "neura.phi"(%34, %35) {dfg_id = 66 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %37 = neura.reserve {dfg_id = 11 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %38 = "neura.data_mov"(%9) {dfg_id = 21 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %39 = "neura.phi"(%37, %38) {dfg_id = 27 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%36) {dfg_id = 75 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %41 = "neura.data_mov"(%39) {dfg_id = 36 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 25 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 4000 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 4000 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %42 = "neura.gep"(%40, %41) <{operandSegmentSizes = array<i32: 1, 1>}> {dfg_id = 81 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %43 = "neura.data_mov"(%42) {dfg_id = 90 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %44 = "neura.load"(%43) {dfg_id = 98 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %45 = "neura.data_mov"(%33) {dfg_id = 59 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %46 = "neura.load"(%45) {dfg_id = 67 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %47 = "neura.data_mov"(%46) {dfg_id = 76 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}, {id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %48 = "neura.data_mov"(%44) {dfg_id = 108 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %49 = "neura.mul"(%47, %48) {dfg_id = 111 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %50 = "neura.data_mov"(%39) {dfg_id = 35 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 25 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 11 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %51 = "neura.gep"(%50) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 47 : i32, lhs_value = "%arg3", mapping_locs = [{id = 0 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %52 = "neura.data_mov"(%51) {dfg_id = 56 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %53 = "neura.load"(%52) {dfg_id = 65 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %54 = "neura.data_mov"(%53) {dfg_id = 73 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %55 = "neura.data_mov"(%49) {dfg_id = 116 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %56 = "neura.add"(%54, %55) {dfg_id = 120 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %57 = "neura.data_mov"(%56) {dfg_id = 125 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %58 = "neura.data_mov"(%51) {dfg_id = 55 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 2 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 2 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 2 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%57, %58) {dfg_id = 130 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %59 = "neura.data_mov"(%39) {dfg_id = 34 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %60 = "neura.gep"(%59) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 46 : i32, lhs_value = "%arg1", mapping_locs = [{id = 8 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %61 = "neura.data_mov"(%60) {dfg_id = 54 : i32, mapping_locs = [{id = 8001 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 8001 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %62 = "neura.load"(%61) {dfg_id = 64 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %63 = "neura.data_mov"(%62) {dfg_id = 72 : i32, mapping_locs = [{id = 8001 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 8001 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %64 = "neura.data_mov"(%44) {dfg_id = 107 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %65 = "neura.mul"(%63, %64) {dfg_id = 110 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %66 = "neura.data_mov"(%30) {dfg_id = 62 : i32, mapping_locs = [{id = 8001 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %67 = "neura.load"(%66) {dfg_id = 68 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %68 = "neura.data_mov"(%67) {dfg_id = 77 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 8000 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %69 = "neura.data_mov"(%65) {dfg_id = 115 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %70 = "neura.add"(%68, %69) {dfg_id = 119 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %71 = "neura.data_mov"(%70) {dfg_id = 124 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %72 = "neura.data_mov"(%30) {dfg_id = 61 : i32, mapping_locs = [{id = 8008 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}, {id = 8008 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}, {id = 8008 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 10 : i32}, {id = 8008 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 11 : i32}, {id = 8008 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%71, %72) {dfg_id = 129 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %73 = "neura.data_mov"(%39) {dfg_id = 33 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %74 = "neura.add"(%73) {dfg_id = 45 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %75 = "neura.data_mov"(%74) {dfg_id = 53 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %76 = "neura.icmp"(%75) <{cmpType = "eq"}> {dfg_id = 63 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 8 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %77 = "neura.data_mov"(%76) {dfg_id = 71 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %78 = "neura.not"(%77) {dfg_id = 80 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %79 = "neura.data_mov"(%74) {dfg_id = 52 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 160 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 160 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %80 = "neura.data_mov"(%78) {dfg_id = 89 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %81 = neura.grant_predicate %79, %80 {dfg_id = 97 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %81 -> %37 {dfg_id = 106 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 161 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 16 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 11 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %82 = "neura.data_mov"(%36) {dfg_id = 74 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 4000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 4000 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %83 = "neura.data_mov"(%78) {dfg_id = 88 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 4008 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}, {id = 4008 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %84 = neura.grant_predicate %82, %83 {dfg_id = 96 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     neura.ctrl_mov %84 -> %34 {dfg_id = 105 : i32, mapping_locs = [{id = 4001 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 4001 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 4001 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 4001 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %85 = "neura.data_mov"(%33) {dfg_id = 58 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %86 = "neura.data_mov"(%78) {dfg_id = 87 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %87 = neura.grant_predicate %85, %86 {dfg_id = 95 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     neura.ctrl_mov %87 -> %31 {dfg_id = 104 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 3 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 11 : i32}, {id = 2000 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 2000 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 2000 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}, {id = 5 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 15 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %88 = "neura.data_mov"(%30) {dfg_id = 60 : i32, mapping_locs = [{id = 8001 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 25 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %89 = "neura.data_mov"(%78) {dfg_id = 86 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 4001 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %90 = neura.grant_predicate %88, %89 {dfg_id = 94 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     neura.ctrl_mov %90 -> %28 {dfg_id = 103 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 11 : i32}, {id = 8000 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 8000 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 8000 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}, {id = 8000 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %91 = "neura.data_mov"(%27) {dfg_id = 40 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 162 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 162 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}, {id = 162 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 162 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %92 = "neura.data_mov"(%78) {dfg_id = 85 : i32, mapping_locs = [{id = 168 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}, {id = 168 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %93 = neura.grant_predicate %91, %92 {dfg_id = 93 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %93 -> %25 {dfg_id = 102 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 161 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %94 = "neura.data_mov"(%24) {dfg_id = 38 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 321 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 321 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 321 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 321 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 321 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 321 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %95 = "neura.data_mov"(%78) {dfg_id = 84 : i32, mapping_locs = [{id = 168 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}, {id = 168 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}, {id = 14 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 10 : i32}, {id = 20 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %96 = neura.grant_predicate %94, %95 {dfg_id = 92 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %96 -> %22 {dfg_id = 101 : i32} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %97 = "neura.data_mov"(%27) {dfg_id = 39 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %98 = "neura.data_mov"(%76) {dfg_id = 70 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %99 = neura.grant_predicate %97, %98 {dfg_id = 79 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %100 = "neura.data_mov"(%24) {dfg_id = 37 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 321 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %101 = "neura.data_mov"(%76) {dfg_id = 69 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %102 = neura.grant_predicate %100, %101 {dfg_id = 78 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %103 = "neura.data_mov"(%99) {dfg_id = 83 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %104 = "neura.add"(%103) {dfg_id = 91 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %105 = "neura.data_mov"(%104) {dfg_id = 100 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %106 = "neura.icmp"(%105) <{cmpType = "eq"}> {dfg_id = 109 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 8 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %107 = "neura.data_mov"(%106) {dfg_id = 114 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %108 = "neura.not"(%107) {dfg_id = 118 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %109 = "neura.data_mov"(%104) {dfg_id = 99 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 169 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 9 : i32}, {id = 169 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %110 = "neura.data_mov"(%108) {dfg_id = 123 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %111 = neura.grant_predicate %109, %110 {dfg_id = 128 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %111 -> %10 {dfg_id = 132 : i32} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %112 = "neura.data_mov"(%102) {dfg_id = 82 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 31 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %113 = "neura.data_mov"(%108) {dfg_id = 122 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %114 = neura.grant_predicate %112, %113 {dfg_id = 127 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %114 -> %7 {dfg_id = 131 : i32} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %115 = "neura.data_mov"(%106) {dfg_id = 112 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %116 = "neura.data_mov"(%106) {dfg_id = 113 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %117 = neura.grant_predicate %115, %116 {dfg_id = 117 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 3 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING-NEXT:     %118 = "neura.data_mov"(%117) {dfg_id = 121 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     neura.return_void %118 : !neura.data<i1, i1> {dfg_id = 126 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 3 : i32}]}
// MAPPING-NEXT:     neura.yield {dfg_id = 12 : i32}
// MAPPING-NEXT:   }

// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 9
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 76
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "MUL"
// YAML-NEXT:                   id: 111
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 120
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "UNRESOLVED"
// YAML-NEXT:                       color: "ERROR"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "STORE"
// YAML-NEXT:                   id: 130
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "UNRESOLVED"
// YAML-NEXT:                       color: "ERROR"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 47
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD"
// YAML-NEXT:                   id: 65
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "UNRESOLVED"
// YAML-NEXT:                       color: "ERROR"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "1"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 95
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:

// ASM: # Compiled II: 9
// ASM-EMPTY:
// ASM-NEXT: PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=10, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   MUL, [$0], [NORTH, RED] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$1], [UNRESOLVED, ERROR] (t=12, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   STORE, [UNRESOLVED, ERROR], [$2] (t=13, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg3], [NORTH, RED] -> [$2] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [UNRESOLVED, ERROR] -> [$1] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-EMPTY:
// ASM-NEXT: PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [$0] (t=9, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [$0] -> [EAST, RED] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [EAST, RED], [NORTH, RED] -> [$0] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [UNRESOLVED, ERROR] -> [WEST, RED] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-EMPTY:
// ASM-NEXT: PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [WEST, RED] -> [$0] (t=12, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [$0] -> [WEST, RED] (t=15, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-EMPTY:
// ASM-NEXT: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [UNRESOLVED, ERROR] -> [SOUTH, RED], [NORTH, RED] (t=9, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$8] (t=9, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$1] (t=9, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [NORTH, RED], [$1] -> [NORTH, RED] (t=10, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$8] -> [$1] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   SHL, [EAST, RED], [#5] (t=5, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg0], [UNRESOLVED, ERROR] (t=6, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=6, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [SOUTH, RED] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$1], [UNRESOLVED, ERROR] -> [$0] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [UNRESOLVED, ERROR], [$0] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-EMPTY:
// ASM-NEXT: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [NORTH, RED], [$0] (t=0, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$9] (t=9, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [$8] -> [$1] (t=10, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [$8] -> [EAST, RED] (t=10, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$9], [NORTH, RED] (t=11, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [$1] -> [NORTH, RED] (t=11, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [UNRESOLVED, ERROR] -> [WEST, RED], [$0] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$1], [UNRESOLVED, ERROR] -> [$2], [NORTH, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg2], [$0] -> [SOUTH, RED] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [NORTH, RED] -> [WEST, RED], [SOUTH, RED], [$8] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
