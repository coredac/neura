// Compile the int ReLU C++ kernel to LLVM IR.
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c++17 \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/relu/relu_int.cpp
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

// The exact CP-SAT mapper emits a routable placement at the II proven by the
// cost model (relu.exact-mapping.json); we replay it and generate code.
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
// RUN:   --map-to-accelerator="import-mapping=%S/relu.exact-mapping.json" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM

// CM: analytical_ii = 5 : i32

// MAPPING:   func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "exact-cpsat", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0 : i32}> {dfg_id = 1 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = neura.reserve {dfg_id = 2 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.data_mov"(%1) {dfg_id = 6 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 1000 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 1000 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %4 = neura.phi_start %3, %2 {dfg_id = 8 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %5 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%0) {dfg_id = 5 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = neura.phi_start %6, %5 {dfg_id = 7 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = "neura.data_mov"(%7) {dfg_id = 11 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = "neura.gep"(%8) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 16 : i32, lhs_value = "%arg0", mapping_locs = [{id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%9) {dfg_id = 20 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %11 = "neura.load"(%10) {dfg_id = 22 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%11) {dfg_id = 27 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %13 = "neura.icmp"(%12) <{cmpType = "sge"}> {dfg_id = 30 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 0 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%13) {dfg_id = 34 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %15 = "neura.data_mov"(%11) {dfg_id = 26 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%4) {dfg_id = 13 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %17 = "neura.sel"(%14, %15, %16) {dfg_id = 38 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%7) {dfg_id = 10 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %19 = "neura.gep"(%18) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 15 : i32, lhs_value = "%arg1", mapping_locs = [{id = 8 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%17) {dfg_id = 41 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %21 = "neura.data_mov"(%19) {dfg_id = 19 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 4000 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%20, %21) {dfg_id = 42 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %22 = "neura.data_mov"(%7) {dfg_id = 9 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %23 = "neura.add"(%22) {dfg_id = 14 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %24 = "neura.data_mov"(%23) {dfg_id = 18 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %25 = "neura.icmp"(%24) <{cmpType = "eq"}> {dfg_id = 21 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%25) {dfg_id = 25 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %27 = "neura.not"(%26) {dfg_id = 29 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%23) {dfg_id = 17 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 4000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%27) {dfg_id = 33 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = neura.grant_predicate %28, %29 {dfg_id = 37 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %30 -> %5 {dfg_id = 40 : i32} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%4) {dfg_id = 12 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}, {id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%27) {dfg_id = 32 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}, {id = 15 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %33 = neura.grant_predicate %31, %32 {dfg_id = 36 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %33 -> %2 {dfg_id = 39 : i32, mapping_locs = [{id = 1008 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 9 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%25) {dfg_id = 23 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%25) {dfg_id = 24 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %36 = neura.grant_predicate %34, %35 {dfg_id = 28 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%36) {dfg_id = 31 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     neura.return_void %37 : !neura.data<i1, i1> {dfg_id = 35 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]}
// MAPPING-NEXT:     neura.yield {dfg_id = 4 : i32}
// MAPPING-NEXT:   }

// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 5
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD"
// YAML-NEXT:                   id: 22
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "UNRESOLVED"
// YAML-NEXT:                       color: "ERROR"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_SGE"
// YAML-NEXT:                   id: 30
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "UNRESOLVED"
// YAML-NEXT:                       color: "ERROR"
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "SEL"
// YAML-NEXT:                   id: 38
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "UNRESOLVED"
// YAML-NEXT:                       color: "ERROR"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "STORE"
// YAML-NEXT:                   id: 42
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "UNRESOLVED"
// YAML-NEXT:                       color: "ERROR"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 16
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg0"
// YAML-NEXT:                       color: "RED"
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
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 8
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 1
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"

// ASM: # Compiled II: 5
// ASM-EMPTY:
// ASM-NEXT: PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [UNRESOLVED, ERROR] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGE, [UNRESOLVED, ERROR], [#0] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   SEL, [UNRESOLVED, ERROR], [$0], [EAST, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   STORE, [UNRESOLVED, ERROR], [NORTH, RED] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg0], [NORTH, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$8] -> [WEST, RED], [EAST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [EAST, RED], [NORTH, RED] -> [$8] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-EMPTY:
// ASM-NEXT: PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [WEST, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-EMPTY:
// ASM-NEXT: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   NOT, [UNRESOLVED, ERROR] -> [EAST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [UNRESOLVED, ERROR] (t=6, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [EAST, RED], [UNRESOLVED, ERROR] -> [SOUTH, RED], [NORTH, RED] (t=2, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$0] -> [SOUTH, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [UNRESOLVED, ERROR], [#1] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [UNRESOLVED, ERROR], [#32] -> [EAST, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [WEST, RED] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [WEST, RED], [WEST, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   RETURN_VOID, [UNRESOLVED, ERROR] (t=7, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-EMPTY:
// ASM-NEXT: PE(0,2):
// ASM-NEXT: {
// ASM-NEXT:   GEP, [arg1], [SOUTH, RED] -> [SOUTH, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:

