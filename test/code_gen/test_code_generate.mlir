// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM


func.func @loop_test() -> f32 {
  %n        = llvm.mlir.constant(10 : i64) : i64
  %c0       = llvm.mlir.constant(0  : i64) : i64
  %c1       = llvm.mlir.constant(1  : i64) : i64
  %c1f      = llvm.mlir.constant(3.0 : f32) : f32
  %acc_init = llvm.mlir.constant(0.0 : f32) : f32

  llvm.br ^bb1(%c0, %acc_init : i64, f32)

^bb1(%i: i64, %acc: f32):
  %next_acc = llvm.fadd %acc, %c1f : f32
  %i_next   = llvm.add  %i, %c1    : i64
  %cmp      = llvm.icmp "slt" %i_next, %n : i64
  llvm.cond_br %cmp, ^bb1(%i_next, %next_acc : i64, f32), ^exit(%next_acc : f32)

^exit(%result: f32):
  return %result : f32
}
// MAPPING: module {
// MAPPING-NEXT:   func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:     %0 = "neura.constant"() <{value = 10 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.data_mov"(%0) {dfg_id = 11 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"(%1) {dfg_id = 16 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = "neura.constant"() <{value = 0 : i64}> {dfg_id = 1 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = "neura.data_mov"(%3) {dfg_id = 12 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %5 = "neura.grant_once"(%4) {dfg_id = 17 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.constant"() <{value = 1 : i64}> {dfg_id = 2 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%6) {dfg_id = 13 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = "neura.grant_once"(%7) {dfg_id = 18 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = "neura.constant"() <{value = 3.000000e+00 : f32}> {dfg_id = 3 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%9) {dfg_id = 14 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %11 = "neura.grant_once"(%10) {dfg_id = 19 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %12 = "neura.constant"() <{value = 0.000000e+00 : f32}> {dfg_id = 4 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %13 = "neura.data_mov"(%12) {dfg_id = 15 : i32, mapping_locs = [{id = 448 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %14 = "neura.grant_once"(%13) {dfg_id = 20 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %15 = neura.reserve {dfg_id = 5 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%2) {dfg_id = 21 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = neura.phi_start %16, %15 {dfg_id = 26 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %18 = neura.reserve {dfg_id = 6 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%8) {dfg_id = 23 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %20 = neura.phi_start %19, %18 {dfg_id = 28 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %21 = neura.reserve {dfg_id = 7 : i32} : !neura.data<f32, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%11) {dfg_id = 24 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %23 = neura.phi_start %22, %21 {dfg_id = 29 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %24 = neura.reserve {dfg_id = 8 : i32} : !neura.data<f32, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%14) {dfg_id = 25 : i32, mapping_locs = [{id = 45 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %26 = neura.phi_start %25, %24 {dfg_id = 30 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %27 = neura.reserve {dfg_id = 9 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%5) {dfg_id = 22 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %29 = neura.phi_start %28, %27 {dfg_id = 27 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %30 = "neura.data_mov"(%26) {dfg_id = 38 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%23) {dfg_id = 37 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %32 = "neura.fadd"(%30, %31) {dfg_id = 40 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %33 = "neura.data_mov"(%29) {dfg_id = 33 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%20) {dfg_id = 35 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %35 = "neura.add"(%33, %34) {dfg_id = 39 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %36 = "neura.data_mov"(%35) {dfg_id = 42 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%17) {dfg_id = 32 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> {dfg_id = 45 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%35) {dfg_id = 41 : i32, mapping_locs = [{id = 328 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}, {id = 328 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}, {id = 328 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%38) {dfg_id = 51 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 320 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %41 = neura.grant_predicate %39, %40 {dfg_id = 57 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %41 -> %27 {dfg_id = 63 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%32) {dfg_id = 44 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 320 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %43 = "neura.data_mov"(%38) {dfg_id = 50 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 336 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 5 : i32}, {id = 336 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 16 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %44 = neura.grant_predicate %42, %43 {dfg_id = 56 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %44 -> %24 {dfg_id = 62 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 321 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %45 = "neura.data_mov"(%23) {dfg_id = 36 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 200 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}, {id = 200 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %46 = "neura.data_mov"(%38) {dfg_id = 49 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 33 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 193 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %47 = neura.grant_predicate %45, %46 {dfg_id = 55 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %47 -> %21 {dfg_id = 61 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}, {id = 168 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %48 = "neura.data_mov"(%20) {dfg_id = 34 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %49 = "neura.data_mov"(%38) {dfg_id = 48 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %50 = neura.grant_predicate %48, %49 {dfg_id = 54 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %50 -> %18 {dfg_id = 60 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 296 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %51 = "neura.data_mov"(%17) {dfg_id = 31 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 160 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %52 = "neura.data_mov"(%38) {dfg_id = 47 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %53 = neura.grant_predicate %51, %52 {dfg_id = 53 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %53 -> %15 {dfg_id = 59 : i32, mapping_locs = [{id = 168 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 5 : i32}, {id = 168 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}, {id = 168 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %54 = "neura.data_mov"(%38) {dfg_id = 46 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %55 = "neura.not"(%54) {dfg_id = 52 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %56 = "neura.data_mov"(%32) {dfg_id = 43 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 192 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 192 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %57 = "neura.data_mov"(%55) {dfg_id = 58 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 28 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}, {id = 33 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %58 = neura.grant_predicate %56, %57 {dfg_id = 64 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %59 = "neura.data_mov"(%58) {dfg_id = 65 : i32, mapping_locs = [{id = 18 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.return_value %59 : !neura.data<f32, i1> {dfg_id = 66 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 3 : i32, y = 1 : i32}]}
// MAPPING-NEXT:     neura.yield {dfg_id = 10 : i32}
// MAPPING-NEXT:   }
// MAPPING-NEXT: }
// MAPPING-EMPTY:


// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 5
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "4"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 3
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#3.000000"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 19
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
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
// YAML-NEXT:                   id: 53
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 0
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#10"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 16
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 26
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 61
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 29
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "6"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "FADD"
// YAML-NEXT:                   id: 40
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 36
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 49
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 55
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 64
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "7"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "RETURN_VALUE"
// YAML-NEXT:                   id: 66
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "8"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 52
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "9"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 2
// YAML-NEXT:                   time_step: 0
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 18
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 580000
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 60
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 28
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_SLT"
// YAML-NEXT:                   id: 45
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "10"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 54
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 51
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 50
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 490000
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 57
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 44
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 56
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$16"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 580001
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 39
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 34
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 30
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "11"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 1
// YAML-NEXT:                   time_step: 0
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 17
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 27
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 3
// YAML-NEXT:       core_id: "14"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 4
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0.000000"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 20
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"


// ASM: # Compiled II: 5
// ASM-EMPTY:
// ASM-NEXT: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#3.000000] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [EAST, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-EMPTY:
// ASM-NEXT: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [$8] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#10] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$8] -> [NORTH, RED], [$0] (t=3, inv_iters=0)
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [$8] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [WEST, RED], [$8] -> [EAST, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   FADD, [NORTH, RED], [WEST, RED] -> [NORTH, RED], [$0] (t=5, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$8] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$8], [$1] -> [WEST, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [EAST, RED] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-EMPTY:
// ASM-NEXT: PE(3,1):
// ASM-NEXT: {
// ASM-NEXT:   RETURN_VALUE, [WEST, RED] (t=9, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(0,2):
// ASM-NEXT: {
// ASM-NEXT:   NOT, [EAST, RED] -> [EAST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(1,2):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#1] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=6, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [$8] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$8] -> [EAST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SLT, [EAST, RED], [SOUTH, RED] -> [EAST, RED], [SOUTH, RED], [WEST, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [WEST, RED] -> [WEST, RED] (t=5, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$16] (t=5, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$8], [$0] -> [EAST, RED] (t=6, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$0] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$16] -> [$1] (t=7, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [EAST, RED], [WEST, RED] -> [WEST, RED], [$8] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [NORTH, RED], [$1] -> [SOUTH, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(3,2):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#0] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [WEST, RED] -> [WEST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-EMPTY:
// ASM-NEXT: PE(2,3):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#0.000000] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [SOUTH, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-EMPTY:
