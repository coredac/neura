// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   | FileCheck %s -check-prefix=CANONICALIZE

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   | FileCheck %s -check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   | FileCheck %s -check-prefix=FUSE

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   | FileCheck %s -check-prefix=MOV

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN:   --architecture-spec=../arch_spec/architecture.yaml \
// RUN:   | FileCheck %s -check-prefix=MAPPING

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN:   --architecture-spec=../arch_spec/architecture.yaml \
// RUN:   --generate-code 
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml -check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM

func.func @loop_test() -> f32 {
  %n = llvm.mlir.constant(10 : i64) : i64
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c1f = llvm.mlir.constant(3.0 : f32) : f32
  %acc_init = llvm.mlir.constant(0.0 : f32) : f32

  llvm.br ^bb1(%c0, %acc_init : i64, f32)

^bb1(%i: i64, %acc: f32):  // loop body + check + increment
  %next_acc = llvm.fadd %acc, %c1f : f32
  %i_next = llvm.add %i, %c1 : i64
  %cmp = llvm.icmp "slt" %i_next, %n : i64
  llvm.cond_br %cmp, ^bb1(%i_next, %next_acc : i64, f32), ^exit(%next_acc : f32)

^exit(%result: f32):
  return %result : f32
}

// CHECK:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CHECK-NEXT:   %0 = "neura.constant"() <{value = 10 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %1 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %2 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %3 = "neura.constant"() <{value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %4 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   neura.br %1, %4 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1
// CHECK-NEXT: ^bb1(%5: !neura.data<i64, i1>, %6: !neura.data<f32, i1>):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:   %7 = "neura.fadd"(%6, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:   %8 = "neura.add"(%5, %2) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:   %9 = "neura.icmp"(%8, %0) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:   neura.cond_br %9 : !neura.data<i1, i1> then %8, %7 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1 else to ^bb2
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT:   "neura.return"(%7) : (!neura.data<f32, i1>) -> ()
// CHECK-NEXT: }

// CANONICALIZE:       func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CANONICALIZE-NEXT:     %0 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %1 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
// CANONICALIZE-NEXT:     neura.br %0, %1 : i64, f32 to ^bb1
// CANONICALIZE-NEXT:   ^bb1(%2: i64, %3: f32):  // 2 preds: ^bb0, ^bb1
// CANONICALIZE-NEXT:     %4 = "neura.fadd"(%3) {rhs_value = 3.000000e+00 : f32} : (f32) -> f32
// CANONICALIZE-NEXT:     %5 = "neura.add"(%2) {rhs_value = 1 : i64} : (i64) -> i64
// CANONICALIZE-NEXT:     %6 = "neura.icmp"(%5) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (i64) -> i1
// CANONICALIZE-NEXT:     neura.cond_br %6 : i1 then %5, %4 : i64, f32 to ^bb1 else %4 : f32 to ^bb2
// CANONICALIZE-NEXT:   ^bb2(%7: f32):  // pred: ^bb1
// CANONICALIZE-NEXT:     "neura.return"(%7) : (f32) -> ()
// CANONICALIZE-NEXT:   }

// CTRL2DATA:        func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %4 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %5 = neura.phi_start %3, %4 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %6 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = neura.phi_start %1, %6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.fadd"(%5) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.add"(%7) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = "neura.icmp"(%9) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %11 = neura.grant_predicate %9, %10 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %11 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.grant_predicate %8, %10 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %12 -> %4 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %13 = "neura.not"(%10) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %14 = neura.grant_predicate %8, %13 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.return_value %14 : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.yield
// CTRL2DATA-NEXT:   }

// FUSE:        func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// FUSE-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// FUSE-NEXT:     %2 = neura.reserve : !neura.data<f32, i1>
// FUSE-NEXT:     %3 = neura.phi_start %1, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     %4 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:     %5 = neura.phi_start %0, %4 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     %6 = "neura.fadd"(%3) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %7 = "neura.add"(%5) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %8 = "neura.icmp"(%7) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %9 = neura.grant_predicate %7, %8 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     neura.ctrl_mov %9 -> %4 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:     %10 = neura.grant_predicate %6, %8 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     neura.ctrl_mov %10 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
// FUSE-NEXT:     %11 = "neura.not"(%8) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %12 = neura.grant_predicate %6, %11 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     neura.return_value %12 : !neura.data<f32, i1>
// FUSE-NEXT:     neura.yield
// FUSE-NEXT:   }

// MOV:      func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// MOV-NEXT:   %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:   %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:   %2 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:   %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// MOV-NEXT:   %5 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:   %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MOV-NEXT:   %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:   neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:   %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %18 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:   neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:   %20 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %22 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %23 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:   %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     neura.return_value %25 : !neura.data<f32, i1>
// MOV-NEXT:     neura.yield

// MAPPING: module {
// MAPPING-NEXT:   func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> {dfg_id = 1 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %2 = neura.reserve {dfg_id = 2 : i32} : !neura.data<f32, i1>
// MAPPING-NEXT:     %3 = "neura.data_mov"(%1) {dfg_id = 6 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %4 = neura.phi_start %3, %2 {dfg_id = 8 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %5 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%0) {dfg_id = 5 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = neura.phi_start %6, %5 {dfg_id = 7 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = "neura.data_mov"(%4) {dfg_id = 10 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %9 = "neura.fadd"(%8) {dfg_id = 12 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%7) {dfg_id = 9 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %11 = "neura.add"(%10) {dfg_id = 11 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%11) {dfg_id = 14 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {dfg_id = 17 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%11) {dfg_id = 13 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 1000 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %15 = "neura.data_mov"(%13) {dfg_id = 20 : i32, mapping_locs = [{id = 1008 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %16 = neura.grant_predicate %14, %15 {dfg_id = 23 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %16 -> %5 {dfg_id = 26 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%9) {dfg_id = 16 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 4000 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%13) {dfg_id = 19 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 4001 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %19 = neura.grant_predicate %17, %18 {dfg_id = 22 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %19 -> %2 {dfg_id = 25 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%13) {dfg_id = 18 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %21 = "neura.not"(%20) {dfg_id = 21 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%9) {dfg_id = 15 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%21) {dfg_id = 24 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %24 = neura.grant_predicate %22, %23 {dfg_id = 27 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%24) {dfg_id = 28 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.return_value %25 : !neura.data<f32, i1> {dfg_id = 29 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]}
// MAPPING-NEXT:     neura.yield {dfg_id = 4 : i32}
// MAPPING-NEXT:   }
// MAPPING-NEXT: }
// MAPPING-EMPTY:

// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 5
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "2"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "RETURN_VALUE"
// YAML-NEXT:                   id: 56
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "5"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 500000
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 0
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#10"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"

// ASM: # Compiled II: 5
// ASM-EMPTY:
// ASM-NEXT: PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   RETURN_VALUE, [NORTH, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-EMPTY:
// ASM-NEXT: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [NORTH, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#10] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$1] -> [EAST, RED], [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [EAST, RED] -> [$1] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [NORTH, RED], [EAST, RED] -> [SOUTH, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [NORTH, RED], [$0] -> [NORTH, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SLT, [NORTH, RED], [WEST, RED] -> [NORTH, RED], [EAST, RED], [$0], [WEST, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [NORTH, RED], [$0] -> [WEST, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(3,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [NORTH, RED], [$0] -> [NORTH, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [WEST, RED] -> [WEST, RED] (t=4, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-EMPTY:
// ASM-NEXT: PE(1,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#1] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [SOUTH, RED] -> [EAST, RED] (t=1, inv_iters=0)
