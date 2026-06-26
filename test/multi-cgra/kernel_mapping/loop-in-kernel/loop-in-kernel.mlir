// RUN: mlir-neura-opt %s \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm \
// RUN: --assign-accelerator \
// RUN: --lower-memref-to-neura \
// RUN: --lower-arith-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: -o %t.neura.mlir
// RUN: FileCheck %s --input-file=%t.neura.mlir --check-prefixes=NEURA

// RUN: mlir-neura-opt %s \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm \
// RUN: --assign-accelerator \
// RUN: --lower-memref-to-neura \
// RUN: --lower-arith-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --promote-input-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: -o %t.dataflow.mlir
// RUN: FileCheck %s --input-file=%t.dataflow.mlir --check-prefixes=DATAFLOW

// RUN: mlir-neura-opt %s \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm \
// RUN: --assign-accelerator \
// RUN: --lower-memref-to-neura \
// RUN: --lower-arith-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --promote-input-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-cast \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic" \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture.yaml \
// RUN: -o %t.mapped.mlir
// RUN: FileCheck %s --input-file=%t.mapped.mlir --check-prefixes=MAPPED

module {
  func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %done_reads:2, %value_outputs = taskflow.task @Task_o will_reads(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
    ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
      %1 = neura.kernel inputs(%arg3, %arg4, %arg5 : memref<?xi32>, memref<?xi32>, i32) {
      ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
        %0 = affine.for %arg9 = 0 to 32 iter_args(%arg10 = %arg8) -> (i32) {
            %1 = affine.load %arg6[%arg9] : memref<?xi32>
            %2 = affine.load %arg7[%arg9] : memref<?xi32>
            %3 = arith.muli %1, %2 : i32
            %4 = arith.addi %arg10, %3 : i32
            affine.yield %4 : i32
        }
        neura.yield results(%0 : i32)
      } : i32
      taskflow.yield done_reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%1 : i32)
    }
    return %value_outputs : i32
  }
}

// NEURA: module {
// NEURA-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// NEURA-NEXT:     %c0_i32 = arith.constant 0 : i32
// NEURA-NEXT:     %done_reads:2, %value_outputs = taskflow.task @Task_o will_reads(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// NEURA-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// NEURA-NEXT:       %0 = neura.kernel inputs(%arg3, %arg4, %arg5 : memref<?xi32>, memref<?xi32>, i32) attributes {accelerator = "neura"} {
// NEURA-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// NEURA-NEXT:         %1 = "neura.constant"() <{value = 1 : index}> : () -> index
// NEURA-NEXT:         %2 = "neura.constant"() <{value = 32 : index}> : () -> index
// NEURA-NEXT:         %3 = "neura.constant"() <{value = 0 : index}> : () -> index
// NEURA-NEXT:         %4 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// NEURA-NEXT:         neura.br %4, %arg8 : i64, i32 to ^bb1
// NEURA-NEXT:       ^bb1(%5: i64, %6: i32):  // 2 preds: ^bb0, ^bb2
// NEURA-NEXT:         %7 = "neura.cast"(%5) <{cast_type = "int_to_index"}> : (i64) -> index
// NEURA-NEXT:         %8 = "neura.icmp"(%7, %2) <{cmpType = "slt"}> : (index, index) -> i1
// NEURA-NEXT:         neura.cond_br %8 : i1 then to ^bb2 else to ^bb3
// NEURA-NEXT:       ^bb2:  // pred: ^bb1
// NEURA-NEXT:         %9 = neura.load_indexed %arg6[%7 : index] memref<?xi32> : i32
// NEURA-NEXT:         %10 = neura.load_indexed %arg7[%7 : index] memref<?xi32> : i32
// NEURA-NEXT:         %11 = "neura.mul"(%9, %10) : (i32, i32) -> i32
// NEURA-NEXT:         %12 = "neura.add"(%6, %11) : (i32, i32) -> i32
// NEURA-NEXT:         %13 = "neura.add"(%7, %1) : (index, index) -> index
// NEURA-NEXT:         %14 = "neura.cast"(%13) <{cast_type = "index_to_int"}> : (index) -> i64
// NEURA-NEXT:         neura.br %14, %12 : i64, i32 to ^bb1
// NEURA-NEXT:       ^bb3:  // pred: ^bb1
// NEURA-NEXT:         neura.yield results(%6 : i32)
// NEURA-NEXT:       } : i32
// NEURA-NEXT:       taskflow.yield done_reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%0 : i32)
// NEURA-NEXT:     }
// NEURA-NEXT:     return %value_outputs : i32
// NEURA-NEXT:   }
// NEURA-NEXT: }


// DATAFLOW: module {
// DATAFLOW-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// DATAFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// DATAFLOW-NEXT:     %done_reads:2, %value_outputs = taskflow.task @Task_o will_reads(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// DATAFLOW-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// DATAFLOW-NEXT:       %0 = neura.kernel inputs(%arg3, %arg4, %arg5 : memref<?xi32>, memref<?xi32>, i32) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// DATAFLOW-NEXT:         %1 = "neura.grant_once"() <{constant_value = "%input2"}> : () -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %2 = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
// DATAFLOW-NEXT:         %3 = "neura.cast"(%2) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// DATAFLOW-NEXT:         %4 = "neura.grant_once"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW-NEXT:         %5 = neura.reserve : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %6 = neura.phi_start %1, %5 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %7 = neura.reserve : !neura.data<i64, i1>
// DATAFLOW-NEXT:         %8 = neura.phi_start %4, %7 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:         %9 = "neura.cast"(%8) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// DATAFLOW-NEXT:         %10 = "neura.icmp"(%9) <{cmpType = "slt"}> {rhs_value = 32 : index} : (!neura.data<index, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:         %11 = neura.grant_predicate %9, %10 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// DATAFLOW-NEXT:         %12 = neura.grant_predicate %6, %10 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %13 = "neura.not"(%10) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:         %14 = neura.grant_predicate %6, %13 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.return_value %14 : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %15 = neura.load_indexed [%11 : !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %16 = neura.load_indexed [%11 : !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %17 = "neura.mul"(%15, %16) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %18 = "neura.add"(%12, %17) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %19 = "neura.add"(%11) {rhs_value = 1 : index} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// DATAFLOW-NEXT:         %20 = "neura.cast"(%19) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// DATAFLOW-NEXT:         neura.ctrl_mov %20 -> %7 : !neura.data<i64, i1> !neura.data<i64, i1>
// DATAFLOW-NEXT:         neura.ctrl_mov %18 -> %5 : !neura.data<i32, i1> !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.yield
// DATAFLOW-NEXT:       } : i32
// DATAFLOW-NEXT:       taskflow.yield done_reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%0 : i32)
// DATAFLOW-NEXT:     }
// DATAFLOW-NEXT:     return %value_outputs : i32
// DATAFLOW-NEXT:   }
// DATAFLOW-NEXT: }


// MAPPED: module {
// MAPPED-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// MAPPED-NEXT:     %c0_i32 = arith.constant 0 : i32
// MAPPED-NEXT:     %done_reads:2, %value_outputs = taskflow.task @Task_o will_reads(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// MAPPED-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// MAPPED-NEXT:       %0 = neura.kernel inputs(%arg3, %arg4, %arg5 : memref<?xi32>, memref<?xi32>, i32) attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 6 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPED-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// MAPPED-NEXT:         %1 = "neura.grant_once"() <{constant_value = "%input2"}> {dfg_id = 0 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPED-NEXT:         %2 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 1 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPED-NEXT:         %3 = neura.reserve {dfg_id = 2 : i32} : !neura.data<i32, i1>
// MAPPED-NEXT:         %4 = "neura.data_mov"(%1) {dfg_id = 5 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %5 = neura.phi_start %4, %3 {dfg_id = 7 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPED-NEXT:         %6 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
// MAPPED-NEXT:         %7 = "neura.data_mov"(%2) {dfg_id = 6 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPED-NEXT:         %8 = neura.phi_start %7, %6 {dfg_id = 8 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPED-NEXT:         %9 = "neura.data_mov"(%8) {dfg_id = 12 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPED-NEXT:         %10 = "neura.icmp"(%9) <{cmpType = "slt"}> {dfg_id = 13 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 32 : index} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %11 = "neura.data_mov"(%8) {dfg_id = 11 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPED-NEXT:         %12 = "neura.data_mov"(%10) {dfg_id = 16 : i32, mapping_locs = [{id = 168 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %13 = neura.grant_predicate %11, %12 {dfg_id = 19 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPED-NEXT:         %14 = "neura.data_mov"(%5) {dfg_id = 10 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %15 = "neura.data_mov"(%10) {dfg_id = 15 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 200 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}, {id = 200 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %16 = neura.grant_predicate %14, %15 {dfg_id = 18 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPED-NEXT:         %17 = "neura.data_mov"(%10) {dfg_id = 14 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 4000 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %18 = "neura.not"(%17) {dfg_id = 17 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %19 = "neura.data_mov"(%5) {dfg_id = 9 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %20 = "neura.data_mov"(%18) {dfg_id = 20 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %21 = neura.grant_predicate %19, %20 {dfg_id = 25 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPED-NEXT:         %22 = "neura.data_mov"(%21) {dfg_id = 29 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         neura.return_value %22 : !neura.data<i32, i1> {dfg_id = 33 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]}
// MAPPED-NEXT:         %23 = "neura.data_mov"(%13) {dfg_id = 24 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 4000 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPED-NEXT:         %24 = neura.load_indexed [%23 : !neura.data<i64, i1>]  {dfg_id = 28 : i32, lhs_value = "%input0", mapping_locs = [{id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i32, i1>
// MAPPED-NEXT:         %25 = "neura.data_mov"(%13) {dfg_id = 23 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPED-NEXT:         %26 = neura.load_indexed [%25 : !neura.data<i64, i1>]  {dfg_id = 27 : i32, lhs_value = "%input1", mapping_locs = [{id = 1 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i32, i1>
// MAPPED-NEXT:         %27 = "neura.data_mov"(%24) {dfg_id = 32 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %28 = "neura.data_mov"(%26) {dfg_id = 31 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 160 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %29 = "neura.mul"(%27, %28) {dfg_id = 34 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %30 = "neura.data_mov"(%16) {dfg_id = 21 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 192 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %31 = "neura.data_mov"(%29) {dfg_id = 35 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %32 = "neura.add"(%30, %31) {dfg_id = 36 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %33 = "neura.data_mov"(%13) {dfg_id = 22 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPED-NEXT:         %34 = "neura.add"(%33) {dfg_id = 26 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 1 : index} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPED-NEXT:         neura.ctrl_mov %34 -> %6 {dfg_id = 30 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 1000 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 1000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPED-NEXT:         neura.ctrl_mov %32 -> %3 {dfg_id = 37 : i32, mapping_locs = [{id = 201 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 7 : i32}, {id = 201 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 8 : i32}, {id = 201 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 9 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPED-NEXT:         neura.yield {dfg_id = 4 : i32}
// MAPPED-NEXT:       } : i32
// MAPPED-NEXT:       taskflow.yield done_reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%0 : i32)
// MAPPED-NEXT:     }
// MAPPED-NEXT:     return %value_outputs : i32
// MAPPED-NEXT:   }
// MAPPED-NEXT: }


