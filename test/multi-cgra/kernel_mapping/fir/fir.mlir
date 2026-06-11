// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: -o %t.taskflow.mlir
// RUN: FileCheck %s --input-file=%t.taskflow.mlir --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: -o %t.hyperblock.mlir
// RUN: FileCheck %s --input-file=%t.hyperblock.mlir --check-prefixes=HYPERBLOCK

// RUN: neura-compiler %s --taskflow-conversion \
// RUN: -o %t.kernel.mlir
// RUN: FileCheck %s --input-file=%t.kernel.mlir --check-prefixes=KERNEL

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --classify-task-and-counter \
// RUN: --convert-taskflow-to-neura \
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

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --classify-task-and-counter \
// RUN: --convert-taskflow-to-neura \
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

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --classify-task-and-counter \
// RUN: --convert-taskflow-to-neura \
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
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic" \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture_with_counter.yaml \
// RUN: -o %t.mapped.mlir
// RUN: FileCheck %s --input-file=%t.mapped.mlir --check-prefixes=MAPPED



module attributes {} {
  func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg3 = 0 to 32 iter_args(%arg4 = %c0_i32) -> (i32) {
      %1 = affine.load %arg0[%arg3] : memref<?xi32>
      %2 = affine.load %arg2[%arg3] : memref<?xi32>
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %arg4, %3 : i32
      affine.yield %4 : i32
    }
    return %0 : i32
  }
}

// TASKFLOW: module {
// TASKFLOW-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// TASKFLOW-NEXT:     %dependency_read_out:2, %value_outputs = taskflow.task @Task_0 dependency_read_in(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg2 : memref<?xi32>, memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// TASKFLOW-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// TASKFLOW-NEXT:       %0 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %arg5) -> (i32) {
// TASKFLOW-NEXT:         %1 = affine.load %arg3[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:         %2 = affine.load %arg4[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:         %3 = arith.muli %1, %2 : i32
// TASKFLOW-NEXT:         %4 = arith.addi %arg7, %3 : i32
// TASKFLOW-NEXT:         affine.yield %4 : i32
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%0 : i32)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     return %value_outputs : i32
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// HYPERBLOCK:      module {
// HYPERBLOCK-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// HYPERBLOCK-NEXT:     %c0_i32 = arith.constant 0 : i32
// HYPERBLOCK-NEXT:     %dependency_read_out:2, %value_outputs = taskflow.task @Task_0 dependency_read_in(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg2 : memref<?xi32>, memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// HYPERBLOCK-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c32 = arith.constant 32 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 : index
// HYPERBLOCK-NEXT:       %1 = "taskflow.hyperblock"(%0, %arg5) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg6: index, %arg7: i32):
// HYPERBLOCK-NEXT:         %2 = memref.load %arg3[%arg6] : memref<?xi32>
// HYPERBLOCK-NEXT:         %3 = memref.load %arg4[%arg6] : memref<?xi32>
// HYPERBLOCK-NEXT:         %4 = arith.muli %2, %3 : i32
// HYPERBLOCK-NEXT:         %5 = arith.addi %arg7, %4 : i32
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield iter_args_next(%5 : i32) results(%5 : i32)
// HYPERBLOCK-NEXT:       }) : (index, i32) -> i32
// HYPERBLOCK-NEXT:       taskflow.yield reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%1 : i32)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     return %value_outputs : i32
// HYPERBLOCK-NEXT:   }
// HYPERBLOCK-NEXT: }

// KERNEL:      module {
// KERNEL-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// KERNEL-NEXT:     %c0_i32 = arith.constant 0 : i32
// KERNEL-NEXT:     %dependency_read_out:2, %value_outputs = taskflow.task @Task_0 dependency_read_in(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg2 : memref<?xi32>, memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// KERNEL-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c32 = arith.constant 32 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// KERNEL-NEXT:       %1 = neura.kernel inputs(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) iter_args_init(%arg5 : i32) {
// KERNEL-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// KERNEL-NEXT:         %c0_0 = arith.constant 0 : index
// KERNEL-NEXT:         %c32_1 = arith.constant 32 : index
// KERNEL-NEXT:         %c1_2 = arith.constant 1 : index
// KERNEL-NEXT:         %2 = neura.counter from %c0_0 : index to %c32_1 : index step %c1_2 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} -> index
// KERNEL-NEXT:         %3 = memref.load %arg6[%2] : memref<?xi32>
// KERNEL-NEXT:         %4 = memref.load %arg7[%2] : memref<?xi32>
// KERNEL-NEXT:         %5 = arith.muli %3, %4 : i32
// KERNEL-NEXT:         %6 = arith.addi %arg8, %5 : i32
// KERNEL-NEXT:         neura.yield iter_args_next(%6 : i32) results(%6 : i32)
// KERNEL-NEXT:       } : i32
// KERNEL-NEXT:       taskflow.yield reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%1 : i32)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     return %value_outputs : i32
// KERNEL-NEXT:   }
// KERNEL-NEXT: }

// NEURA:      module {
// NEURA-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// NEURA-NEXT:     %c0_i32 = arith.constant 0 : i32
// NEURA-NEXT:     %dependency_read_out:2, %value_outputs = taskflow.task @Task_0 dependency_read_in(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg2 : memref<?xi32>, memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// NEURA-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// NEURA-NEXT:       %c0 = arith.constant 0 : index
// NEURA-NEXT:       %c32 = arith.constant 32 : index
// NEURA-NEXT:       %c1 = arith.constant 1 : index
// NEURA-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// NEURA-NEXT:       %1 = neura.kernel inputs(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) iter_args_init(%arg5 : i32) attributes {accelerator = "neura"} {
// NEURA-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// NEURA-NEXT:         %2 = "neura.constant"() <{value = 0 : index}> : () -> index
// NEURA-NEXT:         %3 = "neura.constant"() <{value = 32 : index}> : () -> index
// NEURA-NEXT:         %4 = "neura.constant"() <{value = 1 : index}> : () -> index
// NEURA-NEXT:         %5 = neura.counter from %2 : index to %3 : index step %4 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} -> index
// NEURA-NEXT:         %6 = neura.load_indexed %arg6[%5 : index] memref<?xi32> : i32
// NEURA-NEXT:         %7 = neura.load_indexed %arg7[%5 : index] memref<?xi32> : i32
// NEURA-NEXT:         %8 = "neura.mul"(%6, %7) : (i32, i32) -> i32
// NEURA-NEXT:         %9 = "neura.add"(%arg8, %8) : (i32, i32) -> i32
// NEURA-NEXT:         neura.yield iter_args_next(%9 : i32) results(%9 : i32)
// NEURA-NEXT:       } : i32
// NEURA-NEXT:       taskflow.yield reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%1 : i32)
// NEURA-NEXT:     }
// NEURA-NEXT:     return %value_outputs : i32
// NEURA-NEXT:   }
// NEURA-NEXT: }

// DATAFLOW:      module {
// DATAFLOW-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// DATAFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// DATAFLOW-NEXT:     %dependency_read_out:2, %value_outputs = taskflow.task @Task_0 dependency_read_in(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg2 : memref<?xi32>, memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// DATAFLOW-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// DATAFLOW-NEXT:       %c0 = arith.constant 0 : index
// DATAFLOW-NEXT:       %c32 = arith.constant 32 : index
// DATAFLOW-NEXT:       %c1 = arith.constant 1 : index
// DATAFLOW-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// DATAFLOW-NEXT:       %1 = neura.kernel inputs(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) iter_args_init(%arg5 : i32) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// DATAFLOW-NEXT:         %2 = "neura.grant_once"() <{constant_value = "%iter_arg_init0"}> : () -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %3 = neura.reserve : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %4 = neura.phi_start %2, %3 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 32 : index} -> !neura.data<index, i1>
// DATAFLOW-NEXT:         %6 = neura.load_indexed [%5 : !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %7 = neura.load_indexed [%5 : !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %8 = "neura.mul"(%6, %7) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %9 = "neura.add"(%4, %8) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.ctrl_mov %9 -> %3 : !neura.data<i32, i1> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %10 = neura.extract_predicate %5 : !neura.data<index, i1> -> !neura.data<i1, i1>
// DATAFLOW-NEXT:         %11 = "neura.not"(%10) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:         %12 = neura.grant_predicate %4, %11 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.return_value %12 : !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.yield
// DATAFLOW-NEXT:       } : i32
// DATAFLOW-NEXT:       taskflow.yield reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%1 : i32)
// DATAFLOW-NEXT:     }
// DATAFLOW-NEXT:     return %value_outputs : i32
// DATAFLOW-NEXT:   }
// DATAFLOW-NEXT: }

// MAPPED:      module {
// MAPPED-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// MAPPED-NEXT:     %c0_i32 = arith.constant 0 : i32
// MAPPED-NEXT:     %dependency_read_out:2, %value_outputs = taskflow.task @Task_0 dependency_read_in(%arg0, %arg2 : memref<?xi32>, memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg2 : memref<?xi32>, memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>, memref<?xi32>, i32) {
// MAPPED-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// MAPPED-NEXT:       %c0 = arith.constant 0 : index
// MAPPED-NEXT:       %c32 = arith.constant 32 : index
// MAPPED-NEXT:       %c1 = arith.constant 1 : index
// MAPPED-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// MAPPED-NEXT:       %1 = neura.kernel inputs(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) iter_args_init(%arg5 : i32) attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 2 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPED-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// MAPPED-NEXT:         %2 = "neura.grant_once"() <{constant_value = "%iter_arg_init0"}> {dfg_id = 0 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPED-NEXT:         %3 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i32, i1>
// MAPPED-NEXT:         %4 = "neura.data_mov"(%2) {dfg_id = 4 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %5 = neura.phi_start %4, %3 {dfg_id = 8 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPED-NEXT:         %6 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, dfg_id = 2 : i32, lower_bound_value = 0 : index, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}], step_value = 1 : index, upper_bound_value = 32 : index} -> !neura.data<index, i1>
// MAPPED-NEXT:         %7 = "neura.data_mov"(%6) {dfg_id = 5 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}, {id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %8 = neura.load_indexed [%7 : !neura.data<index, i1>]  {dfg_id = 9 : i32, lhs_value = "%input0", mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i32, i1>
// MAPPED-NEXT:         %9 = "neura.data_mov"(%6) {dfg_id = 6 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 32 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %10 = neura.load_indexed [%9 : !neura.data<index, i1>]  {dfg_id = 10 : i32, lhs_value = "%input1", mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i32, i1>
// MAPPED-NEXT:         %11 = "neura.data_mov"(%8) {dfg_id = 14 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %12 = "neura.data_mov"(%10) {dfg_id = 15 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %13 = "neura.mul"(%11, %12) {dfg_id = 17 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %14 = "neura.data_mov"(%5) {dfg_id = 13 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 160 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %15 = "neura.data_mov"(%13) {dfg_id = 19 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %16 = "neura.add"(%14, %15) {dfg_id = 21 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         neura.ctrl_mov %16 -> %3 {dfg_id = 23 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 288 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPED-NEXT:         %17 = "neura.data_mov"(%6) {dfg_id = 7 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %18 = neura.extract_predicate %17 {dfg_id = 11 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<index, i1> -> !neura.data<i1, i1>
// MAPPED-NEXT:         %19 = "neura.data_mov"(%18) {dfg_id = 16 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %20 = "neura.not"(%19) {dfg_id = 18 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %21 = "neura.data_mov"(%5) {dfg_id = 12 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %22 = "neura.data_mov"(%20) {dfg_id = 20 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %23 = neura.grant_predicate %21, %22 {dfg_id = 22 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPED-NEXT:         %24 = "neura.data_mov"(%23) {dfg_id = 24 : i32, mapping_locs = [{id = 256 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         neura.return_value %24 : !neura.data<i32, i1> {dfg_id = 25 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 2 : i32}]}
// MAPPED-NEXT:         neura.yield {dfg_id = 3 : i32}
// MAPPED-NEXT:       } : i32
// MAPPED-NEXT:       taskflow.yield reads(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) values(%1 : i32)
// MAPPED-NEXT:     }
// MAPPED-NEXT:     return %value_outputs : i32
// MAPPED-NEXT:   }
// MAPPED-NEXT: }
