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
  func.func @_Z6kernelPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 32 {
      %0 = affine.load %arg0[%arg2] : memref<?xi32>
      %1 = arith.cmpi sgt, %0, %c0_i32 : i32
      scf.if %1 {
        %2 = affine.load %arg0[%arg2] : memref<?xi32>
        %3 = affine.load %arg1[%arg2] : memref<?xi32>
        %4 = arith.addi %3, %2 : i32
        affine.store %4, %arg1[%arg2] : memref<?xi32>
      } else {
        %2 = affine.load %arg1[%arg2] : memref<?xi32>
        affine.store %2, %arg1[%arg2] : memref<?xi32>
      }
    }
    return
  }
}

// TASKFLOW: module {
// TASKFLOW-NEXT:   func.func @_Z6kernelPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// TASKFLOW-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) dependency_write_in(%arg1 : memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg1 : memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg1 : memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// TASKFLOW-NEXT:       affine.for %arg6 = 0 to 32 {
// TASKFLOW-NEXT:         %0 = affine.load %arg2[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:         %1 = arith.cmpi sgt, %0, %arg5 : i32
// TASKFLOW-NEXT:         scf.if %1 {
// TASKFLOW-NEXT:           %2 = affine.load %arg2[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:           %3 = affine.load %arg4[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:           %4 = arith.addi %3, %2 : i32
// TASKFLOW-NEXT:           affine.store %4, %arg4[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:         } else {
// TASKFLOW-NEXT:           %2 = affine.load %arg4[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:           affine.store %2, %arg4[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg4 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     return
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// HYPERBLOCK:      module {
// HYPERBLOCK-NEXT:   func.func @_Z6kernelPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// HYPERBLOCK-NEXT:     %c0_i32 = arith.constant 0 : i32
// HYPERBLOCK-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) dependency_write_in(%arg1 : memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg1 : memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg1 : memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c32 = arith.constant 32 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg6: index):
// HYPERBLOCK-NEXT:         %1 = memref.load %arg2[%arg6] : memref<?xi32>
// HYPERBLOCK-NEXT:         %2 = arith.cmpi sgt, %1, %arg5 : i32
// HYPERBLOCK-NEXT:         scf.if %2 {
// HYPERBLOCK-NEXT:           %3 = memref.load %arg2[%arg6] : memref<?xi32>
// HYPERBLOCK-NEXT:           %4 = memref.load %arg4[%arg6] : memref<?xi32>
// HYPERBLOCK-NEXT:           %5 = arith.addi %4, %3 : i32
// HYPERBLOCK-NEXT:           memref.store %5, %arg4[%arg6] : memref<?xi32>
// HYPERBLOCK-NEXT:         } else {
// HYPERBLOCK-NEXT:           %3 = memref.load %arg4[%arg6] : memref<?xi32>
// HYPERBLOCK-NEXT:           memref.store %3, %arg4[%arg6] : memref<?xi32>
// HYPERBLOCK-NEXT:         }
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield writes(%arg4 : memref<?xi32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     return
// HYPERBLOCK-NEXT:   }
// HYPERBLOCK-NEXT: }

// KERNEL:      module {
// KERNEL-NEXT:   func.func @_Z6kernelPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// KERNEL-NEXT:     %c0_i32 = arith.constant 0 : i32
// KERNEL-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) dependency_write_in(%arg1 : memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg1 : memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg1 : memref<?xi32>)] {dlp_replicable = true} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c32 = arith.constant 32 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg5, %arg4 : memref<?xi32>, i32, memref<?xi32>) {
// KERNEL-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: i32, %arg8: memref<?xi32>):
// KERNEL-NEXT:         %c0_0 = arith.constant 0 : index
// KERNEL-NEXT:         %c32_1 = arith.constant 32 : index
// KERNEL-NEXT:         %c1_2 = arith.constant 1 : index
// KERNEL-NEXT:         %1 = neura.counter from %c0_0 : index to %c32_1 : index step %c1_2 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} -> index
// KERNEL-NEXT:         %2 = memref.load %arg6[%1] : memref<?xi32>
// KERNEL-NEXT:         %3 = arith.cmpi sgt, %2, %arg7 : i32
// KERNEL-NEXT:         scf.if %3 {
// KERNEL-NEXT:           %4 = memref.load %arg6[%1] : memref<?xi32>
// KERNEL-NEXT:           %5 = memref.load %arg8[%1] : memref<?xi32>
// KERNEL-NEXT:           %6 = arith.addi %5, %4 : i32
// KERNEL-NEXT:           memref.store %6, %arg8[%1] : memref<?xi32>
// KERNEL-NEXT:         } else {
// KERNEL-NEXT:           %4 = memref.load %arg8[%1] : memref<?xi32>
// KERNEL-NEXT:           memref.store %4, %arg8[%1] : memref<?xi32>
// KERNEL-NEXT:         }
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg4 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     return
// KERNEL-NEXT:   }
// KERNEL-NEXT: }

// NEURA:      module {
// NEURA-NEXT:   func.func @_Z6kernelPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// NEURA-NEXT:     %c0_i32 = arith.constant 0 : i32
// NEURA-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) dependency_write_in(%arg1 : memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg1 : memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg1 : memref<?xi32>)] {dlp_replicable = true} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>) {
// NEURA-NEXT:     ^bb0(%arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// NEURA-NEXT:       %c0 = arith.constant 0 : index
// NEURA-NEXT:       %c32 = arith.constant 32 : index
// NEURA-NEXT:       %c1 = arith.constant 1 : index
// NEURA-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// NEURA-NEXT:       neura.kernel inputs(%arg2, %arg5, %arg4 : memref<?xi32>, i32, memref<?xi32>) attributes {accelerator = "neura"} {
// NEURA-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: i32, %arg8: memref<?xi32>):
// NEURA-NEXT:         %1 = "neura.constant"() <{value = 0 : index}> : () -> index
// NEURA-NEXT:         %2 = "neura.constant"() <{value = 32 : index}> : () -> index
// NEURA-NEXT:         %3 = "neura.constant"() <{value = 1 : index}> : () -> index
// NEURA-NEXT:         %4 = neura.counter from %1 : index to %2 : index step %3 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} -> index
// NEURA-NEXT:         %5 = neura.load_indexed %arg6[%4 : index] memref<?xi32> : i32
// NEURA-NEXT:         %6 = "neura.icmp"(%5, %arg7) <{cmpType = "sgt"}> : (i32, i32) -> i1
// NEURA-NEXT:         neura.cond_br %6 : i1 then to ^bb1 else to ^bb2
// NEURA-NEXT:       ^bb1:  // pred: ^bb0
// NEURA-NEXT:         %7 = neura.load_indexed %arg6[%4 : index] memref<?xi32> : i32
// NEURA-NEXT:         %8 = neura.load_indexed %arg8[%4 : index] memref<?xi32> : i32
// NEURA-NEXT:         %9 = "neura.add"(%8, %7) : (i32, i32) -> i32
// NEURA-NEXT:         neura.store_indexed %9 to %arg8[%4 : index] memref<?xi32> : i32
// NEURA-NEXT:         neura.br to ^bb3
// NEURA-NEXT:       ^bb2:  // pred: ^bb0
// NEURA-NEXT:         %10 = neura.load_indexed %arg8[%4 : index] memref<?xi32> : i32
// NEURA-NEXT:         neura.store_indexed %10 to %arg8[%4 : index] memref<?xi32> : i32
// NEURA-NEXT:         neura.br to ^bb3
// NEURA-NEXT:       ^bb3:  // 2 preds: ^bb1, ^bb2
// NEURA-NEXT:         neura.yield
// NEURA-NEXT:       }
// NEURA-NEXT:       taskflow.yield writes(%arg4 : memref<?xi32>)
// NEURA-NEXT:     }
// NEURA-NEXT:     return
// NEURA-NEXT:   }
// NEURA-NEXT: }

// DATAFLOW:      module {
// DATAFLOW-NEXT:   func.func @_Z6kernelPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// DATAFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// DATAFLOW-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) dependency_write_in(%arg1 : memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg1 : memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg1 : memref<?xi32>)] {dlp_replicable = true} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>) {
// DATAFLOW-NEXT:     ^bb0(%arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// DATAFLOW-NEXT:       %c0 = arith.constant 0 : index
// DATAFLOW-NEXT:       %c32 = arith.constant 32 : index
// DATAFLOW-NEXT:       %c1 = arith.constant 1 : index
// DATAFLOW-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// DATAFLOW-NEXT:       neura.kernel inputs(%arg2, %arg5, %arg4 : memref<?xi32>, i32, memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: i32, %arg8: memref<?xi32>):
// DATAFLOW-NEXT:         %1 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 32 : index} -> !neura.data<index, i1>
// DATAFLOW-NEXT:         %2 = neura.load_indexed [%1 : !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %3 = "neura.icmp"(%2) <{cmpType = "sgt"}> {rhs_value = "%input1"} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:         %4 = neura.grant_predicate %1, %3 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// DATAFLOW-NEXT:         %5 = "neura.not"(%3) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:         %6 = neura.grant_predicate %1, %5 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// DATAFLOW-NEXT:         %7 = neura.load_indexed [%6 : !neura.data<index, i1>]  {lhs_value = "%input2"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.store_indexed %7 to [%6 : !neura.data<index, i1>]  {rhs_value = "%input2"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %8 = neura.load_indexed [%4 : !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %9 = neura.load_indexed [%4 : !neura.data<index, i1>]  {lhs_value = "%input2"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %10 = "neura.add"(%9, %8) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.store_indexed %10 to [%4 : !neura.data<index, i1>]  {rhs_value = "%input2"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.yield {yield_type = "void"}
// DATAFLOW-NEXT:       }
// DATAFLOW-NEXT:       taskflow.yield writes(%arg4 : memref<?xi32>)
// DATAFLOW-NEXT:     }
// DATAFLOW-NEXT:     return
// DATAFLOW-NEXT:   }
// DATAFLOW-NEXT: }

// MAPPED:      module {
// MAPPED-NEXT:   func.func @_Z6kernelPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// MAPPED-NEXT:     %c0_i32 = arith.constant 0 : i32
// MAPPED-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) dependency_write_in(%arg1 : memref<?xi32>) value_inputs(%c0_i32 : i32) [original_read_memrefs(%arg0, %arg1 : memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg1 : memref<?xi32>)] {dlp_replicable = true} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> (memref<?xi32>) {
// MAPPED-NEXT:     ^bb0(%arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// MAPPED-NEXT:       %c0 = arith.constant 0 : index
// MAPPED-NEXT:       %c32 = arith.constant 32 : index
// MAPPED-NEXT:       %c1 = arith.constant 1 : index
// MAPPED-NEXT:       %0 = taskflow.counter from %c0 to %c32 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// MAPPED-NEXT:       neura.kernel inputs(%arg2, %arg5, %arg4 : memref<?xi32>, i32, memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 2 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPED-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: i32, %arg8: memref<?xi32>):
// MAPPED-NEXT:         %1 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, dfg_id = 0 : i32, lower_bound_value = 0 : index, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}], step_value = 1 : index, upper_bound_value = 32 : index} -> !neura.data<index, i1>
// MAPPED-NEXT:         %2 = "neura.data_mov"(%1) {dfg_id = 2 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %3 = neura.load_indexed [%2 : !neura.data<index, i1>]  {dfg_id = 5 : i32, lhs_value = "%input0", mapping_locs = [{id = 5 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>
// MAPPED-NEXT:         %4 = "neura.data_mov"(%3) {dfg_id = 6 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %5 = "neura.icmp"(%4) <{cmpType = "sgt"}> {dfg_id = 7 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = "%input1"} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %6 = "neura.data_mov"(%1) {dfg_id = 3 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 32 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}, {id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %7 = "neura.data_mov"(%5) {dfg_id = 9 : i32, mapping_locs = [{id = 40 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %8 = neura.grant_predicate %6, %7 {dfg_id = 11 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// MAPPED-NEXT:         %9 = "neura.data_mov"(%5) {dfg_id = 8 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %10 = "neura.not"(%9) {dfg_id = 10 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %11 = "neura.data_mov"(%1) {dfg_id = 4 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %12 = "neura.data_mov"(%10) {dfg_id = 12 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPED-NEXT:         %13 = neura.grant_predicate %11, %12 {dfg_id = 16 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// MAPPED-NEXT:         %14 = "neura.data_mov"(%13) {dfg_id = 20 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %15 = neura.load_indexed [%14 : !neura.data<index, i1>]  {dfg_id = 23 : i32, lhs_value = "%input2", mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i32, i1>
// MAPPED-NEXT:         %16 = "neura.data_mov"(%15) {dfg_id = 25 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 5 : i32}, {id = 256 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %17 = "neura.data_mov"(%13) {dfg_id = 19 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 4 : i32}, {id = 128 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         neura.store_indexed %16 to [%17 : !neura.data<index, i1>]  {dfg_id = 27 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 1 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}], rhs_value = "%input2"} : !neura.data<i32, i1>
// MAPPED-NEXT:         %18 = "neura.data_mov"(%8) {dfg_id = 15 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 3 : i32}, {id = 64 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %19 = neura.load_indexed [%18 : !neura.data<index, i1>]  {dfg_id = 18 : i32, lhs_value = "%input0", mapping_locs = [{id = 2 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<i32, i1>
// MAPPED-NEXT:         %20 = "neura.data_mov"(%8) {dfg_id = 14 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         %21 = neura.load_indexed [%20 : !neura.data<index, i1>]  {dfg_id = 17 : i32, lhs_value = "%input2", mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<i32, i1>
// MAPPED-NEXT:         %22 = "neura.data_mov"(%21) {dfg_id = 21 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 4 : i32}, {id = 96 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %23 = "neura.data_mov"(%19) {dfg_id = 22 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %24 = "neura.add"(%22, %23) {dfg_id = 24 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %25 = "neura.data_mov"(%24) {dfg_id = 26 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPED-NEXT:         %26 = "neura.data_mov"(%8) {dfg_id = 13 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 3 : i32}, {id = 7 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 4 : i32}, {id = 18 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 5 : i32}, {id = 224 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// MAPPED-NEXT:         neura.store_indexed %25 to [%26 : !neura.data<index, i1>]  {dfg_id = 28 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 1 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 1 : i32}], rhs_value = "%input2"} : !neura.data<i32, i1>
// MAPPED-NEXT:         neura.yield {dfg_id = 1 : i32, yield_type = "void"}
// MAPPED-NEXT:       }
// MAPPED-NEXT:       taskflow.yield writes(%arg4 : memref<?xi32>)
// MAPPED-NEXT:     }
// MAPPED-NEXT:     return
// MAPPED-NEXT:   }
// MAPPED-NEXT: }