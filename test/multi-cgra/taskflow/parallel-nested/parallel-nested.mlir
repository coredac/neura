// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: -o %t.serialized.mlir
// RUN: FileCheck %s --input-file=%t.serialized.mlir --check-prefixes=SERIALIZED

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: -o %t.taskflow.mlir
// RUN: FileCheck %s --input-file=%t.taskflow.mlir --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --classify-counters \
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
// RUN: '--resource-aware-task-optimization=balance-skip-mapper=false' \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture_with_counter.yaml \
// RUN: -o %t.resopt.mlir
// RUN: FileCheck %s --input-file=%t.resopt.mlir --check-prefixes=RESOPT

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: -o %t.hyperblock.mlir
// RUN: FileCheck %s --input-file=%t.hyperblock.mlir --check-prefixes=HYPERBLOCK

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --allocate-cgra-to-task \
// RUN: -o %t.placement.mlir
// RUN: FileCheck %s --input-file=%t.placement.mlir --check-prefixes=PLACEMENT

module {
  // Example: Parallel nested loops scenario
  // Task 0: Single-level loop (vector scaling)
  // Task 1: Two-level nested loop (matrix multiplication)
  func.func @parallel_nested_example(%A: memref<16xf32>, 
                                      %B: memref<8x8xf32>, 
                                      %C: memref<8x8xf32>,
                                      %D: memref<8x8xf32>,
                                      %scalar: f32) {
    // Task 0: Single-level loop - Vector scaling
    // Computes: A[i] = A[i] * scalar
    affine.for %i = 0 to 16 {
      %v = affine.load %A[%i] : memref<16xf32>
      %scaled = arith.mulf %v, %scalar : f32
      affine.store %scaled, %A[%i] : memref<16xf32>
    }
    
    // Task 1: Two-level nested loop - Matrix multiplication
    // Computes: D[i][j] = B[i][j] * C[i][j] (element-wise)
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        %b_val = affine.load %B[%i, %j] : memref<8x8xf32>
        %c_val = affine.load %C[%i, %j] : memref<8x8xf32>
        %product = arith.mulf %b_val, %c_val : f32
        affine.store %product, %D[%i, %j] : memref<8x8xf32>
      }
    }
    return
  }
}

// SERIALIZED: module {
// SERIALIZED-NEXT:   func.func @parallel_nested_example(%arg0: memref<16xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: f32) {
// SERIALIZED-NEXT:     affine.for %arg5 = 0 to 16 {
// SERIALIZED-NEXT:       %0 = affine.load %arg0[%arg5] : memref<16xf32>
// SERIALIZED-NEXT:       %1 = arith.mulf %0, %arg4 : f32
// SERIALIZED-NEXT:       affine.store %1, %arg0[%arg5] : memref<16xf32>
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg5 = 0 to 8 {
// SERIALIZED-NEXT:       affine.for %arg6 = 0 to 8 {
// SERIALIZED-NEXT:         %0 = affine.load %arg1[%arg5, %arg6] : memref<8x8xf32>
// SERIALIZED-NEXT:         %1 = affine.load %arg2[%arg5, %arg6] : memref<8x8xf32>
// SERIALIZED-NEXT:         %2 = arith.mulf %0, %1 : f32
// SERIALIZED-NEXT:         affine.store %2, %arg3[%arg5, %arg6] : memref<8x8xf32>
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     return
// SERIALIZED-NEXT:   }
// SERIALIZED-NEXT: }

// TASKFLOW: module {
// TASKFLOW-NEXT:   func.func @parallel_nested_example(%arg0: memref<16xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: f32) {
// TASKFLOW-NEXT:     %dependency_read_out, %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<16xf32>) dependency_write_in(%arg0 : memref<16xf32>) value_inputs(%arg4 : f32) [original_read_memrefs(%arg0 : memref<16xf32>), original_write_memrefs(%arg0 : memref<16xf32>)] : (memref<16xf32>, memref<16xf32>, f32) -> (memref<16xf32>, memref<16xf32>) {
// TASKFLOW-NEXT:     ^bb0(%arg5: memref<16xf32>, %arg6: memref<16xf32>, %arg7: f32):
// TASKFLOW-NEXT:       affine.for %arg8 = 0 to 16 {
// TASKFLOW-NEXT:         %0 = affine.load %arg6[%arg8] : memref<16xf32>
// TASKFLOW-NEXT:         %1 = arith.mulf %0, %arg7 : f32
// TASKFLOW-NEXT:         affine.store %1, %arg6[%arg8] : memref<16xf32>
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield reads(%arg6 : memref<16xf32>) writes(%arg6 : memref<16xf32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %dependency_read_out_0:2, %dependency_write_out_1 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<8x8xf32>, memref<8x8xf32>) dependency_write_in(%arg3 : memref<8x8xf32>) [original_read_memrefs(%arg1, %arg2 : memref<8x8xf32>, memref<8x8xf32>), original_write_memrefs(%arg3 : memref<8x8xf32>)] : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) {
// TASKFLOW-NEXT:     ^bb0(%arg5: memref<8x8xf32>, %arg6: memref<8x8xf32>, %arg7: memref<8x8xf32>):
// TASKFLOW-NEXT:       affine.for %arg8 = 0 to 8 {
// TASKFLOW-NEXT:         affine.for %arg9 = 0 to 8 {
// TASKFLOW-NEXT:           %0 = affine.load %arg5[%arg8, %arg9] : memref<8x8xf32>
// TASKFLOW-NEXT:           %1 = affine.load %arg6[%arg8, %arg9] : memref<8x8xf32>
// TASKFLOW-NEXT:           %2 = arith.mulf %0, %1 : f32
// TASKFLOW-NEXT:           affine.store %2, %arg7[%arg8, %arg9] : memref<8x8xf32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield reads(%arg5, %arg6 : memref<8x8xf32>, memref<8x8xf32>) writes(%arg7 : memref<8x8xf32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     return
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// HYPERBLOCK:      module {
// HYPERBLOCK-NEXT:   func.func @parallel_nested_example(%arg0: memref<16xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: f32) {
// HYPERBLOCK-NEXT:     %dependency_read_out, %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<16xf32>) dependency_write_in(%arg0 : memref<16xf32>) value_inputs(%arg4 : f32) [original_read_memrefs(%arg0 : memref<16xf32>), original_write_memrefs(%arg0 : memref<16xf32>)] : (memref<16xf32>, memref<16xf32>, f32) -> (memref<16xf32>, memref<16xf32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg5: memref<16xf32>, %arg6: memref<16xf32>, %arg7: f32):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c16 = arith.constant 16 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %0 = taskflow.counter from %c0 to %c16 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg8: index):
// HYPERBLOCK-NEXT:         %1 = memref.load %arg6[%arg8] : memref<16xf32>
// HYPERBLOCK-NEXT:         %2 = arith.mulf %1, %arg7 : f32
// HYPERBLOCK-NEXT:         memref.store %2, %arg6[%arg8] : memref<16xf32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield reads(%arg6 : memref<16xf32>) writes(%arg6 : memref<16xf32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %dependency_read_out_0:2, %dependency_write_out_1 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<8x8xf32>, memref<8x8xf32>) dependency_write_in(%arg3 : memref<8x8xf32>) [original_read_memrefs(%arg1, %arg2 : memref<8x8xf32>, memref<8x8xf32>), original_write_memrefs(%arg3 : memref<8x8xf32>)] : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg5: memref<8x8xf32>, %arg6: memref<8x8xf32>, %arg7: memref<8x8xf32>):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c8 = arith.constant 8 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %0 = taskflow.counter from %c0 to %c8 step %c1 : index
// HYPERBLOCK-NEXT:       %c0_2 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c8_3 = arith.constant 8 : index
// HYPERBLOCK-NEXT:       %c1_4 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0_2 to %c8_3 step %c1_4 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%0, %1) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg8: index, %arg9: index):
// HYPERBLOCK-NEXT:         %2 = memref.load %arg5[%arg8, %arg9] : memref<8x8xf32>
// HYPERBLOCK-NEXT:         %3 = memref.load %arg6[%arg8, %arg9] : memref<8x8xf32>
// HYPERBLOCK-NEXT:         %4 = arith.mulf %2, %3 : f32
// HYPERBLOCK-NEXT:         memref.store %4, %arg7[%arg8, %arg9] : memref<8x8xf32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield reads(%arg5, %arg6 : memref<8x8xf32>, memref<8x8xf32>) writes(%arg7 : memref<8x8xf32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     return
// HYPERBLOCK-NEXT:   }
// HYPERBLOCK-NEXT: }

// PLACEMENT:      module {
// PLACEMENT-NEXT:   func.func @parallel_nested_example(%arg0: memref<16xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: f32) {
// PLACEMENT-NEXT:     %dependency_read_out, %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<16xf32>) dependency_write_in(%arg0 : memref<16xf32>) value_inputs(%arg4 : f32) [original_read_memrefs(%arg0 : memref<16xf32>), original_write_memrefs(%arg0 : memref<16xf32>)] {task_allocation_info = {cgra_positions = [{col = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}} : (memref<16xf32>, memref<16xf32>, f32) -> (memref<16xf32>, memref<16xf32>) {
// PLACEMENT-NEXT:     ^bb0(%arg5: memref<16xf32>, %arg6: memref<16xf32>, %arg7: f32):
// PLACEMENT-NEXT:       %c0 = arith.constant 0 : index
// PLACEMENT-NEXT:       %c16 = arith.constant 16 : index
// PLACEMENT-NEXT:       %c1 = arith.constant 1 : index
// PLACEMENT-NEXT:       %0 = taskflow.counter from %c0 to %c16 step %c1 : index
// PLACEMENT-NEXT:       "taskflow.hyperblock"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// PLACEMENT-NEXT:       ^bb0(%arg8: index):
// PLACEMENT-NEXT:         %1 = memref.load %arg6[%arg8] : memref<16xf32>
// PLACEMENT-NEXT:         %2 = arith.mulf %1, %arg7 : f32
// PLACEMENT-NEXT:         memref.store %2, %arg6[%arg8] : memref<16xf32>
// PLACEMENT-NEXT:         taskflow.hyperblock.yield
// PLACEMENT-NEXT:       }) : (index) -> ()
// PLACEMENT-NEXT:       taskflow.yield reads(%arg6 : memref<16xf32>) writes(%arg6 : memref<16xf32>)
// PLACEMENT-NEXT:     }
// PLACEMENT-NEXT:     %dependency_read_out_0:2, %dependency_write_out_1 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<8x8xf32>, memref<8x8xf32>) dependency_write_in(%arg3 : memref<8x8xf32>) [original_read_memrefs(%arg1, %arg2 : memref<8x8xf32>, memref<8x8xf32>), original_write_memrefs(%arg3 : memref<8x8xf32>)] {task_allocation_info = {cgra_positions = [{col = 1 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}, {col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 0 : i32}]}} : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) {
// PLACEMENT-NEXT:     ^bb0(%arg5: memref<8x8xf32>, %arg6: memref<8x8xf32>, %arg7: memref<8x8xf32>):
// PLACEMENT-NEXT:       %c0 = arith.constant 0 : index
// PLACEMENT-NEXT:       %c8 = arith.constant 8 : index
// PLACEMENT-NEXT:       %c1 = arith.constant 1 : index
// PLACEMENT-NEXT:       %0 = taskflow.counter from %c0 to %c8 step %c1 : index
// PLACEMENT-NEXT:       %c0_2 = arith.constant 0 : index
// PLACEMENT-NEXT:       %c8_3 = arith.constant 8 : index
// PLACEMENT-NEXT:       %c1_4 = arith.constant 1 : index
// PLACEMENT-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0_2 to %c8_3 step %c1_4 : index
// PLACEMENT-NEXT:       "taskflow.hyperblock"(%0, %1) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// PLACEMENT-NEXT:       ^bb0(%arg8: index, %arg9: index):
// PLACEMENT-NEXT:         %2 = memref.load %arg5[%arg8, %arg9] : memref<8x8xf32>
// PLACEMENT-NEXT:         %3 = memref.load %arg6[%arg8, %arg9] : memref<8x8xf32>
// PLACEMENT-NEXT:         %4 = arith.mulf %2, %3 : f32
// PLACEMENT-NEXT:         memref.store %4, %arg7[%arg8, %arg9] : memref<8x8xf32>
// PLACEMENT-NEXT:         taskflow.hyperblock.yield
// PLACEMENT-NEXT:       }) : (index, index) -> ()
// PLACEMENT-NEXT:       taskflow.yield reads(%arg5, %arg6 : memref<8x8xf32>, memref<8x8xf32>) writes(%arg7 : memref<8x8xf32>)
// PLACEMENT-NEXT:     }
// PLACEMENT-NEXT:     return
// PLACEMENT-NEXT:   }
// PLACEMENT-NEXT: }



// RESOPT:      module {
// RESOPT-NEXT:   func.func @parallel_nested_example(%arg0: memref<16xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: f32) {
// RESOPT-NEXT:     %dependency_read_out:3, %dependency_write_out:2 = taskflow.task @Task_0_Task_1_utilfused dependency_read_in(%arg0, %arg1, %arg2 : memref<16xf32>, memref<8x8xf32>, memref<8x8xf32>) dependency_write_in(%arg0, %arg3 : memref<16xf32>, memref<8x8xf32>) value_inputs(%arg4 : f32) [original_read_memrefs(%arg0, %arg1, %arg2 : memref<16xf32>, memref<8x8xf32>, memref<8x8xf32>), original_write_memrefs(%arg0, %arg3 : memref<16xf32>, memref<8x8xf32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 2 : i32, steps = 4 : i32, trip_count = 64 : i32} : (memref<16xf32>, memref<8x8xf32>, memref<8x8xf32>, memref<16xf32>, memref<8x8xf32>, f32) -> (memref<16xf32>, memref<8x8xf32>, memref<8x8xf32>, memref<16xf32>, memref<8x8xf32>) {
// RESOPT-NEXT:     ^bb0(%arg5: memref<16xf32>, %arg6: memref<8x8xf32>, %arg7: memref<8x8xf32>, %arg8: memref<16xf32>, %arg9: memref<8x8xf32>, %arg10: f32):
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c16 = arith.constant 16 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %0 = taskflow.counter from %c0 to %c16 step %c1 attributes {counter_id = 0 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg8, %arg10, %arg6, %arg7, %arg9 : memref<16xf32>, f32, memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg11: memref<16xf32>, %arg12: f32, %arg13: memref<8x8xf32>, %arg14: memref<8x8xf32>, %arg15: memref<8x8xf32>):
// RESOPT-NEXT:         %3 = neura.counter attributes {counter_id = 0 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 16 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %4 = neura.load_indexed [%3 : !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %5 = "neura.fmul"(%4) {rhs_value = "%input1"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         neura.store_indexed %5 to [%3 : !neura.data<index, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %6 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %7 = neura.counter attributes {counter_id = 1 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %8 = neura.load_indexed [%6, %7 : !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %9 = neura.load_indexed [%6, %7 : !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %10 = "neura.fmul"(%8, %9) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         neura.store_indexed %10 to [%6, %7 : !neura.data<index, i1>, !neura.data<index, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       %c0_0 = arith.constant 0 : index
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c1_1 = arith.constant 1 : index
// RESOPT-NEXT:       %1 = taskflow.counter from %c0_0 to %c8 step %c1_1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0_0 to %c8 step %c1_1 attributes {counter_id = 1 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       taskflow.yield reads(%arg5, %arg6, %arg7 : memref<16xf32>, memref<8x8xf32>, memref<8x8xf32>) writes(%arg8, %arg9 : memref<16xf32>, memref<8x8xf32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     return
// RESOPT-NEXT:   }
// RESOPT-NEXT: }

