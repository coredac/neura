// RUN: python3 %S/gcn_dynamic.py %t.linalg.mlir

// RUN: neura-compiler %t.linalg.mlir \
// RUN:   --linalg-to-affine-conversion \
// RUN:   -o %t.affine.mlir
// RUN: FileCheck --input-file=%t.affine.mlir %s --check-prefix=AFFINE

// RUN: mlir-neura-opt %t.affine.mlir \
// RUN:   --convert-affine-to-taskflow \
// RUN:   -o %t.taskflow.mlir
// RUN: FileCheck --input-file=%t.taskflow.mlir %s --check-prefix=TASKFLOW

// RUN: mlir-neura-opt %t.taskflow.mlir \
// RUN:   --construct-hyperblock-from-task \
// RUN:   --cse \
// RUN:   --classify-task-and-counter \
// RUN:   --convert-taskflow-to-neura \
// RUN:   -o %t.neura.mlir
// RUN: FileCheck --input-file=%t.neura.mlir %s --check-prefix=NEURA

// AFFINE:          %dim_2 = memref.dim %arg1, %c0 : memref<?x?xf32>
// AFFINE-NEXT:     %dim_3 = memref.dim %arg1, %c1 : memref<?x?xf32>
// AFFINE-NEXT:     affine.for %arg2 = 0 to %dim_2 {
// AFFINE-NEXT:       affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg4 = 0 to %dim_3 {
// AFFINE-NEXT:           %4 = affine.load %arg1[%arg2, %arg4] : memref<?x?xf32>
// AFFINE-NEXT:           %5 = affine.load %arg0[%arg4, %arg3] : memref<?x8xf32>
// AFFINE-NEXT:           %6 = affine.load %alloc[%arg2, %arg3] : memref<?x8xf32>
// AFFINE-NEXT:           %7 = arith.mulf %4, %5 : f32
// AFFINE-NEXT:           %8 = arith.addf %6, %7 : f32
// AFFINE-NEXT:           affine.store %8, %alloc[%arg2, %arg3] : memref<?x8xf32>
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }

// TASKFLOW:          %dim_2 = memref.dim %arg1, %c0 : memref<?x?xf32>
// TASKFLOW-NEXT:     %dim_3 = memref.dim %arg1, %c1 : memref<?x?xf32>
// TASKFLOW-NEXT:     %dependency_read_out:3, %dependency_write_out_4 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg0, %dependency_write_out : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>) dependency_write_in(%dependency_write_out : memref<?x8xf32>) value_inputs(%dim_2, %dim_3 : index, index) [original_read_memrefs(%arg1, %arg0, %alloc : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>), original_write_memrefs(%alloc : memref<?x8xf32>)] : (memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, index, index) -> (memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>) {
// TASKFLOW-NEXT:     ^bb0(%arg2: memref<?x?xf32>, %arg3: memref<?x8xf32>, %arg4: memref<?x8xf32>, %arg5: memref<?x8xf32>, %arg6: index, %arg7: index):
// TASKFLOW-NEXT:       affine.for %arg8 = 0 to %arg6 {
// TASKFLOW-NEXT:         affine.for %arg9 = 0 to 8 {
// TASKFLOW-NEXT:           affine.for %arg10 = 0 to %arg7 {
// TASKFLOW-NEXT:             %4 = affine.load %arg2[%arg8, %arg10] : memref<?x?xf32>
// TASKFLOW-NEXT:             %5 = affine.load %arg3[%arg10, %arg9] : memref<?x8xf32>
// TASKFLOW-NEXT:             %6 = affine.load %arg5[%arg8, %arg9] : memref<?x8xf32>
// TASKFLOW-NEXT:             %7 = arith.mulf %4, %5 : f32
// TASKFLOW-NEXT:             %8 = arith.addf %6, %7 : f32
// TASKFLOW-NEXT:             affine.store %8, %arg5[%arg8, %arg9] : memref<?x8xf32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield reads(%arg2, %arg3, %arg5 : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>) writes(%arg5 : memref<?x8xf32>)
// TASKFLOW-NEXT:     }

// NEURA:      %dependency_read_out:3, %dependency_write_out_2 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg0, %dependency_write_out : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>) dependency_write_in(%dependency_write_out : memref<?x8xf32>) value_inputs(%dim, %dim_0 : index, index) [original_read_memrefs(%arg1, %arg0, %alloc : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>), original_write_memrefs(%alloc : memref<?x8xf32>)] {dlp_replicable = true, runtime_managable = true} : (memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, index, index) -> (memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>) {
// NEURA-NEXT:     ^bb0(%arg2: memref<?x?xf32>, %arg3: memref<?x8xf32>, %arg4: memref<?x8xf32>, %arg5: memref<?x8xf32>, %arg6: index, %arg7: index):
// NEURA-NEXT:       %c8 = arith.constant 8 : index
// NEURA-NEXT:       %c0_26 = arith.constant 0 : index
// NEURA-NEXT:       %c1_27 = arith.constant 1 : index
// NEURA-NEXT:       %4 = taskflow.counter from %c0_26 to %arg6 step %c1_27 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
// NEURA-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0_26 to %c8 step %c1_27 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
// NEURA-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0_26 to %arg7 step %c1_27 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
// NEURA-NEXT:       neura.kernel inputs(%arg2, %arg3, %arg5, %arg6, %arg7 : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>, index, index) {
// NEURA-NEXT:       ^bb0(%arg8: memref<?x?xf32>, %arg9: memref<?x8xf32>, %arg10: memref<?x8xf32>, %arg11: index, %arg12: index):
// NEURA-NEXT:         %c8_28 = arith.constant 8 : index
// NEURA-NEXT:         %c0_29 = arith.constant 0 : index
// NEURA-NEXT:         %c1_30 = arith.constant 1 : index
// NEURA-NEXT:         %7 = neura.counter from %c0_29 : index to %arg11 : index step %c1_30 : index attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
// NEURA-NEXT:         %8 = neura.counter from %c0_29 : index to %c8_28 : index step %c1_30 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
// NEURA-NEXT:         %9 = neura.counter from %c0_29 : index to %arg12 : index step %c1_30 : index attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
// NEURA-NEXT:         %10 = memref.load %arg8[%7, %9] : memref<?x?xf32>
// NEURA-NEXT:         %11 = memref.load %arg9[%9, %8] : memref<?x8xf32>
// NEURA-NEXT:         %12 = memref.load %arg10[%7, %8] : memref<?x8xf32>
// NEURA-NEXT:         %13 = arith.mulf %10, %11 : f32
// NEURA-NEXT:         %14 = arith.addf %12, %13 : f32
// NEURA-NEXT:         memref.store %14, %arg10[%7, %8] : memref<?x8xf32>
// NEURA-NEXT:         neura.yield
// NEURA-NEXT:       }
// NEURA-NEXT:       taskflow.yield reads(%arg2, %arg3, %arg5 : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>) writes(%arg5 : memref<?x8xf32>)
// NEURA-NEXT:     }
