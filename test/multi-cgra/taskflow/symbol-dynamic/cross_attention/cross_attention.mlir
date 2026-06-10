// RUN: python3 %S/cross_attention.py %t.linalg.mlir

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


// AFFINE:          %dim_4 = memref.dim %arg0, %c0 : memref<?x64xf32>
// AFFINE-NEXT:     affine.for %arg2 = 0 to %dim_4 {
// AFFINE-NEXT:       affine.for %arg3 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:           %4 = affine.load %arg0[%arg2, %arg4] : memref<?x64xf32>
// AFFINE-NEXT:           %5 = affine.load %alloc[%arg4, %arg3] : memref<64x64xf32>
// AFFINE-NEXT:           %6 = affine.load %alloc_3[%arg2, %arg3] : memref<?x64xf32>
// AFFINE-NEXT:           %7 = arith.mulf %4, %5 : f32
// AFFINE-NEXT:           %8 = arith.addf %6, %7 : f32
// AFFINE-NEXT:           affine.store %8, %alloc_3[%arg2, %arg3] : memref<?x64xf32>
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }

// TASKFLOW:          %dim_7 = memref.dim %arg0, %c0 : memref<?x64xf32>
// TASKFLOW-NEXT:     %dependency_read_out_8:3, %dependency_write_out_9 = taskflow.task @Task_3 dependency_read_in(%arg0, %dependency_write_out, %dependency_write_out_6 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>) dependency_write_in(%dependency_write_out_6 : memref<?x64xf32>) value_inputs(%dim_7 : index) [original_read_memrefs(%arg0, %alloc, %alloc_4 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>), original_write_memrefs(%alloc_4 : memref<?x64xf32>)] : (memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>, memref<?x64xf32>, index) -> (memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>, memref<?x64xf32>) {
// TASKFLOW-NEXT:     ^bb0(%arg2: memref<?x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<?x64xf32>, %arg5: memref<?x64xf32>, %arg6: index):
// TASKFLOW-NEXT:       affine.for %arg7 = 0 to %arg6 {
// TASKFLOW-NEXT:         affine.for %arg8 = 0 to 64 {
// TASKFLOW-NEXT:           affine.for %arg9 = 0 to 64 {
// TASKFLOW-NEXT:             %4 = affine.load %arg2[%arg7, %arg9] : memref<?x64xf32>
// TASKFLOW-NEXT:             %5 = affine.load %arg3[%arg9, %arg8] : memref<64x64xf32>
// TASKFLOW-NEXT:             %6 = affine.load %arg5[%arg7, %arg8] : memref<?x64xf32>
// TASKFLOW-NEXT:             %7 = arith.mulf %4, %5 : f32
// TASKFLOW-NEXT:             %8 = arith.addf %6, %7 : f32
// TASKFLOW-NEXT:             affine.store %8, %arg5[%arg7, %arg8] : memref<?x64xf32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield reads(%arg2, %arg3, %arg5 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>) writes(%arg5 : memref<?x64xf32>)
// TASKFLOW-NEXT:     }

// NEURA:      %dependency_read_out_7:3, %dependency_write_out_8 = taskflow.task @Task_3 dependency_read_in(%arg0, %dependency_write_out, %dependency_write_out_6 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>) dependency_write_in(%dependency_write_out_6 : memref<?x64xf32>) value_inputs(%dim : index) [original_read_memrefs(%arg0, %alloc, %alloc_4 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>), original_write_memrefs(%alloc_4 : memref<?x64xf32>)] {dlp_replicable = true, runtime_managable = true} : (memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>, memref<?x64xf32>, index) -> (memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>, memref<?x64xf32>) {
// NEURA-NEXT:     ^bb0(%arg2: memref<?x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<?x64xf32>, %arg5: memref<?x64xf32>, %arg6: index):
// NEURA-NEXT:       %c64 = arith.constant 64 : index
// NEURA-NEXT:       %c0_58 = arith.constant 0 : index
// NEURA-NEXT:       %c1 = arith.constant 1 : index
// NEURA-NEXT:       %4 = taskflow.counter from %c0_58 to %arg6 step %c1 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
// NEURA-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0_58 to %c64 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
// NEURA-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0_58 to %c64 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
// NEURA-NEXT:       neura.kernel inputs(%arg2, %arg3, %arg5, %arg6 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>, index) {
// NEURA-NEXT:       ^bb0(%arg7: memref<?x64xf32>, %arg8: memref<64x64xf32>, %arg9: memref<?x64xf32>, %arg10: index):
// NEURA-NEXT:         %c64_59 = arith.constant 64 : index
// NEURA-NEXT:         %c0_60 = arith.constant 0 : index
// NEURA-NEXT:         %c1_61 = arith.constant 1 : index
// NEURA-NEXT:         %7 = neura.counter from %c0_60 : index to %arg10 : index step %c1_61 : index attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
// NEURA-NEXT:         %8 = neura.counter from %c0_60 : index to %c64_59 : index step %c1_61 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
// NEURA-NEXT:         %9 = neura.counter from %c0_60 : index to %c64_59 : index step %c1_61 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
// NEURA-NEXT:         %10 = memref.load %arg7[%7, %9] : memref<?x64xf32>
// NEURA-NEXT:         %11 = memref.load %arg8[%9, %8] : memref<64x64xf32>
// NEURA-NEXT:         %12 = memref.load %arg9[%7, %8] : memref<?x64xf32>
// NEURA-NEXT:         %13 = arith.mulf %10, %11 : f32
// NEURA-NEXT:         %14 = arith.addf %12, %13 : f32
// NEURA-NEXT:         memref.store %14, %arg9[%7, %8] : memref<?x64xf32>
// NEURA-NEXT:         neura.yield
// NEURA-NEXT:       }
// NEURA-NEXT:       taskflow.yield reads(%arg2, %arg3, %arg5 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>) writes(%arg5 : memref<?x64xf32>)
// NEURA-NEXT:     }
