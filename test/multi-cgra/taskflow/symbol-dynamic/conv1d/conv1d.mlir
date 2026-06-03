// RUN: python3 %S/conv1d_pipeline.py %t.linalg.mlir

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


// AFFINE:          affine.for %arg1 = 0 to 1 {
// AFFINE:            affine.for %arg2 = 0 to 16 {
// AFFINE:              affine.for %arg3 = 0 to %5 {
// AFFINE:                affine.for %arg4 = 0 to 1 {
// AFFINE-NEXT:             affine.for %arg5 = 0 to 3 {
// AFFINE-NEXT:               %8 = affine.load %arg0[%arg1, %arg4, %arg3 + %arg5] : memref<1x1x?xf32>
// AFFINE-NEXT:               %9 = affine.load %0[%arg2, %arg4, %arg5] : memref<16x1x3xf32>
// AFFINE-NEXT:               %10 = affine.load %alloc[%arg1, %arg2, %arg3] : memref<1x16x?xf32>
// AFFINE-NEXT:               %11 = arith.mulf %8, %9 : f32
// AFFINE-NEXT:               %12 = arith.addf %10, %11 : f32
// AFFINE-NEXT:               affine.store %12, %alloc[%arg1, %arg2, %arg3] : memref<1x16x?xf32>
// AFFINE-NEXT:             }
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }

// TASKFLOW:          %dependency_read_out:3, %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg0, %0, %dependency_write_out : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>) dependency_write_in(%dependency_write_out : memref<1x16x?xf32>) value_inputs(%5 : index) [original_read_memrefs(%arg0, %0, %alloc : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>), original_write_memrefs(%alloc : memref<1x16x?xf32>)] : (memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>, memref<1x16x?xf32>, index) -> (memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>, memref<1x16x?xf32>) {
// TASKFLOW-NEXT:     ^bb0(%arg1: memref<1x1x?xf32>, %arg2: memref<16x1x3xf32>, %arg3: memref<1x16x?xf32>, %arg4: memref<1x16x?xf32>, %arg5: index):
// TASKFLOW-NEXT:       affine.for %arg6 = 0 to 1 {
// TASKFLOW-NEXT:         affine.for %arg7 = 0 to 16 {
// TASKFLOW-NEXT:           affine.for %arg8 = 0 to %arg5 {
// TASKFLOW-NEXT:             affine.for %arg9 = 0 to 1 {
// TASKFLOW-NEXT:               affine.for %arg10 = 0 to 3 {
// TASKFLOW-NEXT:                 %8 = affine.load %arg1[%arg6, %arg9, %arg8 + %arg10] : memref<1x1x?xf32>
// TASKFLOW-NEXT:                 %9 = affine.load %arg2[%arg7, %arg9, %arg10] : memref<16x1x3xf32>
// TASKFLOW-NEXT:                 %10 = affine.load %arg4[%arg6, %arg7, %arg8] : memref<1x16x?xf32>
// TASKFLOW-NEXT:                 %11 = arith.mulf %8, %9 : f32
// TASKFLOW-NEXT:                 %12 = arith.addf %10, %11 : f32
// TASKFLOW-NEXT:                 affine.store %12, %arg4[%arg6, %arg7, %arg8] : memref<1x16x?xf32>
// TASKFLOW-NEXT:               }
// TASKFLOW-NEXT:             }
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield reads(%arg1, %arg2, %arg4 : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>) writes(%arg4 : memref<1x16x?xf32>)
// TASKFLOW-NEXT:     }

// NEURA:      %dependency_read_out:3, %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg0, %0, %dependency_write_out : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>) dependency_write_in(%dependency_write_out : memref<1x16x?xf32>) value_inputs(%5 : index) [original_read_memrefs(%arg0, %0, %alloc : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>), original_write_memrefs(%alloc : memref<1x16x?xf32>)] {task_type = "symbol_dynamic"} : (memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>, memref<1x16x?xf32>, index) -> (memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>, memref<1x16x?xf32>) {
// NEURA-NEXT:     ^bb0(%arg1: memref<1x1x?xf32>, %arg2: memref<16x1x3xf32>, %arg3: memref<1x16x?xf32>, %arg4: memref<1x16x?xf32>, %arg5: index):
// NEURA-NEXT:       %c3 = arith.constant 3 : index
// NEURA-NEXT:       %c16 = arith.constant 16 : index
// NEURA-NEXT:       %c0 = arith.constant 0 : index
// NEURA-NEXT:       %c1 = arith.constant 1 : index
// NEURA-NEXT:       %8 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_dynamism = "static", counter_hierarchy = "root", counter_id = 0 : i32} : index
// NEURA-NEXT:       %9 = taskflow.counter parent(%8 : index) from %c0 to %c16 step %c1 attributes {counter_dynamism = "static", counter_hierarchy = "relay", counter_id = 1 : i32} : index
// NEURA-NEXT:       %10 = taskflow.counter parent(%9 : index) from %c0 to %arg5 step %c1 attributes {counter_dynamism = "symbol_dynamic", counter_hierarchy = "relay", counter_id = 2 : i32} : index
// NEURA-NEXT:       %11 = taskflow.counter parent(%10 : index) from %c0 to %c1 step %c1 attributes {counter_dynamism = "static", counter_hierarchy = "relay", counter_id = 3 : i32} : index
// NEURA-NEXT:       %12 = taskflow.counter parent(%11 : index) from %c0 to %c3 step %c1 attributes {counter_dynamism = "static", counter_hierarchy = "leaf", counter_id = 4 : i32} : index
// NEURA-NEXT:       neura.kernel inputs(%arg1, %arg2, %arg4, %arg5 : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>, index) {
// NEURA-NEXT:       ^bb0(%arg6: memref<1x1x?xf32>, %arg7: memref<16x1x3xf32>, %arg8: memref<1x16x?xf32>, %arg9: index):
// NEURA-NEXT:         %c3_22 = arith.constant 3 : index
// NEURA-NEXT:         %c16_23 = arith.constant 16 : index
// NEURA-NEXT:         %c0_24 = arith.constant 0 : index
// NEURA-NEXT:         %c1_25 = arith.constant 1 : index
// NEURA-NEXT:         %13 = neura.counter from %c0_24 : index to %c1_25 : index step %c1_25 : index attributes {counter_dynamism = "static", counter_hierarchy = "root", counter_id = 0 : i32} -> index
// NEURA-NEXT:         %14 = neura.counter from %c0_24 : index to %c16_23 : index step %c1_25 : index attributes {counter_dynamism = "static", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
// NEURA-NEXT:         %15 = neura.counter from %c0_24 : index to %arg9 : index step %c1_25 : index attributes {counter_dynamism = "symbol_dynamic", counter_hierarchy = "relay", counter_id = 2 : i32} -> index
// NEURA-NEXT:         %16 = neura.counter from %c0_24 : index to %c1_25 : index step %c1_25 : index attributes {counter_dynamism = "static", counter_hierarchy = "relay", counter_id = 3 : i32} -> index
// NEURA-NEXT:         %17 = neura.counter from %c0_24 : index to %c3_22 : index step %c1_25 : index attributes {counter_dynamism = "static", counter_hierarchy = "leaf", counter_id = 4 : i32} -> index
// NEURA-NEXT:         %18 = arith.addi %15, %17 : index
// NEURA-NEXT:         %19 = memref.load %arg6[%13, %16, %18] : memref<1x1x?xf32>
// NEURA-NEXT:         %20 = memref.load %arg7[%14, %16, %17] : memref<16x1x3xf32>
// NEURA-NEXT:         %21 = memref.load %arg8[%13, %14, %15] : memref<1x16x?xf32>
// NEURA-NEXT:         %22 = arith.mulf %19, %20 : f32
// NEURA-NEXT:         %23 = arith.addf %21, %22 : f32
// NEURA-NEXT:         memref.store %23, %arg8[%13, %14, %15] : memref<1x16x?xf32>
// NEURA-NEXT:         neura.yield
// NEURA-NEXT:       }
// NEURA-NEXT:       taskflow.yield reads(%arg1, %arg2, %arg4 : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>) writes(%arg4 : memref<1x16x?xf32>)
// NEURA-NEXT:     }