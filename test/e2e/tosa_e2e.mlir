// RUN: mlir-neura-opt %s --pass-pipeline='builtin.module(func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-fuse-elementwise-ops),one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},func.func(convert-linalg-to-affine-loops),convert-affine-to-taskflow)' \
// RUN: -o %t-taskflow.mlir 
// RUN: FileCheck %s --input-file=%t-taskflow.mlir

// Verifies the end-to-end lowering from TOSA to Taskflow.
func.func @test_e2e(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %1 = tosa.mul %0, %0 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// CHECK:      func.func @test_e2e(%arg0: memref<16xf32>, %arg1: memref<16xf32>) -> memref<16xf32> {
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK-NEXT:   %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0, %arg1 : memref<16xf32>, memref<16xf32>) dependency_write_in(%alloc : memref<16xf32>) [original_read_memrefs(%arg0, %arg1 : memref<16xf32>, memref<16xf32>), original_write_memrefs(%alloc : memref<16xf32>)] : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> (memref<16xf32>) {
// CHECK-NEXT:   ^bb0(%arg2: memref<16xf32>, %arg3: memref<16xf32>, %arg4: memref<16xf32>):
// CHECK-NEXT:     affine.for %arg5 = 0 to 16 {
// CHECK-NEXT:       %0 = affine.load %arg2[%arg5] : memref<16xf32>
// CHECK-NEXT:       %1 = affine.load %arg3[%arg5] : memref<16xf32>
// CHECK-NEXT:       %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:       %3 = arith.mulf %2, %2 : f32
// CHECK-NEXT:       affine.store %3, %arg4[%arg5] : memref<16xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     taskflow.yield writes(%arg4 : memref<16xf32>)
// CHECK-NEXT:   }
// CHECK-NEXT:   return %dependency_write_out : memref<16xf32>
// CHECK-NEXT: }
