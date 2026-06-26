// RUN: mlir-neura-opt --pass-pipeline='builtin.module(func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-fuse-elementwise-ops),one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},func.func(convert-linalg-to-affine-loops),convert-affine-to-taskflow)' %s \
// RUN: -o %t-taskflow.mlir 
// RUN: FileCheck %s --input-file=%t-taskflow.mlir

// Simple TOSA add lowering test

func.func @simple_add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK:      func.func @simple_add(%arg0: memref<16xf32>, %arg1: memref<16xf32>) -> memref<16xf32> {
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK-NEXT:   %done_writes = taskflow.task @Task_0 will_reads(%arg0, %arg1 : memref<16xf32>, memref<16xf32>) will_writes(%alloc : memref<16xf32>) [original_read_memrefs(%arg0, %arg1 : memref<16xf32>, memref<16xf32>), original_write_memrefs(%alloc : memref<16xf32>)] : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> (memref<16xf32>) {
// CHECK-NEXT:   ^bb0(%arg2: memref<16xf32>, %arg3: memref<16xf32>, %arg4: memref<16xf32>):
// CHECK-NEXT:     affine.for %arg5 = 0 to 16 {
// CHECK-NEXT:       %0 = affine.load %arg2[%arg5] : memref<16xf32>
// CHECK-NEXT:       %1 = affine.load %arg3[%arg5] : memref<16xf32>
// CHECK-NEXT:       %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:       affine.store %2, %arg4[%arg5] : memref<16xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     taskflow.yield done_writes(%arg4 : memref<16xf32>)
// CHECK-NEXT:   }
// CHECK-NEXT:   return %done_writes : memref<16xf32>
// CHECK-NEXT: }
