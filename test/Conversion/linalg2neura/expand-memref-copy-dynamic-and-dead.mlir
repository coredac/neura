// RUN: mlir-neura-opt %s --expand-memref-copy \
// RUN: -o %t-expand.mlir
// RUN: FileCheck %s --input-file=%t-expand.mlir

// Dynamic shape: bounds must come from memref.dim, not a constant.
func.func @copy_dynamic(%src: memref<?x4xf32>, %dst: memref<?x4xf32>) {
  memref.copy %src, %dst : memref<?x4xf32> to memref<?x4xf32>
  return
}

// CHECK-LABEL: func.func @copy_dynamic(%arg0: memref<?x4xf32>, %arg1: memref<?x4xf32>)
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %dim = memref.dim %arg0, %c0 : memref<?x4xf32>
// CHECK-NEXT: affine.for %arg2 = 0 to %dim {
// CHECK-NEXT:   affine.for %arg3 = 0 to 4 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg2, %arg3] : memref<?x4xf32>
// CHECK-NEXT:     affine.store %0, %arg1[%arg2, %arg3] : memref<?x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NOT: memref.copy

// Dead copy: target has no other users, so the copy should just disappear
// (and the now-unused alloc is cleaned up too).
func.func @copy_dead(%src: memref<4xf32>) {
  %dst = memref.alloc() : memref<4xf32>
  memref.copy %src, %dst : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @copy_dead(%arg0: memref<4xf32>)
// CHECK-NEXT: return
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
