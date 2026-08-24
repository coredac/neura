// RUN: mlir-neura-opt %s --expand-memref-copy | FileCheck %s

// Dynamic shape: bounds must come from memref.dim, not a constant.
func.func @copy_dynamic(%src: memref<?x4xf32>, %dst: memref<?x4xf32>) {
  memref.copy %src, %dst : memref<?x4xf32> to memref<?x4xf32>
  return
}

// CHECK-LABEL: func.func @copy_dynamic
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[C0]]
// CHECK: affine.for %{{.*}} = 0 to %[[DIM]]
// CHECK: affine.for %{{.*}} = 0 to 4
// CHECK: affine.load %arg0
// CHECK: affine.store %{{.*}}, %arg1
// CHECK-NOT: memref.copy

// Dead copy: target has no other users, so the copy should just disappear.
func.func @copy_dead(%src: memref<4xf32>) {
  %dst = memref.alloc() : memref<4xf32>
  memref.copy %src, %dst : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @copy_dead
// CHECK-NOT: affine.for
// CHECK-NOT: memref.copy
