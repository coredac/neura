// RUN: mlir-neura-opt %s --expand-memref-copy | FileCheck %s

func.func @copy_2d(%src: memref<4x4xf32>, %dst: memref<4x4xf32>) {
  memref.copy %src, %dst : memref<4x4xf32> to memref<4x4xf32>
  return
}

// CHECK-LABEL: func.func @copy_2d
// CHECK: affine.for
// CHECK: affine.for
// CHECK: %[[V:.*]] = affine.load %arg0
// CHECK: affine.store %[[V]], %arg1
// CHECK-NOT: memref.copy
