// RUN: mlir-neura-opt %s --expand-memref-copy \
// RUN: -o %t-expand.mlir
// RUN: FileCheck %s --input-file=%t-expand.mlir

func.func @copy_2d(%src: memref<4x4xf32>, %dst: memref<4x4xf32>) {
  memref.copy %src, %dst : memref<4x4xf32> to memref<4x4xf32>
  return
}

// CHECK-LABEL: func.func @copy_2d(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>)
// CHECK-NEXT: affine.for %arg2 = 0 to 4 {
// CHECK-NEXT:   affine.for %arg3 = 0 to 4 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg2, %arg3] : memref<4x4xf32>
// CHECK-NEXT:     affine.store %0, %arg1[%arg2, %arg3] : memref<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NOT: memref.copy
