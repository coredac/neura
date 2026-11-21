// RUN: mlir-neura-opt %s --assign-accelerator --lower-llvm-to-neura | FileCheck %s

// A[i][j] with int A[N][16].
// CHECK-LABEL: func.func @gep_2d
// CHECK: "neura.gep"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 1, 2>, strides = [16, 1]}>
llvm.func @gep_2d(%A: !llvm.ptr, %i: i64, %j: i64) -> i32 {
  %p = llvm.getelementptr %A[%i, %j] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<16 x i32>
  %v = llvm.load %p : !llvm.ptr -> i32
  llvm.return %v : i32
}

// T[k][i][j] with a global int T[2][8][16]; the leading 0 indexes the global.
// CHECK-LABEL: func.func @gep_3d_global
// CHECK: %[[T:.*]] = "neura.constant"() <{value = @T}>
// CHECK: %[[EIGHT:.*]] = "neura.constant"() <{value = 8 : i64}>
// CHECK: %[[K8:.*]] = "neura.mul"(%arg0, %[[EIGHT]])
// CHECK: %[[OUTER:.*]] = "neura.add"(%arg1, %[[K8]])
// CHECK: "neura.gep"(%[[T]], %[[OUTER]], %arg2) <{operandSegmentSizes = array<i32: 1, 2>, strides = [16, 1]}>
llvm.mlir.global external @T() : !llvm.array<2 x array<8 x array<16 x i32>>>
llvm.func @gep_3d_global(%k: i64, %i: i64, %j: i64) -> i32 {
  %T = llvm.mlir.addressof @T : !llvm.ptr
  %p = llvm.getelementptr %T[0, %k, %i, %j] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<2 x array<8 x array<16 x i32>>>
  %v = llvm.load %p : !llvm.ptr -> i32
  llvm.return %v : i32
}

// Row step in bytes, as clang -O3 emits it: (i << 4) bytes is 4 ints per row.
// CHECK-LABEL: func.func @byte_row_offset
// CHECK: %[[TWO:.*]] = "neura.constant"() <{value = 2 : i64}>
// CHECK: %[[ROW:.*]] = "neura.shl"(%arg1, %[[TWO]])
// CHECK: %[[P:.*]] = "neura.gep"(%arg0, %[[ROW]]) <{operandSegmentSizes = array<i32: 1, 1>, strides = [1]}>
// CHECK: "neura.gep"(%[[P]], %arg2) <{operandSegmentSizes = array<i32: 1, 1>, strides = [1]}>
llvm.func @byte_row_offset(%A: !llvm.ptr, %i: i64, %j: i64) -> i32 {
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %off = llvm.shl %i, %c4 : i64
  %row = llvm.getelementptr %A[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %p = llvm.getelementptr %row[%j] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %v = llvm.load %p : !llvm.ptr -> i32
  llvm.return %v : i32
}

// Constant byte offset: 64 bytes is 16 ints.
// CHECK-LABEL: func.func @byte_const_offset
// CHECK: %[[OFF:.*]] = "neura.constant"() <{value = 16 : i64}>
// CHECK: "neura.gep"(%arg0, %[[OFF]]) <{operandSegmentSizes = array<i32: 1, 1>, strides = [1]}>
llvm.func @byte_const_offset(%A: !llvm.ptr) -> i32 {
  %p = llvm.getelementptr %A[64] : (!llvm.ptr) -> !llvm.ptr, i8
  %v = llvm.load %p : !llvm.ptr -> i32
  llvm.return %v : i32
}

// A char access keeps its index, since each char is one element.
// CHECK-LABEL: func.func @char_access
// CHECK: "neura.gep"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 1, 1>, strides = [1]}>
llvm.func @char_access(%A: !llvm.ptr, %i: i64) -> i8 {
  %p = llvm.getelementptr %A[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %v = llvm.load %p : !llvm.ptr -> i8
  llvm.return %v : i8
}
