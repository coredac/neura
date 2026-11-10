// RUN: mlir-neura-opt %s --assign-accelerator --lower-llvm-to-neura \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s

// The call folds into the local reduction that produced its operand.
// CHECK-LABEL: func.func @fold_into_local_reduce
// CHECK-NOT: "neura.vector.reduce.add"(
// CHECK: "neura.vector.reduce.add.global"(%arg0) : (vector<4xi32>) -> i32
// CHECK-NOT: llvm.call
llvm.func @cgra_allreduce_add(i32) -> i32

llvm.func @fold_into_local_reduce(%v: vector<4xi32>) -> i32 {
  %partial = "llvm.intr.vector.reduce.add"(%v) : (vector<4xi32>) -> i32
  %total = llvm.call @cgra_allreduce_add(%partial) : (i32) -> i32
  llvm.return %total : i32
}

// -----

// A local reduction with other users stays alongside the global one.
// CHECK-LABEL: func.func @local_reduce_still_used
// CHECK: %[[LOCAL:.*]] = "neura.vector.reduce.add"(%arg0)
// CHECK: %[[TOTAL:.*]] = "neura.vector.reduce.add.global"(%arg0)
// CHECK: "neura.add"(%[[LOCAL]], %[[TOTAL]])
llvm.func @cgra_allreduce_add(i32) -> i32

llvm.func @local_reduce_still_used(%v: vector<4xi32>) -> i32 {
  %partial = "llvm.intr.vector.reduce.add"(%v) : (vector<4xi32>) -> i32
  %total = llvm.call @cgra_allreduce_add(%partial) : (i32) -> i32
  %sum = llvm.add %partial, %total : i32
  llvm.return %sum : i32
}

// -----

llvm.func @cgra_allreduce_add(i32) -> i32

llvm.func @scalar_operand(%x: i32) -> i32 {
  // expected-error @+1 {{cgra_allreduce_add expects its operand to be the direct result of a vector reduction (llvm.vector.reduce.add); a loop with a scalar remainder, such as one with a runtime trip count, is not supported}}
  %total = llvm.call @cgra_allreduce_add(%x) : (i32) -> i32
  llvm.return %total : i32
}

// -----

// A module that also defines cgra_allreduce_add, e.g. as a host-side stub,
// still gets the global reduce.
// CHECK-LABEL: func.func @callee_defined_in_module
// CHECK: "neura.vector.reduce.add.global"(%arg0)
// CHECK-NOT: call @cgra_allreduce_add
llvm.func @cgra_allreduce_add(%x: i32) -> i32 {
  llvm.return %x : i32
}

llvm.func @callee_defined_in_module(%v: vector<4xi32>) -> i32 {
  %partial = "llvm.intr.vector.reduce.add"(%v) : (vector<4xi32>) -> i32
  %total = llvm.call @cgra_allreduce_add(%partial) : (i32) -> i32
  llvm.return %total : i32
}
