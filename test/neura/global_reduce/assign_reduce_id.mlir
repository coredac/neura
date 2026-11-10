// RUN: mlir-neura-opt %s --architecture-spec=%S/../../arch_spec/architecture_with_global_reduce.yaml \
// RUN:   --assign-reduce-id --split-input-file --verify-diagnostics | FileCheck %s

// Ids follow program order and skip ids that are already given.
// CHECK-LABEL: func.func @assign_in_program_order
// CHECK: "neura.vector.reduce.add.global"(%arg0) <{reduce_id = 1 : i32}>
// CHECK: "neura.vector.reduce.add.global"(%arg0) <{reduce_id = 0 : i32}>
// CHECK: "neura.vector.reduce.add.global"(%arg0) <{reduce_id = 2 : i32}>
func.func @assign_in_program_order(%v: vector<4xi32>) -> (i32, i32, i32) {
  %0 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  %1 = "neura.vector.reduce.add.global"(%v) {reduce_id = 0 : i32} : (vector<4xi32>) -> i32
  %2 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  return %0, %1, %2 : i32, i32, i32
}

// -----

// Each kernel has its own reduce table.
// CHECK-LABEL: func.func @first_kernel
// CHECK: <{reduce_id = 0 : i32}>
// CHECK-LABEL: func.func @second_kernel
// CHECK: <{reduce_id = 0 : i32}>
func.func @first_kernel(%v: vector<4xi32>) -> i32 {
  %0 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  return %0 : i32
}
func.func @second_kernel(%v: vector<4xi32>) -> i32 {
  %0 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  return %0 : i32
}

// -----

// expected-error @+1 {{5 global reduce operations exceed the 3 entries of the global reduce unit}}
func.func @too_many_reductions(%v: vector<4xi32>) -> (i32, i32, i32, i32, i32) {
  %0 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  %1 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  %2 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  %3 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  %4 = "neura.vector.reduce.add.global"(%v) : (vector<4xi32>) -> i32
  return %0, %1, %2, %3, %4 : i32, i32, i32, i32, i32
}

// -----

func.func @id_out_of_range(%v: vector<4xi32>) -> i32 {
  // expected-error @+1 {{reduce_id 3 is out of range for a global reduce unit with 3 entries}}
  %0 = "neura.vector.reduce.add.global"(%v) {reduce_id = 3 : i32} : (vector<4xi32>) -> i32
  return %0 : i32
}

// -----

func.func @duplicate_id(%v: vector<4xi32>) -> (i32, i32) {
  %0 = "neura.vector.reduce.add.global"(%v) {reduce_id = 1 : i32} : (vector<4xi32>) -> i32
  // expected-error @+1 {{reduce_id 1 is used by more than one global reduce}}
  %1 = "neura.vector.reduce.add.global"(%v) {reduce_id = 1 : i32} : (vector<4xi32>) -> i32
  return %0, %1 : i32, i32
}
