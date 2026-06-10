// RUN: neura-interpreter %s --verbose | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Add two float constants
// ===----------------------------------------------------------------------===//
func.func @test_add_f32() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 32.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 42.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Add a negative and positive float
// ===----------------------------------------------------------------------===//
func.func @test_add_negative() -> f32 {
  %a = arith.constant -5.0 : f32
  %b = arith.constant 3.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -2.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Add two fractional values
// ===----------------------------------------------------------------------===//
func.func @test_add_fraction() -> f32 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 1.25 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 4.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Add zero and a number
// ===----------------------------------------------------------------------===//
func.func @test_add_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 7.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 7.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 5: Mapped form — single SSA operand with rhs_value attribute
// ===----------------------------------------------------------------------===//
// After --map-to-accelerator, compile-time constant operands are folded into
// rhs_value attributes (e.g. "neura.add"(%x) {rhs_value = 1 : i64}).
// Before this fix, the interpreter rejected any op with numOperands < 2.
func.func @test_add_mapped_form() -> !neura.data<i64, i1> {
  %a = "neura.grant_once"() <{constant_value = 10 : i64}> : () -> !neura.data<i64, i1>
  %res = "neura.add"(%a) {rhs_value = 5 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  // CHECK: [neura-interpreter]  → Output: 15.000000
  return %res : !neura.data<i64, i1>
}

// RUN: neura-interpreter %s | FileCheck %s

// TODO: Remove Test 6 because we plan to remove the predicate attribute in
// https://github.com/coredac/dataflow/issues/116

// ===----------------------------------------------------------------------===//
// Test 6: Add with operation embed predicate 0
// ===----------------------------------------------------------------------===//
// func.func @test_add__embed_predicate_zero() -> f32 {
//   %a = "neura.constant"() {value = 10.0 : f32, predicate = false} : () -> f32
//   %b = "neura.constant"() {value = 32.0 : f32, predicate = false} : () -> f32
//   %res = "neura.add"(%a, %b) : (f32, f32) -> f32
//   // [neura-interpreter]  → Output: 0.000000
//   return %res : f32
// }