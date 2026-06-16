// RUN: mlir-neura-opt --emulate-math-with-arith %s | FileCheck %s

// Test: fpowi with exponent 3 -> x * x * x
func.func @test_fpowi(%x: f32) -> f32 {
  %c3 = arith.constant 3 : i32
  %result = math.fpowi %x, %c3 : f32, i32
  return %result : f32
}

// CHECK:   func.func @test_fpowi(%arg0: f32) -> f32 {
// CHECK:     %0 = arith.mulf %arg0, %arg0 : f32
// CHECK:     %1 = arith.mulf %0, %arg0 : f32
// CHECK:     return %1 : f32
// CHECK:   }

// Test: fpowi with exponent 1 -> identity (no mulf)
func.func @test_fpowi_one(%x: f32) -> f32 {
  %c1 = arith.constant 1 : i32
  %result = math.fpowi %x, %c1 : f32, i32
  return %result : f32
}

// CHECK:   func.func @test_fpowi_one(%arg0: f32) -> f32 {
// CHECK:     return %arg0 : f32
// CHECK:   }

// Test: tanh -> (exp(2x) - 1) / (exp(2x) + 1)
func.func @test_tanh(%x: f32) -> f32 {
  %result = math.tanh %x : f32
  return %result : f32
}

// CHECK:   func.func @test_tanh(%arg0: f32) -> f32 {
// CHECK:     %cst = arith.constant 2.000000e+00 : f32
// CHECK:     %cst_0 = arith.constant 1.000000e+00 : f32
// CHECK:     %0 = arith.mulf %arg0, %cst : f32
// CHECK:     %1 = math.exp %0 : f32
// CHECK:     %2 = arith.subf %1, %cst_0 : f32
// CHECK:     %3 = arith.addf %1, %cst_0 : f32
// CHECK:     %4 = arith.divf %2, %3 : f32
// CHECK:     return %4 : f32
// CHECK:   }
