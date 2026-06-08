// RUN: neura-interpreter %s --verbose | FileCheck %s

// Test that neura.yield is handled correctly as a kernel body terminator.
// Before this fix, the interpreter aborted with "Unhandled op: neura.yield".

func.func @test_yield_no_operands() {
  %0 = "neura.grant_once"() <{constant_value = 0 : i32}> : () -> !neura.data<i32, i1>
  neura.yield
  // CHECK: [neura-interpreter]  Executing neura.yield
}
