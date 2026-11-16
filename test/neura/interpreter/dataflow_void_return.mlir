// RUN: neura-interpreter %s --dataflow | head -n 20 | FileCheck %s

// A void return has no predicate to wait on, so in dataflow mode it must end
// execution in the DFG pass where it first fires; it used to never terminate.
// head bounds the output so a regression fails (pipefail) instead of hanging
// the test.

// CHECK:      → Output: (void)
func.func @void_func_return() {
  return
}

// CHECK-NEXT: → Output: (void)
func.func @void_neura_return() {
  "neura.return"() : () -> ()
}

// CHECK-NEXT: → Output: 7.000000
func.func @value_func_return() -> f32 {
  %c = "neura.constant"() <{predicate = true, value = 7.000000e+00 : f32}> : () -> f32
  return %c : f32
}

// A value return still waits for a true predicate: the reserve is unset on
// the first pass and only carries the value on the second.
// CHECK-NEXT: → Output: 0.000000
// CHECK-NEXT: → Output: 42.000000
// CHECK-NOT:  Output
func.func @value_return_waits_for_predicate() -> f32 {
  %c = "neura.grant_once"() <{constant_value = 4.200000e+01 : f32}> : () -> f32
  %r = neura.reserve : f32
  neura.ctrl_mov %c -> %r : f32 f32
  return %r : f32
}
