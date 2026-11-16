// RUN: neura-interpreter %s --dataflow --verbose | head -n 400 \
// RUN:   | FileCheck %s --implicit-check-not="DFG Iteration 1"

// Each return fires before the adds that become ready at later topological
// levels. Execution must not end until the last add has run. head bounds the
// output if execution never ends.

// The void return is ready at level 1.
// CHECK: → Output: (void)
// CHECK: Result  : value = 8.000000e+00
// CHECK: Execution terminated after return at end of DFG iteration 0
func.func @void_return_with_late_ops() {
  %c = "neura.constant"() <{value = 1 : i32}> : () -> i32
  %a = "neura.add"(%c, %c) : (i32, i32) -> i32
  %b = "neura.add"(%a, %a) : (i32, i32) -> i32
  %d = "neura.add"(%b, %b) : (i32, i32) -> i32
  return
}

// The value return is ready at level 2, as soon as %c is valid.
// CHECK: → Output: 1.000000
// CHECK: Result  : value = 8.000000e+00
// CHECK: Execution terminated after return at end of DFG iteration 0
func.func @value_return_with_late_ops() -> i32 {
  %c = "neura.constant"() <{value = 1 : i32}> : () -> i32
  %a = "neura.add"(%c, %c) : (i32, i32) -> i32
  %b = "neura.add"(%a, %a) : (i32, i32) -> i32
  %d = "neura.add"(%b, %b) : (i32, i32) -> i32
  return %c : i32
}
