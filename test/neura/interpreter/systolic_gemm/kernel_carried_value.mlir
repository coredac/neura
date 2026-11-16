// RUN: split-file %s %t
// RUN: neura-interpreter %t/kernel.mlir --dataflow --memory-in=%t/memory.json \
// RUN:   | FileCheck %s
// RUN: neura-interpreter %t/kernel.mlir --memory-in=%t/memory.json | FileCheck %s

// A running sum carried from one pass to the next: phi picks the grant_once
// zero on the first pass and the reserve, which ctrl_mov fills, after that.
// A kernel body is a dataflow graph, so it must behave the same in both modes.

// CHECK: Final memory[input #1] = [1, 3, 6]

//--- kernel.mlir
func.func @prefix_sum(%a: memref<3xi32>, %c: memref<3xi32>) {
  neura.kernel inputs(%a, %c : memref<3xi32>, memref<3xi32>) {
  ^bb0(%arg_a: memref<3xi32>, %arg_c: memref<3xi32>):
    %zero = "neura.grant_once"() <{constant_value = 0 : i32}> : () -> i32
    %carried = neura.reserve : i32
    %acc = "neura.phi"(%carried, %zero) : (i32, i32) -> i32
    %v = "neura.load"(%arg_a) {constants = array<i64: 0, 1, 2>} : (memref<3xi32>) -> i32
    %sum = "neura.add"(%acc, %v) : (i32, i32) -> i32
    "neura.store"(%sum, %arg_c) {constants = array<i64: 0, 1, 2>} : (i32, memref<3xi32>) -> ()
    neura.ctrl_mov %sum -> %carried : i32 i32
    neura.yield
  }
  return
}

//--- memory.json
{"0": [1, 2, 3]}
