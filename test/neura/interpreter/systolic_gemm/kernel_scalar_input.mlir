// RUN: split-file %s %t
// RUN: neura-interpreter %t/kernel.mlir --dataflow --memory-in=%t/memory.json \
// RUN:   | FileCheck %s
// RUN: neura-interpreter %t/kernel.mlir --memory-in=%t/memory.json | FileCheck %s

// A scalar kernel input computed in the function. In dataflow mode the kernel
// runs once %k is ready, like any other op.

// CHECK: Final memory[input #0] = [11]

//--- kernel.mlir
func.func @scalar_input(%a: memref<1xi32>) {
  %k = "neura.constant"() <{value = 1 : i32}> : () -> i32
  neura.kernel inputs(%a, %k : memref<1xi32>, i32) {
  ^bb0(%arg: memref<1xi32>, %kk: i32):
    %v = "neura.load"(%arg) {constants = array<i64: 0>} : (memref<1xi32>) -> i32
    %s = "neura.add"(%v, %kk) : (i32, i32) -> i32
    "neura.store"(%s, %arg) {constants = array<i64: 0>} : (i32, memref<1xi32>) -> ()
    neura.yield
  }
  return
}

//--- memory.json
{"0": [10]}
