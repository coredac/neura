// RUN: split-file %s %t
// RUN: neura-interpreter %t/kernel.mlir --dataflow --verbose \
// RUN:   --memory-in=%t/memory.json | FileCheck %s

// Configured neura.load/neura.store without neura.mac: doubles each element
// of %a in place. The three-element constants arrays give a trip count of 3,
// so the load/add/store chain fires once per offset.

// CHECK: Derived trip count = 3 from load/store constants
// CHECK: Loaded element 0 (firing #0) = 1.000000e+01
// CHECK: Stored element 0 (firing #0) = 20
// CHECK: Loaded element 1 (firing #1) = 2.000000e+01
// CHECK: Stored element 1 (firing #1) = 40
// CHECK: Loaded element 2 (firing #2) = 3.000000e+01
// CHECK: Stored element 2 (firing #2) = 60
// CHECK: Final memory[input #0] = [20, 40, 60]

//--- kernel.mlir
func.func @test_load_add_store(%a: memref<3xi32>) {
  neura.kernel inputs(%a : memref<3xi32>) {
  ^bb0(%arg: memref<3xi32>):
    %v = "neura.load"(%arg) {constants = array<i64: 0, 1, 2>}
         : (memref<3xi32>) -> i32
    %doubled = "neura.add"(%v, %v) : (i32, i32) -> i32
    "neura.store"(%doubled, %arg) {constants = array<i64: 0, 1, 2>}
         : (i32, memref<3xi32>) -> ()
    neura.yield
  }
  return
}

//--- memory.json
{"0": [10, 20, 30]}
