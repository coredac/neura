// RUN: split-file %s %t
// RUN: neura-interpreter %t/kernel.mlir --dataflow --memory-in=%t/memory.json \
// RUN:   | FileCheck %s
// RUN: neura-interpreter %t/kernel.mlir --memory-in=%t/memory.json | FileCheck %s

// a[i] = a[i] + k, written back only where a[i] > 0. The kernel runs three
// passes, so this checks that:
// - the scalar input %k stays valid on every pass, not only the first;
// - a store whose value has a false predicate does not write memory;
// - such a store still uses up its constants[k] slot, so the next pass
//   writes the next element.

// CHECK: Final memory[input #0] = [6, -3, 8]

//--- kernel.mlir
func.func @predicated_store(%a: memref<3xi32>) {
  %k = "neura.constant"() <{value = 1 : i32}> : () -> i32
  neura.kernel inputs(%a, %k : memref<3xi32>, i32) {
  ^bb0(%arg: memref<3xi32>, %kk: i32):
    %v = "neura.load"(%arg) {constants = array<i64: 0, 1, 2>} : (memref<3xi32>) -> i32
    %sum = "neura.add"(%v, %kk) : (i32, i32) -> i32
    %positive = "neura.icmp"(%v) {cmpType = "sgt", rhs_value = 0 : i32} : (i32) -> i1
    %kept = neura.grant_predicate %sum, %positive : i32, i1 -> i32
    "neura.store"(%kept, %arg) {constants = array<i64: 0, 1, 2>} : (i32, memref<3xi32>) -> ()
    neura.yield
  }
  return
}

//--- memory.json
{"0": [5, -3, 7]}
