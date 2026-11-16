// RUN: split-file %s %t
// RUN: neura-interpreter %t/kernel.mlir --dataflow --verbose \
// RUN:   --memory-in=%t/memory.json | FileCheck %s

// neura.kernel entry without load/store/mac: input #0 gets its contents from
// --memory-in, input #1 has no entry and gets zeros, and a body without
// load/store runs once.

// CHECK:      Entering neura.kernel
// CHECK-NEXT: Bound input #0 to memref buffer (4 elements)
// CHECK-NEXT: Bound input #1 to memref buffer (3 elements)
// CHECK-NEXT: No load/store found in kernel body; trip count = 1
// CHECK:      Final memory[input #0] = [1, 2, 3, 4]
// CHECK-NEXT: Final memory[input #1] = [0, 0, 0]

//--- kernel.mlir
func.func @test_kernel_entry(%a: memref<2x2xi32>, %b: memref<3xi32>) {
  neura.kernel inputs(%a, %b : memref<2x2xi32>, memref<3xi32>) {
  ^bb0(%arg_a: memref<2x2xi32>, %arg_b: memref<3xi32>):
    neura.yield
  }
  return
}

//--- memory.json
{"0": [1, 2, 3, 4]}
