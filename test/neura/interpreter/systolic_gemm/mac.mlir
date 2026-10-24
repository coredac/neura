// RUN: split-file %s %t
// RUN: neura-interpreter %t/kernel.mlir --dataflow --verbose \
// RUN:   --memory-in=%t/memory.json | FileCheck %s

// Two chained neura.mac operations. The stationary map (x, y) -> (0) selects
// weight 3 from %w for both, and the load gives input0 = 5 from %a.
// - The first MAC has no input1: 5 * 3 + 0 = 15.
// - The second MAC takes the first MAC's forwarded input0 and its result as
//   input1: 5 * 3 + 15 = 30.

// CHECK: input0 = 5.000000e+00, stationary = 3, input1 = 0.000000e+00
// CHECK: result = 1.500000e+01, forwarded = 5.000000e+00
// CHECK: input0 = 5.000000e+00, stationary = 3, input1 = 1.500000e+01
// CHECK: result = 3.000000e+01, forwarded = 5.000000e+00
// CHECK: Final memory[input #2] = [15, 30]

//--- kernel.mlir
func.func @test_mac(%a: memref<1xi32>, %w: memref<1xi32>, %c: memref<2xi32>) {
  neura.kernel inputs(%a, %w, %c : memref<1xi32>, memref<1xi32>, memref<2xi32>)
    attributes {
      kernel_metadata = {kind = "template", template = {name = "systolic_array",
        stationary = {kernel_input = 1 : i32, map = affine_map<(x, y) -> (0)>}}}
    } {
  ^bb0(%arg_a: memref<1xi32>, %arg_w: memref<1xi32>, %arg_c: memref<2xi32>):
    %v = "neura.load"(%arg_a) {constants = array<i64: 0>} : (memref<1xi32>) -> i32
    %first:2 = "neura.mac"(%v) {placement = {x = 0 : i32, y = 0 : i32}}
               : (i32) -> (i32, i32)
    %second:2 = "neura.mac"(%first#1, %first#0) {placement = {x = 1 : i32, y = 0 : i32}}
                : (i32, i32) -> (i32, i32)
    "neura.store"(%first#0, %arg_c) {constants = array<i64: 0>} : (i32, memref<2xi32>) -> ()
    "neura.store"(%second#0, %arg_c) {constants = array<i64: 1>} : (i32, memref<2xi32>) -> ()
    neura.yield
  }
  return
}

//--- memory.json
{"0": [5], "1": [3]}
