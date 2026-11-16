// RUN: split-file %s %t
// RUN: not neura-interpreter %t/scalar_input.mlir --dataflow 2>&1 | FileCheck %t/scalar_input.mlir
// RUN: not neura-interpreter %t/kernel_result.mlir --dataflow 2>&1 | FileCheck %t/kernel_result.mlir
// RUN: not neura-interpreter %t/memref_type.mlir --dataflow 2>&1 | FileCheck %t/memref_type.mlir
// RUN: not neura-interpreter %t/trip_count.mlir --dataflow 2>&1 | FileCheck %t/trip_count.mlir
// RUN: not neura-interpreter %t/map_arity.mlir --dataflow 2>&1 | FileCheck %t/map_arity.mlir
// RUN: not neura-interpreter %t/nested_kernel.mlir --dataflow 2>&1 | FileCheck %t/nested_kernel.mlir
// RUN: not neura-interpreter %t/folded_store.mlir --dataflow 2>&1 | FileCheck %t/folded_store.mlir
// RUN: not neura-interpreter %t/one_input.mlir --memory-in=%t/size.json 2>&1 | FileCheck %s --check-prefix=SIZE
// RUN: not neura-interpreter %t/one_input.mlir --memory-in=%t/index.json 2>&1 | FileCheck %s --check-prefix=INDEX
// RUN: not neura-interpreter %t/one_input.mlir --memory-in=%t/not_integer.json 2>&1 | FileCheck %s --check-prefix=NOT-INTEGER
// RUN: not neura-interpreter %t/one_input.mlir --memory-in=%t/range.json 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not neura-interpreter %t/one_input.mlir --memory-in=%t/bad.json 2>&1 | FileCheck %s --check-prefix=BAD-JSON
// RUN: not neura-interpreter %t/one_input.mlir --memory-in=%t/not_object.json 2>&1 | FileCheck %s --check-prefix=NOT-OBJECT
// RUN: not neura-interpreter %t/one_input.mlir --memory-in=%t/missing.json 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not neura-interpreter %t/two_kernels.mlir --memory-in=%t/size_ok.json 2>&1 | FileCheck %s --check-prefix=TWO-KERNELS

// neura.kernel inputs and forms that the interpreter does not support, and bad
// --memory-in files, give an error instead of a wrong result.

// SIZE: neura.kernel: --memory-in entry for input #0 has 2 elements, expected 3
// INDEX: neura.kernel: --memory-in entry 1 is not a memref input of the kernel
// NOT-INTEGER: --memory-in: entry "0" must contain only integers
// RANGE: neura.kernel: --memory-in value 4294967296 for input #0 does not fit in i32
// BAD-JSON: --memory-in: {{.*}}bad.json:
// NOT-OBJECT: --memory-in: expected an object that maps kernel input indices to arrays of integers
// MISSING: --memory-in: cannot read {{.*}}missing.json
// TWO-KERNELS: --memory-in and --memory-out support only one neura.kernel per run

//--- scalar_input.mlir
// The interpreter does not bind function arguments.
func.func @scalar_input(%k: i32) {
  neura.kernel inputs(%k : i32) {
  ^bb0(%arg: i32):
    neura.yield
  }
  return
}
// CHECK: neura.kernel: scalar input #0 has no value

//--- kernel_result.mlir
func.func @kernel_result(%a: memref<1xi32>) {
  %r = neura.kernel inputs(%a : memref<1xi32>) {
  ^bb0(%arg: memref<1xi32>):
    %v = "neura.load"(%arg) {constants = array<i64: 0>} : (memref<1xi32>) -> i32
    neura.yield results(%v : i32)
  } : i32
  return
}
// CHECK: neura.kernel: results and iter_args_init are not supported

//--- memref_type.mlir
func.func @memref_type(%a: memref<2xf32>) {
  neura.kernel inputs(%a : memref<2xf32>) {
  ^bb0(%arg: memref<2xf32>):
    neura.yield
  }
  return
}
// CHECK: neura.kernel: memref inputs must be static identity-layout i32 memrefs

//--- trip_count.mlir
func.func @trip_count(%a: memref<3xi32>) {
  neura.kernel inputs(%a : memref<3xi32>) {
  ^bb0(%arg: memref<3xi32>):
    %v = "neura.load"(%arg) {constants = array<i64: 0, 1>} : (memref<3xi32>) -> i32
    "neura.store"(%v, %arg) {constants = array<i64: 0, 1, 2>} : (i32, memref<3xi32>) -> ()
    neura.yield
  }
  return
}
// CHECK: neura.kernel: load/store constants arrays disagree on trip count

//--- map_arity.mlir
func.func @map_arity(%a: memref<1xi32>, %w: memref<1xi32>) {
  neura.kernel inputs(%a, %w : memref<1xi32>, memref<1xi32>) attributes {
    kernel_metadata = {kind = "template", template = {name = "systolic_array",
      stationary = {kernel_input = 1 : i32, map = affine_map<(x) -> (0)>}}}
  } {
  ^bb0(%arg_a: memref<1xi32>, %arg_w: memref<1xi32>):
    %v = "neura.load"(%arg_a) {constants = array<i64: 0>} : (memref<1xi32>) -> i32
    %r:2 = "neura.mac"(%v) {placement = {x = 0 : i32, y = 0 : i32}} : (i32) -> (i32, i32)
    neura.yield
  }
  return
}
// CHECK: neura.mac: stationary map must have two dimensions (x, y) and no symbols

//--- nested_kernel.mlir
func.func @nested_kernel(%a: memref<1xi32>) {
  neura.kernel inputs(%a : memref<1xi32>) {
  ^bb0(%arg: memref<1xi32>):
    neura.kernel inputs(%arg : memref<1xi32>) {
    ^bb0(%inner: memref<1xi32>):
      neura.yield
    }
    neura.yield
  }
  return
}
// CHECK: neura.kernel: nested kernels are not supported

//--- folded_store.mlir
func.func @folded_store(%a: memref<1xi32>) {
  neura.kernel inputs(%a : memref<1xi32>) {
  ^bb0(%arg: memref<1xi32>):
    "neura.store"(%arg) {lhs_value = 5 : i32, constants = array<i64: 0>} : (memref<1xi32>) -> ()
    neura.yield
  }
  return
}
// CHECK: neura.store: the folded-value (lhs_value) form is not supported

//--- one_input.mlir
func.func @one_input(%a: memref<3xi32>) {
  neura.kernel inputs(%a : memref<3xi32>) {
  ^bb0(%arg: memref<3xi32>):
    neura.yield
  }
  return
}

//--- two_kernels.mlir
func.func @two_kernels(%a: memref<3xi32>) {
  neura.kernel inputs(%a : memref<3xi32>) {
  ^bb0(%arg: memref<3xi32>):
    neura.yield
  }
  neura.kernel inputs(%a : memref<3xi32>) {
  ^bb0(%arg: memref<3xi32>):
    neura.yield
  }
  return
}

//--- size.json
{"0": [1, 2]}

//--- size_ok.json
{"0": [1, 2, 3]}

//--- index.json
{"1": [1, 2, 3]}

//--- not_integer.json
{"0": [1, 2.5, 3]}

//--- range.json
{"0": [1, 2, 4294967296]}

//--- bad.json
{"0": [1, 2,

//--- not_object.json
[1, 2, 3]
