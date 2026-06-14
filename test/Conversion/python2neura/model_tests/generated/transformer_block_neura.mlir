// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module {
  memref.global "private" constant @__constant_8xf32_2 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32_3> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8xf32_1 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32_2> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense_resource<torch_tensor_8_16_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x8xf32 : memref<16x8xf32> = dense_resource<torch_tensor_16_8_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8xf32_0 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32_1> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32_1 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32_2> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32_0 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32_1> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32> {alignment = 64 : i64}
  func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %cst_1 = arith.constant 8.000000e+00 : f64
    %cst_2 = arith.constant 2.8284271247461903 : f64
    %cst_3 = arith.constant 1.000000e-05 : f64
    %cst_4 = arith.constant 5.000000e-01 : f32
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant 7.977240e-01 : f32
    %cst_7 = arith.constant 4.471500e-02 : f32
    %cst_8 = arith.constant 0.000000e+00 : f64
    %cst_9 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_10 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %1 = memref.get_global @__constant_8x8xf32_0 : memref<8x8xf32>
    %2 = memref.get_global @__constant_8x8xf32_1 : memref<8x8xf32>
    %3 = memref.get_global @__constant_8xf32 : memref<8xf32>
    %4 = memref.get_global @__constant_8xf32_0 : memref<8xf32>
    %5 = memref.get_global @__constant_16x8xf32 : memref<16x8xf32>
    %6 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
    %7 = memref.get_global @__constant_8xf32_1 : memref<8xf32>
    %8 = memref.get_global @__constant_8xf32_2 : memref<8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
    neura.kernel inputs(%0, %alloc : memref<8x8xf32>, memref<8x8xf32>) {
    ^bb0(%arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>):
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%10, %9] : memref<8x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<8x8xf32>
      neura.yield
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%cst_10, %alloc_12 : f32, memref<4x8xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc_12, %alloc_13 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %alloc, %alloc_13 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %11 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %12 = memref.load %arg1[%9, %11] : memref<4x8xf32>
      %13 = memref.load %arg2[%11, %10] : memref<8x8xf32>
      %14 = memref.load %arg3[%9, %10] : memref<4x8xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg3[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%1, %alloc : memref<8x8xf32>, memref<8x8xf32>) {
    ^bb0(%arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>):
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%10, %9] : memref<8x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<8x8xf32>
      neura.yield
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc_12, %alloc_14 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %alloc, %alloc_14 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %11 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %12 = memref.load %arg1[%9, %11] : memref<4x8xf32>
      %13 = memref.load %arg2[%11, %10] : memref<8x8xf32>
      %14 = memref.load %arg3[%9, %10] : memref<4x8xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg3[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%2, %alloc : memref<8x8xf32>, memref<8x8xf32>) {
    ^bb0(%arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>):
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%10, %9] : memref<8x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<8x8xf32>
      neura.yield
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc_12, %alloc_15 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %alloc, %alloc_15 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %11 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %12 = memref.load %arg1[%9, %11] : memref<4x8xf32>
      %13 = memref.load %arg2[%11, %10] : memref<8x8xf32>
      %14 = memref.load %arg3[%9, %10] : memref<4x8xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg3[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<8x4xf32>
    neura.kernel inputs(%alloc_14, %alloc_16 : memref<4x8xf32>, memref<8x4xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>):
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%10, %9] : memref<4x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<8x4xf32>
      neura.yield
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
    neura.kernel inputs(%cst_10, %alloc_17 : f32, memref<4x4xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%9, %10] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_13, %alloc_16, %alloc_17 : memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>, %arg3: memref<4x4xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %11 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %12 = memref.load %arg1[%9, %11] : memref<4x8xf32>
      %13 = memref.load %arg2[%11, %10] : memref<8x4xf32>
      %14 = memref.load %arg3[%9, %10] : memref<4x4xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg3[%9, %10] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_17, %cst_2 : memref<4x4xf32>, f64) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: f64):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x4xf32>
      %12 = arith.truncf %arg2 : f64 to f32
      %13 = arith.divf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x4xf32>
      neura.yield
    }
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
    neura.kernel inputs(%c0_i64, %alloc_18 : i64, memref<4xi64>) {
    ^bb0(%arg1: i64, %arg2: memref<4xi64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} -> index
      memref.store %arg1, %arg2[%9] : memref<4xi64>
      neura.yield
    }
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    neura.kernel inputs(%cst_9, %alloc_19 : f32, memref<4xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} -> index
      memref.store %arg1, %arg2[%9] : memref<4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_17, %alloc_19, %alloc_18 : memref<4x4xf32>, memref<4xf32>, memref<4xi64>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4xf32>, %arg3: memref<4xi64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x4xf32>
      %12 = memref.load %arg2[%9] : memref<4xf32>
      %13 = memref.load %arg3[%9] : memref<4xi64>
      %14 = arith.index_cast %10 : index to i64
      %15 = arith.maximumf %11, %12 : f32
      %16 = arith.cmpf ogt, %11, %12 : f32
      %17 = arith.select %16, %14, %13 : i64
      memref.store %15, %arg2[%9] : memref<4xf32>
      memref.store %17, %arg3[%9] : memref<4xi64>
      neura.yield
    }
    neura.kernel inputs(%alloc_17, %alloc_19 : memref<4x4xf32>, memref<4xf32>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x4xf32>
      %12 = memref.load %arg2[%9] : memref<4xf32>
      %13 = arith.subf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_17 : memref<4x4xf32>) {
    ^bb0(%arg1: memref<4x4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x4xf32>
      %12 = math.exp %11 : f32
      memref.store %12, %arg1[%9, %10] : memref<4x4xf32>
      neura.yield
    }
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%cst_10, %alloc_21 : f32, memref<4x1xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%alloc_21, %alloc_22 : memref<4x1xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf32>
      memref.store %11, %arg2[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_17, %alloc_22 : memref<4x4xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x4xf32>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf32>
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg2[%9, %c0] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_17, %alloc_22 : memref<4x4xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x4xf32>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf32>
      %13 = arith.divf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x4xf32>
      neura.yield
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc_12, %alloc_23 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_17, %alloc_15, %alloc_23 : memref<4x4xf32>, memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %11 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %12 = memref.load %arg1[%9, %11] : memref<4x4xf32>
      %13 = memref.load %arg2[%11, %10] : memref<4x8xf32>
      %14 = memref.load %arg3[%9, %10] : memref<4x8xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg3[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %alloc_23, %alloc_11 : memref<4x8xf32>, memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%9, %10] : memref<4x8xf32>
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg3[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf64>
    neura.kernel inputs(%alloc_11, %alloc_24 : memref<4x8xf32>, memref<4x8xf64>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = arith.extf %11 : f32 to f64
      memref.store %12, %arg2[%9, %10] : memref<4x8xf64>
      neura.yield
    }
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
    neura.kernel inputs(%cst_8, %alloc_26 : f64, memref<4x1xf64>) {
    ^bb0(%arg1: f64, %arg2: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%9, %10] : memref<4x1xf64>
      neura.yield
    }
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
    neura.kernel inputs(%alloc_26, %alloc_27 : memref<4x1xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      memref.store %11, %arg2[%9, %10] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_24, %alloc_27 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf64>
      %13 = arith.addf %11, %12 : f64
      memref.store %13, %arg2[%9, %c0] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_27, %cst_1, %alloc_25 : memref<4x1xf64>, f64, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      %12 = arith.divf %11, %arg2 : f64
      memref.store %12, %arg3[%9, %10] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_24, %alloc_25 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf64>
      %13 = arith.subf %11, %12 : f64
      memref.store %13, %arg1[%9, %10] : memref<4x8xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_24 : memref<4x8xf64>) {
    ^bb0(%arg1: memref<4x8xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %12 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %13 = arith.mulf %11, %12 : f64
      memref.store %13, %arg1[%9, %10] : memref<4x8xf64>
      neura.yield
    }
    %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
    neura.kernel inputs(%alloc_26, %alloc_28 : memref<4x1xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      memref.store %11, %arg2[%9, %10] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_24, %alloc_28 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf64>
      %13 = arith.addf %11, %12 : f64
      memref.store %13, %arg2[%9, %c0] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_28, %cst_1, %alloc_25 : memref<4x1xf64>, f64, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      %12 = arith.divf %11, %arg2 : f64
      memref.store %12, %arg3[%9, %10] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_25, %alloc_20 : memref<4x1xf64>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      %12 = arith.truncf %11 : f64 to f32
      memref.store %12, %arg2[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%alloc_21, %alloc_29 : memref<4x1xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf32>
      memref.store %11, %arg2[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_29 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf32>
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg2[%9, %c0] : memref<4x1xf32>
      neura.yield
    }
    %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%alloc_29, %cst_0, %alloc_30 : memref<4x1xf32>, f32, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f32, %arg3: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf32>
      %12 = arith.divf %11, %arg2 : f32
      memref.store %12, %arg3[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_20, %cst_3 : memref<4x1xf32>, f64) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f64):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf32>
      %12 = arith.truncf %arg2 : f64 to f32
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_20 : memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf32>
      %12 = math.rsqrt %11 : f32
      memref.store %12, %arg1[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_30 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf32>
      %13 = arith.subf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_20 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf32>
      %13 = arith.mulf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %3 : memref<4x8xf32>, memref<8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%10] : memref<8xf32>
      %13 = arith.mulf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %4 : memref<4x8xf32>, memref<8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%10] : memref<8xf32>
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
    neura.kernel inputs(%5, %alloc_31 : memref<16x8xf32>, memref<8x16xf32>) {
    ^bb0(%arg1: memref<16x8xf32>, %arg2: memref<8x16xf32>):
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%10, %9] : memref<16x8xf32>
      memref.store %11, %arg2[%9, %10] : memref<8x16xf32>
      neura.yield
    }
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<4x16xf32>
    neura.kernel inputs(%cst_10, %alloc_32 : f32, memref<4x16xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x16xf32>):
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%9, %10] : memref<4x16xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_31, %alloc_32 : memref<4x8xf32>, memref<8x16xf32>, memref<4x16xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x16xf32>, %arg3: memref<4x16xf32>):
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %11 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %12 = memref.load %arg1[%9, %11] : memref<4x8xf32>
      %13 = memref.load %arg2[%11, %10] : memref<8x16xf32>
      %14 = memref.load %arg3[%9, %10] : memref<4x16xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg3[%9, %10] : memref<4x16xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_32, %cst_7, %cst_6, %cst, %cst_5, %cst_4 : memref<4x16xf32>, f32, f32, f32, f32, f32) {
    ^bb0(%arg1: memref<4x16xf32>, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x16xf32>
      %12 = arith.mulf %11, %11 : f32
      %13 = arith.mulf %12, %11 : f32
      %14 = arith.mulf %13, %arg2 : f32
      %15 = arith.addf %11, %14 : f32
      %16 = arith.mulf %15, %arg3 : f32
      %17 = arith.mulf %16, %arg4 : f32
      %18 = math.exp %17 : f32
      %19 = arith.subf %18, %arg5 : f32
      %20 = arith.addf %18, %arg5 : f32
      %21 = arith.divf %19, %20 : f32
      %22 = arith.addf %21, %arg5 : f32
      %23 = arith.mulf %22, %arg6 : f32
      %24 = arith.mulf %11, %23 : f32
      memref.store %24, %arg1[%9, %10] : memref<4x16xf32>
      neura.yield
    }
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<16x8xf32>
    neura.kernel inputs(%6, %alloc_33 : memref<8x16xf32>, memref<16x8xf32>) {
    ^bb0(%arg1: memref<8x16xf32>, %arg2: memref<16x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%10, %9] : memref<8x16xf32>
      memref.store %11, %arg2[%9, %10] : memref<16x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_32, %alloc_33, %alloc_12 : memref<4x16xf32>, memref<16x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x16xf32>, %arg2: memref<16x8xf32>, %arg3: memref<4x8xf32>):
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %11 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %12 = memref.load %arg1[%9, %11] : memref<4x16xf32>
      %13 = memref.load %arg2[%11, %10] : memref<16x8xf32>
      %14 = memref.load %arg3[%9, %10] : memref<4x8xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg3[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_12 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%9, %10] : memref<4x8xf32>
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_24 : memref<4x8xf32>, memref<4x8xf64>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = arith.extf %11 : f32 to f64
      memref.store %12, %arg2[%9, %10] : memref<4x8xf64>
      neura.yield
    }
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
    neura.kernel inputs(%alloc_26, %alloc_34 : memref<4x1xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      memref.store %11, %arg2[%9, %10] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_24, %alloc_34 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf64>
      %13 = arith.addf %11, %12 : f64
      memref.store %13, %arg2[%9, %c0] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_34, %cst_1, %alloc_25 : memref<4x1xf64>, f64, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      %12 = arith.divf %11, %arg2 : f64
      memref.store %12, %arg3[%9, %10] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_24, %alloc_25 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf64>
      %13 = arith.subf %11, %12 : f64
      memref.store %13, %arg1[%9, %10] : memref<4x8xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_24 : memref<4x8xf64>) {
    ^bb0(%arg1: memref<4x8xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %12 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %13 = arith.mulf %11, %12 : f64
      memref.store %13, %arg1[%9, %10] : memref<4x8xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_24, %alloc_26 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf64>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf64>
      %13 = arith.addf %11, %12 : f64
      memref.store %13, %arg2[%9, %c0] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_26, %cst_1, %alloc_25 : memref<4x1xf64>, f64, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      %12 = arith.divf %11, %arg2 : f64
      memref.store %12, %arg3[%9, %10] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_25, %alloc_20 : memref<4x1xf64>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf64>
      %12 = arith.truncf %11 : f64 to f32
      memref.store %12, %arg2[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_21 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf32>
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg2[%9, %c0] : memref<4x1xf32>
      neura.yield
    }
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%alloc_21, %cst_0, %alloc_35 : memref<4x1xf32>, f32, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f32, %arg3: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf32>
      %12 = arith.divf %11, %arg2 : f32
      memref.store %12, %arg3[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_20, %cst_3 : memref<4x1xf32>, f64) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f64):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf32>
      %12 = arith.truncf %arg2 : f64 to f32
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_20 : memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x1xf32>
      %12 = math.rsqrt %11 : f32
      memref.store %12, %arg1[%9, %10] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_35 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf32>
      %13 = arith.subf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %alloc_20 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%9, %c0] : memref<4x1xf32>
      %13 = arith.mulf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %7 : memref<4x8xf32>, memref<8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%10] : memref<8xf32>
      %13 = arith.mulf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %8 : memref<4x8xf32>, memref<8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %9 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %10 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %11 = memref.load %arg1[%9, %10] : memref<4x8xf32>
      %12 = memref.load %arg2[%10] : memref<8xf32>
      %13 = arith.addf %11, %12 : f32
      memref.store %13, %arg1[%9, %10] : memref<4x8xf32>
      neura.yield
    }
    return %alloc_11 : memref<4x8xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_torch.float32_3: "0x040000000000000000000000000000000000000000000000000000000000000000000000",
      torch_tensor_8_torch.float32_2: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F",
      torch_tensor_8_16_torch.float32: "0x04000000D2ED31BEE00553BDE6963D3EC23C033E76FE21BE56484DBEECF02EBE04A47CBE228A45BE883B7DBDC0C42C3E40662B3D0AB742BE8C5E4DBE14B5FE3D626C3EBE402AFCBC089CF53DB4F5ECBD5050DFBC6044B2BCE84272BDDCCC01BE4E3564BE7CF04EBE041409BE2441773EB44DF7BD42E92BBE3036783DCC1C8D3DD0430C3E4097423E4E8F0E3E02D37DBE60A2B53CA2121B3E9049BDBC00DB16BE140E743E9480BFBD90C011BE50CC66BE20F6363C58E2E23DF0AC623DE87E4A3DAE2542BE3C136FBE8041903BF66D693E1EB1133E4E0D15BE88EC04BD10B83CBE3400F7BDD871393DE0660B3E0210543E00096BBE5E2A2B3E068E34BE2CBABF3D88A3583E0012663B3EE9683E0C205ABE2890C3BD1050153EC0185FBDF09C51BDB865D5BD5E76303E3C23FB3D0812A43DDADC0FBED2CE4FBE5083DD3CDCB1973D8065ECBDA4418FBDE6E42C3EA024A33C00CE383CC8017CBDBAD467BEB6B470BE64BEF4BD5A2102BEF4839F3D780C95BD444EC8BDFA13743E6057B23DC480363EA0DDF7BD9022D1BD442EBC3D02912ABE265927BEE0D545BCFC44BBBD96E93FBE44D9173E9EDD4D3EE01D263DC04A32BD3A206DBE807DBABD0C59823D806CF13D987301BDB06BCABD5EA60E3EDAE04BBEEECB213E04A2C6BD80C37A3BD05B4ABD204CF83CC0B99ABDFE243A3E00F8D4BB5ED5473E9623763E1871F9BD1CC13ABEAA634D3E",
      torch_tensor_16_8_torch.float32: "0x04000000E86A673E64919B3E0D5D0FBD5F1EA2BD1E0EA93BFCEEACBC8C13AE3D2BE9CA3D7B69A4BE1E9B05BE1471983EB2080D3E2E0E90BC73465ABE427F5DBEB026A2BEA205ECBD2186F43DFCD7663EFC26273ECE02A0BE6AB759BEFD8A64BDE51BAF3E867C513DCD83BBBDEAC7153ED3E409BEE2546ABEAB1F843EA09E24BE472791BD2D14B4BE464D723E8225893EC0F1033EB3707CBEBBA7B2BE250593BE75FC863E1DD12D3E1D54983E41A93D3E2042B73DB3E662BB92AA89BEA3179BBEF250A9BE7234143ED8BF31BEF1B891BDEB5A50BEA7F183BDC1D07EBECE8F6CBE2D2FF03D7331D7BD8A805F3E554AE8BD34CA84BEC7827FBD4F7F2FBE7E86DDBD6154ACBEAE8E4A3E8F0D7CBE90F7353E9549243E7D53813EECD98ABE4330823E84272BBE6657063E77FFA93E673C4CBD233033BB6AB4A6BDFA2597BE4FBE2D3E7DB3B3BEEAC0603E7671873EEE30AB3E87CBAABDABBB94BE64CBA23D0300483E8D2BB4BE8E5AA4BDAC0659BE7152FDBC483332BE5BFD13BE62DDE5BDFE05ACBE1987943E11C1973EB02B63BD39EB24BD54C113BECA78A3BE7028B0BE038E063EA4C646BE0FBF68BEB4FAE1BC3620F1BD0A41EABDAF1F3A3CF79599BD0F4FF9BDA4582DBE895193BE38C9973E1AE210BE9ADDBF3DEB3AFBBD985BEB3CB0C3A83EA7CC263E0DDF9CBEB4B10F3ED0D4AB3E207EBE3D3CB8723EF776B23E38E65DBD1549963D",
      torch_tensor_8_torch.float32_1: "0x040000000000000000000000000000000000000000000000000000000000000000000000",
      torch_tensor_8_torch.float32: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F",
      torch_tensor_8_8_torch.float32_2: "0x040000007DC7973E0CB48FBD9ABF9B3E0A9CE13D604999BE238B7A3EE339C7BD63C70ABE414296BE59F6B3BE8E2ACF3D962B9EBD35F30C3E538E94BEDF64863EBFE084BEC60E7ABDE13C973D32E73A3E8723923EB7E6A43EC4888FBE003AB63DE4B81BBE1DB21EBDBE7D87BE0EE2A43EF6DB84BEF57D413E2E78FE3D114BEB3D1EBC43BEAD89A43EC51D9F3D6B4A3A3D2F869FBECBFA173EC74059BDD4DB25BE667C9B3EA26EA13DB54E48BEEB3D37BEBA508ABCFC254A3EB806B9BDF7914EBE83F4F7BD283C87BE601D013EC21D8C3E3D6BAABEDF2AA83DB5063B3EE74B833D4FEE00BE97F63C3E8F413E3E4960073EC6787EBD91BCBFBDC7EC1A3D8DD27FBD9BC6D7BD",
      torch_tensor_8_8_torch.float32_1: "0x040000007CDEDD3DF8DA96BDCE22963EBD9456BE4CEB57BEA3EE57BE08D1A23EB14CF13D9C2FAE3E166495BE7B8CB3BE7B9F8DBE1A8A73BEE4A3123E26A3013ED369963E49F73ABE58CE76BEFD16403E705612BEDABA5B3EF2D2ABBD4D1A4F3E6FA58CBEE6B336BE0EC1DC3D4E13993D1F9CB8BDD9CC573E021E763E394583BEEB4741BEA5C0A53EC553F4BD405900BE4127AFBE2A544FBE8EE0B43D83253FBD3F6683BE2CE0073C4A4D77BE829399BE755C47BE296E9EBE188666BED3F2B43E94C2883DA721DF3D7AD5A8BE0BC76DBE6B03F1BD1673623D64489FBE22FE1BBEBDBD58BE396C803A3BB706BEB4B3C8BC025475BE688078BEC93653BE8DD9F7BD07E08EBE",
      torch_tensor_8_8_torch.float32: "0x0400000071658A3E5B3F963EA3A1A9BD5249A63EDFA59EBDB01C923DA64230BE799E543E89939F3E06CD84BE60579D3E8F84873D1BBD853EDA1F443D1C922E3E67774CBD958B8B3ED70C563D9C0329BEEB90B83DF2CD26BE48D429BD860B13BE7B2A703E2AE48EBE50E726BE0E76CCBD13AF59BE6DAE083DFDC9B2BEEF7AA33E2AC599BE7BC08B3E2C00713D771CEBBDC7B85F3E2EB2613DCA41923ED34F1E3DC55AE4BD3A8CC23DDA5AC4BDE05D183E3A9EA13EA847513EFB451EBEFCFD503E728E813D1DDB373E16AA5CBE4C31B3BEA1E00BBE91D88ABE8D88943E958ED03D19F6153E5FFFE43D5789C9BBCFAA8D3EE09D80BEC85CB63C1F1B77BE4C45DF3D025CF9BD"
    }
  }
#-}


// DATAFLOW_IR: module {
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8xf32_2 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32_3> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8xf32_1 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32_2> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense_resource<torch_tensor_8_16_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_16x8xf32 : memref<16x8xf32> = dense_resource<torch_tensor_16_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8xf32_0 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32_1> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x8xf32_1 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32_2> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x8xf32_0 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32_1> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
// DATAFLOW_IR-NEXT: %cst = arith.constant 2.000000e+00 : f32
// DATAFLOW_IR-NEXT: %cst_0 = arith.constant 8.000000e+00 : f32
// DATAFLOW_IR-NEXT: %cst_1 = arith.constant 8.000000e+00 : f64
// DATAFLOW_IR-NEXT: %cst_2 = arith.constant 2.8284271247461903 : f64
// DATAFLOW_IR-NEXT: %cst_3 = arith.constant 1.000000e-05 : f64
// DATAFLOW_IR-NEXT: %cst_4 = arith.constant 5.000000e-01 : f32
// DATAFLOW_IR-NEXT: %cst_5 = arith.constant 1.000000e+00 : f32
// DATAFLOW_IR-NEXT: %cst_6 = arith.constant 7.977240e-01 : f32
// DATAFLOW_IR-NEXT: %cst_7 = arith.constant 4.471500e-02 : f32
// DATAFLOW_IR-NEXT: %cst_8 = arith.constant 0.000000e+00 : f64
// DATAFLOW_IR-NEXT: %cst_9 = arith.constant 0xFF800000 : f32
// DATAFLOW_IR-NEXT: %c0_i64 = arith.constant 0 : i64
// DATAFLOW_IR-NEXT: %cst_10 = arith.constant 0.000000e+00 : f32
// DATAFLOW_IR-NEXT: %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
// DATAFLOW_IR-NEXT: %1 = memref.get_global @__constant_8x8xf32_0 : memref<8x8xf32>
// DATAFLOW_IR-NEXT: %2 = memref.get_global @__constant_8x8xf32_1 : memref<8x8xf32>
// DATAFLOW_IR-NEXT: %3 = memref.get_global @__constant_8xf32 : memref<8xf32>
// DATAFLOW_IR-NEXT: %4 = memref.get_global @__constant_8xf32_0 : memref<8xf32>
// DATAFLOW_IR-NEXT: %5 = memref.get_global @__constant_16x8xf32 : memref<16x8xf32>
// DATAFLOW_IR-NEXT: %6 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
// DATAFLOW_IR-NEXT: %7 = memref.get_global @__constant_8xf32_1 : memref<8xf32>
// DATAFLOW_IR-NEXT: %8 = memref.get_global @__constant_8xf32_2 : memref<8xf32>
// DATAFLOW_IR-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%0, %alloc : memref<8x8xf32>, memref<8x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_10, %alloc_12 : f32, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<memref<4x8xf32>, i1>) -> !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %12 to %12[%13, %14 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x8xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_12, %alloc_13 : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc, %alloc_13 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %20 = neura.load_indexed [%18, %19 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.fmul_fadd"(%21, %22, %23) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %25 to [%26, %27 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%1, %alloc : memref<8x8xf32>, memref<8x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_12, %alloc_14 : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc, %alloc_14 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %20 = neura.load_indexed [%18, %19 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.fmul_fadd"(%21, %22, %23) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %25 to [%26, %27 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%2, %alloc : memref<8x8xf32>, memref<8x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_12, %alloc_15 : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc, %alloc_15 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %20 = neura.load_indexed [%18, %19 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.fmul_fadd"(%21, %22, %23) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %25 to [%26, %27 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<8x4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_14, %alloc_16 : memref<4x8xf32>, memref<8x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_10, %alloc_17 : f32, memref<4x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x4xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x4xf32>, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<memref<4x4xf32>, i1>) -> !neura.data<memref<4x4xf32>, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %12 to %12[%13, %14 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x4xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x4xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_13, %alloc_16, %alloc_17 : memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>, %arg3: memref<4x4xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %20 = neura.load_indexed [%18, %19 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.fmul_fadd"(%21, %22, %23) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %25 to [%26, %27 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_17, %cst_2 : memref<4x4xf32>, f64) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: f64):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.cast"(%15) <{cast_type = "truncf"}> : (!neura.data<f64, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.fdiv"(%17, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%19) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %20 to [%21, %22 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%c0_i64, %alloc_18 : i64, memref<4xi64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: i64, %arg2: memref<4xi64>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4xi64>, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<memref<4xi64>, i1>) -> !neura.data<memref<4xi64>, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %11 to %11[%12 : !neura.data<i64, i1>] !neura.data<memref<4xi64>, i1> {lhs_value = "%input0"} : !neura.data<memref<4xi64>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_9, %alloc_19 : f32, memref<4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4xf32>, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<memref<4xf32>, i1>) -> !neura.data<memref<4xf32>, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %11 to %11[%12 : !neura.data<i64, i1>] !neura.data<memref<4xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_17, %alloc_19, %alloc_18 : memref<4x4xf32>, memref<4xf32>, memref<4xi64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4xf32>, %arg3: memref<4xi64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = neura.load_indexed [%14 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%16 : !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fcmp"(%18, %19) <{cmpType = "ogt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.sel"(%21, %22, %23) : (!neura.data<i1, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.fcmp"(%25, %26) <{cmpType = "ogt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT: %28 = "neura.data_mov"(%27) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT: %29 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %30 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %31 = "neura.sel"(%28, %29, %30) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %32 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %33 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %32 to [%33 : !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %34 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %35 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %34 to [%35 : !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_17, %alloc_19 : memref<4x4xf32>, memref<4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = neura.load_indexed [%14 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fsub"(%16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_17 : memref<4x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.exp"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_10, %alloc_21 : f32, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x1xf32>, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<memref<4x1xf32>, i1>) -> !neura.data<memref<4x1xf32>, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %12 to %12[%13, %14 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x1xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x1xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_21, %alloc_22 : memref<4x1xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_17, %alloc_22 : memref<4x4xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fadd"(%18, %19) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_17, %alloc_22 : memref<4x4xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fdiv"(%18, %19) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_12, %alloc_23 : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_17, %alloc_15, %alloc_23 : memref<4x4xf32>, memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %20 = neura.load_indexed [%18, %19 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.fmul_fadd"(%21, %22, %23) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %25 to [%26, %27 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc_23, %alloc_11 : memref<4x8xf32>, memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = neura.load_indexed [%14, %15 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.fadd"(%17, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%19) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %20 to [%21, %22 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_24 : memref<4x8xf32>, memref<4x8xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.cast"(%14) <{cast_type = "extf"}> : (!neura.data<f32, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
// DATAFLOW_IR-NEXT: %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_8, %alloc_26 : f64, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f64, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x1xf64>, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<memref<4x1xf64>, i1>) -> !neura.data<memref<4x1xf64>, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %12 to %12[%13, %14 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x1xf64>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x1xf64>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_26, %alloc_27 : memref<4x1xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_24, %alloc_27 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fadd"(%18, %19) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_27, %cst_1, %alloc_25 : memref<4x1xf64>, f64, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.fdiv"(%14) {rhs_value = "%input1"} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_24, %alloc_25 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fsub"(%18, %19) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_24 : memref<4x8xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = neura.load_indexed [%14, %15 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%16) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.fmul"(%17, %18) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%19) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %20 to [%21, %22 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_26, %alloc_28 : memref<4x1xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_24, %alloc_28 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fadd"(%18, %19) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_28, %cst_1, %alloc_25 : memref<4x1xf64>, f64, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.fdiv"(%14) {rhs_value = "%input1"} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_25, %alloc_20 : memref<4x1xf64>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.cast"(%14) <{cast_type = "truncf"}> : (!neura.data<f64, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_21, %alloc_29 : memref<4x1xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_29 : memref<4x8xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fadd"(%18, %19) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_29, %cst_0, %alloc_30 : memref<4x1xf32>, f32, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>, %arg2: f32, %arg3: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.fdiv"(%14) {rhs_value = "%input1"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_20, %cst_3 : memref<4x1xf32>, f64) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>, %arg2: f64):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.cast"(%15) <{cast_type = "truncf"}> : (!neura.data<f64, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.fadd"(%17, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%19) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %20 to [%21, %22 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_20 : memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.rsqrt"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_30 : memref<4x8xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fsub"(%18, %19) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_20 : memref<4x8xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fmul"(%18, %19) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %3 : memref<4x8xf32>, memref<8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = neura.load_indexed [%14 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fmul"(%16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %4 : memref<4x8xf32>, memref<8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = neura.load_indexed [%14 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fadd"(%16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%5, %alloc_31 : memref<16x8xf32>, memref<8x16xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<16x8xf32>, %arg2: memref<8x16xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<4x16xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_10, %alloc_32 : f32, memref<4x16xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x16xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x16xf32>, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<memref<4x16xf32>, i1>) -> !neura.data<memref<4x16xf32>, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %12 to %12[%13, %14 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x16xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x16xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_31, %alloc_32 : memref<4x8xf32>, memref<8x16xf32>, memref<4x16xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x16xf32>, %arg3: memref<4x16xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %20 = neura.load_indexed [%18, %19 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.fmul_fadd"(%21, %22, %23) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %25 to [%26, %27 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_32, %cst_7, %cst_6, %cst, %cst_5, %cst_4 : memref<4x16xf32>, f32, f32, f32, f32, f32) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x16xf32>, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.fmul"(%14, %15) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.fmul"(%17, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%19) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.fmul"(%20) {rhs_value = "%input1"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%21) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.fadd"(%22, %23) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.fmul"(%25) {rhs_value = "%input2"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%26) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %28 = "neura.fmul"(%27) {rhs_value = "%input3"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %29 = "neura.data_mov"(%28) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %30 = "neura.exp"(%29) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %31 = "neura.data_mov"(%30) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %32 = "neura.fsub"(%31) {rhs_value = "%input4"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %33 = "neura.data_mov"(%30) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %34 = "neura.fadd"(%33) {rhs_value = "%input4"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %35 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %36 = "neura.data_mov"(%34) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %37 = "neura.fdiv"(%35, %36) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %38 = "neura.data_mov"(%37) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %39 = "neura.fadd"(%38) {rhs_value = "%input4"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %40 = "neura.data_mov"(%39) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %41 = "neura.fmul"(%40) {rhs_value = "%input5"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %42 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %43 = "neura.data_mov"(%41) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %44 = "neura.fmul"(%42, %43) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %45 = "neura.data_mov"(%44) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %46 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %47 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %45 to [%46, %47 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<16x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%6, %alloc_33 : memref<8x16xf32>, memref<16x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<8x16xf32>, %arg2: memref<16x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_32, %alloc_33, %alloc_12 : memref<4x16xf32>, memref<16x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x16xf32>, %arg2: memref<16x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %20 = neura.load_indexed [%18, %19 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.fmul_fadd"(%21, %22, %23) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %25 to [%26, %27 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_12 : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = neura.load_indexed [%14, %15 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.fadd"(%17, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%19) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %20 to [%21, %22 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_24 : memref<4x8xf32>, memref<4x8xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.cast"(%14) <{cast_type = "extf"}> : (!neura.data<f32, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_26, %alloc_34 : memref<4x1xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_24, %alloc_34 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fadd"(%18, %19) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_34, %cst_1, %alloc_25 : memref<4x1xf64>, f64, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.fdiv"(%14) {rhs_value = "%input1"} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_24, %alloc_25 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fsub"(%18, %19) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_24 : memref<4x8xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = neura.load_indexed [%14, %15 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%16) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.fmul"(%17, %18) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%19) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %20 to [%21, %22 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_24, %alloc_26 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fadd"(%18, %19) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_26, %cst_1, %alloc_25 : memref<4x1xf64>, f64, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.fdiv"(%14) {rhs_value = "%input1"} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_25, %alloc_20 : memref<4x1xf64>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.cast"(%14) <{cast_type = "truncf"}> : (!neura.data<f64, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_21 : memref<4x8xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fadd"(%18, %19) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_21, %cst_0, %alloc_35 : memref<4x1xf32>, f32, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>, %arg2: f32, %arg3: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.fdiv"(%14) {rhs_value = "%input1"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_20, %cst_3 : memref<4x1xf32>, f64) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>, %arg2: f64):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%9) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.cast"(%15) <{cast_type = "truncf"}> : (!neura.data<f64, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.fadd"(%17, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%19) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %20 to [%21, %22 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_20 : memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.rsqrt"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %16 to [%17, %18 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_35 : memref<4x8xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fsub"(%18, %19) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %alloc_20 : memref<4x8xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %9 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = neura.load_indexed [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fmul"(%18, %19) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %21 to [%22, %23 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %7 : memref<4x8xf32>, memref<8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = neura.load_indexed [%14 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fmul"(%16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %8 : memref<4x8xf32>, memref<8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
// DATAFLOW_IR-NEXT: %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = neura.load_indexed [%14 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fadd"(%16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: return %alloc_11 : memref<4x8xf32>
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: }

// INTERPRETER_OUTPUT: [neura-interpreter]  Executing func.return:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  → Output memref<4x8xf32>:
// INTERPRETER_OUTPUT-NEXT: [-8.428015e-01 1.860268e+00 -1.042649e+00 1.205288e-01 -8.270965e-01 1.266027e+00 1.186914e-01 -6.529682e-01
// INTERPRETER_OUTPUT-NEXT: -8.428012e-01 1.860268e+00 -1.042648e+00 1.205283e-01 -8.270968e-01 1.266027e+00 1.186911e-01 -6.529682e-01
// INTERPRETER_OUTPUT-NEXT: -8.428012e-01 1.860268e+00 -1.042648e+00 1.205283e-01 -8.270968e-01 1.266027e+00 1.186911e-01 -6.529682e-01
// INTERPRETER_OUTPUT-NEXT: -8.428012e-01 1.860268e+00 -1.042648e+00 1.205283e-01 -8.270968e-01 1.266027e+00 1.186911e-01 -6.529682e-01]
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
