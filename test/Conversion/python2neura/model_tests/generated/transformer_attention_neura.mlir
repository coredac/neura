// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module {
  memref.global "private" constant @__constant_8x8xf32_1 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32_2> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32_0 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32_1> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32> {alignment = 64 : i64}
  func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
    %cst = arith.constant 2.8284271247461903 : f64
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %1 = memref.get_global @__constant_8x8xf32_0 : memref<8x8xf32>
    %2 = memref.get_global @__constant_8x8xf32_1 : memref<8x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%cst_1, %alloc : f32, memref<4x8xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc, %alloc_2 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      memref.store %5, %arg2[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %0, %alloc_2 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %5 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %6 = memref.load %arg1[%3, %5] : memref<4x8xf32>
      %7 = memref.load %arg2[%5, %4] : memref<8x8xf32>
      %8 = memref.load %arg3[%3, %4] : memref<4x8xf32>
      %9 = arith.mulf %6, %7 : f32
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %arg3[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc, %alloc_3 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      memref.store %5, %arg2[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %1, %alloc_3 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %5 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %6 = memref.load %arg1[%3, %5] : memref<4x8xf32>
      %7 = memref.load %arg2[%5, %4] : memref<8x8xf32>
      %8 = memref.load %arg3[%3, %4] : memref<4x8xf32>
      %9 = arith.mulf %6, %7 : f32
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %arg3[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc, %alloc_4 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      memref.store %5, %arg2[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %2, %alloc_4 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %5 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %6 = memref.load %arg1[%3, %5] : memref<4x8xf32>
      %7 = memref.load %arg2[%5, %4] : memref<8x8xf32>
      %8 = memref.load %arg3[%3, %4] : memref<4x8xf32>
      %9 = arith.mulf %6, %7 : f32
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %arg3[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<8x4xf32>
    neura.kernel inputs(%alloc_3, %alloc_5 : memref<4x8xf32>, memref<8x4xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>):
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%4, %3] : memref<4x8xf32>
      memref.store %5, %arg2[%3, %4] : memref<8x4xf32>
      neura.yield
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
    neura.kernel inputs(%cst_1, %alloc_6 : f32, memref<4x4xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%3, %4] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_2, %alloc_5, %alloc_6 : memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>, %arg3: memref<4x4xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %5 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %6 = memref.load %arg1[%3, %5] : memref<4x8xf32>
      %7 = memref.load %arg2[%5, %4] : memref<8x4xf32>
      %8 = memref.load %arg3[%3, %4] : memref<4x4xf32>
      %9 = arith.mulf %6, %7 : f32
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %arg3[%3, %4] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_6, %cst : memref<4x4xf32>, f64) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: f64):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x4xf32>
      %6 = arith.truncf %arg2 : f64 to f32
      %7 = arith.divf %5, %6 : f32
      memref.store %7, %arg1[%3, %4] : memref<4x4xf32>
      neura.yield
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
    neura.kernel inputs(%c0_i64, %alloc_7 : i64, memref<4xi64>) {
    ^bb0(%arg1: i64, %arg2: memref<4xi64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} -> index
      memref.store %arg1, %arg2[%3] : memref<4xi64>
      neura.yield
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    neura.kernel inputs(%cst_0, %alloc_8 : f32, memref<4xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} -> index
      memref.store %arg1, %arg2[%3] : memref<4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_6, %alloc_8, %alloc_7 : memref<4x4xf32>, memref<4xf32>, memref<4xi64>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4xf32>, %arg3: memref<4xi64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x4xf32>
      %6 = memref.load %arg2[%3] : memref<4xf32>
      %7 = memref.load %arg3[%3] : memref<4xi64>
      %8 = arith.index_cast %4 : index to i64
      %9 = arith.maximumf %5, %6 : f32
      %10 = arith.cmpf ogt, %5, %6 : f32
      %11 = arith.select %10, %8, %7 : i64
      memref.store %9, %arg2[%3] : memref<4xf32>
      memref.store %11, %arg3[%3] : memref<4xi64>
      neura.yield
    }
    neura.kernel inputs(%alloc_6, %alloc_8 : memref<4x4xf32>, memref<4xf32>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x4xf32>
      %6 = memref.load %arg2[%3] : memref<4xf32>
      %7 = arith.subf %5, %6 : f32
      memref.store %7, %arg1[%3, %4] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_6 : memref<4x4xf32>) {
    ^bb0(%arg1: memref<4x4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x4xf32>
      %6 = math.exp %5 : f32
      memref.store %6, %arg1[%3, %4] : memref<4x4xf32>
      neura.yield
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%cst_1, %alloc_9 : f32, memref<4x1xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%3, %4] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_6, %alloc_9 : memref<4x4xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x4xf32>
      %6 = memref.load %arg2[%3, %c0] : memref<4x1xf32>
      %7 = arith.addf %5, %6 : f32
      memref.store %7, %arg2[%3, %c0] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_6, %alloc_9 : memref<4x4xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x4xf32>
      %6 = memref.load %arg2[%3, %c0] : memref<4x1xf32>
      %7 = arith.divf %5, %6 : f32
      memref.store %7, %arg1[%3, %4] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_6, %alloc_4, %alloc : memref<4x4xf32>, memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %5 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %6 = memref.load %arg1[%3, %5] : memref<4x4xf32>
      %7 = memref.load %arg2[%5, %4] : memref<4x8xf32>
      %8 = memref.load %arg3[%3, %4] : memref<4x8xf32>
      %9 = arith.mulf %6, %7 : f32
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %arg3[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    return %alloc : memref<4x8xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_8_torch.float32_2: "0x0400000037C0453ECF3ACF3D891613BE3265E7BD05E15EBC4877273E4617863D75EE6B3DE5C7E93D6254F33A19EA38BE7187BD3DB6BD19BD8893D33D49A08CBD606B823D2034C7BDCB4AC43D5FCE253EDE8A143E60C2DC3C0B57ACBCF81396BDAFE12A3C44D90E3DCD29C63DF9BE3EBDCC54243EDDF67DBEA3062BBD35D4F4BDDA5DA63DDD9D42BEF93FBB3C29EB223B83B30DBD33F9EA3C31AD95BD77368F3C6709E0BDAC0F24BE62890A3ED8F9033E9F5CAB3BC2651EBE31F99A3D8FD39E3D6A844F3EFBBB6A3B9492453CFBFFA4BD040CAABCE8DCBEBD3AEA22BED8A5E8BD550E56BD537D54BD0FBB19BE164A45BEB278513C1F7ED13D58A763BD2D3C903DF361913D",
      torch_tensor_8_8_torch.float32_1: "0x0400000051FB133EAF64AF3D8721633EE249563D8AFC0D3D67A7A1BCD5FAD7BDE3DD023EE30E8DBC338B563DD389B93B649C2E3DA7856B3DD36C83BD6AEF61BEB4C399BD71738E3AB6BF0ABD1C4909BE8DC46FBD679F5B3DABE2563DD5B7E93DCA39A93B835C983DC94145BD40EED6BD625B773DEF5C30BEEE86A9BD70AC083EC50E463D1A7D80BE9FE2473DF3AEA03D01BE3B3B083A833DEDE56E3DB081DA3D016238BD68C597BC3F2A9A3DEFC9253D0833923C8D03D93C585F023E3B7409B953B6F8BC253315BE83AA27BCC06975BDFE66433D9AB8943DDD57153C725C1FBD0F3C583D6745A6BA0A4BC53C5D25593C38849C3D0B42E03D27380B3D0873933D3783283D",
      torch_tensor_8_8_torch.float32: "0x04000000EE50453E434C183E8677B83DF89A57BEABF08A3DB5D5FCBD9F1F8DBB625124BE8D099ABD49D4283E5FC220BDB8BA0FBEF21195BD822465BD4D759DBD19269C3D5B2C283E04BE82BCEABB4BBD470E343DE8439BBDE7D6DC3D0301A43D76182C3E7BFB023EF3C0043E0D0C7A3D5AAD083E24BFBDBC67D6883B2A17CEBC5A19B03D64CA0DBEDE6DB2BD3BFBB6BC9ADB2F3E079D023D0BE22DBD5372FA3CF5A29EBDD47E1FBE02E8CB3D1E2EB4BD503A76BD1C7902BE8A5F593E62DBFCBD7BD947BDA526BBBD59C986BD2BABFF3B0B5F573DA4E147BD0DFEF33D72B5A6BD38BB96BD4DB10FBE22F46B3B9800D0BBAE5D8A3D2F3F20BCEDE23C3EE097F2BDEBAC0D3E"
    }
  }
#-}


// DATAFLOW_IR: module {
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x8xf32_1 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32_2> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x8xf32_0 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32_1> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
// DATAFLOW_IR-NEXT: %cst = arith.constant 2.8284271247461903 : f64
// DATAFLOW_IR-NEXT: %cst_0 = arith.constant 0xFF800000 : f32
// DATAFLOW_IR-NEXT: %c0_i64 = arith.constant 0 : i64
// DATAFLOW_IR-NEXT: %cst_1 = arith.constant 0.000000e+00 : f32
// DATAFLOW_IR-NEXT: %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
// DATAFLOW_IR-NEXT: %1 = memref.get_global @__constant_8x8xf32_0 : memref<8x8xf32>
// DATAFLOW_IR-NEXT: %2 = memref.get_global @__constant_8x8xf32_1 : memref<8x8xf32>
// DATAFLOW_IR-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_1, %alloc : f32, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<memref<4x8xf32>, i1>) -> !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %6 to %6[%7, %8 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x8xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc, %alloc_2 : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %8 to [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %0, %alloc_2 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fmul_fadd"(%15, %16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc, %alloc_3 : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %8 to [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %1, %alloc_3 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fmul_fadd"(%15, %16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc, %alloc_4 : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %8 to [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %2, %alloc_4 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fmul_fadd"(%15, %16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<8x4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_3, %alloc_5 : memref<4x8xf32>, memref<8x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %8 to [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_1, %alloc_6 : f32, memref<4x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x4xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x4xf32>, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<memref<4x4xf32>, i1>) -> !neura.data<memref<4x4xf32>, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %6 to %6[%7, %8 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x4xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x4xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_2, %alloc_5, %alloc_6 : memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>, %arg3: memref<4x4xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fmul_fadd"(%15, %16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_6, %cst : memref<4x4xf32>, f64) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: f64):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.cast"(%9) <{cast_type = "truncf"}> : (!neura.data<f64, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.fdiv"(%11, %12) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%c0_i64, %alloc_7 : i64, memref<4xi64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: i64, %arg2: memref<4xi64>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4xi64>, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<memref<4xi64>, i1>) -> !neura.data<memref<4xi64>, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %5 to %5[%6 : !neura.data<i64, i1>] !neura.data<memref<4xi64>, i1> {lhs_value = "%input0"} : !neura.data<memref<4xi64>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_0, %alloc_8 : f32, memref<4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4xf32>, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<memref<4xf32>, i1>) -> !neura.data<memref<4xf32>, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %5 to %5[%6 : !neura.data<i64, i1>] !neura.data<memref<4xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_6, %alloc_8, %alloc_7 : memref<4x4xf32>, memref<4xf32>, memref<4xi64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4xf32>, %arg3: memref<4xi64>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = neura.load_indexed [%8 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%10 : !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.fcmp"(%12, %13) <{cmpType = "ogt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.sel"(%15, %16, %17) : (!neura.data<i1, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.fcmp"(%19, %20) <{cmpType = "ogt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.sel"(%22, %23, %24) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %26 to [%27 : !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %28 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %29 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %28 to [%29 : !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_6, %alloc_8 : memref<4x4xf32>, memref<4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = neura.load_indexed [%8 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.fsub"(%10, %11) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %13 to [%14, %15 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_6 : memref<4x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.exp"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %10 to [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_1, %alloc_9 : f32, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x1xf32>, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<memref<4x1xf32>, i1>) -> !neura.data<memref<4x1xf32>, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %6 to %6[%7, %8 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x1xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x1xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_6, %alloc_9 : memref<4x4xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.fadd"(%12, %13) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %15 to [%16, %17 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_6, %alloc_9 : memref<4x4xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.fdiv"(%12, %13) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %15 to [%16, %17 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_6, %alloc_4, %alloc : memref<4x4xf32>, memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x4xf32>, %arg2: memref<4x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = neura.load_indexed [%12, %13 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fmul_fadd"(%15, %16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %19 to [%20, %21 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: return %alloc : memref<4x8xf32>
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: }

// INTERPRETER_OUTPUT: [neura-interpreter]  Executing func.return:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  → Output memref<4x8xf32>:
// INTERPRETER_OUTPUT-NEXT: [5.590250e-03 -1.148392e-02 1.275721e-02 -9.011777e-03 1.415298e-02 2.921528e-04 5.243274e-03 -1.181881e-02
// INTERPRETER_OUTPUT-NEXT: 5.590250e-03 -1.148392e-02 1.275721e-02 -9.011777e-03 1.415298e-02 2.921528e-04 5.243274e-03 -1.181881e-02
// INTERPRETER_OUTPUT-NEXT: 5.590250e-03 -1.148392e-02 1.275721e-02 -9.011777e-03 1.415298e-02 2.921528e-04 5.243274e-03 -1.181881e-02
// INTERPRETER_OUTPUT-NEXT: 5.590250e-03 -1.148392e-02 1.275721e-02 -9.011777e-03 1.415298e-02 2.921528e-04 5.243274e-03 -1.181881e-02]
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
