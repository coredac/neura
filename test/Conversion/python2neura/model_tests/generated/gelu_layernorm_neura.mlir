// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module {
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8xf32_0 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32_1> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32> {alignment = 64 : i64}
  func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %cst_1 = arith.constant 8.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e-05 : f64
    %cst_3 = arith.constant 5.000000e-01 : f32
    %cst_4 = arith.constant 1.000000e+00 : f32
    %cst_5 = arith.constant 7.977240e-01 : f32
    %cst_6 = arith.constant 4.471500e-02 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 0.000000e+00 : f64
    %0 = memref.get_global @__constant_8xf32 : memref<8xf32>
    %1 = memref.get_global @__constant_8xf32_0 : memref<8xf32>
    %2 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf64>
    neura.kernel inputs(%arg0, %alloc : memref<4x8xf32>, memref<4x8xf64>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      %6 = arith.extf %5 : f32 to f64
      memref.store %6, %arg2[%3, %4] : memref<4x8xf64>
      neura.yield
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
    neura.kernel inputs(%cst_8, %alloc_10 : f64, memref<4x1xf64>) {
    ^bb0(%arg1: f64, %arg2: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%3, %4] : memref<4x1xf64>
      neura.yield
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
    neura.kernel inputs(%alloc_10, %alloc_11 : memref<4x1xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x1xf64>
      memref.store %5, %arg2[%3, %4] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc, %alloc_11 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf64>
      %6 = memref.load %arg2[%3, %c0] : memref<4x1xf64>
      %7 = arith.addf %5, %6 : f64
      memref.store %7, %arg2[%3, %c0] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_11, %cst_1, %alloc_9 : memref<4x1xf64>, f64, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x1xf64>
      %6 = arith.divf %5, %arg2 : f64
      memref.store %6, %arg3[%3, %4] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc, %alloc_9 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf64>
      %6 = memref.load %arg2[%3, %c0] : memref<4x1xf64>
      %7 = arith.subf %5, %6 : f64
      memref.store %7, %arg1[%3, %4] : memref<4x8xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc : memref<4x8xf64>) {
    ^bb0(%arg1: memref<4x8xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf64>
      %6 = memref.load %arg1[%3, %4] : memref<4x8xf64>
      %7 = arith.mulf %5, %6 : f64
      memref.store %7, %arg1[%3, %4] : memref<4x8xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc, %alloc_10 : memref<4x8xf64>, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf64>
      %6 = memref.load %arg2[%3, %c0] : memref<4x1xf64>
      %7 = arith.addf %5, %6 : f64
      memref.store %7, %arg2[%3, %c0] : memref<4x1xf64>
      neura.yield
    }
    neura.kernel inputs(%alloc_10, %cst_1, %alloc_9 : memref<4x1xf64>, f64, memref<4x1xf64>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x1xf64>
      %6 = arith.divf %5, %arg2 : f64
      memref.store %6, %arg3[%3, %4] : memref<4x1xf64>
      neura.yield
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%alloc_9, %alloc_13 : memref<4x1xf64>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x1xf64>
      %6 = arith.truncf %5 : f64 to f32
      memref.store %6, %arg2[%3, %4] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%cst_7, %alloc_12 : f32, memref<4x1xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%3, %4] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %alloc_12 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      %6 = memref.load %arg2[%3, %c0] : memref<4x1xf32>
      %7 = arith.addf %5, %6 : f32
      memref.store %7, %arg2[%3, %c0] : memref<4x1xf32>
      neura.yield
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%alloc_12, %cst_0, %alloc_14 : memref<4x1xf32>, f32, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f32, %arg3: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x1xf32>
      %6 = arith.divf %5, %arg2 : f32
      memref.store %6, %arg3[%3, %4] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_13, %cst_2, %alloc_12 : memref<4x1xf32>, f64, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f64, %arg3: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x1xf32>
      %6 = arith.truncf %arg2 : f64 to f32
      %7 = arith.addf %5, %6 : f32
      memref.store %7, %arg3[%3, %4] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_12 : memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x1xf32>
      %6 = math.rsqrt %5 : f32
      memref.store %6, %arg1[%3, %4] : memref<4x1xf32>
      neura.yield
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%arg0, %alloc_14, %alloc_15 : memref<4x8xf32>, memref<4x1xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      %6 = memref.load %arg2[%3, %c0] : memref<4x1xf32>
      %7 = arith.subf %5, %6 : f32
      memref.store %7, %arg3[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_15, %alloc_12 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      %6 = memref.load %arg2[%3, %c0] : memref<4x1xf32>
      %7 = arith.mulf %5, %6 : f32
      memref.store %7, %arg1[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_15, %0 : memref<4x8xf32>, memref<8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      %6 = memref.load %arg2[%4] : memref<8xf32>
      %7 = arith.mulf %5, %6 : f32
      memref.store %7, %arg1[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_15, %1 : memref<4x8xf32>, memref<8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      %6 = memref.load %arg2[%4] : memref<8xf32>
      %7 = arith.addf %5, %6 : f32
      memref.store %7, %arg1[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc_15, %cst_6, %cst_5, %cst, %cst_4, %cst_3, %alloc_16 : memref<4x8xf32>, f32, f32, f32, f32, f32, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      %6 = arith.mulf %5, %5 : f32
      %7 = arith.mulf %6, %5 : f32
      %8 = arith.mulf %7, %arg2 : f32
      %9 = arith.addf %5, %8 : f32
      %10 = arith.mulf %9, %arg3 : f32
      %11 = arith.mulf %10, %arg4 : f32
      %12 = math.exp %11 : f32
      %13 = arith.subf %12, %arg5 : f32
      %14 = arith.addf %12, %arg5 : f32
      %15 = arith.divf %13, %14 : f32
      %16 = arith.addf %15, %arg5 : f32
      %17 = arith.mulf %16, %arg6 : f32
      %18 = arith.mulf %5, %17 : f32
      memref.store %18, %arg7[%3, %4] : memref<4x8xf32>
      neura.yield
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
    neura.kernel inputs(%2, %alloc_17 : memref<8x8xf32>, memref<8x8xf32>) {
    ^bb0(%arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>):
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%4, %3] : memref<8x8xf32>
      memref.store %5, %arg2[%3, %4] : memref<8x8xf32>
      neura.yield
    }
    neura.kernel inputs(%cst_7, %alloc_15 : f32, memref<4x8xf32>) {
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
    neura.kernel inputs(%alloc_16, %alloc_17, %alloc_15 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
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
    return %alloc_15 : memref<4x8xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_8_torch.float32: "0x0400000071658A3E5B3F963EA3A1A9BD5249A63EDFA59EBDB01C923DA64230BE799E543E89939F3E06CD84BE60579D3E8F84873D1BBD853EDA1F443D1C922E3E67774CBD958B8B3ED70C563D9C0329BEEB90B83DF2CD26BE48D429BD860B13BE7B2A703E2AE48EBE50E726BE0E76CCBD13AF59BE6DAE083DFDC9B2BEEF7AA33E2AC599BE7BC08B3E2C00713D771CEBBDC7B85F3E2EB2613DCA41923ED34F1E3DC55AE4BD3A8CC23DDA5AC4BDE05D183E3A9EA13EA847513EFB451EBEFCFD503E728E813D1DDB373E16AA5CBE4C31B3BEA1E00BBE91D88ABE8D88943E958ED03D19F6153E5FFFE43D5789C9BBCFAA8D3EE09D80BEC85CB63C1F1B77BE4C45DF3D025CF9BD",
      torch_tensor_8_torch.float32_1: "0x040000000000000000000000000000000000000000000000000000000000000000000000",
      torch_tensor_8_torch.float32: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F"
    }
  }
#-}


// DATAFLOW_IR: module {
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8xf32_0 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32_1> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
// DATAFLOW_IR-NEXT: %cst = arith.constant 2.000000e+00 : f32
// DATAFLOW_IR-NEXT: %cst_0 = arith.constant 8.000000e+00 : f32
// DATAFLOW_IR-NEXT: %cst_1 = arith.constant 8.000000e+00 : f64
// DATAFLOW_IR-NEXT: %cst_2 = arith.constant 1.000000e-05 : f64
// DATAFLOW_IR-NEXT: %cst_3 = arith.constant 5.000000e-01 : f32
// DATAFLOW_IR-NEXT: %cst_4 = arith.constant 1.000000e+00 : f32
// DATAFLOW_IR-NEXT: %cst_5 = arith.constant 7.977240e-01 : f32
// DATAFLOW_IR-NEXT: %cst_6 = arith.constant 4.471500e-02 : f32
// DATAFLOW_IR-NEXT: %cst_7 = arith.constant 0.000000e+00 : f32
// DATAFLOW_IR-NEXT: %cst_8 = arith.constant 0.000000e+00 : f64
// DATAFLOW_IR-NEXT: %0 = memref.get_global @__constant_8xf32 : memref<8xf32>
// DATAFLOW_IR-NEXT: %1 = memref.get_global @__constant_8xf32_0 : memref<8xf32>
// DATAFLOW_IR-NEXT: %2 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
// DATAFLOW_IR-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc : memref<4x8xf32>, memref<4x8xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf64>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.cast"(%8) <{cast_type = "extf"}> : (!neura.data<f32, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%9) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %10 to [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
// DATAFLOW_IR-NEXT: %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_8, %alloc_10 : f64, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f64, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x1xf64>, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<memref<4x1xf64>, i1>) -> !neura.data<memref<4x1xf64>, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %6 to %6[%7, %8 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x1xf64>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x1xf64>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf64>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_10, %alloc_11 : memref<4x1xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %8 to [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc, %alloc_11 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%8) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.fadd"(%12, %13) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %15 to [%16, %17 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_11, %cst_1, %alloc_9 : memref<4x1xf64>, f64, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.fdiv"(%8) {rhs_value = "%input1"} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%9) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %10 to [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc, %alloc_9 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%8) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.fsub"(%12, %13) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %15 to [%16, %17 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc : memref<4x8xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.load_indexed [%8, %9 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%7) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.fmul"(%11, %12) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc, %alloc_10 : memref<4x8xf64>, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf64>, %arg2: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%8) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.fadd"(%12, %13) : (!neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%14) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %15 to [%16, %17 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_10, %cst_1, %alloc_9 : memref<4x1xf64>, f64, memref<4x1xf64>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: f64, %arg3: memref<4x1xf64>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.fdiv"(%8) {rhs_value = "%input1"} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%9) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %10 to [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_9, %alloc_13 : memref<4x1xf64>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf64>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.cast"(%8) <{cast_type = "truncf"}> : (!neura.data<f64, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %10 to [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_7, %alloc_12 : f32, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
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
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc_12 : memref<4x8xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
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
// DATAFLOW_IR-NEXT: %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_12, %cst_0, %alloc_14 : memref<4x1xf32>, f32, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>, %arg2: f32, %arg3: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.fdiv"(%8) {rhs_value = "%input1"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %10 to [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_13, %cst_2, %alloc_12 : memref<4x1xf32>, f64, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>, %arg2: f64, %arg3: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.cast"(%9) <{cast_type = "truncf"}> : (!neura.data<f64, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.fadd"(%11, %12) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %14 to [%15, %16 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_12 : memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.rsqrt"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %10 to [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc_14, %alloc_15 : memref<4x8xf32>, memref<4x1xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.fsub"(%12, %13) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %15 to [%16, %17 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_15, %alloc_12 : memref<4x8xf32>, memref<4x1xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
// DATAFLOW_IR-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = neura.load_indexed [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.fmul"(%12, %13) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %15 to [%16, %17 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_15, %0 : memref<4x8xf32>, memref<8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = neura.load_indexed [%8 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.fmul"(%10, %11) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %13 to [%14, %15 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_15, %1 : memref<4x8xf32>, memref<8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = neura.load_indexed [%8 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.fadd"(%10, %11) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %13 to [%14, %15 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_15, %cst_6, %cst_5, %cst, %cst_4, %cst_3, %alloc_16 : memref<4x8xf32>, f32, f32, f32, f32, f32, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.fmul"(%8, %9) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.fmul"(%11, %12) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.fmul"(%14) {rhs_value = "%input1"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%15) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.fadd"(%16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.fmul"(%19) {rhs_value = "%input2"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %22 = "neura.fmul"(%21) {rhs_value = "%input3"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %23 = "neura.data_mov"(%22) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %24 = "neura.exp"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %26 = "neura.fsub"(%25) {rhs_value = "%input4"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %27 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %28 = "neura.fadd"(%27) {rhs_value = "%input4"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %29 = "neura.data_mov"(%26) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %30 = "neura.data_mov"(%28) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %31 = "neura.fdiv"(%29, %30) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %32 = "neura.data_mov"(%31) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %33 = "neura.fadd"(%32) {rhs_value = "%input4"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %34 = "neura.data_mov"(%33) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %35 = "neura.fmul"(%34) {rhs_value = "%input5"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %36 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %37 = "neura.data_mov"(%35) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %38 = "neura.fmul"(%36, %37) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %39 = "neura.data_mov"(%38) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %40 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %41 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %39 to [%40, %41 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input6"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%2, %alloc_17 : memref<8x8xf32>, memref<8x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>):
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %8 to [%9, %10 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst_7, %alloc_15 : f32, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
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
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_16, %alloc_17, %alloc_15 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
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
// DATAFLOW_IR-NEXT: return %alloc_15 : memref<4x8xf32>
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: }

// INTERPRETER_OUTPUT: [neura-interpreter]  Executing func.return:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  → Output memref<4x8xf32>:
// INTERPRETER_OUTPUT-NEXT: [-1.730864e-08 -2.530729e-07 6.680588e-08 1.748581e-07 -1.578036e-07 -1.868163e-07 -1.573871e-07 1.365300e-07
// INTERPRETER_OUTPUT-NEXT: 2.716384e-07 -1.898723e-07 3.537913e-07 -2.151499e-07 -3.830347e-07 -5.339828e-07 -5.222387e-07 9.477055e-07
// INTERPRETER_OUTPUT-NEXT: 1.493372e-07 -1.604693e-07 2.156446e-07 -3.824857e-08 -3.173584e-07 -5.712441e-07 -6.084985e-07 1.019423e-06
// INTERPRETER_OUTPUT-NEXT: 2.716384e-07 -1.898723e-07 3.537913e-07 -2.151499e-07 -3.830347e-07 -5.339828e-07 -5.222387e-07 9.477055e-07]
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
