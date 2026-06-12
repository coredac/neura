// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module attributes {torch.debug_module_name = "Model"} {
  memref.global "private" constant @__constant_8xf32_0 : memref<8xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%cst_1, %alloc_2 : f32, memref<4x1xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%0, %1] : memref<4x1xf32>
      neura.yield
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%alloc_2, %alloc_3 : memref<4x1xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %1] : memref<4x1xf32>
      memref.store %2, %arg2[%0, %1] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %alloc_3 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %1] : memref<4x8xf32>
      %3 = memref.load %arg2[%0, %c0] : memref<4x1xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg2[%0, %c0] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_3, %cst, %alloc : memref<4x1xf32>, f32, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f32, %arg3: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %c0] : memref<4x1xf32>
      %3 = arith.divf %2, %arg2 : f32
      memref.store %3, %arg3[%0, %1] : memref<4x1xf32>
      neura.yield
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc, %alloc_4 : memref<4x1xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %c0] : memref<4x1xf32>
      memref.store %2, %arg2[%0, %1] : memref<4x8xf32>
      neura.yield
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%arg0, %alloc_4, %alloc_5 : memref<4x8xf32>, memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %1] : memref<4x8xf32>
      %3 = memref.load %arg2[%0, %1] : memref<4x8xf32>
      %4 = arith.subf %2, %3 : f32
      memref.store %4, %arg3[%0, %1] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_5, %alloc_4 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %1] : memref<4x8xf32>
      %3 = memref.load %arg1[%0, %1] : memref<4x8xf32>
      %4 = arith.mulf %2, %3 : f32
      memref.store %4, %arg2[%0, %1] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_4, %alloc_2 : memref<4x8xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x1xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %1] : memref<4x8xf32>
      %3 = memref.load %arg2[%0, %c0] : memref<4x1xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg2[%0, %c0] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_2, %cst, %alloc : memref<4x1xf32>, f32, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f32, %arg3: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %c0] : memref<4x1xf32>
      %3 = arith.divf %2, %arg2 : f32
      memref.store %3, %arg3[%0, %1] : memref<4x1xf32>
      neura.yield
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x1xf32>
    neura.kernel inputs(%alloc, %cst_0, %alloc_6 : memref<4x1xf32>, f64, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: f64, %arg3: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %c0] : memref<4x1xf32>
      %3 = arith.truncf %arg2 : f64 to f32
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg3[%0, %1] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_6, %alloc : memref<4x1xf32>, memref<4x1xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: memref<4x1xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %c0] : memref<4x1xf32>
      %3 = math.rsqrt %2 : f32
      memref.store %3, %arg2[%0, %1] : memref<4x1xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc, %alloc_4 : memref<4x1xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x1xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %c0] : memref<4x1xf32>
      memref.store %2, %arg2[%0, %1] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_5, %alloc_4 : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %1] : memref<4x8xf32>
      %3 = memref.load %arg2[%0, %1] : memref<4x8xf32>
      %4 = arith.mulf %2, %3 : f32
      memref.store %4, %arg2[%0, %1] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_4 : memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %1] : memref<4x8xf32>
      memref.store %2, %arg1[%0, %1] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_4, %cst_1 : memref<4x8xf32>, f32) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: f32):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %0 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %1 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %2 = memref.load %arg1[%0, %1] : memref<4x8xf32>
      %3 = arith.addf %2, %arg2 : f32
      memref.store %3, %arg1[%0, %1] : memref<4x8xf32>
      neura.yield
    }
    return %alloc_4 : memref<4x8xf32>
  }
}


// DATAFLOW_IR-DAG: module attributes {torch.debug_module_name
// DATAFLOW_IR-DAG: func.func @forward
// DATAFLOW_IR-DAG: dataflow_mode = "predicate"
// DATAFLOW_IR-DAG: neura.kernel
// DATAFLOW_IR-DAG: neura.counter
// DATAFLOW_IR-DAG: neura.load_indexed
// DATAFLOW_IR-DAG: neura.store_indexed
// DATAFLOW_IR-DAG: neura.fadd
// DATAFLOW_IR-DAG: neura.fsub
// DATAFLOW_IR-DAG: neura.fdiv
// DATAFLOW_IR-DAG: neura.fmul
// DATAFLOW_IR-DAG: neura.yield
// INTERPRETER_OUTPUT-DAG: Store to m
// INTERPRETER_OUTPUT-DAG: Output: 0.000000
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
