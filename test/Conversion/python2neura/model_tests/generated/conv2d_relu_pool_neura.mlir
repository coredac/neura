// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module attributes {torch.debug_module_name = "Model"} {
  memref.global "private" constant @__constant_9x4xf32 : memref<9x4xf32> = dense<[[0.192691535, 0.148728415, 0.0900717229, -0.210552096], [0.0678418502, -0.123454489, -0.00430674804, -0.160466701], [-0.0752135292, 0.164872304, -0.0392478667, -0.140360713], [-0.0727881342, -0.0559430197, -0.0768838897, 0.0762445405], [0.164231703, -0.0159597471, -0.0497397557, 0.0439589284], [0.0318880342, -4.245190e-02, 0.0305720922, -0.0774592533], [0.0034912345, 3.211030e-02, 0.157360017, -0.0845467448], [-0.127415121, 0.212278515, -0.123465315, -0.0487913899], [-0.141805992, 0.0896268338, 0.00499053858, 0.2266718]]> {alignment = 64 : i64}
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x9xf32>) -> memref<1x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_9x4xf32 : memref<9x4xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    neura.kernel inputs(%cst, %alloc : f32, memref<1x4xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<1x4xf32>):
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %1 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %2 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%1, %2] : memref<1x4xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %0, %alloc : memref<1x9xf32>, memref<9x4xf32>, memref<1x4xf32>) {
    ^bb0(%arg1: memref<1x9xf32>, %arg2: memref<9x4xf32>, %arg3: memref<1x4xf32>):
      %c9 = arith.constant 9 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %1 = neura.counter from %c0 : index to %c1 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %2 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %3 = neura.counter from %c0 : index to %c9 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %4 = memref.load %arg1[%1, %3] : memref<1x9xf32>
      %5 = memref.load %arg2[%3, %2] : memref<9x4xf32>
      %6 = memref.load %arg3[%1, %2] : memref<1x4xf32>
      %7 = arith.mulf %4, %5 : f32
      %8 = arith.addf %6, %7 : f32
      memref.store %8, %arg3[%1, %2] : memref<1x4xf32>
      neura.yield
    }
    return %alloc : memref<1x4xf32>
  }
}


// DATAFLOW_IR-DAG: module attributes {torch.debug_module_name
// DATAFLOW_IR-DAG: func.func @forward
// DATAFLOW_IR-DAG: dataflow_mode = "predicate"
// DATAFLOW_IR-DAG: neura.kernel
// DATAFLOW_IR-DAG: neura.counter
// DATAFLOW_IR-DAG: neura.load_indexed
// DATAFLOW_IR-DAG: neura.store_indexed
// DATAFLOW_IR-DAG: neura.fmul_fadd
// DATAFLOW_IR-DAG: neura.yield
// INTERPRETER_OUTPUT-DAG: Store to m
// INTERPRETER_OUTPUT-DAG: Output: 0.000000
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
