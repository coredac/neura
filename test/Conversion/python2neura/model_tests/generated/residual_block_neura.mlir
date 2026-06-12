// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module attributes {torch.debug_module_name = "Model"} {
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense<[[0.192691535, 0.148728415, 0.0900717229, -0.210552096, 0.0678418502, -0.123454489, -0.00430674804, -0.160466701], [-0.0752135292, 0.164872304, -0.0392478667, -0.140360713, -0.0727881342, -0.0559430197, -0.0768838897, 0.0762445405], [0.164231703, -0.0159597471, -0.0497397557, 0.0439589284, -0.0758131146, 0.107831769, 0.0800800547, 0.168062061], [0.127912447, 0.129642293, 0.0610466488, 0.133473784, -0.023162432, 0.00417594938, -0.0251575299, 0.0859858543], [-0.138467371, -0.0871236175, -0.0223365929, 0.171736151, 0.0318880342, -4.245190e-02, 0.0305720922, -0.0774592533], [-0.155757248, 0.0995636135, -0.0879785866, -0.0601142049, -0.127415121, 0.212278515, -0.123465315, -0.0487913899], [-9.138230e-02, -0.0658137277, 0.00780238723, 0.0525808744, -0.048799172, 0.119136907, -0.081400767, -0.0735992789], [-0.140324786, 0.00360036688, -0.00634772703, 0.0675614923, -0.00978068914, 0.184459403, -0.118453741, 0.138354942]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<[0.0133141968, 0.0863977671, -0.10156747, -0.0888748541, 0.0149779711, -0.0208893921, -0.0387020968, 0.0991237759]> {alignment = 64 : i64}
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_8xf32 : memref<8xf32>
    %1 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%cst, %alloc : f32, memref<4x8xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%2, %3] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %1, %alloc : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %5 = memref.load %arg1[%2, %4] : memref<4x8xf32>
      %6 = memref.load %arg2[%4, %3] : memref<8x8xf32>
      %7 = memref.load %arg3[%2, %3] : memref<4x8xf32>
      %8 = arith.mulf %5, %6 : f32
      %9 = arith.addf %7, %8 : f32
      memref.store %9, %arg3[%2, %3] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc, %0 : memref<4x8xf32>, memref<8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %4 = memref.load %arg1[%2, %3] : memref<4x8xf32>
      %5 = memref.load %arg2[%3] : memref<8xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg1[%2, %3] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc, %cst : memref<4x8xf32>, f32) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: f32):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %4 = memref.load %arg1[%2, %3] : memref<4x8xf32>
      %5 = arith.cmpf ugt, %4, %arg2 : f32
      %6 = arith.select %5, %4, %arg2 : f32
      memref.store %6, %arg1[%2, %3] : memref<4x8xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %alloc : memref<4x8xf32>, memref<4x8xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %4 = memref.load %arg1[%2, %3] : memref<4x8xf32>
      %5 = memref.load %arg2[%2, %3] : memref<4x8xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg2[%2, %3] : memref<4x8xf32>
      neura.yield
    }
    return %alloc : memref<4x8xf32>
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
// DATAFLOW_IR-DAG: neura.fadd
// DATAFLOW_IR-DAG: neura.fcmp
// DATAFLOW_IR-DAG: neura.yield
// INTERPRETER_OUTPUT-DAG: Store to m
// INTERPRETER_OUTPUT-DAG: Output: 0.000000
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
