// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module attributes {torch.debug_module_name = "Model"} {
  memref.global "private" constant @__constant_8x8xf32_1 : memref<8x8xf32> = dense<[[0.192691535, 0.148728415, 0.0900717229, -0.210552096, 0.0678418502, -0.123454489, -0.00430674804, -0.160466701], [-0.0752135292, 0.164872304, -0.0392478667, -0.140360713, -0.0727881342, -0.0559430197, -0.0768838897, 0.0762445405], [0.164231703, -0.0159597471, -0.0497397557, 0.0439589284, -0.0758131146, 0.107831769, 0.0800800547, 0.168062061], [0.127912447, 0.129642293, 0.0610466488, 0.133473784, -0.023162432, 0.00417594938, -0.0251575299, 0.0859858543], [-0.138467371, -0.0871236175, -0.0223365929, 0.171736151, 0.0318880342, -4.245190e-02, 0.0305720922, -0.0774592533], [-0.155757248, 0.0995636135, -0.0879785866, -0.0601142049, -0.127415121, 0.212278515, -0.123465315, -0.0487913899], [-9.138230e-02, -0.0658137277, 0.00780238723, 0.0525808744, -0.048799172, 0.119136907, -0.081400767, -0.0735992789], [-0.140324786, 0.00360036688, -0.00634772703, 0.0675614923, -0.00978068914, 0.184459403, -0.118453741, 0.138354942]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32_0 : memref<8x8xf32> = dense<[[0.144513384, 0.0856412574, 0.221807584, 0.0523165539, 0.0346646681, -0.019733144, -0.105458893, 0.127799556], [-0.0172190126, 0.0523788445, 0.00566218188, 0.0426296145, 0.0575005077, -0.0641724095, -0.220639855, -0.0750803053], [0.00108681445, -0.033874236, -0.134067953, -0.0585370548, 0.0536188148, 0.0524622612, 0.114120163, 0.0051643597], [7.439520e-02, -0.0481584407, -0.104946613, 0.0603898838, -0.172229514, -0.0827768892, 0.133470297, 0.0483539291], [-0.250954449, 0.0488001071, 0.0784586891, 0.00286471867, 0.0640755296, 0.0583247431, 0.106692672, -0.0450153388], [-0.0185267478, 0.0752758905, 0.040475782, 1.784660e-02, 0.026490951, 0.127316833, -1.31086374E-4, -0.0303603765], [-0.145702913, -0.0102335243, -0.0599153041, 0.0477056429, 0.0726177245, 0.00911518652, -0.0389065221, 0.0527916513], [-0.00126854784, 0.024083633, 0.013253537, 0.0764240622, 0.109500967, 0.0339890979, 0.0719967484, 0.0411407612]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense<[[0.193116054, 0.101186387, -0.143640652, -0.112985983, -0.0136034535, 0.163540959, 0.0654740781, 0.0576004572], [0.1141508, 0.00185645767, -0.180580512, 0.0925434902, -0.0375344381, 0.103308737, -0.0686650947, 0.063681364], [-0.0972673892, 0.0958457812, 0.161920056, 0.145060986, 0.026948154, -2.103760e-02, -0.0732802749, 0.0104297837], [0.0348751694, 0.0967594161, -0.0465688445, 0.160479724, -0.248012021, -0.0417543761, -0.119545378, 0.0812336951], [-0.190055326, 0.022857653, 0.00248594047, -0.0345950238, 0.0286832806, -0.0730842426, 0.0174820255, -0.109392934], [-0.160216033, 0.135289699, 0.128882766, 0.00522954715, -0.15468505, 0.0756706074, 0.0775519534, 0.202653557], [0.00358176115, 0.0120588727, -0.0805663689, -0.0207576826, -0.0931947827, -0.159096628, -0.113597572, -0.0522597618], [-0.051877331, -0.150127634, -0.192665428, 0.0127851237, 0.102291338, -0.0555795133, 0.0704272762, 0.0709876046]]> {alignment = 64 : i64}
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
    %cst = arith.constant 2.8284271247461903 : f64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %1 = memref.get_global @__constant_8x8xf32_0 : memref<8x8xf32>
    %2 = memref.get_global @__constant_8x8xf32_1 : memref<8x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%cst_0, %alloc : f32, memref<4x8xf32>) {
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
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    neura.kernel inputs(%alloc, %alloc_1 : memref<4x8xf32>, memref<4x8xf32>) {
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
    neura.kernel inputs(%arg0, %2, %alloc_1 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
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
    neura.kernel inputs(%arg0, %1, %alloc_2 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
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
    neura.kernel inputs(%arg0, %0, %alloc_3 : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
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
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x4xf32>
    neura.kernel inputs(%alloc_2, %alloc_4 : memref<4x8xf32>, memref<8x4xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x4xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x8xf32>
      memref.store %5, %arg2[%4, %3] : memref<8x4xf32>
      neura.yield
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
    neura.kernel inputs(%cst_0, %alloc_5 : f32, memref<4x4xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x4xf32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%3, %4] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_1, %alloc_4, %alloc_5 : memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) {
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
    neura.kernel inputs(%alloc_5, %cst : memref<4x4xf32>, f64) {
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
    neura.kernel inputs(%alloc_5, %cst_0 : memref<4x4xf32>, f32) {
    ^bb0(%arg1: memref<4x4xf32>, %arg2: f32):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %4 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %5 = memref.load %arg1[%3, %4] : memref<4x4xf32>
      %6 = arith.cmpf ugt, %5, %arg2 : f32
      %7 = arith.select %6, %5, %arg2 : f32
      memref.store %7, %arg1[%3, %4] : memref<4x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_5, %alloc_3, %alloc : memref<4x4xf32>, memref<4x8xf32>, memref<4x8xf32>) {
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


// DATAFLOW_IR-DAG: module attributes {torch.debug_module_name
// DATAFLOW_IR-DAG: func.func @forward
// DATAFLOW_IR-DAG: dataflow_mode = "predicate"
// DATAFLOW_IR-DAG: neura.kernel
// DATAFLOW_IR-DAG: neura.counter
// DATAFLOW_IR-DAG: neura.load_indexed
// DATAFLOW_IR-DAG: neura.store_indexed
// DATAFLOW_IR-DAG: neura.fmul_fadd
// DATAFLOW_IR-DAG: neura.fcmp
// DATAFLOW_IR-DAG: neura.fdiv
// DATAFLOW_IR-DAG: neura.yield
// INTERPRETER_OUTPUT-DAG: Store to m
// INTERPRETER_OUTPUT-DAG: Output: 0.000000
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
