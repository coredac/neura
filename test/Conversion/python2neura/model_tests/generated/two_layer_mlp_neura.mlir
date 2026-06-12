// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module attributes {torch.debug_module_name = "Model"} {
  memref.global "private" constant @__constant_16x8xf32 : memref<16x8xf32> = dense<"0x71658A3E5B3F963EA3A1A9BD5249A63EDFA59EBDB01C923DA64230BE799E543E89939F3E06CD84BE60579D3E8F84873D1BBD853EDA1F443D1C922E3E67774CBD958B8B3ED70C563D9C0329BEEB90B83DF2CD26BE48D429BD860B13BE7B2A703E2AE48EBE50E726BE0E76CCBD13AF59BE6DAE083DFDC9B2BEEF7AA33E2AC599BE7BC08B3E2C00713D771CEBBDC7B85F3E2EB2613DCA41923ED34F1E3DC55AE4BD3A8CC23DDA5AC4BDE05D183E3A9EA13EA847513EFB451EBEFCFD503E728E813D1DDB373E16AA5CBE4C31B3BEA1E00BBE91D88ABE8D88943E958ED03D19F6153E5FFFE43D5789C9BBCFAA8D3EE09D80BEC85CB63C1F1B77BE4C45DF3D025CF9BD7CDEDD3DF8DA96BDCE22963EBD9456BE4CEB57BEA3EE57BE08D1A23EB14CF13D9C2FAE3E166495BE7B8CB3BE7B9F8DBE1A8A73BEE4A3123E26A3013ED369963E49F73ABE58CE76BEFD16403E705612BEDABA5B3EF2D2ABBD4D1A4F3E6FA58CBEE6B336BE0EC1DC3D4E13993D1F9CB8BDD9CC573E021E763E394583BEEB4741BEA5C0A53EC553F4BD405900BE4127AFBE2A544FBE8EE0B43D83253FBD3F6683BE2CE0073C4A4D77BE829399BE755C47BE296E9EBE188666BED3F2B43E94C2883DA721DF3D7AD5A8BE0BC76DBE6B03F1BD1673623D64489FBE22FE1BBEBDBD58BE396C803A3BB706BEB4B3C8BC025475BE688078BEC93653BE8DD9F7BD07E08EBE"> {alignment = 64 : i64}
  memref.global "private" constant @__constant_4x16xf32 : memref<4x16xf32> = dense<[[0.209617466, -0.049616009, 0.215099186, 0.0778955221, -0.211699247, 0.1730088, -0.0687862039, -0.0958315134, -0.207517624, -0.248540163, 0.0715276599, -0.0546109676, 0.0973307788, -0.205165863, 0.185607284, -0.183513433], [-0.0431683362, 0.0522174239, 0.129062951, 0.201827586, 0.227739811, -0.198230535, 0.0629168153, -0.107531488, -0.0273962021, -0.187122524, 0.227714658, -0.183487624, 0.133612812, 0.0878598988, 0.081238985, -0.135161549], [0.22723788, 5.493760e-02, 0.0321600139, -0.220313698, 0.10494712, -0.0375051498, -0.11453107, 0.21473664, 0.0557371974, -0.138319105, -0.12653473, -0.0119389296, 0.139590323, -0.0638834536, -0.14264372, -0.0856107175], [-0.186768711, 0.0891581177, 0.193510056, -0.235360086, 0.0580626726, 0.129147947, 0.0453323424, -0.0890311598, 0.130485535, 0.131378293, 0.0934818089, -0.0439303517, -0.066200316, 0.02674523, -0.0441635251, -0.0745002627]]> {alignment = 64 : i64}
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<2x8xf32>) -> memref<2x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_4x16xf32 : memref<4x16xf32>
    %1 = memref.get_global @__constant_16x8xf32 : memref<16x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
    neura.kernel inputs(%1, %alloc : memref<16x8xf32>, memref<8x16xf32>) {
    ^bb0(%arg1: memref<16x8xf32>, %arg2: memref<8x16xf32>):
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %4 = memref.load %arg1[%2, %3] : memref<16x8xf32>
      memref.store %4, %arg2[%3, %2] : memref<8x16xf32>
      neura.yield
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x16xf32>
    neura.kernel inputs(%cst, %alloc_0 : f32, memref<2x16xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<2x16xf32>):
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c2 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%2, %3] : memref<2x16xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %alloc, %alloc_0 : memref<2x8xf32>, memref<8x16xf32>, memref<2x16xf32>) {
    ^bb0(%arg1: memref<2x8xf32>, %arg2: memref<8x16xf32>, %arg3: memref<2x16xf32>):
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c2 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %4 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %5 = memref.load %arg1[%2, %4] : memref<2x8xf32>
      %6 = memref.load %arg2[%4, %3] : memref<8x16xf32>
      %7 = memref.load %arg3[%2, %3] : memref<2x16xf32>
      %8 = arith.mulf %5, %6 : f32
      %9 = arith.addf %7, %8 : f32
      memref.store %9, %arg3[%2, %3] : memref<2x16xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_0, %cst : memref<2x16xf32>, f32) {
    ^bb0(%arg1: memref<2x16xf32>, %arg2: f32):
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c2 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %4 = memref.load %arg1[%2, %3] : memref<2x16xf32>
      %5 = arith.cmpf ugt, %4, %arg2 : f32
      %6 = arith.select %5, %4, %arg2 : f32
      memref.store %6, %arg1[%2, %3] : memref<2x16xf32>
      neura.yield
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x4xf32>
    neura.kernel inputs(%0, %alloc_1 : memref<4x16xf32>, memref<16x4xf32>) {
    ^bb0(%arg1: memref<4x16xf32>, %arg2: memref<16x4xf32>):
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %4 = memref.load %arg1[%2, %3] : memref<4x16xf32>
      memref.store %4, %arg2[%3, %2] : memref<16x4xf32>
      neura.yield
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
    neura.kernel inputs(%cst, %alloc_2 : f32, memref<2x4xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<2x4xf32>):
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c2 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%2, %3] : memref<2x4xf32>
      neura.yield
    }
    neura.kernel inputs(%alloc_0, %alloc_1, %alloc_2 : memref<2x16xf32>, memref<16x4xf32>, memref<2x4xf32>) {
    ^bb0(%arg1: memref<2x16xf32>, %arg2: memref<16x4xf32>, %arg3: memref<2x4xf32>):
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c2 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %4 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %5 = memref.load %arg1[%2, %4] : memref<2x16xf32>
      %6 = memref.load %arg2[%4, %3] : memref<16x4xf32>
      %7 = memref.load %arg3[%2, %3] : memref<2x4xf32>
      %8 = arith.mulf %5, %6 : f32
      %9 = arith.addf %7, %8 : f32
      memref.store %9, %arg3[%2, %3] : memref<2x4xf32>
      neura.yield
    }
    return %alloc_2 : memref<2x4xf32>
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
// DATAFLOW_IR-DAG: neura.yield
// INTERPRETER_OUTPUT-DAG: Store to m
// INTERPRETER_OUTPUT-DAG: Output: 0.000000
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
