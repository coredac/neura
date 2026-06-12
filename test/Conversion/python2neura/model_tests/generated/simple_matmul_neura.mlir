// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module attributes {torch.debug_module_name = "Model"} {
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<"0xEE50453E434C183E8677B83DF89A57BEABF08A3DB5D5FCBD9F1F8DBB625124BE8D099ABD49D4283E5FC220BDB8BA0FBEF21195BD822465BD4D759DBD19269C3D5B2C283E04BE82BCEABB4BBD470E343DE8439BBDE7D6DC3D0301A43D76182C3E7BFB023EF3C0043E0D0C7A3D5AAD083E24BFBDBC67D6883B2A17CEBC5A19B03D64CA0DBEDE6DB2BD3BFBB6BC9ADB2F3E079D023D0BE22DBD5372FA3CF5A29EBDD47E1FBE02E8CB3D1E2EB4BD503A76BD1C7902BE8A5F593E62DBFCBD7BD947BDA526BBBD59C986BD2BABFF3B0B5F573DA4E147BD0DFEF33D72B5A6BD38BB96BD4DB10FBE22F46B3B9800D0BBAE5D8A3D2F3F20BCEDE23C3EE097F2BDEBAC0D3E51FB133EAF64AF3D8721633EE249563D8AFC0D3D67A7A1BCD5FAD7BDE3DD023EE30E8DBC338B563DD389B93B649C2E3DA7856B3DD36C83BD6AEF61BEB4C399BD71738E3AB6BF0ABD1C4909BE8DC46FBD679F5B3DABE2563DD5B7E93DCA39A93B835C983DC94145BD40EED6BD625B773DEF5C30BEEE86A9BD70AC083EC50E463D1A7D80BE9FE2473DF3AEA03D01BE3B3B083A833DEDE56E3DB081DA3D016238BD68C597BC3F2A9A3DEFC9253D0833923C8D03D93C585F023E3B7409B953B6F8BC253315BE83AA27BCC06975BDFE66433D9AB8943DDD57153C725C1FBD0F3C583D6745A6BA0A4BC53C5D25593C38849C3D0B42E03D27380B3D0873933D3783283D"> {alignment = 64 : i64}
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16xf32>
    neura.kernel inputs(%cst, %alloc : f32, memref<4x16xf32>) {
    ^bb0(%arg1: f32, %arg2: memref<4x16xf32>):
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %1 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %2 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      memref.store %arg1, %arg2[%1, %2] : memref<4x16xf32>
      neura.yield
    }
    neura.kernel inputs(%arg0, %0, %alloc : memref<4x8xf32>, memref<8x16xf32>, memref<4x16xf32>) {
    ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x16xf32>, %arg3: memref<4x16xf32>):
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %1 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %2 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
      %3 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
      %4 = memref.load %arg1[%1, %3] : memref<4x8xf32>
      %5 = memref.load %arg2[%3, %2] : memref<8x16xf32>
      %6 = memref.load %arg3[%1, %2] : memref<4x16xf32>
      %7 = arith.mulf %4, %5 : f32
      %8 = arith.addf %6, %7 : f32
      memref.store %8, %arg3[%1, %2] : memref<4x16xf32>
      neura.yield
    }
    return %alloc : memref<4x16xf32>
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
