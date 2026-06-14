// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module {
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense_resource<torch_tensor_8_16_torch.float32> {alignment = 64 : i64}
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

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_16_torch.float32: "0x04000000EE50453E434C183E8677B83DF89A57BEABF08A3DB5D5FCBD9F1F8DBB625124BE8D099ABD49D4283E5FC220BDB8BA0FBEF21195BD822465BD4D759DBD19269C3D5B2C283E04BE82BCEABB4BBD470E343DE8439BBDE7D6DC3D0301A43D76182C3E7BFB023EF3C0043E0D0C7A3D5AAD083E24BFBDBC67D6883B2A17CEBC5A19B03D64CA0DBEDE6DB2BD3BFBB6BC9ADB2F3E079D023D0BE22DBD5372FA3CF5A29EBDD47E1FBE02E8CB3D1E2EB4BD503A76BD1C7902BE8A5F593E62DBFCBD7BD947BDA526BBBD59C986BD2BABFF3B0B5F573DA4E147BD0DFEF33D72B5A6BD38BB96BD4DB10FBE22F46B3B9800D0BBAE5D8A3D2F3F20BCEDE23C3EE097F2BDEBAC0D3E51FB133EAF64AF3D8721633EE249563D8AFC0D3D67A7A1BCD5FAD7BDE3DD023EE30E8DBC338B563DD389B93B649C2E3DA7856B3DD36C83BD6AEF61BEB4C399BD71738E3AB6BF0ABD1C4909BE8DC46FBD679F5B3DABE2563DD5B7E93DCA39A93B835C983DC94145BD40EED6BD625B773DEF5C30BEEE86A9BD70AC083EC50E463D1A7D80BE9FE2473DF3AEA03D01BE3B3B083A833DEDE56E3DB081DA3D016238BD68C597BC3F2A9A3DEFC9253D0833923C8D03D93C585F023E3B7409B953B6F8BC253315BE83AA27BCC06975BDFE66433D9AB8943DDD57153C725C1FBD0F3C583D6745A6BA0A4BC53C5D25593C38849C3D0B42E03D27380B3D0873933D3783283D"
    }
  }
#-}


// DATAFLOW_IR: module {
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense_resource<torch_tensor_8_16_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x16xf32> {
// DATAFLOW_IR-NEXT: %cst = arith.constant 0.000000e+00 : f32
// DATAFLOW_IR-NEXT: %0 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
// DATAFLOW_IR-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst, %alloc : f32, memref<4x16xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x16xf32>):
// DATAFLOW_IR-NEXT: %1 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x16xf32>, i1>
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = "neura.data_mov"(%1) : (!neura.data<memref<4x16xf32>, i1>) -> !neura.data<memref<4x16xf32>, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %4 to %4[%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x16xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x16xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %0, %alloc : memref<4x8xf32>, memref<8x16xf32>, memref<4x16xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x16xf32>, %arg3: memref<4x16xf32>):
// DATAFLOW_IR-NEXT: %1 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = "neura.data_mov"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = neura.load_indexed [%4, %5 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = neura.load_indexed [%7, %8 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = neura.load_indexed [%10, %11 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%6) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.fmul_fadd"(%13, %14, %15) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.data_mov"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %17 to [%18, %19 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: return %alloc : memref<4x16xf32>
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: }

// INTERPRETER_OUTPUT: [neura-interpreter]  Executing func.return:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  → Output memref<4x16xf32>:
// INTERPRETER_OUTPUT-NEXT: [4.509688e-03 -2.552128e-03 -4.804209e-03 -3.712754e-03 -7.278085e-03 -5.864510e-03 -3.687961e-03 9.904162e-05 7.492973e-03 -1.823241e-02 4.295052e-03 -7.196470e-03 7.706895e-03 -1.306991e-02 1.306762e-02 -8.578561e-03
// INTERPRETER_OUTPUT-NEXT: 4.509643e-03 -2.552100e-03 -4.804159e-03 -3.712715e-03 -7.278007e-03 -5.864450e-03 -3.687921e-03 9.904022e-05 7.492893e-03 -1.823221e-02 4.295005e-03 -7.196395e-03 7.706810e-03 -1.306977e-02 1.306748e-02 -8.578470e-03
// INTERPRETER_OUTPUT-NEXT: 4.509640e-03 -2.552098e-03 -4.804155e-03 -3.712712e-03 -7.278002e-03 -5.864445e-03 -3.687917e-03 9.903999e-05 7.492888e-03 -1.823220e-02 4.295002e-03 -7.196389e-03 7.706803e-03 -1.306976e-02 1.306747e-02 -8.578463e-03
// INTERPRETER_OUTPUT-NEXT: 4.509651e-03 -2.552104e-03 -4.804166e-03 -3.712721e-03 -7.278020e-03 -5.864459e-03 -3.687927e-03 9.904080e-05 7.492906e-03 -1.823224e-02 4.295012e-03 -7.196407e-03 7.706823e-03 -1.306980e-02 1.306750e-02 -8.578483e-03]
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
