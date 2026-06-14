// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module {
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32> {alignment = 64 : i64}
  func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %1 = memref.get_global @__constant_8xf32 : memref<8xf32>
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
    neura.kernel inputs(%arg0, %0, %alloc : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) {
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
    neura.kernel inputs(%alloc, %1 : memref<4x8xf32>, memref<8xf32>) {
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

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_torch.float32: "0x04000000CA235A3C50F1B03D9B02D0BD0504B6BD2A66753C3B20ABBC17861EBD6801CB3D",
      torch_tensor_8_8_torch.float32: "0x04000000EE50453E434C183E8677B83DF89A57BEABF08A3DB5D5FCBD9F1F8DBB625124BE8D099ABD49D4283E5FC220BDB8BA0FBEF21195BD822465BD4D759DBD19269C3D5B2C283E04BE82BCEABB4BBD470E343DE8439BBDE7D6DC3D0301A43D76182C3E7BFB023EF3C0043E0D0C7A3D5AAD083E24BFBDBC67D6883B2A17CEBC5A19B03D64CA0DBEDE6DB2BD3BFBB6BC9ADB2F3E079D023D0BE22DBD5372FA3CF5A29EBDD47E1FBE02E8CB3D1E2EB4BD503A76BD1C7902BE8A5F593E62DBFCBD7BD947BDA526BBBD59C986BD2BABFF3B0B5F573DA4E147BD0DFEF33D72B5A6BD38BB96BD4DB10FBE22F46B3B9800D0BBAE5D8A3D2F3F20BCEDE23C3EE097F2BDEBAC0D3E"
    }
  }
#-}


// DATAFLOW_IR: module {
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense_resource<torch_tensor_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense_resource<torch_tensor_8_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: func.func @forward(%arg0: memref<4x8xf32>) -> memref<4x8xf32> {
// DATAFLOW_IR-NEXT: %cst = arith.constant 0.000000e+00 : f32
// DATAFLOW_IR-NEXT: %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
// DATAFLOW_IR-NEXT: %1 = memref.get_global @__constant_8xf32 : memref<8xf32>
// DATAFLOW_IR-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst, %alloc : f32, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %2 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%2) : (!neura.data<memref<4x8xf32>, i1>) -> !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %5 to %5[%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<4x8xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<4x8xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %0, %alloc : memref<4x8xf32>, memref<8x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = neura.load_indexed [%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %10 = neura.load_indexed [%8, %9 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %13 = neura.load_indexed [%11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %16 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %17 = "neura.fmul_fadd"(%14, %15, %16) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %18 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %19 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %20 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %18 to [%19, %20 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input2"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc, %1 : memref<4x8xf32>, memref<8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<8xf32>):
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = neura.load_indexed [%4, %5 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = neura.load_indexed [%7 : !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%6) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.fadd"(%9, %10) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %12 to [%13, %14 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc : memref<4x8xf32>, memref<4x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>):
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = neura.load_indexed [%4, %5 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = neura.load_indexed [%7, %8 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %10 = "neura.data_mov"(%6) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %11 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %12 = "neura.fadd"(%10, %11) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %14 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %15 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %13 to [%14, %15 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: return %alloc : memref<4x8xf32>
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: }

// INTERPRETER_OUTPUT: [neura-interpreter]  Executing func.return:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  → Output memref<4x8xf32>:
// INTERPRETER_OUTPUT-NEXT: [-1.882851e-02 3.629323e-02 -1.362410e-01 -1.273681e-01 -1.200995e-02 -7.203141e-02 -6.347172e-02 5.881016e-02
// INTERPRETER_OUTPUT-NEXT: -1.882816e-02 3.629377e-02 -1.362406e-01 -1.273677e-01 -1.200966e-02 -7.203087e-02 -6.347145e-02 5.881060e-02
// INTERPRETER_OUTPUT-NEXT: -1.882813e-02 3.629382e-02 -1.362406e-01 -1.273677e-01 -1.200964e-02 -7.203083e-02 -6.347144e-02 5.881062e-02
// INTERPRETER_OUTPUT-NEXT: -1.882821e-02 3.629369e-02 -1.362407e-01 -1.273678e-01 -1.200971e-02 -7.203096e-02 -6.347150e-02 5.881053e-02]
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
