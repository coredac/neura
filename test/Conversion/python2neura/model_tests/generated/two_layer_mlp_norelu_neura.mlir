// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module {
  memref.global "private" constant @__constant_4x16xf32 : memref<4x16xf32> = dense_resource<torch_tensor_4_16_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x8xf32 : memref<16x8xf32> = dense_resource<torch_tensor_16_8_torch.float32> {alignment = 64 : i64}
  func.func @forward(%arg0: memref<2x8xf32>) -> memref<2x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_16x8xf32 : memref<16x8xf32>
    %1 = memref.get_global @__constant_4x16xf32 : memref<4x16xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
    neura.kernel inputs(%0, %alloc : memref<16x8xf32>, memref<8x16xf32>) {
    ^bb0(%arg1: memref<16x8xf32>, %arg2: memref<8x16xf32>):
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c8 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %4 = memref.load %arg1[%3, %2] : memref<16x8xf32>
      memref.store %4, %arg2[%2, %3] : memref<8x16xf32>
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
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x4xf32>
    neura.kernel inputs(%1, %alloc_1 : memref<4x16xf32>, memref<16x4xf32>) {
    ^bb0(%arg1: memref<4x16xf32>, %arg2: memref<16x4xf32>):
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %2 = neura.counter from %c0 : index to %c16 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
      %3 = neura.counter from %c0 : index to %c4 : index step %c1 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} -> index
      %4 = memref.load %arg1[%3, %2] : memref<4x16xf32>
      memref.store %4, %arg2[%2, %3] : memref<16x4xf32>
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

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_4_16_torch.float32: "0x04000000F6A5563E283A4BBDF6425C3EB0879F3DB0C758BE3829313EC8DF8CBD5043C4BD807F54BE50817EBE187D923DC0AF5FBD5C55C73D001752BED60F3E3EF2EA3BBE48D130BDF0E1553D1429043EE4AB4E3EA034693EF2FC4ABE88DA803D7839DCBD006EE0BC0C9D3FBE082E693E2EE43BBECCD1083EE4EFB33DA060A63DCA670ABE0CB1683E4006613D38BA033DEA9961BE84EED63D009F19BD448FEABDECE35B3EB04C643D86A30DBE529201BE809B43BCC4F00E3E54D582BD321112BEAC54AFBD4C403FBE8898B63D8027463E3C0271BE20D36D3D5C3F043E68AE393DF855B6BD009E053E0888063E6473BF3D50F033BD089487BDD018DB3CD0E434BD989398BD",
      torch_tensor_16_8_torch.float32: "0x0400000071658A3E5B3F963EA3A1A9BD5249A63EDFA59EBDB01C923DA64230BE799E543E89939F3E06CD84BE60579D3E8F84873D1BBD853EDA1F443D1C922E3E67774CBD958B8B3ED70C563D9C0329BEEB90B83DF2CD26BE48D429BD860B13BE7B2A703E2AE48EBE50E726BE0E76CCBD13AF59BE6DAE083DFDC9B2BEEF7AA33E2AC599BE7BC08B3E2C00713D771CEBBDC7B85F3E2EB2613DCA41923ED34F1E3DC55AE4BD3A8CC23DDA5AC4BDE05D183E3A9EA13EA847513EFB451EBEFCFD503E728E813D1DDB373E16AA5CBE4C31B3BEA1E00BBE91D88ABE8D88943E958ED03D19F6153E5FFFE43D5789C9BBCFAA8D3EE09D80BEC85CB63C1F1B77BE4C45DF3D025CF9BD7CDEDD3DF8DA96BDCE22963EBD9456BE4CEB57BEA3EE57BE08D1A23EB14CF13D9C2FAE3E166495BE7B8CB3BE7B9F8DBE1A8A73BEE4A3123E26A3013ED369963E49F73ABE58CE76BEFD16403E705612BEDABA5B3EF2D2ABBD4D1A4F3E6FA58CBEE6B336BE0EC1DC3D4E13993D1F9CB8BDD9CC573E021E763E394583BEEB4741BEA5C0A53EC553F4BD405900BE4127AFBE2A544FBE8EE0B43D83253FBD3F6683BE2CE0073C4A4D77BE829399BE755C47BE296E9EBE188666BED3F2B43E94C2883DA721DF3D7AD5A8BE0BC76DBE6B03F1BD1673623D64489FBE22FE1BBEBDBD58BE396C803A3BB706BEB4B3C8BC025475BE688078BEC93653BE8DD9F7BD07E08EBE"
    }
  }
#-}


// DATAFLOW_IR: module {
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_4x16xf32 : memref<4x16xf32> = dense_resource<torch_tensor_4_16_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_16x8xf32 : memref<16x8xf32> = dense_resource<torch_tensor_16_8_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: func.func @forward(%arg0: memref<2x8xf32>) -> memref<2x4xf32> {
// DATAFLOW_IR-NEXT: %cst = arith.constant 0.000000e+00 : f32
// DATAFLOW_IR-NEXT: %0 = memref.get_global @__constant_16x8xf32 : memref<16x8xf32>
// DATAFLOW_IR-NEXT: %1 = memref.get_global @__constant_4x16xf32 : memref<4x16xf32>
// DATAFLOW_IR-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%0, %alloc : memref<16x8xf32>, memref<8x16xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<16x8xf32>, %arg2: memref<8x16xf32>):
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 8 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = neura.load_indexed [%4, %5 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%6) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %7 to [%8, %9 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x16xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst, %alloc_0 : f32, memref<2x16xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<2x16xf32>):
// DATAFLOW_IR-NEXT: %2 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<2x16xf32>, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%2) : (!neura.data<memref<2x16xf32>, i1>) -> !neura.data<memref<2x16xf32>, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %5 to %5[%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<2x16xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<2x16xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %alloc, %alloc_0 : memref<2x8xf32>, memref<8x16xf32>, memref<2x16xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<2x8xf32>, %arg2: memref<8x16xf32>, %arg3: memref<2x16xf32>):
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
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
// DATAFLOW_IR-NEXT: %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%1, %alloc_1 : memref<4x16xf32>, memref<16x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<4x16xf32>, %arg2: memref<16x4xf32>):
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = neura.load_indexed [%4, %5 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%6) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: %8 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %9 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %7 to [%8, %9 : !neura.data<i64, i1>, !neura.data<i64, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst, %alloc_2 : f32, memref<2x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<2x4xf32>):
// DATAFLOW_IR-NEXT: %2 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<2x4xf32>, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%2) : (!neura.data<memref<2x4xf32>, i1>) -> !neura.data<memref<2x4xf32>, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %7 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %5 to %5[%6, %7 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<2x4xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<2x4xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%alloc_0, %alloc_1, %alloc_2 : memref<2x16xf32>, memref<16x4xf32>, memref<2x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<2x16xf32>, %arg2: memref<16x4xf32>, %arg3: memref<2x4xf32>):
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 16 : i64} -> !neura.data<i64, i1>
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
// DATAFLOW_IR-NEXT: return %alloc_2 : memref<2x4xf32>
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: }

// INTERPRETER_OUTPUT: [neura-interpreter]  Executing func.return:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  → Output memref<2x4xf32>:
// INTERPRETER_OUTPUT-NEXT: [-8.152244e-03 1.069074e-02 -3.069830e-02 -1.667813e-02
// INTERPRETER_OUTPUT-NEXT: -8.152161e-03 1.069063e-02 -3.069797e-02 -1.667796e-02]
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
