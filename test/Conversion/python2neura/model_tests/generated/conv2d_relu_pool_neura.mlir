// RUN: mlir-neura-opt --assign-accelerator --lower-affine-to-neura --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-return --canonicalize-cast --promote-input-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --fuse-pattern --insert-data-mov %s -o %t_dataflow.mlir
// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt
// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir
// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt
module {
  memref.global "private" constant @__constant_9x4xf32 : memref<9x4xf32> = dense_resource<torch_tensor_9_4_torch.float32> {alignment = 64 : i64}
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

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_9_4_torch.float32: "0x04000000EE50453E434C183E8677B83DF89A57BEABF08A3DB5D5FCBD9F1F8DBB625124BE8D099ABD49D4283E5FC220BDB8BA0FBEF21195BD822465BD4D759DBD19269C3D5B2C283E04BE82BCEABB4BBD470E343D079D023D0BE22DBD5372FA3CF5A29EBD32CD643B1786033DFC22213ED826ADBD1C7902BE8A5F593E62DBFCBD7BD947BD973511BE468EB73DAC87A33BA71C683E"
    }
  }
#-}


// DATAFLOW_IR: module {
// DATAFLOW_IR-NEXT: memref.global "private" constant @__constant_9x4xf32 : memref<9x4xf32> = dense_resource<torch_tensor_9_4_torch.float32> {alignment = 64 : i64}
// DATAFLOW_IR-NEXT: func.func @forward(%arg0: memref<1x9xf32>) -> memref<1x4xf32> {
// DATAFLOW_IR-NEXT: %cst = arith.constant 0.000000e+00 : f32
// DATAFLOW_IR-NEXT: %0 = memref.get_global @__constant_9x4xf32 : memref<9x4xf32>
// DATAFLOW_IR-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
// DATAFLOW_IR-NEXT: neura.kernel inputs(%cst, %alloc : f32, memref<1x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: f32, %arg2: memref<1x4xf32>):
// DATAFLOW_IR-NEXT: %1 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<1x4xf32>, i1>
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %4 = "neura.data_mov"(%1) : (!neura.data<memref<1x4xf32>, i1>) -> !neura.data<memref<1x4xf32>, i1>
// DATAFLOW_IR-NEXT: %5 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %6 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: neura.store_indexed %4 to %4[%5, %6 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<1x4xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<1x4xf32>, i1>
// DATAFLOW_IR-NEXT: neura.yield {yield_type = "void"}
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: neura.kernel inputs(%arg0, %0, %alloc : memref<1x9xf32>, memref<9x4xf32>, memref<1x4xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT: ^bb0(%arg1: memref<1x9xf32>, %arg2: memref<9x4xf32>, %arg3: memref<1x4xf32>):
// DATAFLOW_IR-NEXT: %1 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %2 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
// DATAFLOW_IR-NEXT: %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 9 : i64} -> !neura.data<i64, i1>
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
// DATAFLOW_IR-NEXT: return %alloc : memref<1x4xf32>
// DATAFLOW_IR-NEXT: }
// DATAFLOW_IR-NEXT: }

// INTERPRETER_OUTPUT: [neura-interpreter]  Executing func.return:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  → Output memref<1x4xf32>:
// INTERPRETER_OUTPUT-NEXT: [-1.561197e-03 -1.490603e-02 3.873464e-04 1.365095e-02]
// INTERPRETER_OUTPUT-NOT: Error
// INTERPRETER_OUTPUT-NOT: Failed
// INTERPRETER_OUTPUT-NOT: Unhandled
