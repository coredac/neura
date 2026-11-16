// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: | FileCheck %s

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: | FileCheck %s --check-prefix=CAST

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-input-arg-to-const \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: -o %t-ctrl2data.mlir
// RUN: FileCheck %s --input-file=%t-ctrl2data.mlir --check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-input-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN: --architecture-spec=../../arch_spec/architecture.yaml \
// RUN: -o %t-mapped.mlir
// RUN: FileCheck %s --input-file=%t-mapped.mlir --check-prefix=MAPPING

module attributes {} {
  func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 128 {
        %0 = affine.load %arg0[0, 0, 0, 0, 0, %arg3] : memref<?x1x1x1x1x128xi8>
        affine.store %0, %arg1[0, 0, %arg2, 0, 0, %arg3] : memref<?x1x128x1x1x128xi8>
      }
    }
    return
  }
}


// CHECK:   func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %3 = "neura.cast"(%2) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %3 : i64 to ^bb1
// CHECK-NEXT:   ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %5 = "neura.cast"(%4) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %6 = "neura.icmp"(%5, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %6 : i1 then to ^bb2 else to ^bb6
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %7 = "neura.cast"(%2) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %7 : i64 to ^bb3
// CHECK-NEXT:   ^bb3(%8: i64):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT:     %9 = "neura.cast"(%8) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %10 = "neura.icmp"(%9, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %10 : i1 then to ^bb4 else to ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %11 = neura.load_indexed %arg0[%2, %2, %2, %2, %2, %9 : index, index, index, index, index, index] memref<?x1x1x1x1x128xi8> : i8
// CHECK-NEXT:     neura.store_indexed %11 to %arg1[%2, %2, %5, %2, %2, %9 : index, index, index, index, index, index] memref<?x1x128x1x1x128xi8> : i8
// CHECK-NEXT:     %12 = "neura.add"(%9, %0) : (index, index) -> index
// CHECK-NEXT:     %13 = "neura.cast"(%12) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %13 : i64 to ^bb3
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %14 = "neura.add"(%5, %0) : (index, index) -> index
// CHECK-NEXT:     %15 = "neura.cast"(%14) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %15 : i64 to ^bb1
// CHECK-NEXT:   ^bb6:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   }
// CAST:     func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CAST-NEXT:     %0 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CAST-NEXT:     %1 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// CAST-NEXT:     %2 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CAST-NEXT:     neura.br %2 : i64 to ^bb1
// CAST-NEXT:   ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb5
// CAST-NEXT:     %4 = "neura.icmp"(%3, %1) <{cmpType = "slt"}> : (i64, i64) -> i1
// CAST-NEXT:     neura.cond_br %4 : i1 then to ^bb2 else to ^bb6
// CAST-NEXT:   ^bb2:  // pred: ^bb1
// CAST-NEXT:     neura.br %2 : i64 to ^bb3
// CAST-NEXT:   ^bb3(%5: i64):  // 2 preds: ^bb2, ^bb4
// CAST-NEXT:     %6 = "neura.icmp"(%5, %1) <{cmpType = "slt"}> : (i64, i64) -> i1
// CAST-NEXT:     neura.cond_br %6 : i1 then to ^bb4 else to ^bb5
// CAST-NEXT:   ^bb4:  // pred: ^bb3
// CAST-NEXT:     %7 = neura.load_indexed %arg0[%2, %2, %2, %2, %2, %5 : i64, i64, i64, i64, i64, i64] memref<?x1x1x1x1x128xi8> : i8
// CAST-NEXT:     neura.store_indexed %7 to %arg1[%2, %2, %3, %2, %2, %5 : i64, i64, i64, i64, i64, i64] memref<?x1x128x1x1x128xi8> : i8
// CAST-NEXT:     %8 = "neura.add"(%5, %0) : (i64, i64) -> i64
// CAST-NEXT:     neura.br %8 : i64 to ^bb3
// CAST-NEXT:   ^bb5:  // pred: ^bb3
// CAST-NEXT:     %9 = "neura.add"(%3, %0) : (i64, i64) -> i64
// CAST-NEXT:     neura.br %9 : i64 to ^bb1
// CAST-NEXT:   ^bb6:  // pred: ^bb1
// CAST-NEXT:     "neura.return"() : () -> ()
// CAST-NEXT:   }


// CTRL2DATA:      func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?x1x1x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?x1x128x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = neura.phi_start %5, %10 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %13 = neura.phi_start %3, %12 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<memref<?x1x128x1x1x128xi8>, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %15 = neura.phi_start %1, %14 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<memref<?x1x1x1x1x128xi8>, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = neura.phi_start %9, %16 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = neura.phi_start %7, %18 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %21 = neura.phi_start %9, %20 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %22 = "neura.icmp"(%21, %19) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %23 = "neura.not"(%22) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %24 = neura.grant_predicate %17, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = neura.grant_predicate %19, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %15, %22 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %13, %22 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %21, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = neura.grant_predicate %11, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %23, %23 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     neura.return_void %30 : !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %31 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = neura.phi_start %29, %31 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %34 = neura.phi_start %28, %33 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = neura.reserve : !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %36 = neura.phi_start %27, %35 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<memref<?x1x128x1x1x128xi8>, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %37 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %38 = neura.phi_start %24, %37 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %39 = neura.reserve : !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %40 = neura.phi_start %26, %39 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<memref<?x1x1x1x1x128xi8>, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %41 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %42 = neura.phi_start %25, %41 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %43 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %44 = neura.phi_start %24, %43 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %45 = "neura.icmp"(%44, %42) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %46 = neura.grant_predicate %40, %45 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %47 = neura.grant_predicate %38, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %48 = neura.grant_predicate %44, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %49 = neura.grant_predicate %36, %45 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %50 = neura.grant_predicate %34, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %51 = neura.grant_predicate %32, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %52 = neura.grant_predicate %42, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %53 = "neura.not"(%45) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %54 = neura.grant_predicate %34, %53 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %55 = neura.grant_predicate %32, %53 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %56 = neura.grant_predicate %42, %53 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %57 = neura.grant_predicate %38, %53 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %58 = neura.grant_predicate %40, %53 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %59 = neura.grant_predicate %36, %53 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %60 = "neura.add"(%54, %55) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %60 -> %20 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %56 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %57 -> %16 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %58 -> %14 : !neura.data<memref<?x1x1x1x1x128xi8>, i1> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %59 -> %12 : !neura.data<memref<?x1x128x1x1x128xi8>, i1> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %55 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %61 = neura.load_indexed %46[%47, %47, %47, %47, %47, %48 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x1x1x1x1x128xi8>, i1> : !neura.data<i8, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %61 to %49[%47, %47, %50, %47, %47, %48 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x1x128x1x1x128xi8>, i1> : !neura.data<i8, i1>
// CTRL2DATA-NEXT:     %62 = "neura.add"(%48, %51) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %62 -> %43 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %52 -> %41 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %46 -> %39 : !neura.data<memref<?x1x1x1x1x128xi8>, i1> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %47 -> %37 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %49 -> %35 : !neura.data<memref<?x1x128x1x1x128xi8>, i1> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %50 -> %33 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %51 -> %31 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.yield
// CTRL2DATA-NEXT:   }

// MAPPING:      func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 8 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 8 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}}