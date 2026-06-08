// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -O1 -o %t-relu.ll relu.cpp
// RUN: mlir-translate --import-llvm %t-relu.ll -o %t-relu.mlir

// RUN: mlir-neura-opt %t-relu.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --canonicalize-live-in \
// RUN:   -o %t-relu-neura.mlir
// RUN:   FileCheck %s --input-file=%t-relu-neura.mlir --check-prefix=CHECK

// RUN: mlir-neura-opt %t-relu.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   -o %t-relu-neura-dataflow.mlir
// RUN:  FileCheck %s  --input-file=%t-relu-neura-dataflow.mlir --check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %t-relu.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN:   -o %t-relu-neura-mapping.mlir
// RUN:   FileCheck %s --input-file=%t-relu-neura-mapping.mlir --check-prefix=MAPPING

// CHECK:      func.func @_Z6kernelPiS_
// CHECK-SAME: accelerator = "neura"
// CHECK-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !llvm.ptr
// CHECK-NEXT:     %1 = "neura.constant"() <{value = "%arg1"}> : () -> !llvm.ptr
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:     %3 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %4 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:     %5 = "neura.constant"() <{value = 32 : i64}> : () -> i64
// CHECK-NEXT:     neura.br %2, %0, %3, %1, %4, %5 : i64, !llvm.ptr, i32, !llvm.ptr, i64, i64 to ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb4
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   ^bb2(%6: i64, %7: !llvm.ptr, %8: i32, %9: !llvm.ptr, %10: i64, %11: i64):  // 2 preds: ^bb0, ^bb4
// CHECK-NEXT:     %12 = "neura.gep"(%7, %6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:     %13 = "neura.load"(%12) : (!llvm.ptr) -> i32
// CHECK-NEXT:     %14 = "neura.icmp"(%13, %8) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CHECK-NEXT:     neura.cond_br %14 : i1 then %9, %6, %13, %10, %11, %7, %8 : !llvm.ptr, i64, i32, i64, i64, !llvm.ptr, i32 to ^bb3 else %6, %10, %11, %7, %8, %9 : i64, i64, i64, !llvm.ptr, i32, !llvm.ptr to ^bb4
// CHECK-NEXT:   ^bb3(%15: !llvm.ptr, %16: i64, %17: i32, %18: i64, %19: i64, %20: !llvm.ptr, %21: i32):  // pred: ^bb2
// CHECK-NEXT:     %22 = "neura.gep"(%15, %16) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:     %23 = "neura.load"(%22) : (!llvm.ptr) -> i32
// CHECK-NEXT:     %24 = "neura.add"(%23, %17) : (i32, i32) -> i32
// CHECK-NEXT:     "neura.store"(%24, %22) : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:     neura.br %16, %18, %19, %20, %21, %15 : i64, i64, i64, !llvm.ptr, i32, !llvm.ptr to ^bb4
// CHECK-NEXT:   ^bb4(%25: i64, %26: i64, %27: i64, %28: !llvm.ptr, %29: i32, %30: !llvm.ptr):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:     %31 = "neura.add"(%25, %26) : (i64, i64) -> i64
// CHECK-NEXT:     %32 = "neura.icmp"(%31, %27) <{cmpType = "eq"}> : (i64, i64) -> i1
// CHECK-NEXT:     neura.cond_br %32 : i1 then to ^bb1 else %31, %28, %29, %30, %26, %27 : i64, !llvm.ptr, i32, !llvm.ptr, i64, i64 to ^bb2
// CHECK-NEXT:   }


// CTRL2DATA:      func.func @_Z6kernelPiS_
// CTRL2DATA-SAME: accelerator = "neura"
// CTRL2DATA-SAME: dataflow_mode = "predicate"
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{value = 32 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = neura.phi_start %11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = neura.phi_start %9, %14 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %17 = neura.phi_start %3, %16 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = neura.phi_start %7, %18 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %21 = neura.phi_start %1, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %22 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %23 = neura.phi_start %5, %22 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %24 = "neura.gep"(%21, %23) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %25 = "neura.load"(%24) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %26 = "neura.icmp"(%25, %19) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %17, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %23, %26 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = neura.grant_predicate %25, %26 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %15, %26 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.grant_predicate %13, %26 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = neura.grant_predicate %21, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %33 = neura.grant_predicate %19, %26 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %34 = "neura.not"(%26) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %35 = neura.grant_predicate %23, %34 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = neura.grant_predicate %15, %34 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %37 = neura.grant_predicate %13, %34 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %38 = neura.grant_predicate %21, %34 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %39 = neura.grant_predicate %19, %34 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %40 = neura.grant_predicate %17, %34 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %41 = "neura.gep"(%27, %28) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %42 = "neura.load"(%41) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %43 = "neura.add"(%42, %29) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     "neura.store"(%43, %41) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CTRL2DATA-NEXT:     %44 = "neura.phi"(%40, %27) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %45 = "neura.phi"(%39, %33) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %46 = "neura.phi"(%38, %32) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %47 = "neura.phi"(%37, %31) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %48 = "neura.phi"(%36, %30) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %49 = "neura.phi"(%35, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %50 = "neura.add"(%49, %48) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %51 = "neura.icmp"(%50, %47) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %52 = "neura.not"(%51) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %53 = neura.grant_predicate %50, %52 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %53 -> %22 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %54 = neura.grant_predicate %46, %52 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %54 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %55 = neura.grant_predicate %45, %52 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %55 -> %18 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %56 = neura.grant_predicate %44, %52 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %56 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %57 = neura.grant_predicate %48, %52 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %57 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %58 = neura.grant_predicate %47, %52 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %58 -> %12 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %59 = neura.grant_predicate %51, %51 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     neura.return_void %59 : !neura.data<i1, i1>
// CTRL2DATA-NEXT:     neura.yield
// CTRL2DATA-NEXT:   }


// MAPPING: func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
