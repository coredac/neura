// Compiles an attention-style GEMM+Softmax+GEMM kernel to LLVM IR, imports to MLIR,
// then lowers via Neura.
// RUN: clang -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c11 \
// RUN:   -o %t-kernel-full.ll %S/kernel.c
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir
//
// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   -o %t-dataflow.mlir
// RUN:   FileCheck %s --input-file=%t-dataflow.mlir --check-prefix=DATAFLOW

// DATAFLOW:      func.func @kernel_gemv_relu_gemv() -> (i32 {llvm.range = #llvm.constant_range<i32, 0, 256>}) attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// DATAFLOW-NEXT:     %0 = "neura.constant"() <{value = @run_gemv_relu_gemv.y}> : () -> !neura.data<!llvm.ptr, i1>
// DATAFLOW-NEXT:     %1 = "neura.constant"() <{value = 0 : i8}> : () -> !neura.data<i8, i1>
// DATAFLOW-NEXT:     %2 = "neura.constant"() <{value = 16 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %3 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %4 = "neura.grant_once"() <{constant_value = 0 : i32}> : () -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     "neura.memset"(%0, %1, %2) <{is_volatile = false}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i8, i1>, !neura.data<i64, i1>) -> ()
// DATAFLOW-NEXT:     %5 = neura.reserve : !neura.data<i32, i1>
// DATAFLOW-NEXT:     %6 = neura.phi_start %4, %5 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %7 = neura.reserve : !neura.data<i64, i1>
// DATAFLOW-NEXT:     %8 = neura.phi_start %3, %7 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %9 = neura.reserve : !neura.data<i64, i1>
// DATAFLOW-NEXT:     %10 = neura.phi_start %3, %9 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %11 = "neura.gep"(%10) <{operandSegmentSizes = array<i32: 0, 1>, strides = [1]}> {lhs_value = @run_gemv_relu_gemv.y} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// DATAFLOW-NEXT:     %12 = "neura.load"(%11) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:     %14 = neura.grant_predicate %11, %13 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// DATAFLOW-NEXT:     %15 = neura.grant_predicate %8, %13 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %16 = neura.grant_predicate %6, %13 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %17 = "neura.not"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:     %18 = neura.grant_predicate %8, %17 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %19 = neura.grant_predicate %6, %17 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     "neura.store"(%14) {lhs_value = 0 : i32} : (!neura.data<!llvm.ptr, i1>) -> ()
// DATAFLOW-NEXT:     %20 = "neura.phi"(%19, %16) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %21 = "neura.phi"(%18, %15) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %22 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %23 = "neura.icmp"(%22) <{cmpType = "eq"}> {rhs_value = 4 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:     %24 = "neura.not"(%23) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:     %25 = neura.grant_predicate %22, %24 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     neura.ctrl_mov %25 -> %9 : !neura.data<i64, i1> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %26 = neura.grant_predicate %21, %24 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     neura.ctrl_mov %26 -> %7 : !neura.data<i64, i1> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %27 = neura.grant_predicate %20, %24 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     neura.ctrl_mov %27 -> %5 : !neura.data<i32, i1> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %28 = neura.grant_predicate %21, %23 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %29 = neura.grant_predicate %20, %23 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %30 = neura.reserve : !neura.data<i32, i1>
// DATAFLOW-NEXT:     %31 = neura.phi_start %29, %30 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %32 = neura.reserve : !neura.data<i64, i1>
// DATAFLOW-NEXT:     %33 = neura.phi_start %28, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %34 = "neura.gep"(%33) <{operandSegmentSizes = array<i32: 0, 1>, strides = [1]}> {lhs_value = @run_gemv_relu_gemv.y} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// DATAFLOW-NEXT:     %35 = "neura.load"(%34) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %36 = "neura.add"(%35, %31) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %37 = "neura.add"(%33) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %38 = "neura.icmp"(%37) <{cmpType = "eq"}> {rhs_value = 4 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:     %39 = "neura.not"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:     %40 = neura.grant_predicate %37, %39 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// DATAFLOW-NEXT:     neura.ctrl_mov %40 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
// DATAFLOW-NEXT:     %41 = neura.grant_predicate %36, %39 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     neura.ctrl_mov %41 -> %30 : !neura.data<i32, i1> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %42 = neura.grant_predicate %36, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     %43 = "neura.and"(%42) {rhs_value = 255 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:     neura.return_value %43 : !neura.data<i32, i1>
// DATAFLOW-NEXT:     neura.yield
// DATAFLOW-NEXT:   }
