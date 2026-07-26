// Constructed regression: deep loop-carried recurrence. A serial add/sub chain
// threads the accumulator each iteration, giving RecMII = 6 (dominant). Shows
// the model predicts II from recurrence depth.
// RUN: mlir-neura-opt %s --cost-model-analytical --architecture-spec=%S/../arch_spec/architecture.yaml -o %t
// RUN: FileCheck %s --input-file=%t
// CHECK: module attributes {dlti.dl_spec = #dlti.dl_spec<i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
// CHECK-NEXT:   func.func @chain_kernel(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: i32 {llvm.noundef}) -> i64 attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", analytical_cost_model = {analytical_ii = 6 : i32, dominant = "rec", issue_mii = 3 : i32, max_ii = 20 : i32, mem_mii = 1 : i32, rec_mii = 6 : i32, reg_mii = 1 : i32, res_mii = 1 : i32, route_mii = 1 : i32}, dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// CHECK-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<i32, i1>
// CHECK-NEXT:     %1 = "neura.constant"() <{value = "%arg2"}> : () -> !neura.data<i32, i1>
// CHECK-NEXT:     %2 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %3 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %4 = "neura.data_mov"(%1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-NEXT:     %5 = "neura.icmp"(%4) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %6 = "neura.data_mov"(%5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %8 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-NEXT:     %9 = "neura.data_mov"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %10 = neura.grant_predicate %8, %9 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CHECK-NEXT:     %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %12 = "neura.data_mov"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %13 = neura.grant_predicate %11, %12 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %14 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %15 = "neura.data_mov"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %17 = "neura.data_mov"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %18 = "neura.not"(%17) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %19 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %20 = "neura.data_mov"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %21 = neura.grant_predicate %19, %20 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %22 = "neura.data_mov"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-NEXT:     %23 = neura.zext %22 : !neura.data<i32, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %24 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %25 = "neura.data_mov"(%23) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %26 = "neura.phi"(%24, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %27 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %28 = "neura.data_mov"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %30 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %31 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %32 = "neura.phi"(%30, %31) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %33 = "neura.data_mov"(%32) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %34 = "neura.gep"(%33) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %35 = "neura.data_mov"(%34) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %36 = "neura.load"(%35) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %37 = "neura.data_mov"(%32) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %38 = "neura.gep"(%37) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %39 = "neura.data_mov"(%38) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %40 = "neura.load"(%39) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %41 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %42 = "neura.mul"(%41) {rhs_value = -3 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %43 = "neura.data_mov"(%36) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %44 = "neura.shl"(%43) {rhs_value = 2 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %45 = "neura.data_mov"(%36) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %46 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %47 = "neura.add"(%45, %46) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %48 = "neura.data_mov"(%47) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %49 = "neura.data_mov"(%42) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %50 = "neura.add"(%48, %49) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %51 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %52 = "neura.shl"(%51) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %53 = "neura.data_mov"(%50) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %54 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %55 = "neura.sub"(%53, %54) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %56 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %57 = "neura.data_mov"(%44) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %58 = "neura.add"(%56, %57) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %59 = "neura.data_mov"(%32) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %60 = "neura.add"(%59) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %61 = "neura.data_mov"(%60) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %62 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %63 = "neura.icmp"(%61, %62) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %64 = "neura.data_mov"(%63) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %65 = "neura.not"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %66 = "neura.data_mov"(%60) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %67 = "neura.data_mov"(%65) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %68 = neura.grant_predicate %66, %67 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %68 -> %30 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %69 = "neura.data_mov"(%58) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %70 = "neura.data_mov"(%65) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %71 = neura.grant_predicate %69, %70 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %71 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %72 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %73 = "neura.data_mov"(%65) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %74 = neura.grant_predicate %72, %73 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %74 -> %24 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %75 = "neura.data_mov"(%58) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %76 = "neura.data_mov"(%63) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %77 = neura.grant_predicate %75, %76 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %78 = "neura.data_mov"(%21) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %79 = "neura.data_mov"(%77) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %80 = "neura.phi"(%78, %79) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %81 = "neura.data_mov"(%80) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.return_value %81 : !neura.data<i64, i1>
// CHECK-NEXT:     neura.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
module attributes {dlti.dl_spec = #dlti.dl_spec<i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  func.func @chain_kernel(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: i32 {llvm.noundef}) -> i64 attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<i32, i1>
    %1 = "neura.constant"() <{value = "%arg2"}> : () -> !neura.data<i32, i1>
    %2 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
    %3 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
    %4 = "neura.data_mov"(%1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %5 = "neura.icmp"(%4) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
    %6 = "neura.data_mov"(%5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %7 = "neura.grant_once"(%6) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %8 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %9 = "neura.data_mov"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %10 = neura.grant_predicate %8, %9 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %11 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %12 = "neura.data_mov"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %13 = neura.grant_predicate %11, %12 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %14 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %15 = "neura.data_mov"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %17 = "neura.data_mov"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %18 = "neura.not"(%17) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %19 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %20 = "neura.data_mov"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %21 = neura.grant_predicate %19, %20 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %22 = "neura.data_mov"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %23 = neura.zext %22 : !neura.data<i32, i1> -> !neura.data<i64, i1>
    %24 = neura.reserve : !neura.data<i64, i1>
    %25 = "neura.data_mov"(%23) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %26 = "neura.phi"(%24, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %27 = neura.reserve : !neura.data<i64, i1>
    %28 = "neura.data_mov"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %30 = neura.reserve : !neura.data<i64, i1>
    %31 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %32 = "neura.phi"(%30, %31) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %33 = "neura.data_mov"(%32) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %34 = "neura.gep"(%33) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %35 = "neura.data_mov"(%34) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %36 = "neura.load"(%35) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %37 = "neura.data_mov"(%32) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %38 = "neura.gep"(%37) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %39 = "neura.data_mov"(%38) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %40 = "neura.load"(%39) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %41 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %42 = "neura.mul"(%41) {rhs_value = -3 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %43 = "neura.data_mov"(%36) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %44 = "neura.shl"(%43) {rhs_value = 2 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %45 = "neura.data_mov"(%36) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %46 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %47 = "neura.add"(%45, %46) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %48 = "neura.data_mov"(%47) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %49 = "neura.data_mov"(%42) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %50 = "neura.add"(%48, %49) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %51 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %52 = "neura.shl"(%51) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %53 = "neura.data_mov"(%50) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %54 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %55 = "neura.sub"(%53, %54) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %56 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %57 = "neura.data_mov"(%44) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %58 = "neura.add"(%56, %57) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %59 = "neura.data_mov"(%32) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %60 = "neura.add"(%59) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %61 = "neura.data_mov"(%60) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %62 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %63 = "neura.icmp"(%61, %62) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    %64 = "neura.data_mov"(%63) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %65 = "neura.not"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %66 = "neura.data_mov"(%60) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %67 = "neura.data_mov"(%65) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %68 = neura.grant_predicate %66, %67 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %68 -> %30 : !neura.data<i64, i1> !neura.data<i64, i1>
    %69 = "neura.data_mov"(%58) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %70 = "neura.data_mov"(%65) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %71 = neura.grant_predicate %69, %70 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %71 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
    %72 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %73 = "neura.data_mov"(%65) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %74 = neura.grant_predicate %72, %73 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %74 -> %24 : !neura.data<i64, i1> !neura.data<i64, i1>
    %75 = "neura.data_mov"(%58) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %76 = "neura.data_mov"(%63) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %77 = neura.grant_predicate %75, %76 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %78 = "neura.data_mov"(%21) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %79 = "neura.data_mov"(%77) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %80 = "neura.phi"(%78, %79) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %81 = "neura.data_mov"(%80) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    neura.return_value %81 : !neura.data<i64, i1>
    neura.yield
  }
}

