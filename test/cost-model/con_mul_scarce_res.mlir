// Constructed regression: FU-class saturation. 10 independent integer muls;
// the 'mul' FU exists on ONE tile (scarce_mul.yaml), so ResMII(mul)=10 and the
// analytical II is 10 (dominant=res) -- strictly tighter than the crude
// ceil(#ops/#tiles) baseline (issue_mii=7). Frozen pre-map DFG; the model runs
// without invoking the mapper.
// RUN: mlir-neura-opt %s --cost-model-analytical --architecture-spec=%S/arch/scarce_mul.yaml -o %t
// RUN: FileCheck %s --input-file=%t
// CHECK: module attributes {dlti.dl_spec = #dlti.dl_spec<i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
// CHECK-NEXT:   func.func @mul_kernel(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}) -> i64 attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", analytical_cost_model = {analytical_ii = 10 : i32, dominant = "res", issue_mii = 7 : i32, max_ii = 64 : i32, mem_mii = 1 : i32, rec_mii = 5 : i32, reg_mii = 1 : i32, res_mii = 10 : i32, route_mii = 4 : i32}, dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// CHECK-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<i32, i1>
// CHECK-NEXT:     %1 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<i32, i1>
// CHECK-NEXT:     %2 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %3 = "neura.data_mov"(%1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-NEXT:     %4 = "neura.icmp"(%3) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %5 = "neura.data_mov"(%4) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %6 = "neura.grant_once"(%5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %7 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-NEXT:     %8 = "neura.data_mov"(%6) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %9 = neura.grant_predicate %7, %8 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CHECK-NEXT:     %10 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %11 = "neura.data_mov"(%6) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %13 = "neura.data_mov"(%6) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %14 = "neura.not"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %15 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %16 = "neura.data_mov"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %17 = neura.grant_predicate %15, %16 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %18 = "neura.data_mov"(%9) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-NEXT:     %19 = neura.zext %18 : !neura.data<i32, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %20 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %21 = "neura.data_mov"(%19) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %22 = "neura.phi"(%20, %21) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %23 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %24 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %25 = "neura.phi"(%23, %24) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %26 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %27 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %28 = "neura.phi"(%26, %27) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %29 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %30 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %31 = "neura.phi"(%29, %30) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %32 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %33 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %34 = "neura.phi"(%32, %33) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %35 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %36 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %37 = "neura.phi"(%35, %36) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %38 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %39 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %40 = "neura.phi"(%38, %39) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %41 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %42 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %43 = "neura.phi"(%41, %42) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %44 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %45 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %46 = "neura.phi"(%44, %45) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %47 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %48 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %49 = "neura.phi"(%47, %48) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %50 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %51 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %52 = "neura.phi"(%50, %51) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %53 = neura.reserve : !neura.data<i64, i1>
// CHECK-NEXT:     %54 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %55 = "neura.phi"(%53, %54) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %56 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %57 = "neura.gep"(%56) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %58 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %59 = "neura.load"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %60 = "neura.data_mov"(%59) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %61 = "neura.data_mov"(%59) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %62 = "neura.mul"(%60, %61) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %63 = "neura.data_mov"(%62) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %64 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %65 = "neura.add"(%63, %64) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %66 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %67 = "neura.add"(%66) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %68 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %69 = "neura.gep"(%68) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %70 = "neura.data_mov"(%69) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %71 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %72 = "neura.data_mov"(%71) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %73 = "neura.data_mov"(%71) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %74 = "neura.mul"(%72, %73) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %75 = "neura.data_mov"(%74) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %76 = "neura.data_mov"(%28) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %77 = "neura.add"(%75, %76) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %78 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %79 = "neura.gep"(%78) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 16 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %80 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %82 = "neura.data_mov"(%81) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %83 = "neura.data_mov"(%81) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %84 = "neura.mul"(%82, %83) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %85 = "neura.data_mov"(%84) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %86 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %87 = "neura.add"(%85, %86) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %88 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %89 = "neura.gep"(%88) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 24 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %90 = "neura.data_mov"(%89) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %92 = "neura.data_mov"(%91) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %93 = "neura.data_mov"(%91) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %94 = "neura.mul"(%92, %93) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %95 = "neura.data_mov"(%94) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %96 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %97 = "neura.add"(%95, %96) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %98 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %99 = "neura.gep"(%98) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 32 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %100 = "neura.data_mov"(%99) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %101 = "neura.load"(%100) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %102 = "neura.data_mov"(%101) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %103 = "neura.data_mov"(%101) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %104 = "neura.mul"(%102, %103) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %105 = "neura.data_mov"(%104) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %106 = "neura.data_mov"(%37) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %107 = "neura.add"(%105, %106) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %108 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %109 = "neura.gep"(%108) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 40 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %110 = "neura.data_mov"(%109) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %111 = "neura.load"(%110) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %112 = "neura.data_mov"(%111) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %113 = "neura.data_mov"(%111) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %114 = "neura.mul"(%112, %113) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %115 = "neura.data_mov"(%114) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %116 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %117 = "neura.add"(%115, %116) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %118 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %119 = "neura.gep"(%118) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 48 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %120 = "neura.data_mov"(%119) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %121 = "neura.load"(%120) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %122 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %123 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %124 = "neura.mul"(%122, %123) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %125 = "neura.data_mov"(%124) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %126 = "neura.data_mov"(%43) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %127 = "neura.add"(%125, %126) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %128 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %129 = "neura.gep"(%128) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 56 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %130 = "neura.data_mov"(%129) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %131 = "neura.load"(%130) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %132 = "neura.data_mov"(%131) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %133 = "neura.data_mov"(%131) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %134 = "neura.mul"(%132, %133) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %135 = "neura.data_mov"(%134) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %136 = "neura.data_mov"(%46) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %137 = "neura.add"(%135, %136) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %138 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %139 = "neura.gep"(%138) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 64 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %141 = "neura.load"(%140) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %142 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %143 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %144 = "neura.mul"(%142, %143) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %145 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %146 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %147 = "neura.add"(%145, %146) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %148 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %149 = "neura.gep"(%148) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 72 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %150 = "neura.data_mov"(%149) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %151 = "neura.load"(%150) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %152 = "neura.data_mov"(%151) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %153 = "neura.data_mov"(%151) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %154 = "neura.mul"(%152, %153) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %155 = "neura.data_mov"(%154) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %156 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %157 = "neura.add"(%155, %156) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %158 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %159 = "neura.data_mov"(%22) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %160 = "neura.icmp"(%158, %159) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %161 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %162 = "neura.not"(%161) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %163 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %164 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %165 = neura.grant_predicate %163, %164 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %165 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %166 = "neura.data_mov"(%157) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %167 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %168 = neura.grant_predicate %166, %167 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %168 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %169 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %170 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %171 = neura.grant_predicate %169, %170 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %171 -> %47 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %172 = "neura.data_mov"(%137) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %173 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %174 = neura.grant_predicate %172, %173 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %174 -> %44 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %175 = "neura.data_mov"(%127) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %176 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %177 = neura.grant_predicate %175, %176 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %177 -> %41 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %178 = "neura.data_mov"(%117) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %179 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %180 = neura.grant_predicate %178, %179 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %180 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %181 = "neura.data_mov"(%107) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %182 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %183 = neura.grant_predicate %181, %182 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %183 -> %35 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %184 = "neura.data_mov"(%97) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %185 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %186 = neura.grant_predicate %184, %185 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %186 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %187 = "neura.data_mov"(%87) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %188 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %189 = neura.grant_predicate %187, %188 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %189 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %190 = "neura.data_mov"(%77) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %191 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %192 = neura.grant_predicate %190, %191 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %192 -> %26 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %193 = "neura.data_mov"(%65) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %194 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %195 = neura.grant_predicate %193, %194 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %195 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %196 = "neura.data_mov"(%22) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %197 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %198 = neura.grant_predicate %196, %197 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.ctrl_mov %198 -> %20 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-NEXT:     %199 = "neura.data_mov"(%77) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %200 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %201 = neura.grant_predicate %199, %200 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %202 = "neura.data_mov"(%65) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %203 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %204 = neura.grant_predicate %202, %203 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %205 = "neura.data_mov"(%87) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %206 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %207 = neura.grant_predicate %205, %206 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %208 = "neura.data_mov"(%97) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %209 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %210 = neura.grant_predicate %208, %209 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %211 = "neura.data_mov"(%107) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %212 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %213 = neura.grant_predicate %211, %212 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %214 = "neura.data_mov"(%117) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %215 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %216 = neura.grant_predicate %214, %215 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %217 = "neura.data_mov"(%127) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %218 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %219 = neura.grant_predicate %217, %218 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %220 = "neura.data_mov"(%137) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %221 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %222 = neura.grant_predicate %220, %221 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %223 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %224 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %225 = neura.grant_predicate %223, %224 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %226 = "neura.data_mov"(%157) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %227 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     %228 = neura.grant_predicate %226, %227 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-NEXT:     %229 = "neura.data_mov"(%201) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %230 = "neura.data_mov"(%204) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %231 = "neura.add"(%229, %230) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %232 = "neura.data_mov"(%231) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %233 = "neura.data_mov"(%207) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %234 = "neura.add"(%232, %233) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %235 = "neura.data_mov"(%234) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %236 = "neura.data_mov"(%210) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %237 = "neura.add"(%235, %236) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %238 = "neura.data_mov"(%237) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %239 = "neura.data_mov"(%213) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %240 = "neura.add"(%238, %239) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %241 = "neura.data_mov"(%240) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %242 = "neura.data_mov"(%216) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %243 = "neura.add"(%241, %242) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %244 = "neura.data_mov"(%243) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %245 = "neura.data_mov"(%219) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %246 = "neura.add"(%244, %245) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %247 = "neura.data_mov"(%246) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %248 = "neura.data_mov"(%222) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %249 = "neura.add"(%247, %248) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %250 = "neura.data_mov"(%249) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %251 = "neura.data_mov"(%225) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %252 = "neura.add"(%250, %251) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %253 = "neura.data_mov"(%252) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %254 = "neura.data_mov"(%228) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %255 = "neura.add"(%253, %254) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %256 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %257 = "neura.data_mov"(%255) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %258 = "neura.phi"(%256, %257) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %259 = "neura.data_mov"(%258) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     neura.return_value %259 : !neura.data<i64, i1>
// CHECK-NEXT:     neura.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
module attributes {dlti.dl_spec = #dlti.dl_spec<i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  func.func @mul_kernel(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}) -> i64 attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<i32, i1>
    %1 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<i32, i1>
    %2 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
    %3 = "neura.data_mov"(%1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %4 = "neura.icmp"(%3) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
    %5 = "neura.data_mov"(%4) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %6 = "neura.grant_once"(%5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %7 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %8 = "neura.data_mov"(%6) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %9 = neura.grant_predicate %7, %8 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %10 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %11 = "neura.data_mov"(%6) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %13 = "neura.data_mov"(%6) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %14 = "neura.not"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %15 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %16 = "neura.data_mov"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %17 = neura.grant_predicate %15, %16 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %18 = "neura.data_mov"(%9) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %19 = neura.zext %18 : !neura.data<i32, i1> -> !neura.data<i64, i1>
    %20 = neura.reserve : !neura.data<i64, i1>
    %21 = "neura.data_mov"(%19) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %22 = "neura.phi"(%20, %21) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %23 = neura.reserve : !neura.data<i64, i1>
    %24 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %25 = "neura.phi"(%23, %24) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %26 = neura.reserve : !neura.data<i64, i1>
    %27 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %28 = "neura.phi"(%26, %27) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %29 = neura.reserve : !neura.data<i64, i1>
    %30 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %31 = "neura.phi"(%29, %30) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %32 = neura.reserve : !neura.data<i64, i1>
    %33 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %34 = "neura.phi"(%32, %33) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %35 = neura.reserve : !neura.data<i64, i1>
    %36 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %37 = "neura.phi"(%35, %36) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %38 = neura.reserve : !neura.data<i64, i1>
    %39 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %40 = "neura.phi"(%38, %39) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %41 = neura.reserve : !neura.data<i64, i1>
    %42 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %43 = "neura.phi"(%41, %42) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %44 = neura.reserve : !neura.data<i64, i1>
    %45 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %46 = "neura.phi"(%44, %45) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %47 = neura.reserve : !neura.data<i64, i1>
    %48 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %49 = "neura.phi"(%47, %48) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %50 = neura.reserve : !neura.data<i64, i1>
    %51 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %52 = "neura.phi"(%50, %51) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %53 = neura.reserve : !neura.data<i64, i1>
    %54 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %55 = "neura.phi"(%53, %54) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %56 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %57 = "neura.gep"(%56) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %58 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %59 = "neura.load"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %60 = "neura.data_mov"(%59) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %61 = "neura.data_mov"(%59) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %62 = "neura.mul"(%60, %61) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %63 = "neura.data_mov"(%62) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %64 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %65 = "neura.add"(%63, %64) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %66 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %67 = "neura.add"(%66) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %68 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %69 = "neura.gep"(%68) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %70 = "neura.data_mov"(%69) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %71 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %72 = "neura.data_mov"(%71) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %73 = "neura.data_mov"(%71) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %74 = "neura.mul"(%72, %73) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %75 = "neura.data_mov"(%74) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %76 = "neura.data_mov"(%28) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %77 = "neura.add"(%75, %76) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %78 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %79 = "neura.gep"(%78) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 16 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %80 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %82 = "neura.data_mov"(%81) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %83 = "neura.data_mov"(%81) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %84 = "neura.mul"(%82, %83) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %85 = "neura.data_mov"(%84) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %86 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %87 = "neura.add"(%85, %86) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %88 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %89 = "neura.gep"(%88) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 24 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %90 = "neura.data_mov"(%89) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %92 = "neura.data_mov"(%91) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %93 = "neura.data_mov"(%91) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %94 = "neura.mul"(%92, %93) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %95 = "neura.data_mov"(%94) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %96 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %97 = "neura.add"(%95, %96) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %98 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %99 = "neura.gep"(%98) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 32 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %100 = "neura.data_mov"(%99) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %101 = "neura.load"(%100) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %102 = "neura.data_mov"(%101) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %103 = "neura.data_mov"(%101) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %104 = "neura.mul"(%102, %103) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %105 = "neura.data_mov"(%104) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %106 = "neura.data_mov"(%37) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %107 = "neura.add"(%105, %106) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %108 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %109 = "neura.gep"(%108) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 40 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %110 = "neura.data_mov"(%109) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %111 = "neura.load"(%110) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %112 = "neura.data_mov"(%111) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %113 = "neura.data_mov"(%111) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %114 = "neura.mul"(%112, %113) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %115 = "neura.data_mov"(%114) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %116 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %117 = "neura.add"(%115, %116) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %118 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %119 = "neura.gep"(%118) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 48 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %120 = "neura.data_mov"(%119) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %121 = "neura.load"(%120) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %122 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %123 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %124 = "neura.mul"(%122, %123) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %125 = "neura.data_mov"(%124) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %126 = "neura.data_mov"(%43) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %127 = "neura.add"(%125, %126) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %128 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %129 = "neura.gep"(%128) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 56 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %130 = "neura.data_mov"(%129) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %131 = "neura.load"(%130) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %132 = "neura.data_mov"(%131) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %133 = "neura.data_mov"(%131) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %134 = "neura.mul"(%132, %133) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %135 = "neura.data_mov"(%134) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %136 = "neura.data_mov"(%46) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %137 = "neura.add"(%135, %136) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %138 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %139 = "neura.gep"(%138) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 64 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %141 = "neura.load"(%140) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %142 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %143 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %144 = "neura.mul"(%142, %143) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %145 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %146 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %147 = "neura.add"(%145, %146) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %148 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %149 = "neura.gep"(%148) <{operandSegmentSizes = array<i32: 1, 0>}> {operand_1_value = 72 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %150 = "neura.data_mov"(%149) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %151 = "neura.load"(%150) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i64, i1>
    %152 = "neura.data_mov"(%151) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %153 = "neura.data_mov"(%151) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %154 = "neura.mul"(%152, %153) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %155 = "neura.data_mov"(%154) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %156 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %157 = "neura.add"(%155, %156) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %158 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %159 = "neura.data_mov"(%22) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %160 = "neura.icmp"(%158, %159) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    %161 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %162 = "neura.not"(%161) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %163 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %164 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %165 = neura.grant_predicate %163, %164 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %165 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1>
    %166 = "neura.data_mov"(%157) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %167 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %168 = neura.grant_predicate %166, %167 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %168 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
    %169 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %170 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %171 = neura.grant_predicate %169, %170 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %171 -> %47 : !neura.data<i64, i1> !neura.data<i64, i1>
    %172 = "neura.data_mov"(%137) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %173 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %174 = neura.grant_predicate %172, %173 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %174 -> %44 : !neura.data<i64, i1> !neura.data<i64, i1>
    %175 = "neura.data_mov"(%127) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %176 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %177 = neura.grant_predicate %175, %176 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %177 -> %41 : !neura.data<i64, i1> !neura.data<i64, i1>
    %178 = "neura.data_mov"(%117) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %179 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %180 = neura.grant_predicate %178, %179 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %180 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
    %181 = "neura.data_mov"(%107) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %182 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %183 = neura.grant_predicate %181, %182 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %183 -> %35 : !neura.data<i64, i1> !neura.data<i64, i1>
    %184 = "neura.data_mov"(%97) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %185 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %186 = neura.grant_predicate %184, %185 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %186 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
    %187 = "neura.data_mov"(%87) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %188 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %189 = neura.grant_predicate %187, %188 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %189 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1>
    %190 = "neura.data_mov"(%77) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %191 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %192 = neura.grant_predicate %190, %191 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %192 -> %26 : !neura.data<i64, i1> !neura.data<i64, i1>
    %193 = "neura.data_mov"(%65) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %194 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %195 = neura.grant_predicate %193, %194 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %195 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
    %196 = "neura.data_mov"(%22) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %197 = "neura.data_mov"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %198 = neura.grant_predicate %196, %197 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %198 -> %20 : !neura.data<i64, i1> !neura.data<i64, i1>
    %199 = "neura.data_mov"(%77) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %200 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %201 = neura.grant_predicate %199, %200 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %202 = "neura.data_mov"(%65) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %203 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %204 = neura.grant_predicate %202, %203 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %205 = "neura.data_mov"(%87) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %206 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %207 = neura.grant_predicate %205, %206 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %208 = "neura.data_mov"(%97) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %209 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %210 = neura.grant_predicate %208, %209 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %211 = "neura.data_mov"(%107) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %212 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %213 = neura.grant_predicate %211, %212 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %214 = "neura.data_mov"(%117) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %215 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %216 = neura.grant_predicate %214, %215 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %217 = "neura.data_mov"(%127) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %218 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %219 = neura.grant_predicate %217, %218 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %220 = "neura.data_mov"(%137) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %221 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %222 = neura.grant_predicate %220, %221 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %223 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %224 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %225 = neura.grant_predicate %223, %224 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %226 = "neura.data_mov"(%157) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %227 = "neura.data_mov"(%160) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %228 = neura.grant_predicate %226, %227 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %229 = "neura.data_mov"(%201) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %230 = "neura.data_mov"(%204) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %231 = "neura.add"(%229, %230) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %232 = "neura.data_mov"(%231) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %233 = "neura.data_mov"(%207) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %234 = "neura.add"(%232, %233) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %235 = "neura.data_mov"(%234) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %236 = "neura.data_mov"(%210) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %237 = "neura.add"(%235, %236) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %238 = "neura.data_mov"(%237) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %239 = "neura.data_mov"(%213) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %240 = "neura.add"(%238, %239) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %241 = "neura.data_mov"(%240) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %242 = "neura.data_mov"(%216) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %243 = "neura.add"(%241, %242) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %244 = "neura.data_mov"(%243) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %245 = "neura.data_mov"(%219) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %246 = "neura.add"(%244, %245) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %247 = "neura.data_mov"(%246) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %248 = "neura.data_mov"(%222) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %249 = "neura.add"(%247, %248) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %250 = "neura.data_mov"(%249) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %251 = "neura.data_mov"(%225) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %252 = "neura.add"(%250, %251) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %253 = "neura.data_mov"(%252) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %254 = "neura.data_mov"(%228) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %255 = "neura.add"(%253, %254) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %256 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %257 = "neura.data_mov"(%255) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %258 = "neura.phi"(%256, %257) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %259 = "neura.data_mov"(%258) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    neura.return_value %259 : !neura.data<i64, i1>
    neura.yield
  }
}

