// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: -o %t.perfect.mlir
// RUN: FileCheck %s --input-file=%t.perfect.mlir --check-prefixes=PERFECT

module attributes {} {
  func.func @_Z18attention_pipelinePKfS0_S0_S0_PfS1_S1_S1_S1_S1_(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 9.99999974E-5 : f32
    %cst_0 = arith.constant 0.166666672 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 1.250000e-01 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<256xf32>
    %alloca_5 = memref.alloca() : memref<256xf32>
    // PERFECT:        affine.for %arg10 = 0 to 256 {
    // PERFECT-NEXT:      affine.for %arg11 = 0 to 64 {
    // PERFECT-NEXT:        %0 = affine.for %arg12 = 0 to 64 iter_args(%arg13 = %cst_4) -> (f32) {
    // PERFECT-NEXT:          %1 = affine.load %arg0[%arg12 + %arg10 * 64] : memref<?xf32>
    // PERFECT-NEXT:          %2 = affine.load %arg1[%arg11 + %arg12 * 64] : memref<?xf32>
    // PERFECT-NEXT:          %3 = arith.mulf %1, %2 : f32
    // PERFECT-NEXT:          %4 = arith.addf %arg13, %3 : f32
    // PERFECT-NEXT:          %c1 = arith.constant 1 : index
    // PERFECT-NEXT:          %5 = arith.addi %arg12, %c1 : index
    // PERFECT-NEXT:          %c64 = arith.constant 64 : index
    // PERFECT-NEXT:          %6 = arith.cmpi sge, %5, %c64 : index
    // PERFECT-NEXT:          scf.if %6 {
    // PERFECT-NEXT:           affine.store %4, %arg4[%arg11 + %arg10 * 64] : memref<?xf32>
    // PERFECT-NEXT:          }
    // PERFECT-NEXT:          affine.yield %4 : f32
    // PERFECT-NEXT:        }
    // PERFECT-NEXT:      }
    // PERFECT-NEXT:    }
    affine.for %arg10 = 0 to 256 {
      affine.for %arg11 = 0 to 64 {
        %0 = affine.for %arg12 = 0 to 64 iter_args(%arg13 = %cst_4) -> (f32) {
          %1 = affine.load %arg0[%arg12 + %arg10 * 64] : memref<?xf32>
          %2 = affine.load %arg1[%arg11 + %arg12 * 64] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg13, %3 : f32
          affine.yield %4 : f32
        }
        affine.store %0, %arg4[%arg11 + %arg10 * 64] : memref<?xf32>
      }
    }
    affine.for %arg10 = 0 to 256 {
      affine.for %arg11 = 0 to 64 {
        %0 = affine.for %arg12 = 0 to 64 iter_args(%arg13 = %cst_4) -> (f32) {
          %1 = affine.load %arg0[%arg12 + %arg10 * 64] : memref<?xf32>
          %2 = affine.load %arg2[%arg11 + %arg12 * 64] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg13, %3 : f32
          affine.yield %4 : f32
        }
        affine.store %0, %arg5[%arg11 + %arg10 * 64] : memref<?xf32>
      }
    }
    affine.for %arg10 = 0 to 256 {
      affine.for %arg11 = 0 to 64 {
        %0 = affine.for %arg12 = 0 to 64 iter_args(%arg13 = %cst_4) -> (f32) {
          %1 = affine.load %arg0[%arg12 + %arg10 * 64] : memref<?xf32>
          %2 = affine.load %arg3[%arg11 + %arg12 * 64] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg13, %3 : f32
          affine.yield %4 : f32
        }
        affine.store %0, %arg6[%arg11 + %arg10 * 64] : memref<?xf32>
      }
    }
    affine.for %arg10 = 0 to 256 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.for %arg12 = 0 to 64 iter_args(%arg13 = %cst_4) -> (f32) {
          %2 = affine.load %arg4[%arg12 + %arg10 * 64] : memref<?xf32>
          %3 = affine.load %arg5[%arg12 + %arg11 * 64] : memref<?xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = arith.addf %arg13, %4 : f32
          affine.yield %5 : f32
        }
        %1 = arith.mulf %0, %cst_3 : f32
        affine.store %1, %arg7[%arg11 + %arg10 * 256] : memref<?xf32>
      }
    }
    affine.for %arg10 = 0 to 256 {
      %0 = affine.load %arg7[%arg10 * 256] : memref<?xf32>
      %1 = affine.for %arg11 = 1 to 256 iter_args(%arg12 = %0) -> (f32) {
        %2 = affine.load %arg7[%arg11 + %arg10 * 256] : memref<?xf32>
        %3 = arith.cmpf ogt, %2, %arg12 : f32
        %4 = arith.select %3, %2, %arg12 : f32
        affine.yield %4 : f32
      }
      affine.store %1, %alloca_5[%arg10] : memref<256xf32>
    }
    // Softmax uses a third-order Taylor approximation for exp(x), not a math.exp
    // op: exp(x) ~= 1 + x + 0.5*x^2 + (1/6)*x^3. The input is shifted by the
    // row maximum for numerical stability, clamped to 1e-4, stored as the
    // unnormalized probability, and accumulated into the row denominator.
    affine.for %arg10 = 0 to 256 {
      %0 = affine.load %alloca_5[%arg10] : memref<256xf32>
      %1 = affine.for %arg11 = 0 to 256 iter_args(%arg12 = %cst_4) -> (f32) {
        %2 = affine.load %arg7[%arg11 + %arg10 * 256] : memref<?xf32>
        %3 = arith.subf %2, %0 : f32
        %4 = arith.mulf %3, %3 : f32
        %5 = arith.mulf %4, %3 : f32
        %6 = arith.addf %3, %cst_2 : f32
        %7 = arith.mulf %4, %cst_1 : f32
        %8 = arith.addf %6, %7 : f32
        %9 = arith.mulf %5, %cst_0 : f32
        %10 = arith.addf %8, %9 : f32
        %11 = arith.cmpf olt, %10, %cst : f32
        %12 = arith.select %11, %cst, %10 : f32
        affine.store %12, %arg8[%arg11 + %arg10 * 256] : memref<?xf32>
        %13 = arith.addf %arg12, %12 : f32
        affine.yield %13 : f32
      }
      affine.store %1, %alloca[%arg10] : memref<256xf32>
    }
    // Normalize the Taylor-approximated softmax values by the row denominator.
    affine.for %arg10 = 0 to 256 {
      %0 = affine.load %alloca[%arg10] : memref<256xf32>
      affine.for %arg11 = 0 to 256 {
        %1 = affine.load %arg8[%arg11 + %arg10 * 256] : memref<?xf32>
        %2 = arith.divf %1, %0 : f32
        affine.store %2, %arg8[%arg11 + %arg10 * 256] : memref<?xf32>
      }
    }
    affine.for %arg10 = 0 to 256 {
      affine.for %arg11 = 0 to 64 {
        %0 = affine.for %arg12 = 0 to 256 iter_args(%arg13 = %cst_4) -> (f32) {
          %1 = affine.load %arg8[%arg12 + %arg10 * 256] : memref<?xf32>
          %2 = affine.load %arg6[%arg11 + %arg12 * 64] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg13, %3 : f32
          affine.yield %4 : f32
        }
        affine.store %0, %arg9[%arg11 + %arg10 * 64] : memref<?xf32>
      }
    }
    return
  }
}
