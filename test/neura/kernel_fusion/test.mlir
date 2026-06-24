// RUN: mlir-neura-opt %s \
// RUN:   --affine-loop-tree-serialization \
// RUN:   --convert-affine-to-taskflow \
// RUN:   --construct-hyperblock-from-task \
// RUN:   --fuse-task \
// RUN:   -o %t-kernel-fuse.mlir
// RUN:   FileCheck --input-file=%t-kernel-fuse.mlir %s

module {

// =============================================================================
// TEST 1: Producer-Consumer Fusion
// Loops share intermediate memref C: loop1 writes C, loop2 reads C.
// Expected: Fused into one task; intermediate store/load eliminated.
// =============================================================================

// CHECK-LABEL: func.func @test_producer_consumer
// CHECK:         taskflow.task @fused_pc
// CHECK:           arith.addf
// CHECK-NEXT:      arith.mulf
// CHECK-NEXT:      memref.store
// CHECK-NOT:     taskflow.task
// CHECK:         return

func.func @test_producer_consumer(%A: memref<64xf32>, %B: memref<64xf32>,
                                  %C: memref<64xf32>, %D: memref<64xf32>) {
  %cst = arith.constant 2.000000e+00 : f32
  affine.for %i = 0 to 64 {
    %v = affine.load %A[%i] : memref<64xf32>
    %w = affine.load %B[%i] : memref<64xf32>
    %s = arith.addf %v, %w : f32
    affine.store %s, %C[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %C[%i] : memref<64xf32>
    %r = arith.mulf %v, %cst : f32
    affine.store %r, %D[%i] : memref<64xf32>
  }
  return
}

// =============================================================================
// TEST 2: Sibling Fusion
// Both loops read from A without data dependency.
// Expected: Fused into one task with deduplicated read of A.
// =============================================================================

// CHECK-LABEL: func.func @test_sibling
// CHECK:         taskflow.task @fused_sibling
// CHECK:           arith.mulf
// CHECK:           memref.store
// CHECK:           arith.addf
// CHECK:           memref.store
// CHECK-NOT:     taskflow.task
// CHECK:         return

func.func @test_sibling(%A: memref<64xf32>, %B: memref<64xf32>,
                         %C: memref<64xf32>) {
  %cst3 = arith.constant 3.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  affine.for %i = 0 to 64 {
    %v = affine.load %A[%i] : memref<64xf32>
    %r = arith.mulf %v, %cst3 : f32
    affine.store %r, %B[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %A[%i] : memref<64xf32>
    %r = arith.addf %v, %cst1 : f32
    affine.store %r, %C[%i] : memref<64xf32>
  }
  return
}

// =============================================================================
// TEST 3: No Shared Input (No Fusion)
// Loops read from different arrays (A and B) with no dependency.
// Expected: Two separate tasks remain.
// =============================================================================

// CHECK-LABEL: func.func @test_no_shared_input
// CHECK:         taskflow.task @Task_0
// CHECK:         taskflow.task @Task_1
// CHECK:         return

func.func @test_no_shared_input(%A: memref<64xf32>, %B: memref<64xf32>,
                                 %C: memref<64xf32>, %D: memref<64xf32>) {
  %cst2 = arith.constant 2.000000e+00 : f32
  %cst3 = arith.constant 3.000000e+00 : f32
  affine.for %i = 0 to 64 {
    %v = affine.load %A[%i] : memref<64xf32>
    %r = arith.mulf %v, %cst2 : f32
    affine.store %r, %C[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %B[%i] : memref<64xf32>
    %r = arith.addf %v, %cst3 : f32
    affine.store %r, %D[%i] : memref<64xf32>
  }
  return
}

// =============================================================================
// TEST 4: Chain Fusion (A -> B -> C)
// Three loops chained via intermediates C and D.
// Expected: All fused into one task; all intermediate store/load eliminated.
// =============================================================================

// CHECK-LABEL: func.func @test_chain
// CHECK:         taskflow.task @fused_pc
// CHECK:           arith.addf
// CHECK-NEXT:      arith.mulf
// CHECK-NEXT:      arith.addf
// CHECK-NEXT:      memref.store
// CHECK-NOT:     taskflow.task
// CHECK:         return

func.func @test_chain(%A: memref<64xf32>, %B: memref<64xf32>,
                       %C: memref<64xf32>, %D: memref<64xf32>,
                       %E: memref<64xf32>) {
  %cst2 = arith.constant 2.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  affine.for %i = 0 to 64 {
    %v = affine.load %A[%i] : memref<64xf32>
    %w = affine.load %B[%i] : memref<64xf32>
    %s = arith.addf %v, %w : f32
    affine.store %s, %C[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %C[%i] : memref<64xf32>
    %r = arith.mulf %v, %cst2 : f32
    affine.store %r, %D[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %D[%i] : memref<64xf32>
    %r = arith.addf %v, %cst1 : f32
    affine.store %r, %E[%i] : memref<64xf32>
  }
  return
}

// =============================================================================
// TEST 5: Sibling Fusion Rejected - Compute Pressure
// Both loops read from shared array A and each performs a long multiply-add
// chain. The fused kernel's operation count is large enough to push res_mii
// above the individual res_mii values, making fusion unprofitable.
// Expected: Two separate tasks remain.
// =============================================================================

// CHECK-LABEL: func.func @test_sibling_rejected_compute
// CHECK:         taskflow.task @Task_0
// CHECK:         taskflow.task @Task_1
// CHECK-NOT:     fused_sibling
// CHECK:         return

func.func @test_sibling_rejected_compute(%A: memref<64xf32>,
                                          %B: memref<64xf32>,
                                          %C: memref<64xf32>) {
  %c1 = arith.constant 1.5e+00 : f32
  %c2 = arith.constant 2.5e+00 : f32
  affine.for %i = 0 to 64 {
    %v  = affine.load %A[%i] : memref<64xf32>
    %r1 = arith.mulf %v,  %c1 : f32
    %r2 = arith.mulf %r1, %c2 : f32
    %r3 = arith.mulf %r2, %c1 : f32
    %r4 = arith.mulf %r3, %c2 : f32
    %r5 = arith.mulf %r4, %c1 : f32
    %r6 = arith.mulf %r5, %c2 : f32
    %r7 = arith.addf %r6, %v  : f32
    affine.store %r7, %B[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v  = affine.load %A[%i] : memref<64xf32>
    %r1 = arith.mulf %v,  %c2 : f32
    %r2 = arith.mulf %r1, %c1 : f32
    %r3 = arith.mulf %r2, %c2 : f32
    %r4 = arith.mulf %r3, %c1 : f32
    %r5 = arith.mulf %r4, %c2 : f32
    %r6 = arith.mulf %r5, %c1 : f32
    %r7 = arith.addf %r6, %v  : f32
    affine.store %r7, %C[%i] : memref<64xf32>
  }
  return
}

// =============================================================================
// TEST 6: Sibling Fusion Rejected - Memory Pressure
// Both loops share array A but each also loads from two additional independent
// arrays, so no deduplication occurs for those loads. The accumulated load and
// compute operations in the fused kernel raise res_mii beyond either individual,
// making fusion unprofitable.
// Expected: Two separate tasks remain.
// =============================================================================

// CHECK-LABEL: func.func @test_sibling_rejected_memory
// CHECK-NOT:         taskflow.task @fused_sibling
// CHECK:         return

func.func @test_sibling_rejected_memory(%A: memref<64xf32>,
                                         %B: memref<64xf32>,
                                         %C: memref<64xf32>,
                                         %D: memref<64xf32>,
                                         %E: memref<64xf32>,
                                         %F: memref<64xf32>,
                                         %G: memref<64xf32>) {
  affine.for %i = 0 to 64 {
    %va = affine.load %A[%i] : memref<64xf32>
    %vb = affine.load %B[%i] : memref<64xf32>
    %vc = affine.load %C[%i] : memref<64xf32>
    %s1 = arith.addf %va, %vb : f32
    %s2 = arith.mulf %s1, %vc : f32
    %s3 = arith.mulf %s2, %va : f32
    %s4 = arith.addf %s3, %vb : f32
    %s5 = arith.mulf %s4, %vc : f32
    %s6 = arith.addf %s5, %va : f32
    affine.store %s6, %D[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %va = affine.load %A[%i] : memref<64xf32>
    %ve = affine.load %E[%i] : memref<64xf32>
    %vf = affine.load %F[%i] : memref<64xf32>
    %s1 = arith.addf %va, %ve : f32
    %s2 = arith.mulf %s1, %vf : f32
    %s3 = arith.mulf %s2, %va : f32
    %s4 = arith.addf %s3, %ve : f32
    %s5 = arith.mulf %s4, %vf : f32
    %s6 = arith.addf %s5, %va : f32
    affine.store %s6, %G[%i] : memref<64xf32>
  }
  return
}

}
