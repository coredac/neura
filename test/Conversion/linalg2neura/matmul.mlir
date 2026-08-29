// Compiles a linalg.matmul (tensor-level, the shape torch-mlir produces for
// real models) through bufferization and the affine-based Neura pipeline,
// all the way to a mapped, code-generated program. This exercises the
// linalg -> affine -> neura ingestion path end to end: bufferize, expand the
// trailing memref.copy that -buffer-results-to-out-params leaves behind,
// lower to affine loops, then through the existing affine-sourced Neura
// pipeline (see test/neura/for_loop for the analogous LLVM-sourced path).
//
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -buffer-results-to-out-params \
// RUN:   | mlir-neura-opt --expand-memref-copy \
// RUN:   | mlir-opt -convert-linalg-to-affine-loops \
// RUN:   | mlir-neura-opt \
// RUN:       --assign-accelerator \
// RUN:       --lower-affine-to-neura \
// RUN:       --lower-arith-to-neura \
// RUN:       --promote-input-arg-to-const \
// RUN:       --fold-constant \
// RUN:       --canonicalize-return \
// RUN:       --canonicalize-live-in \
// RUN:       --leverage-predicated-value \
// RUN:       --insert-data-mov \
// RUN:       --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:       --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   -o %t-mapped.mlir
// RUN: FileCheck %s --input-file=%t-mapped.mlir

func.func @matmul(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %c = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @matmul
// CHECK-SAME: mapping_info
// CHECK: neura.loop_control
// CHECK: neura.load_indexed
// CHECK: neura.fmul
// CHECK: neura.fadd
// CHECK: neura.store_indexed
// CHECK-NOT: linalg.matmul
// CHECK-NOT: memref.copy
// CHECK-NOT: affine.for
