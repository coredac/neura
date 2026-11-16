// RUN: neura-interpreter %S/../../template-mapping/weight_stationary_gemm_4x4.mlir \
// RUN:   --dataflow --memory-in=%S/Inputs/ws_gemm_3x3.json \
// RUN:   --memory-out=%t-dataflow.json | FileCheck %s
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t-dataflow.json
// RUN: neura-interpreter %S/../../template-mapping/weight_stationary_gemm_4x4.mlir \
// RUN:   --memory-in=%S/Inputs/ws_gemm_3x3.json \
// RUN:   --memory-out=%t-control-flow.json | FileCheck %s
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t-control-flow.json

// Runs the kernel of test/neura/template-mapping/weight_stationary_gemm_4x4.mlir
// unchanged, in dataflow and control-flow modes: a 3x3 weight-stationary GEMM
// on a 4x4 CGRA, with one load column, one store row, and neura.mac on the
// other 3x3 Tiles.
//
// Inputs/ws_gemm_3x3.json holds the same A and B as the VectorCGRA RTL test
// (tancheng/VectorCGRA@neura-systolic-integration,
// cgra/test/CgraRTL_neura_systolic_test.py). As in that test, C starts as 123
// everywhere, so every output must be written.

// A and B are unchanged, and C = A @ B (row-major).
// CHECK:      Final memory[input #0] = [1, -2, 3, 4, 0, -1, 2, 5, 1]
// CHECK-NEXT: Final memory[input #1] = [2, 1, -1, 0, 3, 4, -2, 5, 2]
// CHECK-NEXT: Final memory[input #2] = [-4, 10, -3, 10, -1, -6, 2, 22, 20]

// JSON: {"0":[1,-2,3,4,0,-1,2,5,1],"1":[2,1,-1,0,3,4,-2,5,2],"2":[-4,10,-3,10,-1,-6,2,22,20]}
