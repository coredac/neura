// Compiles a vectorized kernel with two reductions. Each is emitted as a global
// reduce with its own reduce id.
// RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -o %t-kernel-full.ll %S/global_reduce_kernel.cpp
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

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
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture_with_global_reduce.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=IDS

// MAPPING-COUNT-2: "neura.vector.reduce.add"(%{{.*}}) {{.*}} : (!neura.data<vector<4xi32>, i1>) -> !neura.data<i32, i1>

// The two reductions sit on different tiles, so their order in the YAML
// follows the mapping.
// YAML-COUNT-2: opcode: "VECTOR.REDUCE.ADD.GLOBAL"
// IDS-DAG: reduce_id: 0
// IDS-DAG: reduce_id: 1
