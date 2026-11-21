// A kernel with more reductions than the global reduce unit has entries
// (global_reduce_entries is 3 in this architecture spec) is rejected.
// RUN: not mlir-neura-opt %s \
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
// RUN:   --generate-code -o %t-mapping.mlir 2>&1 | FileCheck %s

func.func @kernel(%a: vector<4xi32>, %b: vector<4xi32>, %c: vector<4xi32>, %d: vector<4xi32>) -> i32 {
  %0 = "neura.vector.reduce.add"(%a) : (vector<4xi32>) -> i32
  %1 = "neura.vector.reduce.add"(%b) : (vector<4xi32>) -> i32
  %2 = "neura.vector.reduce.add"(%c) : (vector<4xi32>) -> i32
  %3 = "neura.vector.reduce.add"(%d) : (vector<4xi32>) -> i32
  %4 = llvm.add %0, %1 : i32
  %5 = llvm.add %2, %3 : i32
  %6 = llvm.add %4, %5 : i32
  return %6 : i32
}

// CHECK: kernel has more vector reductions than the 3 entries of the global reduce unit
