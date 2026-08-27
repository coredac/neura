// RUN: not mlir-neura-opt %s \
// RUN:   --architecture-spec=%S/../arch_spec/architecture.yaml \
// RUN:   --cost-model-analytical="x-tiles=4 y-tiles=4 valid-tiles=1_1" \
// RUN:   2>&1 | FileCheck %s

// Tile (1,1) has no memory FU. The model must report infeasibility instead of
// inventing capacity one or attaching a finite analytical II.

module {
  func.func @missing_mem_fu() attributes {accelerator = "neura"} {
    %ptr = "neura.grant_once"() <{constant_value = "%arg0"}>
        : () -> !neura.data<!llvm.ptr, i1>
    %value = "neura.load"(%ptr)
        : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    neura.return_value %value : !neura.data<i32, i1>
    neura.yield
  }
}

// CHECK: compute_mii=0 (infeasible: no tile provides fu class mem)
// CHECK: final_ii=0 (dominant=infeasible, INFEASIBLE:
// CHECK: error: analytical cost model is infeasible:
