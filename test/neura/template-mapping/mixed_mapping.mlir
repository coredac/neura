// RUN: mlir-neura-opt %s \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-mode=spatial-only" \
// RUN:   | FileCheck %s

module {
  func.func @mixed_mapping() {
    neura.kernel attributes {
      accelerator = "neura",
      kernel_metadata = {kind = "template"}
    } {
      %lhs = "neura.constant"() <{value = 1 : i32}>
        {placement = {x = 0 : i32, y = 0 : i32}}
        : () -> !neura.data<i32, i1>
      %rhs = "neura.constant"() <{value = 2 : i32}>
        {placement = {x = 2 : i32, y = 0 : i32}}
        : () -> !neura.data<i32, i1>
      %result = "neura.add"(%lhs, %rhs)
        {placement = {x = 1 : i32, y = 0 : i32}}
        : (!neura.data<i32, i1>, !neura.data<i32, i1>)
          -> !neura.data<i32, i1>
      neura.yield
    }

    neura.kernel attributes {accelerator = "neura"} {
      %lhs = "neura.constant"() <{value = 3 : i32}>
        : () -> !neura.data<i32, i1>
      %rhs = "neura.constant"() <{value = 4 : i32}>
        : () -> !neura.data<i32, i1>
      %result = "neura.add"(%lhs, %rhs)
        : (!neura.data<i32, i1>, !neura.data<i32, i1>)
          -> !neura.data<i32, i1>
      neura.yield
    }
    return
  }
}

// CHECK: neura.kernel
// CHECK-SAME: kernel_metadata = {kind = "template"}
// CHECK-SAME: mapping_info = {
// CHECK-SAME: mapping_strategy = "template"
// CHECK: neura.kernel
// CHECK-SAME: mapping_info = {
// CHECK-SAME: mapping_strategy = "heuristic"
