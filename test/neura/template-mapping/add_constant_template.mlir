// RUN: mlir-neura-opt %s \
// RUN:   --insert-data-mov \
// RUN:   -o %t-data-mov.mlir
// RUN:   FileCheck %s --input-file=%t-data-mov.mlir --check-prefix=DATA-MOV

// RUN: mlir-neura-opt %s \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=template mapping-mode=spatial-only" \
// RUN:   -o %t-mapping.mlir
// RUN:   FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING

module {
  func.func @add_constant() {
    neura.kernel attributes {accelerator = "neura"} {
      %lhs = "neura.constant"() <{value = 1 : i32}> {placement = {x = 0 : i32, y = 0 : i32}} : () -> !neura.data<i32, i1>

      %rhs = "neura.constant"() <{value = 2 : i32}> {placement = {x = 2 : i32, y = 0 : i32}} : () -> !neura.data<i32, i1>

      %result = "neura.add"(%lhs, %rhs) {placement = {x = 1 : i32, y = 0 : i32}} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>

      neura.yield
    }
    return
  }
}

// DATA-MOV:     module {
// DATA-MOV-NEXT:  func.func @add_constant() {
// DATA-MOV-NEXT:    neura.kernel attributes {accelerator = "neura"} {
// DATA-MOV-NEXT:      %0 = "neura.constant"() <{value = 1 : i32}> {placement = {x = 0 : i32, y = 0 : i32}} : () -> !neura.data<i32, i1>
// DATA-MOV-NEXT:      %1 = "neura.constant"() <{value = 2 : i32}> {placement = {x = 2 : i32, y = 0 : i32}} : () -> !neura.data<i32, i1>
// DATA-MOV-NEXT:      %2 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATA-MOV-NEXT:      %3 = "neura.data_mov"(%1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATA-MOV-NEXT:      %4 = "neura.add"(%2, %3) {placement = {x = 1 : i32, y = 0 : i32}} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATA-MOV-NEXT:      neura.yield
// DATA-MOV-NEXT:    }
// DATA-MOV-NEXT:    return
// DATA-MOV-NEXT:  }
// DATA-MOV-NEXT:}

// MAPPING:     module {
// MAPPING-NEXT:  func.func @add_constant() {
// MAPPING-NEXT:    neura.kernel attributes {accelerator = "neura", mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-only", mapping_strategy = "template", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:      %0 = "neura.constant"() <{value = 1 : i32}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:      %1 = "neura.constant"() <{value = 2 : i32}> {dfg_id = 1 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:      %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:      %3 = "neura.data_mov"(%1) {dfg_id = 4 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:      %4 = "neura.add"(%2, %3) {dfg_id = 5 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:      neura.yield {dfg_id = 2 : i32}
// MAPPING-NEXT:    }
// MAPPING-NEXT:    return
// MAPPING-NEXT:  }
// MAPPING-NEXT:}