// RUN: mlir-neura-opt %s \
// RUN:   --promote-input-arg-to-const \
// RUN:   --leverage-predicated-value \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=template mapping-mode=spatial-only" \
// RUN:   -o %t-mapping.mlir
// RUN:   FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING

module {
  func.func @ws_mac(%a0: i32, %a1: i32, %B: memref<?x?xi32>) -> i32 {
    %result = neura.kernel inputs(%a0, %a1, %B : i32, i32, memref<?x?xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {
          name = "systolic_array",
          stationary = {map = affine_map<(x, y) -> (1 - y, x)>, kernel_input = 2 : i32},
          input_ports = [
              {kernel_input = 0 : i32, direction = "west", x = 0 : i32, y = 1 : i32},
              {kernel_input = 1 : i32, direction = "west", x = 0 : i32, y = 0 : i32}
          ],
          output_ports = [
              {kernel_result = 0 : i32, direction = "south", x = 0 : i32, y = 0 : i32}
          ]
        }
      }
    } {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi32>):
      %m0:2 = "neura.mac"(%arg0) {placement = {x = 0 : i32, y = 1 : i32}} : (i32) -> (i32, i32)
      %m1:2 = "neura.mac"(%arg1, %m0#0) {placement = {x = 0 : i32, y = 0 : i32}} : (i32, i32) -> (i32, i32)

      neura.yield results(%m1#0 : i32)
    } : i32
    return %result : i32
  }
}

// MAPPING:      #map = affine_map<(d0, d1) -> (-d1 + 1, d0)>
// MAPPING-NEXT: module {
// MAPPING-NEXT:   func.func @ws_mac(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi32>) -> i32 {
// MAPPING-NEXT:     %0 = neura.kernel inputs(%arg0, %arg1, %arg2 : i32, i32, memref<?x?xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {input_ports = [{direction = "west", kernel_input = 0 : i32, x = 0 : i32, y = 1 : i32}, {direction = "west", kernel_input = 1 : i32, x = 0 : i32, y = 0 : i32}], name = "systolic_array", output_ports = [{direction = "south", kernel_result = 0 : i32, x = 0 : i32, y = 0 : i32}], stationary = {kernel_input = 2 : i32, map = #map}}}, mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-only", mapping_strategy = "template", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:     ^bb0(%arg3: !neura.data<i32, i1>, %arg4: !neura.data<i32, i1>, %arg5: !neura.data<memref<?x?xi32>, i1>):
// MAPPING-NEXT:       %1 = "neura.data_mov"(%arg3) {dfg_id = 0 : i32, mapping_locs = [{direction = "west", id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, io = "input", resource = "boundary_port", time_step = 0 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result, %forwarded = "neura.mac"(%1) {dfg_id = 2 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %2 = "neura.data_mov"(%arg4) {dfg_id = 1 : i32, mapping_locs = [{direction = "west", id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, io = "input", resource = "boundary_port", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %3 = "neura.data_mov"(%result) {dfg_id = 3 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_0, %forwarded_1 = "neura.mac"(%2, %3) {dfg_id = 4 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %4 = "neura.data_mov"(%result_0) {dfg_id = 5 : i32, mapping_locs = [{direction = "south", id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, io = "output", resource = "boundary_port", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       neura.yield results(%4 : !neura.data<i32, i1>) {dfg_id = 6 : i32}
// MAPPING-NEXT:     } : i32
// MAPPING-NEXT:     return %0 : i32
// MAPPING-NEXT:   }
// MAPPING-NEXT: }
