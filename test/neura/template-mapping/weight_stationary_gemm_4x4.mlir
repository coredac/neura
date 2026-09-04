// RUN: mlir-neura-opt %s \
// RUN:   --promote-input-arg-to-const \
// RUN:   --leverage-predicated-value \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=template mapping-mode=spatial-only" \
// RUN:   -o %t-mapping.mlir
// RUN:   FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING

module {
  func.func @ws_gemm_4x4(%a0: i32, %a1: i32, %a2: i32, %a3: i32, %B: memref<4x4xi32>) -> (i32, i32, i32, i32) {
    %result:4 = neura.kernel inputs(%a0, %a1, %a2, %a3, %B : i32, i32, i32, i32, memref<4x4xi32>) attributes {
      accelerator = "neura",
      kernel_metadata = {kind = "template",
        template = {
          name = "systolic_array",
          stationary = {
            map = affine_map<(x, y) -> (3 - y, x)>,
            kernel_input = 4 : i32
          },
          input_ports = [
            {kernel_input = 0 : i32, direction = "west", x = 0 : i32, y = 3 : i32},
            {kernel_input = 1 : i32, direction = "west", x = 0 : i32, y = 2 : i32},
            {kernel_input = 2 : i32, direction = "west", x = 0 : i32, y = 1 : i32},
            {kernel_input = 3 : i32, direction = "west", x = 0 : i32, y = 0 : i32}
          ],
          output_ports = [
            {kernel_result = 0 : i32, direction = "south", x = 0 : i32, y = 0 : i32},
            {kernel_result = 1 : i32, direction = "south", x = 1 : i32, y = 0 : i32},
            {kernel_result = 2 : i32, direction = "south", x = 2 : i32, y = 0 : i32},
            {kernel_result = 3 : i32, direction = "south", x = 3 : i32, y = 0 : i32}
          ]
        }
      }
    } {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<4x4xi32>):
      %m_0_3:2 = "neura.mac"(%arg0) {placement = {x = 0 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
      %m_1_3:2 = "neura.mac"(%m_0_3#1) {placement = {x = 1 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
      %m_2_3:2 = "neura.mac"(%m_1_3#1) {placement = {x = 2 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
      %m_3_3:2 = "neura.mac"(%m_2_3#1) {placement = {x = 3 : i32, y = 3 : i32}} : (i32) -> (i32, i32)

      %m_0_2:2 = "neura.mac"(%arg1, %m_0_3#0) {placement = {x = 0 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
      %m_1_2:2 = "neura.mac"(%m_0_2#1, %m_1_3#0) {placement = {x = 1 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
      %m_2_2:2 = "neura.mac"(%m_1_2#1, %m_2_3#0) {placement = {x = 2 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
      %m_3_2:2 = "neura.mac"(%m_2_2#1, %m_3_3#0) {placement = {x = 3 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)

      %m_0_1:2 = "neura.mac"(%arg2, %m_0_2#0) {placement = {x = 0 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
      %m_1_1:2 = "neura.mac"(%m_0_1#1, %m_1_2#0) {placement = {x = 1 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
      %m_2_1:2 = "neura.mac"(%m_1_1#1, %m_2_2#0) {placement = {x = 2 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
      %m_3_1:2 = "neura.mac"(%m_2_1#1, %m_3_2#0) {placement = {x = 3 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)

      %m_0_0:2 = "neura.mac"(%arg3, %m_0_1#0) {placement = {x = 0 : i32, y = 0 : i32}} : (i32, i32) -> (i32, i32)
      %m_1_0:2 = "neura.mac"(%m_0_0#1, %m_1_1#0) {placement = {x = 1 : i32, y = 0 : i32}} : (i32, i32) -> (i32, i32)
      %m_2_0:2 = "neura.mac"(%m_1_0#1, %m_2_1#0) {placement = {x = 2 : i32, y = 0 : i32}} : (i32, i32) -> (i32, i32)
      %m_3_0:2 = "neura.mac"(%m_2_0#1, %m_3_1#0) {placement = {x = 3 : i32, y = 0 : i32}} : (i32, i32) -> (i32, i32)

      neura.yield results(%m_0_0#0, %m_1_0#0, %m_2_0#0, %m_3_0#0 : i32, i32, i32, i32)
    } : i32, i32, i32, i32
    return %result#0, %result#1, %result#2, %result#3 : i32, i32, i32, i32
  }
}

// MAPPING:      #map = affine_map<(d0, d1) -> (-d1 + 3, d0)>
// MAPPING-NEXT: module {
// MAPPING-NEXT:   func.func @ws_gemm_4x4(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<4x4xi32>) -> (i32, i32, i32, i32) {
// MAPPING-NEXT:     %0:4 = neura.kernel inputs(%arg0, %arg1, %arg2, %arg3, %arg4 : i32, i32, i32, i32, memref<4x4xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {input_ports = [{direction = "west", kernel_input = 0 : i32, x = 0 : i32, y = 3 : i32}, {direction = "west", kernel_input = 1 : i32, x = 0 : i32, y = 2 : i32}, {direction = "west", kernel_input = 2 : i32, x = 0 : i32, y = 1 : i32}, {direction = "west", kernel_input = 3 : i32, x = 0 : i32, y = 0 : i32}], name = "systolic_array", output_ports = [{direction = "south", kernel_result = 0 : i32, x = 0 : i32, y = 0 : i32}, {direction = "south", kernel_result = 1 : i32, x = 1 : i32, y = 0 : i32}, {direction = "south", kernel_result = 2 : i32, x = 2 : i32, y = 0 : i32}, {direction = "south", kernel_result = 3 : i32, x = 3 : i32, y = 0 : i32}], stationary = {kernel_input = 4 : i32, map = #map}}}, mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-only", mapping_strategy = "template", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:     ^bb0(%arg5: !neura.data<i32, i1>, %arg6: !neura.data<i32, i1>, %arg7: !neura.data<i32, i1>, %arg8: !neura.data<i32, i1>, %arg9: !neura.data<memref<4x4xi32>, i1>):
// MAPPING-NEXT:       %1 = "neura.data_mov"(%arg5) {dfg_id = 0 : i32, mapping_locs = [{direction = "west", id = 20 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, io = "input", resource = "boundary_port", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result, %forwarded = "neura.mac"(%1) {dfg_id = 4 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %2 = "neura.data_mov"(%forwarded) {dfg_id = 6 : i32, mapping_locs = [{id = 38 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_0, %forwarded_1 = "neura.mac"(%2) {dfg_id = 8 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %3 = "neura.data_mov"(%forwarded_1) {dfg_id = 12 : i32, mapping_locs = [{id = 41 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_2, %forwarded_3 = "neura.mac"(%3) {dfg_id = 15 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %4 = "neura.data_mov"(%forwarded_3) {dfg_id = 21 : i32, mapping_locs = [{id = 44 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_4, %forwarded_5 = "neura.mac"(%4) {dfg_id = 25 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %5 = "neura.data_mov"(%arg6) {dfg_id = 1 : i32, mapping_locs = [{direction = "west", id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, io = "input", resource = "boundary_port", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %6 = "neura.data_mov"(%result) {dfg_id = 5 : i32, mapping_locs = [{id = 39 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_6, %forwarded_7 = "neura.mac"(%5, %6) {dfg_id = 7 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %7 = "neura.data_mov"(%forwarded_7) {dfg_id = 10 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %8 = "neura.data_mov"(%result_0) {dfg_id = 11 : i32, mapping_locs = [{id = 42 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_8, %forwarded_9 = "neura.mac"(%7, %8) {dfg_id = 14 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %9 = "neura.data_mov"(%forwarded_9) {dfg_id = 19 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %10 = "neura.data_mov"(%result_2) {dfg_id = 20 : i32, mapping_locs = [{id = 45 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_10, %forwarded_11 = "neura.mac"(%9, %10) {dfg_id = 24 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %11 = "neura.data_mov"(%forwarded_11) {dfg_id = 31 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %12 = "neura.data_mov"(%result_4) {dfg_id = 32 : i32, mapping_locs = [{id = 47 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_12, %forwarded_13 = "neura.mac"(%11, %12) {dfg_id = 35 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %13 = "neura.data_mov"(%arg7) {dfg_id = 2 : i32, mapping_locs = [{direction = "west", id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, io = "input", resource = "boundary_port", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %14 = "neura.data_mov"(%result_6) {dfg_id = 9 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_14, %forwarded_15 = "neura.mac"(%13, %14) {dfg_id = 13 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %15 = "neura.data_mov"(%forwarded_15) {dfg_id = 17 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %16 = "neura.data_mov"(%result_8) {dfg_id = 18 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_16, %forwarded_17 = "neura.mac"(%15, %16) {dfg_id = 23 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %17 = "neura.data_mov"(%forwarded_17) {dfg_id = 29 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %18 = "neura.data_mov"(%result_10) {dfg_id = 30 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_18, %forwarded_19 = "neura.mac"(%17, %18) {dfg_id = 34 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %19 = "neura.data_mov"(%forwarded_19) {dfg_id = 39 : i32, mapping_locs = [{id = 18 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %20 = "neura.data_mov"(%result_12) {dfg_id = 40 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_20, %forwarded_21 = "neura.mac"(%19, %20) {dfg_id = 42 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %21 = "neura.data_mov"(%arg8) {dfg_id = 3 : i32, mapping_locs = [{direction = "west", id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, io = "input", resource = "boundary_port", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %22 = "neura.data_mov"(%result_14) {dfg_id = 16 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_22, %forwarded_23 = "neura.mac"(%21, %22) {dfg_id = 22 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %23 = "neura.data_mov"(%forwarded_23) {dfg_id = 27 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %24 = "neura.data_mov"(%result_16) {dfg_id = 28 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_24, %forwarded_25 = "neura.mac"(%23, %24) {dfg_id = 33 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %25 = "neura.data_mov"(%forwarded_25) {dfg_id = 37 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %26 = "neura.data_mov"(%result_18) {dfg_id = 38 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_26, %forwarded_27 = "neura.mac"(%25, %26) {dfg_id = 41 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %27 = "neura.data_mov"(%forwarded_27) {dfg_id = 44 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %28 = "neura.data_mov"(%result_20) {dfg_id = 45 : i32, mapping_locs = [{id = 22 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_28, %forwarded_29 = "neura.mac"(%27, %28) {dfg_id = 46 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 6 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %29 = "neura.data_mov"(%result_22) {dfg_id = 26 : i32, mapping_locs = [{direction = "south", id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, io = "output", resource = "boundary_port", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %30 = "neura.data_mov"(%result_24) {dfg_id = 36 : i32, mapping_locs = [{direction = "south", id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, io = "output", resource = "boundary_port", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %31 = "neura.data_mov"(%result_26) {dfg_id = 43 : i32, mapping_locs = [{direction = "south", id = 7 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, io = "output", resource = "boundary_port", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %32 = "neura.data_mov"(%result_28) {dfg_id = 47 : i32, mapping_locs = [{direction = "south", id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 6 : i32, io = "output", resource = "boundary_port", time_step = 6 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       neura.yield results(%29, %30, %31, %32 : !neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) {dfg_id = 48 : i32}
// MAPPING-NEXT:     } : i32, i32, i32, i32
// MAPPING-NEXT:     return %0#0, %0#1, %0#2, %0#3 : i32, i32, i32, i32
// MAPPING-NEXT:   }
// MAPPING-NEXT: }
