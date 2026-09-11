// RUN: mkdir -p %t.dir
// RUN: cd %t.dir && mlir-neura-opt %s --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --promote-input-arg-to-const --leverage-predicated-value \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=template mapping-mode=spatial-only" \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING
// RUN: FileCheck %s --input-file=%t.dir/tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=%t.dir/tmp-generated-instructions.asm --check-prefix=ASM
// RUN: sed 's/array<i64: 0, 3, 6>/array<i64: 0, 3, 9>/g' %t-mapping.mlir | \
// RUN:   not mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --generate-code 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: sed 's/memref<3x3xi32>/memref<3x3xi32, strided<[4, 1]>>/g' %t-mapping.mlir | \
// RUN:   not mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --generate-code 2>&1 | FileCheck %s --check-prefix=LAYOUT

// BOUNDS: constant element offset is out of bounds
// LAYOUT: stationary input requires a static identity-layout i32 memref

// SRAM addresses are configured through per-Tile constant queues.
// A and C use row-major element offsets; their SRAM bases are bound at launch.
// Physical coordinates: west LD column x=0; south ST row y=0.
module {
  func.func @ws_gemm_3x3(%A: memref<3x3xi32>, %B: memref<3x3xi32>, %C: memref<3x3xi32>) {
    neura.kernel inputs(%A, %B, %C : memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) attributes {
      accelerator = "neura",
      kernel_metadata = {kind = "template", template = {
        name = "systolic_array",
        stationary = {kernel_input = 1 : i32, map = affine_map<(x, y) -> (3 - y, x - 1)>}
      }}
    } {
    ^bb0(%a: memref<3x3xi32>, %b: memref<3x3xi32>, %c: memref<3x3xi32>):
      %a0 = "neura.load"(%a) {constants = array<i64: 0, 3, 6>, placement = {x = 0 : i32, y = 3 : i32}} : (memref<3x3xi32>) -> i32
      %m_1_3:2 = "neura.mac"(%a0) {placement = {x = 1 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
      %m_2_3:2 = "neura.mac"(%m_1_3#1) {placement = {x = 2 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
      %m_3_3:2 = "neura.mac"(%m_2_3#1) {placement = {x = 3 : i32, y = 3 : i32}} : (i32) -> (i32, i32)

      %a1 = "neura.load"(%a) {constants = array<i64: 1, 4, 7>, placement = {x = 0 : i32, y = 2 : i32}} : (memref<3x3xi32>) -> i32
      %m_1_2:2 = "neura.mac"(%a1, %m_1_3#0) {placement = {x = 1 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
      %m_2_2:2 = "neura.mac"(%m_1_2#1, %m_2_3#0) {placement = {x = 2 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
      %m_3_2:2 = "neura.mac"(%m_2_2#1, %m_3_3#0) {placement = {x = 3 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)

      %a2 = "neura.load"(%a) {constants = array<i64: 2, 5, 8>, placement = {x = 0 : i32, y = 1 : i32}} : (memref<3x3xi32>) -> i32
      %m_1_1:2 = "neura.mac"(%a2, %m_1_2#0) {placement = {x = 1 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
      %m_2_1:2 = "neura.mac"(%m_1_1#1, %m_2_2#0) {placement = {x = 2 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
      %m_3_1:2 = "neura.mac"(%m_2_1#1, %m_3_2#0) {placement = {x = 3 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)

      "neura.store"(%m_1_1#0, %c) {constants = array<i64: 0, 3, 6>, placement = {x = 1 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
      "neura.store"(%m_2_1#0, %c) {constants = array<i64: 1, 4, 7>, placement = {x = 2 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
      "neura.store"(%m_3_1#0, %c) {constants = array<i64: 2, 5, 8>, placement = {x = 3 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
      neura.yield
    }
    return
  }
}

// MAPPING: #map = affine_map<(d0, d1) -> (-d1 + 3, d0 - 1)>
// MAPPING-NEXT: module {
// MAPPING-NEXT:   func.func @ws_gemm_3x3(%arg0: memref<3x3xi32>, %arg1: memref<3x3xi32>, %arg2: memref<3x3xi32>) {
// MAPPING-NEXT:     neura.kernel inputs(%arg0, %arg1, %arg2 : memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {name = "systolic_array", stationary = {kernel_input = 1 : i32, map = #map}}}, mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-only", mapping_strategy = "template", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:     ^bb0(%arg3: !neura.data<memref<3x3xi32>, i1>, %arg4: !neura.data<memref<3x3xi32>, i1>, %arg5: !neura.data<memref<3x3xi32>, i1>):
// MAPPING-NEXT:       %0 = "neura.load"(%arg3) {constants = array<i64: 0, 3, 6>, dfg_id = 0 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<memref<3x3xi32>, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %1 = "neura.data_mov"(%0) {dfg_id = 4 : i32, mapping_locs = [{id = 38 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result, %forwarded = "neura.mac"(%1) {dfg_id = 7 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %2 = "neura.data_mov"(%forwarded) {dfg_id = 9 : i32, mapping_locs = [{id = 41 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_0, %forwarded_1 = "neura.mac"(%2) {dfg_id = 11 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %3 = "neura.data_mov"(%forwarded_1) {dfg_id = 15 : i32, mapping_locs = [{id = 44 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_2, %forwarded_3 = "neura.mac"(%3) {dfg_id = 18 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %4 = "neura.load"(%arg3) {constants = array<i64: 1, 4, 7>, dfg_id = 1 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<memref<3x3xi32>, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %5 = "neura.data_mov"(%4) {dfg_id = 5 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %6 = "neura.data_mov"(%result) {dfg_id = 8 : i32, mapping_locs = [{id = 42 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_4, %forwarded_5 = "neura.mac"(%5, %6) {dfg_id = 10 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %7 = "neura.data_mov"(%forwarded_5) {dfg_id = 13 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %8 = "neura.data_mov"(%result_0) {dfg_id = 14 : i32, mapping_locs = [{id = 45 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_6, %forwarded_7 = "neura.mac"(%7, %8) {dfg_id = 17 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %9 = "neura.data_mov"(%forwarded_7) {dfg_id = 22 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %10 = "neura.data_mov"(%result_2) {dfg_id = 23 : i32, mapping_locs = [{id = 47 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_8, %forwarded_9 = "neura.mac"(%9, %10) {dfg_id = 26 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %11 = "neura.load"(%arg3) {constants = array<i64: 2, 5, 8>, dfg_id = 2 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<memref<3x3xi32>, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %12 = "neura.data_mov"(%11) {dfg_id = 6 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %13 = "neura.data_mov"(%result_4) {dfg_id = 12 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_10, %forwarded_11 = "neura.mac"(%12, %13) {dfg_id = 16 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %14 = "neura.data_mov"(%forwarded_11) {dfg_id = 20 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %15 = "neura.data_mov"(%result_6) {dfg_id = 21 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_12, %forwarded_13 = "neura.mac"(%14, %15) {dfg_id = 25 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %16 = "neura.data_mov"(%forwarded_13) {dfg_id = 28 : i32, mapping_locs = [{id = 18 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %17 = "neura.data_mov"(%result_8) {dfg_id = 29 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_14, %forwarded_15 = "neura.mac"(%16, %17) {dfg_id = 31 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %18 = "neura.data_mov"(%result_10) {dfg_id = 19 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       "neura.store"(%18, %arg5) {constants = array<i64: 0, 3, 6>, dfg_id = 24 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<memref<3x3xi32>, i1>) -> ()
// MAPPING-NEXT:       %19 = "neura.data_mov"(%result_12) {dfg_id = 27 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       "neura.store"(%19, %arg5) {constants = array<i64: 1, 4, 7>, dfg_id = 30 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<memref<3x3xi32>, i1>) -> ()
// MAPPING-NEXT:       %20 = "neura.data_mov"(%result_14) {dfg_id = 32 : i32, mapping_locs = [{id = 22 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       "neura.store"(%20, %arg5) {constants = array<i64: 2, 5, 8>, dfg_id = 33 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 6 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<memref<3x3xi32>, i1>) -> ()
// MAPPING-NEXT:       neura.yield {dfg_id = 3 : i32}
// MAPPING-NEXT:     }
// MAPPING-NEXT:     return
// MAPPING-NEXT:   }
// MAPPING-NEXT: }

// YAML: array_config:
// YAML-NEXT: columns: 4
// YAML-NEXT: rows: 4
// YAML-NEXT: compiled_ii: 1
// YAML-NEXT: cores:
// YAML: entries:
// YAML: instructions:
// YAML: index_per_ii: 0
// YAML-NEXT: operations:
// YAML-NEXT: - opcode: "STORE"
// YAML: operand: "arg2"
// YAML-NEXT: color: "RED"
// YAML-NEXT: access: "address"
// YAML-NEXT: offsets: [0, 3, 6]
// YAML: opcode: "LOAD"
// YAML: access: "address"
// YAML: opcode: "MUL_ADD"
// YAML: operand: "arg1"
// YAML-NEXT: color: "RED"
// YAML-NEXT: access: "value"
// YAML-NEXT: offsets: [6]

// ASM: # Compiled II: 1
// ASM-EMPTY:
// ASM-NEXT: PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   STORE, [NORTH, RED], [address(arg2, 0, 3, 6)] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   STORE, [NORTH, RED], [address(arg2, 1, 4, 7)] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(3,0):
// ASM-NEXT: {
// ASM-NEXT:   STORE, [NORTH, RED], [address(arg2, 2, 5, 8)] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [address(arg0, 2, 5, 8)] -> [EAST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   MUL_ADD, [WEST, RED], [value(arg1, 6)], [NORTH, RED] -> [SOUTH, RED] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   MUL_ADD, [WEST, RED], [value(arg1, 7)], [NORTH, RED] -> [SOUTH, RED] (t=4, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(3,1):
// ASM-NEXT: {
// ASM-NEXT:   MUL_ADD, [WEST, RED], [value(arg1, 8)], [NORTH, RED] -> [SOUTH, RED] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(0,2):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [address(arg0, 1, 4, 7)] -> [EAST, RED] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(1,2):
// ASM-NEXT: {
// ASM-NEXT:   MUL_ADD, [WEST, RED], [value(arg1, 3)], [NORTH, RED] -> [SOUTH, RED] (t=2, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   MUL_ADD, [WEST, RED], [value(arg1, 4)], [NORTH, RED] -> [SOUTH, RED] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(3,2):
// ASM-NEXT: {
// ASM-NEXT:   MUL_ADD, [WEST, RED], [value(arg1, 5)], [NORTH, RED] -> [SOUTH, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(0,3):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [address(arg0, 0, 3, 6)] -> [EAST, RED] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(1,3):
// ASM-NEXT: {
// ASM-NEXT:   MUL, [WEST, RED], [value(arg1, 0)] -> [SOUTH, RED] (t=1, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(2,3):
// ASM-NEXT: {
// ASM-NEXT:   MUL, [WEST, RED], [value(arg1, 1)] -> [SOUTH, RED] (t=2, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(3,3):
// ASM-NEXT: {
// ASM-NEXT:   MUL, [WEST, RED], [value(arg1, 2)] -> [SOUTH, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
