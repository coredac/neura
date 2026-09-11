// RUN: mkdir -p %t.dir
// RUN: cd %t.dir && mlir-neura-opt %s --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --promote-input-arg-to-const --leverage-predicated-value \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=template mapping-mode=spatial-only" \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING
// RUN: FileCheck %s --input-file=%t.dir/tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=%t.dir/tmp-generated-instructions.asm --check-prefix=ASM

// SRAM addresses are configured through per-Tile constant queues.
// A and C use row-major element offsets; their SRAM bases are bound at launch.
// Physical coordinates: west LD column x=0; south ST row y=0.
module {
  func.func @ws_mac(%A: memref<2x2xi32>, %B: memref<2x1xi32>, %C: memref<2x1xi32>) {
    neura.kernel inputs(%A, %B, %C : memref<2x2xi32>, memref<2x1xi32>, memref<2x1xi32>) attributes {
      accelerator = "neura",
      kernel_metadata = {kind = "template", template = {
        name = "systolic_array",
        stationary = {kernel_input = 1 : i32, map = affine_map<(x, y) -> (2 - y, x - 1)>}
      }}
    } {
    ^bb0(%a: memref<2x2xi32>, %b: memref<2x1xi32>, %c: memref<2x1xi32>):
      %a0 = "neura.load"(%a) {constants = array<i64: 0, 2>, placement = {x = 0 : i32, y = 2 : i32}} : (memref<2x2xi32>) -> i32
      %m_1_2:2 = "neura.mac"(%a0) {placement = {x = 1 : i32, y = 2 : i32}} : (i32) -> (i32, i32)

      %a1 = "neura.load"(%a) {constants = array<i64: 1, 3>, placement = {x = 0 : i32, y = 1 : i32}} : (memref<2x2xi32>) -> i32
      %m_1_1:2 = "neura.mac"(%a1, %m_1_2#0) {placement = {x = 1 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)

      "neura.store"(%m_1_1#0, %c) {constants = array<i64: 0, 1>, placement = {x = 1 : i32, y = 0 : i32}} : (i32, memref<2x1xi32>) -> ()
      neura.yield
    }
    return
  }
}

// MAPPING: #map = affine_map<(d0, d1) -> (-d1 + 2, d0 - 1)>
// MAPPING-NEXT: module {
// MAPPING-NEXT:   func.func @ws_mac(%arg0: memref<2x2xi32>, %arg1: memref<2x1xi32>, %arg2: memref<2x1xi32>) {
// MAPPING-NEXT:     neura.kernel inputs(%arg0, %arg1, %arg2 : memref<2x2xi32>, memref<2x1xi32>, memref<2x1xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {name = "systolic_array", stationary = {kernel_input = 1 : i32, map = #map}}}, mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-only", mapping_strategy = "template", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:     ^bb0(%arg3: !neura.data<memref<2x2xi32>, i1>, %arg4: !neura.data<memref<2x1xi32>, i1>, %arg5: !neura.data<memref<2x1xi32>, i1>):
// MAPPING-NEXT:       %0 = "neura.load"(%arg3) {constants = array<i64: 0, 2>, dfg_id = 0 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<memref<2x2xi32>, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %1 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result, %forwarded = "neura.mac"(%1) {dfg_id = 5 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %2 = "neura.load"(%arg3) {constants = array<i64: 1, 3>, dfg_id = 1 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<memref<2x2xi32>, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %3 = "neura.data_mov"(%2) {dfg_id = 4 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %4 = "neura.data_mov"(%result) {dfg_id = 6 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       %result_0, %forwarded_1 = "neura.mac"(%3, %4) {dfg_id = 7 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
// MAPPING-NEXT:       %5 = "neura.data_mov"(%result_0) {dfg_id = 8 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:       "neura.store"(%5, %arg5) {constants = array<i64: 0, 1>, dfg_id = 9 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<memref<2x1xi32>, i1>) -> ()
// MAPPING-NEXT:       neura.yield {dfg_id = 2 : i32}
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
// YAML-NEXT: offsets: [0, 1]
// YAML: opcode: "LOAD"
// YAML: access: "address"
// YAML: opcode: "MUL_ADD"
// YAML: operand: "arg1"
// YAML-NEXT: color: "RED"
// YAML-NEXT: access: "value"
// YAML-NEXT: offsets: [1]

// ASM: # Compiled II: 1
// ASM-EMPTY:
// ASM-NEXT: PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   STORE, [NORTH, RED], [address(arg2, 0, 1)] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [address(arg0, 1, 3)] -> [EAST, RED] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   MUL_ADD, [WEST, RED], [value(arg1, 1)], [NORTH, RED] -> [SOUTH, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(0,2):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [address(arg0, 0, 2)] -> [EAST, RED] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-EMPTY:
// ASM-NEXT: PE(1,2):
// ASM-NEXT: {
// ASM-NEXT:   MUL, [WEST, RED], [value(arg1, 0)] -> [SOUTH, RED] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
