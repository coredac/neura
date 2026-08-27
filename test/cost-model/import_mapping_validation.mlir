// RUN: not mlir-neura-opt %s \
// RUN:   --architecture-spec=%S/../arch_spec/architecture_4x4.yaml \
// RUN:   --map-to-accelerator="import-mapping=%S/import_duplicate_placement.json" \
// RUN:   2>&1 | FileCheck %s --check-prefix=DUPLICATE

// The importer must reject duplicate IDs instead of overwriting the first
// placement and continuing with a partial witness.

module {
  func.func @duplicate_placement() attributes {accelerator = "neura"} {
    %c = "neura.grant_once"() <{constant_value = 0 : i64}>
        : () -> !neura.data<i64, i1>
    neura.return_void %c : !neura.data<i64, i1>
    neura.yield
  }
}

// DUPLICATE: import-mapping error: duplicate placement id 0
