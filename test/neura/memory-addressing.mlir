// RUN: mlir-neura-opt %s --split-input-file --verify-diagnostics -o /dev/null

// Each operation selects its address mode through operands and constants only.
func.func @address_forms(%addr: i64, %base: memref<3x3xi32>, %value: i32) {
  %a = "neura.load"(%addr) : (i64) -> i32
  %b = "neura.load"(%base) {constants = array<i64: 0, 3, 6>} : (memref<3x3xi32>) -> i32
  %c = "neura.load"() {constants = array<i64: 32, 35, 38>} : () -> i32
  "neura.store"(%value, %addr) : (i32, i64) -> ()
  "neura.store"(%value, %base) {constants = array<i64: 0, 3, 6>} : (i32, memref<3x3xi32>) -> ()
  "neura.store"(%value) {constants = array<i64: 32, 35, 38>} : (i32) -> ()
  return
}

// -----

func.func @missing_load_address() {
  // expected-error @+1 {{requires an address operand or constants}}
  %a = "neura.load"() : () -> i32
  return
}

// -----

func.func @missing_store_address(%value: i32) {
  // expected-error @+1 {{requires an address operand or constants}}
  "neura.store"(%value) : (i32) -> ()
  return
}

// -----

func.func @empty_constants(%addr: i64) {
  // expected-error @+1 {{constants must be a nonempty i64 array}}
  %a = "neura.load"(%addr) {constants = array<i64>} : (i64) -> i32
  return
}

// -----

func.func @wrong_constants(%value: i32) {
  // expected-error @+1 {{constants must be a nonempty i64 array}}
  "neura.store"(%value) {constants = array<i32: 2>} : (i32) -> ()
  return
}

// -----

func.func @dynamic_base_with_constants(%base: i64) {
  // expected-error @+1 {{operand plus constants requires a memref base}}
  %a = "neura.load"(%base) {constants = array<i64: 0, 3, 6>} : (i64) -> i32
  return
}

// -----

func.func @negative_address() {
  // expected-error @+1 {{absolute constant addresses must be nonnegative}}
  %a = "neura.load"() {constants = array<i64: -1>} : () -> i32
  return
}

// -----
