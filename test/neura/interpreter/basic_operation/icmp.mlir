// RUN: neura-interpreter %s --verbose | FileCheck %s

// ====== Equal comparison (eq) ======
// Positive case: Equal numbers
func.func @test_icmp_eq_true() -> i1 {
  %a = arith.constant 42 : i32
  %eq = "neura.icmp"(%a, %a) {cmpType = "eq"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %eq : i1
}

// Negative case: Unequal numbers
func.func @test_icmp_eq_false() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant 43 : i32
  %eq = "neura.icmp"(%a, %b) {cmpType = "eq"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %eq : i1
}

// ====== Not equal comparison (ne) ======
// Positive case: Unequal numbers
func.func @test_icmp_ne_true() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant 43 : i32
  %ne = "neura.icmp"(%a, %b) {cmpType = "ne"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %ne : i1
}

// Negative case: Equal numbers
func.func @test_icmp_ne_false() -> i1 {
  %a = arith.constant 42 : i32
  %ne = "neura.icmp"(%a, %a) {cmpType = "ne"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %ne : i1
}

// ====== Signed less than (slt) ======
// Positive case: lhs < rhs
func.func @test_icmp_slt_true() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant 43 : i32
  %slt = "neura.icmp"(%a, %b) {cmpType = "slt"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %slt : i1
}

// Negative case: lhs >= rhs
func.func @test_icmp_slt_false() -> i1 {
  %a = arith.constant 43 : i32
  %b = arith.constant 42 : i32
  %slt = "neura.icmp"(%a, %b) {cmpType = "slt"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %slt : i1
}

// ====== Signed less than or equal (sle) ======
// Positive case: lhs <= rhs
func.func @test_icmp_sle_true() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant 43 : i32
  %sle = "neura.icmp"(%a, %b) {cmpType = "sle"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %sle : i1
}

// Negative case: lhs > rhs
func.func @test_icmp_sle_false() -> i1 {
  %a = arith.constant 43 : i32
  %b = arith.constant 42 : i32
  %sle = "neura.icmp"(%a, %b) {cmpType = "sle"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %sle : i1
}

// ====== Signed greater than (sgt) ======
// Positive case: lhs > rhs
func.func @test_icmp_sgt_true() -> i1 {
  %a = arith.constant 43 : i32
  %b = arith.constant 42 : i32
  %sgt = "neura.icmp"(%a, %b) {cmpType = "sgt"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %sgt : i1
}

// Negative case: lhs <= rhs
func.func @test_icmp_sgt_false() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant 43 : i32
  %sgt = "neura.icmp"(%a, %b) {cmpType = "sgt"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %sgt : i1
}

// ====== Signed greater than or equal (sge) ======
// Positive case: lhs >= rhs
func.func @test_icmp_sge_true() -> i1 {
  %a = arith.constant 43 : i32
  %b = arith.constant 42 : i32
  %sge = "neura.icmp"(%a, %b) {cmpType = "sge"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %sge : i1
}

// Negative case: lhs < rhs
func.func @test_icmp_sge_false() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant 43 : i32
  %sge = "neura.icmp"(%a, %b) {cmpType = "sge"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %sge : i1
}

// ====== Unsigned less than (ult) ======
// Positive case: lhs < rhs (unsigned)
func.func @test_icmp_ult_true() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant 43 : i32
  %ult = "neura.icmp"(%a, %b) {cmpType = "ult"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %ult : i1
}

// Negative case: lhs >= rhs (unsigned)
func.func @test_icmp_ult_false() -> i1 {
  %a = arith.constant -1 : i32  // Unsigned: 4294967295
  %b = arith.constant 42 : i32
  %ult = "neura.icmp"(%a, %b) {cmpType = "ult"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %ult : i1
}

// ====== Unsigned less than or equal (ule) ======
// Positive case: lhs <= rhs (unsigned)
func.func @test_icmp_ule_true() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant 43 : i32
  %ule = "neura.icmp"(%a, %b) {cmpType = "ule"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %ule : i1
}

// Negative case: lhs > rhs (unsigned)
func.func @test_icmp_ule_false() -> i1 {
  %a = arith.constant -1 : i32  // Unsigned: 4294967295
  %b = arith.constant 42 : i32
  %ule = "neura.icmp"(%a, %b) {cmpType = "ule"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %ule : i1
}

// ====== Unsigned greater than (ugt) ======
// Positive case: lhs > rhs (unsigned)
func.func @test_icmp_ugt_true() -> i1 {
  %a = arith.constant -1 : i32  // Unsigned: 4294967295
  %b = arith.constant 42 : i32
  %ugt = "neura.icmp"(%a, %b) {cmpType = "ugt"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %ugt : i1
}

// Negative case: lhs <= rhs (unsigned)
func.func @test_icmp_ugt_false() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant -1 : i32  // Unsigned: 4294967295
  %ugt = "neura.icmp"(%a, %b) {cmpType = "ugt"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %ugt : i1
}

// ====== Mapped form — single SSA operand with rhs_value attribute ======
// After --map-to-accelerator, compile-time constant operands are folded into
// rhs_value attributes (e.g. "neura.icmp"(%x) {cmpType = "sge", rhs_value = 0 : i32}).
// Before this fix, the interpreter rejected any op with numOperands < 2.
func.func @test_icmp_mapped_form_true() -> !neura.data<i1, i1> {
  %a = "neura.grant_once"() <{constant_value = 5 : i32}> : () -> !neura.data<i32, i1>
  %res = "neura.icmp"(%a) {cmpType = "sge", rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %res : !neura.data<i1, i1>
}

func.func @test_icmp_mapped_form_false() -> !neura.data<i1, i1> {
  %a = "neura.grant_once"() <{constant_value = -5 : i32}> : () -> !neura.data<i32, i1>
  %res = "neura.icmp"(%a) {cmpType = "sge", rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : !neura.data<i1, i1>
}

// ====== Unsigned greater than or equal (uge) ======
// Positive case: lhs >= rhs (unsigned)
func.func @test_icmp_uge_true() -> i1 {
  %a = arith.constant -1 : i32  // Unsigned: 4294967295
  %b = arith.constant 42 : i32
  %uge = "neura.icmp"(%a, %b) {cmpType = "uge"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %uge : i1
}

// Negative case: lhs < rhs (unsigned)
func.func @test_icmp_uge_false() -> i1 {
  %a = arith.constant 42 : i32
  %b = arith.constant -1 : i32  // Unsigned: 4294967295
  %uge = "neura.icmp"(%a, %b) {cmpType = "uge"} : (i32, i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %uge : i1
}