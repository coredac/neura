// RUN: python3 %S/gather.py %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s --implicit-check-not="aten.index"

// Unit test for the neura::gather PyTorch custom op (python/neura_ops.py).
// The driver script first verifies eagerly that the op equals fancy indexing
// (table[indices]); a mismatch aborts it and fails this RUN line. FileCheck
// then verifies that the gather call survives torch.export and appears in the
// Torch Dialect MLIR as a torch.operator, and that it is not decomposed back
// into aten.index. This test is independent of any application kernel.

// CHECK-LABEL: func.func @forward
// CHECK: torch.operator "torch.neura.gather"
