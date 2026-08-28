// Unit test for the neura::gather custom op (python/neura_ops.py), covering
// every layer of its compilation path. It is independent of any application
// kernel.
//
// Layer 1 (frontend contract): the driver verifies eagerly that the op equals
// fancy indexing (table[indices]); a mismatch aborts it and fails the RUN
// line. FileCheck then confirms the op survives torch.export as an opaque
// torch.operator and is not decomposed back into aten.index.
//
// Layer 2 (lowering): the Python bridge emits a torch-free neutral module,
// which mlir-neura-opt lowers with -lower-torch-to-neura. FileCheck confirms
// the marker becomes a native neura.gather op and no torch marker remains.

// REQUIRES: neura-torch-mlir
// RUN: %neura_python %S/compile_gather.py %t.torch.mlir
// RUN: FileCheck --input-file=%t.torch.mlir %s --check-prefix=L1 --implicit-check-not="aten.index"

// L1-LABEL: func.func @forward
// L1: torch.operator "torch.neura.gather"

// RUN: %neura_python %S/compile_gather.py --neutral %t.neutral.mlir
// RUN: mlir-neura-opt -allow-unregistered-dialect -lower-torch-to-neura %t.neutral.mlir | FileCheck %s --check-prefix=L2 --implicit-check-not="torch.neura.gather"

// L2-LABEL: func.func @forward
// L2: neura.gather
