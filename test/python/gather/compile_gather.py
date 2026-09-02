"""Compiles the neura::gather custom op through the frontend and bridge.

This is a self-contained unit test driver for the neura::gather custom op
(registered in python/neura_ops.py). It supports two output modes:

Default mode:
    Exports a single-op module via torch-mlir's FX importer. The custom op
    stays opaque through torch.export, appearing as
    ``torch.operator "torch.neura.gather"`` instead of being decomposed into
    aten.index. Before emitting it also verifies eagerly that the op equals
    fancy indexing (table[indices]).

Neutral mode (--neutral):
    Runs the frontend through torch-mlir's own partial conversion pipeline so
    the surrounding ops lower to linalg while the gather stays opaque, then
    neutralizes the gather island (folds the torch materialization casts and
    re-emits the gather as a generic op on builtin tensors). The result
    contains no torch dialect dependency, so mlir-neura-opt can parse it (with
    -allow-unregistered-dialect) and lower it to neura.gather.

This is the same lowering route used for real application kernels, so the op
rides through the standard pipeline instead of a bespoke bridge.

Usage:
    python3 compile_gather.py [--neutral] [output.mlir]
"""

"""
Unit test for the neura::gather custom op (python/neura_ops.py), covering
every layer of its compilation path. It is independent of any application
kernel.

Layer 1 (frontend contract): the driver verifies eagerly that the op equals
fancy indexing (table[indices]); a mismatch aborts it and fails the RUN
line. FileCheck then confirms the op survives torch.export as an opaque
torch.operator and is not decomposed back into aten.index.

Layer 2 (lowering): the Python bridge emits a torch-free neutral module,
which mlir-neura-opt lowers with -lower-torch-to-neura. FileCheck confirms
the marker becomes a native neura.gather op and no torch marker remains.
"""

# RUN: python3 %S/compile_gather.py %t.torch.mlir
# RUN: FileCheck --input-file=%t.torch.mlir %s --check-prefix=L1 --implicit-check-not="aten.index"

# L1-LABEL: func.func @forward
# L1: torch.operator "torch.neura.gather"

# RUN: python3 %S/compile_gather.py --neutral %t.neutral.mlir
# RUN: mlir-neura-opt -allow-unregistered-dialect -lower-torch-to-neura %t.neutral.mlir | FileCheck %s --check-prefix=L2 --implicit-check-not="torch.neura.gather"

# L2-LABEL: func.func @forward
# L2: neura.gather

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
import neura_ops  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch_mlir import ir  # noqa: E402
from torch_mlir.compiler_utils import (  # noqa: E402
    OutputType,
    run_pipeline_with_repro_report,
)
from torch_mlir.fx import export_and_import  # noqa: E402

# Names the opaque frontend marker and the torch materialization cast ops.
GATHER_OP_NAME = "torch.neura.gather"
FROM_CAST = "torch_c.from_builtin_tensor"
TO_CAST = "torch_c.to_builtin_tensor"

# Lowers the surrounding torch ops to linalg while leaving unmatched ops (the
# gather marker) untouched. The final backend-contract verification is skipped
# on purpose so the opaque gather can survive the pipeline.
PARTIAL_PIPELINE = (
    "builtin.module("
    "func.func(torch-decompose-complex-ops),"
    "torch-func-backend-type-conversion,"
    "func.func(convert-torch-to-linalg,convert-torch-to-arith,"
    "convert-torch-to-scf,convert-torch-to-tensor))"
)


class GatherModule(nn.Module):
    """Wraps a single neura::gather call for the frontend unit test."""

    def forward(self, table, indices):
        """Performs one batched indirect-address read.

        Args:
            table: Embedding table of shape [T, C], float32.
            indices: Index vector of shape [K], int64, with values in [0, T).

        Returns:
            Rows selected by ``indices``, shape [K, C], float32.
        """
        return torch.ops.neura.gather(table, indices)


def verify_semantics():
    """Checks that neura::gather is numerically equal to fancy indexing.

    Runs the custom op eagerly and compares it against the reference
    ``table[indices]``. The index vector deliberately contains repeated and
    out-of-order entries so the check exercises true random-address gather
    semantics rather than a contiguous slice. Raises AssertionError on
    mismatch.
    """
    table = torch.randn(16, 4)
    indices = torch.tensor([3, 0, 3, 7, 15, 1, 7, 0])

    actual = torch.ops.neura.gather(table, indices)
    expected = table[indices]

    assert actual.shape == expected.shape, (
        f"shape mismatch: {actual.shape} vs {expected.shape}"
    )
    assert torch.equal(actual, expected), "value mismatch against table[indices]"


def export_module():
    """Exports the gather module to a torch-typed MLIR module.

    Returns:
        The torch-mlir Module produced by the FX importer (RAW output).
    """
    model = GatherModule().eval()
    table = torch.randn(16, 4)
    indices = torch.randint(0, 16, (8,))
    return export_and_import(
        model,
        table,
        indices,
        output_type=OutputType.RAW,
        func_name="forward",
    )


def _collect_gathers(module):
    """Returns every torch.operator that carries the gather marker.

    Args:
        module: The MLIR module to scan.

    Returns:
        A list of matching operations.
    """
    found = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    operation = inner.operation
                    if operation.name == "torch.operator" and GATHER_OP_NAME in str(
                        operation
                    ):
                        found.append(operation)
                    walk(operation)

    walk(module.operation)
    return found


def _neutralize_one(gather_op):
    """Rewrites a single gather island into a builtin-typed generic op.

    Traces each operand back through its from_builtin_tensor cast to the
    builtin value, reads the builtin result type from the to_builtin_tensor
    consumer, emits a generic gather on builtin tensors, and erases the now
    dead cast ops.

    Args:
        gather_op: The torch.operator gather marker to rewrite.
    """
    new_operands = []
    dead_casts = []
    for operand in gather_op.operands:
        owner = operand.owner
        if isinstance(owner, ir.Operation) and owner.name == FROM_CAST:
            new_operands.append(owner.operands[0])
            dead_casts.append(owner)
        else:
            new_operands.append(operand)

    consumer = next(
        u.owner for u in gather_op.results[0].uses if u.owner.name == TO_CAST
    )
    result_type = consumer.results[0].type

    with ir.InsertionPoint(gather_op), gather_op.location:
        rewritten = ir.Operation.create(
            GATHER_OP_NAME, results=[result_type], operands=new_operands
        )

    consumer.results[0].replace_all_uses_with(rewritten.results[0])
    consumer.erase()
    gather_op.erase()
    for cast in dead_casts:
        if len([u for u in cast.results[0].uses]) == 0:
            cast.erase()


def _all_operations(module):
    """Returns a flat snapshot list of every operation in the module.

    Args:
        module: The MLIR module to walk.

    Returns:
        A list of operations captured before any mutation.
    """
    collected = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    collected.append(inner.operation)
                    walk(inner.operation)

    walk(module.operation)
    return collected


def _fold_builtin_casts(module):
    """Folds the torch materialization cast round-trips left by conversion.

    Repeatedly rewrites ``to_builtin_tensor(from_builtin_tensor(x))`` back to
    ``x`` and erases the resulting dead casts. This clears the boundary casts
    that a partial conversion leaves behind, without invoking a torch dialect
    legality pass that would reject the opaque gather marker.

    Args:
        module: The MLIR module to clean up.
    """
    changed = True
    while changed:
        changed = False
        for op in _all_operations(module):
            if op.name != TO_CAST:
                continue
            source = op.operands[0].owner
            if isinstance(source, ir.Operation) and source.name == FROM_CAST:
                op.results[0].replace_all_uses_with(source.operands[0])
                op.erase()
                changed = True
        for op in _all_operations(module):
            if op.name != FROM_CAST:
                continue
            if len([u for u in op.results[0].uses]) == 0:
                op.erase()
                changed = True


def build_neutral_module(module):
    """Lowers a torch export into a neutral, torch-free module.

    Runs the partial conversion pipeline so the neighbours become linalg while
    the gather stays opaque, neutralizes every gather island, and folds the
    leftover materialization casts. The result is free of torch and torch_c
    constructs.

    Args:
        module: The torch-mlir Module produced by ``export_module``.

    Returns:
        The neutral MLIR module string.
    """
    run_pipeline_with_repro_report(
        module, PARTIAL_PIPELINE, "Partial lowering to linalg"
    )
    for gather_op in _collect_gathers(module):
        _neutralize_one(gather_op)
    _fold_builtin_casts(module)
    return str(module)


def compile_gather(output_file, neutral=False):
    """Compiles the gather module and writes MLIR to disk.

    Args:
        output_file: Path to the output MLIR file.
        neutral: When True, emits the torch-free neutral module; otherwise
            emits the raw torch-typed export.

    Returns:
        The MLIR module string.
    """
    module = export_module()
    mlir_str = build_neutral_module(module) if neutral else str(module)

    with open(output_file, "w") as f:
        f.write(mlir_str)

    return mlir_str


if __name__ == "__main__":
    argv = sys.argv[1:]
    emit_neutral = "--neutral" in argv
    positional = [a for a in argv if not a.startswith("--")]
    out = positional[0] if positional else "gather_torch.mlir"

    # Verifies op semantics eagerly before emitting MLIR, so a broken custom op
    # fails the test even if the op still survives tracing as an opaque marker.
    verify_semantics()
    compile_gather(out, neutral=emit_neutral)
