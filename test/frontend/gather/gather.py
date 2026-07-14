"""Compiles a minimal gather module to Torch Dialect MLIR.

Exports a single-op module that calls torch.ops.neura.gather through
torch-mlir's FX importer. The custom op neura::gather (registered in
python/neura_ops.py via torch.library.custom_op) stays opaque through
torch.export, so it appears in the output as
``torch.operator "torch.neura.gather"`` instead of being decomposed into
aten.index.

Before emitting MLIR it also verifies eagerly that the op is numerically
equal to fancy indexing (table[indices]), so a semantically broken custom op
fails the test even when the op still survives tracing.

This is a self-contained unit test for the frontend custom op and does not
depend on any application kernel.

Usage:
    python3 gather.py [output.mlir]
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
import neura_ops  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch_mlir.fx import export_and_import  # noqa: E402
from torch_mlir.compiler_utils import OutputType  # noqa: E402


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
        f"shape mismatch: {actual.shape} vs {expected.shape}")
    assert torch.equal(actual, expected), "value mismatch against table[indices]"


def compile_gather(output_file):
    """Compiles the gather module and writes Torch Dialect MLIR to disk.

    Args:
        output_file: Path to the output MLIR file.

    Returns:
        The MLIR module string.
    """
    model = GatherModule().eval()

    table = torch.randn(16, 4)
    indices = torch.randint(0, 16, (8,))

    module = export_and_import(
        model, table, indices,
        output_type=OutputType.RAW,
        func_name="forward",
    )
    mlir_str = str(module)

    with open(output_file, "w") as f:
        f.write(mlir_str)

    return mlir_str


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "gather_torch.mlir"
    # Verifies op semantics eagerly before emitting MLIR, so a broken custom op
    # fails the test even if the op still survives tracing as an opaque marker.
    verify_semantics()
    compile_gather(out)
