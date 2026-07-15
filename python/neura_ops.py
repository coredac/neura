"""Neura custom ops -- PyTorch frontend interface for hardware primitives.

Defines PyTorch custom ops that map to Neura dialect hardware primitives.
These ops are semantically equivalent to standard operations on the PyTorch
side (training and verification work as usual), but torch_mlir preserves
their op identity after tracing so that downstream compiler passes can
recognize and lower them to the corresponding Neura IR operations.

Supported custom ops:
  - neura::gather  ->  neura.gather (batched random-address read).

Usage:
  import neura_ops
  features = torch.ops.neura.gather(table, indices)
"""

import torch


# ============================================================================
#  neura::gather -- Batched indirect-address read.
#
#  Semantics:  table[indices]  (fancy indexing).
#  Hardware:   neura.gather -- Issues multiple random-address read requests
#              in a single cycle, exploiting memory-level parallelism for
#              hash-table lookups.
#
#  Registered via torch.library.custom_op so that torch.export keeps the op
#  opaque (it is not decomposed back into aten.index). After torch_mlir
#  tracing the op appears as torch.operator "neura.gather".
# ============================================================================


@torch.library.custom_op("neura::gather", mutates_args=())
def gather(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Performs batched indirect-address read (hardware gather primitive).

    Args:
        table: Embedding table of shape [T, C].
        indices: Index vector of shape [K] with values in [0, T).

    Returns:
        Rows selected by ``indices``, shape [K, C].
    """
    return table[indices.long()]


@gather.register_fake
def _gather_fake(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Infers the output shape during tracing without executing the op."""
    return torch.empty(
        (*indices.shape, *table.shape[1:]),
        dtype=table.dtype,
        device=table.device,
    )
