from .._mlir_libs._NeuraExtensionPybind11 import (
    neura as _neura_extension,
)
from ..ir import Context
from ._neura_ops_gen import *


def register_dialect(
    context: Context | None = None,
    load: bool = True,
) -> None:
    """Register and optionally load the Neura dialect."""

    if context is None:
        _neura_extension.register_dialect(load=load)
    else:
        _neura_extension.register_dialect(context, load)
