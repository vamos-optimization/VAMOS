"""
Foundation layer: backend kernels for vectorized compute-heavy operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import KernelBackend


def default_kernel() -> KernelBackend:
    """Return the kernel an algorithm uses when it is built without one: NumPy.

    NumPy is the stable 1.x default when no engine is requested, as in
    ``vamos.optimize``; pass ``NumbaKernel()`` or ``MooCoreKernel()`` explicitly, or
    use ``engine="auto"`` with ``optimize``, to opt in to another backend.
    """
    from .numpy_backend import NumPyKernel

    return NumPyKernel()
