"""
Kernel backend registry.

This lightweight registry maps backend names to their kernel classes so that
orchestration layers can resolve kernels without hard-coding if/elif chains.
Backends that rely on optional dependencies (numba, moocore) are lazy-loaded so
`import vamos` remains safe on a minimal install.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast

from vamos.foundation.exceptions import _suggest_names

from .backend import KernelBackend
from .numpy_backend import NumPyKernel


def _load_numba() -> KernelBackend:
    try:
        module = import_module("vamos.foundation.kernel.numba_backend")
        return cast(KernelBackend, module.NumbaKernel())
    except ImportError as exc:
        raise ImportError(
            "Kernel 'numba' requires the [compute] extra (numba>=0.57). Install with `pip install \"vamos-optimization[compute]\"`."
        ) from exc


def _load_moocore() -> KernelBackend:
    try:
        module = import_module("vamos.foundation.kernel.moocore_backend")
        return cast(KernelBackend, module.MooCoreKernel())
    except ImportError as exc:
        raise ImportError(
            "Kernel 'moocore' requires the [compute] extra (moocore>=0.2.0). Install with `pip install \"vamos-optimization[compute]\"`."
        ) from exc


KERNELS: dict[str, Callable[[], KernelBackend]] = {
    "numpy": NumPyKernel,
    "numba": _load_numba,
    "moocore": _load_moocore,
}

_ENGINE_DOCS = "docs/reference/algorithms.md"
_TROUBLESHOOTING_DOCS = "docs/guide/troubleshooting.md"


def _format_unknown_engine(name: str, options: list[str]) -> str:
    parts = [f"Unknown engine '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_ENGINE_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


def resolve_kernel(name: str) -> KernelBackend:
    key = name.lower()
    try:
        factory = KERNELS[key]
    except KeyError as exc:
        available = sorted(KERNELS)
        raise ValueError(_format_unknown_engine(name, available)) from exc
    return factory()


__all__ = ["KERNELS", "resolve_kernel"]
