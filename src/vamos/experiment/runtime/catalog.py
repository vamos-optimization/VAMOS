from __future__ import annotations

import logging
from importlib import import_module

from vamos.engine.algorithm.registry import get_builtin_algorithm_names
from vamos.foundation.kernel.registry import resolve_kernel

TITLE = "VAMOS Experiment Runner"
DEFAULT_ENGINE = "numpy"
DEFAULT_PROBLEM = "zdt1"

EXTERNAL_ALGORITHM_NAMES = ("pymoo_nsga2", "jmetalpy_nsga2", "pygmo_nsga2")
EXPERIMENT_TYPES = ("backends",)
EXPERIMENT_BACKENDS = (
    "numpy",
    "numba",
    "moocore",
)


def _has_numba() -> bool:
    """Check loadability without compiling kernels or consuming random draws.

    Distribution discovery alone misses incompatible NumPy/Python/LLVM installs.
    Keep this check at selection time: importing VAMOS must not import Numba.
    Compilation and execution failures are deliberately not retried with another
    engine, since evaluation may already have consumed RNG or user-side effects.
    """
    try:
        numba = import_module("numba")
        if numba.config.DISABLE_JIT:
            return False
        resolve_kernel("numba")
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        logging.getLogger(__name__).info("engine='auto' selected numpy: Numba could not be loaded (%s).", type(exc).__name__)
        return False
    return True


def resolve_engine_details(engine: str | None, *, algorithm: str | None = None) -> tuple[str, str]:
    """
    Resolve the effective engine and explain where the decision came from.

    ``engine=None`` is deterministic and always uses the documented default.
    Explicit ``engine="auto"`` prefers loadable Numba for the built-in algorithms
    and falls back to NumPy. Unknown/plugin algorithms keep the reference engine.
    """
    if engine is None:
        return DEFAULT_ENGINE, "default"
    engine_name = str(engine).lower()
    if engine_name == "auto":
        if algorithm and algorithm.lower() in get_builtin_algorithm_names() and _has_numba():
            return "numba", "auto"
        return DEFAULT_ENGINE, "auto"
    return engine_name, "explicit"


def resolve_engine(engine: str | None, *, algorithm: str | None = None) -> str:
    """
    Resolve the effective engine for a run.

    ``engine=None`` is deterministic and uses the default engine.
    ``engine="auto"`` prefers Numba for built-ins, with a NumPy fallback.
    """
    resolved_engine, _source = resolve_engine_details(engine, algorithm=algorithm)
    return resolved_engine


__all__ = [
    "TITLE",
    "DEFAULT_ENGINE",
    "DEFAULT_PROBLEM",
    "EXTERNAL_ALGORITHM_NAMES",
    "EXPERIMENT_TYPES",
    "EXPERIMENT_BACKENDS",
    "resolve_engine_details",
    "resolve_engine",
]
