"""
Algorithm registry.

Maps algorithm names to builder callables so orchestration code avoids
hard-coded conditionals. Builders are expected to accept (config_dict, kernel)
and return an initialized algorithm instance.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping
from importlib import metadata
from typing import Any, Protocol, cast

from vamos.engine.algorithm.config.types import AlgorithmConfigMapping
from vamos.foundation.exceptions import InvalidAlgorithmError, _suggest_names
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.registry import Registry

from .ibea import IBEA
from .moead import MOEAD
from .nsgaii import NSGAII
from .nsgaiii import NSGAIII
from .smpso import SMPSO
from .smsemoa import SMSEMOA
from .spea2 import SPEA2


class AlgorithmLike(Protocol):
    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: Any | None = None,
        live_viz: Any | None = None,
    ) -> Mapping[str, Any]: ...


AlgorithmBuilder = Callable[[AlgorithmConfigMapping, KernelBackend | None], AlgorithmLike]

_ALGORITHMS: Registry[AlgorithmBuilder] | None = None
_ALGORITHM_PLUGINS_LOADED = False
_ALGO_DOCS = "docs/reference/algorithms.md"
_TROUBLESHOOTING_DOCS = "docs/guide/troubleshooting.md"
_ENTRY_POINT_GROUP = "vamos.algorithms"


def _format_unknown_algorithm(name: str, options: list[str]) -> str:
    parts = [f"Unsupported algorithm '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_ALGO_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


def _build_nsgaii(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return NSGAII(dict(cfg), kernel=kernel)


def _build_nsgaiii(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return NSGAIII(dict(cfg), kernel=kernel)


def _build_moead(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return MOEAD(dict(cfg), kernel=kernel)


def _build_smsemoa(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return SMSEMOA(dict(cfg), kernel=kernel)


def _build_spea2(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return SPEA2(dict(cfg), kernel=kernel)


def _build_ibea(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return IBEA(dict(cfg), kernel=kernel)


def _build_smpso(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    """
    Note: SMPSO typically requires a different config structure, but the builder
    signature remains consistent.
    """
    return SMPSO(dict(cfg), kernel=kernel)


def _build_agemoea(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    from .agemoea import AGEMOEA

    return AGEMOEA(dict(cfg), kernel=kernel)


def _build_rvea(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    from .rvea import RVEA

    return RVEA(dict(cfg), kernel=kernel)


def _build_ccmo(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    from .ccmo import CCMO

    return CCMO(dict(cfg), kernel=kernel)


def _build_ctaea(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    from .ctaea import CTAEA

    return CTAEA(dict(cfg), kernel=kernel)


def _build_top(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    from .top import ToP

    return ToP(dict(cfg), kernel=kernel)


def _build_cmoeams(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    from .cmoeams import CMOEAMS

    return CMOEAMS(dict(cfg), kernel=kernel)


def _register_algorithms(registry: Registry[AlgorithmBuilder]) -> None:
    registry.register("nsgaii", _build_nsgaii)
    registry.register("nsgaiii", _build_nsgaiii)
    registry.register("moead", _build_moead)
    registry.register("smsemoa", _build_smsemoa)
    registry.register("spea2", _build_spea2)
    registry.register("ibea", _build_ibea)
    registry.register("smpso", _build_smpso)
    registry.register("agemoea", _build_agemoea)
    registry.register("rvea", _build_rvea)
    registry.register("ccmo", _build_ccmo)
    registry.register("ctaea", _build_ctaea)
    registry.register("top", _build_top)
    registry.register("cmoeams", _build_cmoeams)


def get_builtin_algorithm_names() -> tuple[str, ...]:
    """Return stable built-in IDs without discovering entry points."""
    return ("agemoea", "ccmo", "cmoeams", "ctaea", "ibea", "moead", "nsgaii", "nsgaiii", "rvea", "smpso", "smsemoa", "spea2", "top")


def resolve_builtin_algorithm(name: str) -> AlgorithmBuilder:
    """Resolve only a statically registered built-in algorithm."""
    registry: Registry[AlgorithmBuilder] = Registry("Built-in algorithms")
    _register_algorithms(registry)
    key = name.lower()
    try:
        return registry[key]
    except KeyError as exc:
        raise InvalidAlgorithmError(name, available=list(get_builtin_algorithm_names())) from exc


def _entry_points_for_group(group: str) -> list[metadata.EntryPoint]:
    return list(metadata.entry_points(group=group))


def _load_algorithm_plugins(registry: Registry[AlgorithmBuilder], *, force: bool = False) -> tuple[str, ...]:
    global _ALGORITHM_PLUGINS_LOADED
    if _ALGORITHM_PLUGINS_LOADED and not force:
        return ()

    loaded: list[str] = []
    for entry_point in _entry_points_for_group(_ENTRY_POINT_GROUP):
        name = entry_point.name.strip().lower()
        if not name:
            warnings.warn("Ignoring VAMOS algorithm entry point with an empty name.", RuntimeWarning, stacklevel=2)
            continue
        if name in get_builtin_algorithm_names():
            warnings.warn(
                f"Ignoring VAMOS algorithm plugin '{entry_point.name}': '{name}' is the name of a built-in algorithm.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if name in registry:
            continue
        try:
            builder = entry_point.load()
        except Exception as exc:
            warnings.warn(f"Failed to load VAMOS algorithm plugin '{entry_point.name}': {exc}", RuntimeWarning, stacklevel=2)
            continue
        if not callable(builder):
            warnings.warn(f"Ignoring VAMOS algorithm plugin '{entry_point.name}' because it is not callable.", RuntimeWarning, stacklevel=2)
            continue
        registry.register(name, cast(AlgorithmBuilder, builder))
        loaded.append(name)

    _ALGORITHM_PLUGINS_LOADED = True
    return tuple(loaded)


def _plugins_disabled() -> bool:
    """Whether ``VAMOS_DISABLE_PLUGINS`` turns off the implicit loading of algorithm plugins."""
    return os.environ.get("VAMOS_DISABLE_PLUGINS", "").strip().lower() in ("1", "true", "yes", "on")


def get_algorithms_registry() -> Registry[AlgorithmBuilder]:
    """The algorithm registry; its first use also loads the installed plugins unless ``VAMOS_DISABLE_PLUGINS`` is set."""
    global _ALGORITHMS
    if _ALGORITHMS is None:
        registry: Registry[AlgorithmBuilder] = Registry("Algorithms")
        _register_algorithms(registry)
        _ALGORITHMS = registry
    if not _plugins_disabled():
        _load_algorithm_plugins(_ALGORITHMS)
    return _ALGORITHMS


def discover_algorithm_plugins(*, force: bool = False) -> tuple[str, ...]:
    """Load algorithm builders exposed through the ``vamos.algorithms`` entry point group.

    This explicit call loads them even when ``VAMOS_DISABLE_PLUGINS`` is set.
    """
    global _ALGORITHMS
    if _ALGORITHMS is None:
        registry: Registry[AlgorithmBuilder] = Registry("Algorithms")
        _register_algorithms(registry)
        _ALGORITHMS = registry
    registry = _ALGORITHMS
    return _load_algorithm_plugins(registry, force=force)


def resolve_algorithm(name: str) -> AlgorithmBuilder:
    key = name.lower()
    registry = get_algorithms_registry()
    try:
        return registry[key]
    except KeyError as exc:
        available = sorted(registry.keys())
        raise InvalidAlgorithmError(name, available=available) from exc


def __getattr__(name: str) -> Any:
    if name == "ALGORITHMS":
        return get_algorithms_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AlgorithmBuilder",
    "AlgorithmLike",
    "discover_algorithm_plugins",
    "get_algorithms_registry",
    "get_builtin_algorithm_names",
    "resolve_algorithm",
    "resolve_builtin_algorithm",
]
