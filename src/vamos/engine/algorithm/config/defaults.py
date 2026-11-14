"""Default config builders for built-in algorithms."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any, cast

from .agemoea import AGEMOEAConfig
from .ccmo import CCMOConfig
from .ctaea import CTAEAConfig
from .ibea import IBEAConfig
from .moead import MOEADConfig
from .nsgaii import NSGAIIConfig
from .nsgaiii import NSGAIIIConfig
from .rvea import RVEAConfig
from .smpso import SMPSOConfig
from .smsemoa import SMSEMOAConfig
from .spea2 import SPEA2Config
from .types import AlgorithmConfigProtocol

DefaultConfigBuilder = Callable[[int | None, int | None, int | None, str | None], AlgorithmConfigProtocol]

_DEFAULT_FACTORIES: dict[str, Callable[..., AlgorithmConfigProtocol]] = {
    "nsgaii": NSGAIIConfig.default,
    "moead": MOEADConfig.default,
    "smsemoa": SMSEMOAConfig.default,
    "nsgaiii": NSGAIIIConfig.default,
    "spea2": SPEA2Config.default,
    "ibea": IBEAConfig.default,
    "smpso": SMPSOConfig.default,
    "agemoea": AGEMOEAConfig.default,
    "rvea": RVEAConfig.default,
    "ccmo": CCMOConfig.default,
    "ctaea": CTAEAConfig.default,
}

# The built-in algorithm each configuration class configures.
_CONFIG_ALGORITHMS: dict[type, str] = {
    NSGAIIConfig: "nsgaii",
    MOEADConfig: "moead",
    SMSEMOAConfig: "smsemoa",
    NSGAIIIConfig: "nsgaiii",
    SPEA2Config: "spea2",
    IBEAConfig: "ibea",
    SMPSOConfig: "smpso",
    AGEMOEAConfig: "agemoea",
    RVEAConfig: "rvea",
    CCMOConfig: "ccmo",
    CTAEAConfig: "ctaea",
}


def config_algorithm(config: object) -> str | None:
    """The built-in algorithm that ``config``'s class configures, or ``None`` for another class."""
    return _CONFIG_ALGORITHMS.get(type(config))


def build_default_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None = None,
) -> AlgorithmConfigProtocol | None:
    """The ``default()`` configuration of a built-in algorithm for a problem.

    ``pop_size=None`` leaves the size to the algorithm: 100, or the size of its
    reference lattice for MOEA/D, NSGA-III and RVEA. ``n_obj`` (3 when unknown)
    reaches the algorithms that use it, and ``encoding`` selects the operators of
    discrete and mixed problems. Returns ``None`` for an algorithm that is not
    built in.
    """
    factory = _DEFAULT_FACTORIES.get(algorithm.lower())
    if factory is None:
        return None
    arguments = {"pop_size": pop_size, "n_var": n_var, "n_obj": 3 if n_obj is None else n_obj, "encoding": encoding}
    accepted = inspect.signature(factory).parameters
    return factory(**{name: value for name, value in arguments.items() if name in accepted})


def _with_result_mode(config: AlgorithmConfigProtocol, result_mode: str) -> AlgorithmConfigProtocol:
    """``config`` with ``result_mode``, when it is a dataclass with that field."""
    try:
        names = {field.name for field in fields(cast(Any, config))}
    except TypeError:
        return config
    if "result_mode" not in names:
        return config
    return cast(AlgorithmConfigProtocol, replace(cast(Any, config), result_mode=result_mode))


def build_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None = None,
    result_mode: str,
) -> AlgorithmConfigProtocol:
    """The configuration of a run whose caller gives none: ``optimize``, studies and Studio.

    A built-in algorithm gets its ``default()`` configuration for the problem (see
    :func:`build_default_algorithm_config`) with ``result_mode``; another registered algorithm
    gets a ``GenericAlgorithmConfig`` with the sizes that are given. An unknown name raises
    ``InvalidAlgorithmError``.
    """
    from vamos.engine.algorithm.registry import get_algorithms_registry
    from vamos.foundation.exceptions import InvalidAlgorithmError

    from .generic import GenericAlgorithmConfig

    algorithm = algorithm.lower()
    default_config = build_default_algorithm_config(algorithm, pop_size=pop_size, n_var=n_var, n_obj=n_obj, encoding=encoding)
    if default_config is not None:
        return _with_result_mode(default_config, result_mode)

    registry = get_algorithms_registry()
    if algorithm in registry:
        sizes = {"pop_size": pop_size, "n_var": n_var, "n_obj": n_obj}
        return GenericAlgorithmConfig({name: value for name, value in sizes.items() if value is not None})
    raise InvalidAlgorithmError(algorithm, available=sorted(registry.keys()))


__all__ = ["config_algorithm", "build_algorithm_config", "build_default_algorithm_config", "DefaultConfigBuilder"]
