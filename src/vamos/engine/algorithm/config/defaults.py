"""Default config builders for built-in algorithms."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from .agemoea import AGEMOEAConfig
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
}


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


__all__ = ["build_default_algorithm_config", "DefaultConfigBuilder"]
