"""
Algorithm facade: configuration builders and registry helpers.

Use this module for algorithm config objects and discovery helpers.
For running optimizations, use `vamos.optimize`.

The nine configuration classes of VAMOS 1.1.0, the probability types and the
``available_*`` functions are stable. ``CCMOConfig``, ``CTAEAConfig``, ``ToPConfig`` and
``CMOEAMSConfig`` are new in VAMOS 1.2.0 and not yet covered by the 1.0 compatibility baseline. ``GenericAlgorithmConfig``,
``discover_algorithm_plugins`` and ``resolve_algorithm`` serve custom and
plugin algorithms and are experimental. ``moead_populations`` is new after VAMOS
1.4.0 (development) and not yet covered by the 1.0 compatibility baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    CCMOConfig,
    CMOEAMSConfig,
    CTAEAConfig,
    GenericAlgorithmConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    ProbabilityExpression,
    ProbabilityValue,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
    ToPConfig,
)
from vamos.engine.variation.helpers import operator_names

if TYPE_CHECKING:
    from vamos.engine.algorithm.registry import AlgorithmBuilder


def available_algorithms() -> tuple[str, ...]:
    """Return the canonical algorithm identifiers supported by the engine."""
    from vamos.engine.algorithm.registry import get_algorithms_registry

    return tuple(sorted(get_algorithms_registry().keys()))


def discover_algorithm_plugins(*, force: bool = False) -> tuple[str, ...]:
    """Load algorithm builders exposed via the ``vamos.algorithms`` entry point group.

    Experimental: plugin discovery is not covered by the 1.x stability policy.
    """
    from vamos.engine.algorithm.registry import discover_algorithm_plugins as _discover_algorithm_plugins

    return _discover_algorithm_plugins(force=force)


def available_crossover_methods(encoding: str = "real") -> tuple[str, ...]:
    """
    Return the available crossover method identifiers for a given encoding.

    Parameters
    ----------
    encoding : str, default ``"real"``
        Variable encoding type.

    Returns
    -------
    tuple[str, ...]
        Supported crossover method identifiers for the encoding.
    """
    return operator_names("crossover", encoding)


def available_mutation_methods(encoding: str = "real") -> tuple[str, ...]:
    """
    Return the available mutation method identifiers for a given encoding.

    Parameters
    ----------
    encoding : str, default ``"real"``
        Variable encoding type.

    Returns
    -------
    tuple[str, ...]
        Supported mutation method identifiers for the encoding.
    """
    return operator_names("mutation", encoding)


def moead_populations(n_obj: int, *, max_pop_size: int = 1000) -> tuple[int, ...]:
    """Return the populations, up to ``max_pop_size``, for which MOEA/D has weight vectors by default.

    MOEA/D gives each individual one weight vector. With one or two objectives every population
    from 2 has them (uniform weights). With three or more, a population has them when a complete
    Das-Dennis lattice has exactly that many vectors (``comb(H + n_obj - 1, n_obj - 1)`` for ``H``
    divisions) or when VAMOS ships jMetal's file ``W{n_obj}D_{size}.dat`` (100 with three or four
    objectives, 156 with eight). ``MOEADConfig.default(pop_size=...)`` and ``optimize`` accept exactly
    these sizes, and ``vamos tune`` runs the largest one up to the sampled population.

    Other populations need explicit weights: ``weight_vectors(path=...)`` with a file, or
    ``weight_vectors(divisions=H1, inner_divisions=H2)`` with Deb and Jain's two-layer lattice of
    ``comb(H1 + n_obj - 1, n_obj - 1) + comb(H2 + n_obj - 1, n_obj - 1)`` vectors.

    **Development after 1.4.0:** new.

    Examples
    --------
    >>> moead_populations(4, max_pop_size=200)
    (4, 10, 20, 35, 56, 84, 100, 120, 165)
    """
    from vamos.engine.algorithm.components.weight_vectors import weighted_populations

    return weighted_populations(n_obj, max_pop_size)


def resolve_algorithm(name: str) -> AlgorithmBuilder:
    """Return the registered builder for a named algorithm.

    Experimental: the builder is an internal type, and this function is not
    covered by the 1.x stability policy. Run algorithms through ``vamos.optimize``.
    """
    from vamos.engine.algorithm.registry import resolve_algorithm as _resolve_algorithm

    return _resolve_algorithm(name)


__all__ = [
    "NSGAIIConfig",
    "NSGAIIIConfig",
    "MOEADConfig",
    "SMSEMOAConfig",
    "SMPSOConfig",
    "SPEA2Config",
    "IBEAConfig",
    "AGEMOEAConfig",
    "RVEAConfig",
    "CCMOConfig",
    "CTAEAConfig",
    "ToPConfig",
    "CMOEAMSConfig",
    "GenericAlgorithmConfig",
    "ProbabilityExpression",
    "ProbabilityValue",
    "available_algorithms",
    "discover_algorithm_plugins",
    "available_crossover_methods",
    "available_mutation_methods",
    "moead_populations",
    "resolve_algorithm",
]
