"""
Algorithm facade: configuration builders and registry helpers.

Use this module for algorithm config objects and discovery helpers.
For running optimizations, use `vamos.optimize`.

The nine configuration classes of VAMOS 1.1.0, the probability types and the
``available_*`` functions are stable. ``CCMOConfig`` and ``CTAEAConfig`` are new
after 1.1.0 and not yet covered by the 1.0 compatibility baseline. ``GenericAlgorithmConfig``,
``discover_algorithm_plugins`` and ``resolve_algorithm`` serve custom and
plugin algorithms and are experimental.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    CCMOConfig,
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
)
from vamos.engine.variation.helpers import (
    BINARY_CROSSOVER,
    BINARY_MUTATION,
    INT_CROSSOVER,
    INT_MUTATION,
    MIXED_CROSSOVER,
    MIXED_MUTATION,
    PERM_CROSSOVER,
    PERM_MUTATION,
    REAL_CROSSOVER,
    REAL_MUTATION,
)
from vamos.foundation.encoding import normalize_encoding

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
    try:
        normalized = normalize_encoding(encoding)
    except ValueError:
        return ()
    if normalized == "real":
        return tuple(sorted(REAL_CROSSOVER.keys()))
    if normalized == "binary":
        return tuple(sorted(BINARY_CROSSOVER.keys()))
    if normalized == "permutation":
        return tuple(sorted(PERM_CROSSOVER.keys()))
    if normalized == "integer":
        return tuple(sorted(INT_CROSSOVER.keys()))
    if normalized == "mixed":
        return tuple(sorted(MIXED_CROSSOVER.keys()))
    return ()


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
    try:
        normalized = normalize_encoding(encoding)
    except ValueError:
        return ()
    if normalized == "real":
        return tuple(sorted(REAL_MUTATION.keys()))
    if normalized == "binary":
        return tuple(sorted(BINARY_MUTATION.keys()))
    if normalized == "permutation":
        return tuple(sorted(PERM_MUTATION.keys()))
    if normalized == "integer":
        return tuple(sorted(INT_MUTATION.keys()))
    if normalized == "mixed":
        return tuple(sorted(MIXED_MUTATION.keys()))
    return ()


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
    "GenericAlgorithmConfig",
    "ProbabilityExpression",
    "ProbabilityValue",
    "available_algorithms",
    "discover_algorithm_plugins",
    "available_crossover_methods",
    "available_mutation_methods",
    "resolve_algorithm",
]
