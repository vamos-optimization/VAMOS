"""
Engine layer: multi-objective algorithm implementations.

This package contains concrete algorithms (e.g. NSGA-II/III, MOEA/D, SMPSO, etc.)
and shared building blocks under `vamos.engine.algorithm.components`.

Algorithms
----------
- NSGAII: Non-dominated Sorting Genetic Algorithm II
- NSGAIII: Reference-point based NSGA-III
- MOEAD: Multi-Objective Evolutionary Algorithm based on Decomposition
- SPEA2: Strength Pareto Evolutionary Algorithm 2
- SMSEMOA: S-Metric Selection EMOA (hypervolume-based)
- IBEA: Indicator-Based Evolutionary Algorithm
- SMPSO: Speed-constrained Multi-objective PSO
- AGEMOEA: Adaptive Geometry Estimation MOEA
- RVEA: Reference Vector Guided Evolutionary Algorithm
- CCMO: Coevolutionary framework for constrained multi-objective optimization
- CTAEA: Two-archive evolutionary algorithm for constrained multi-objective optimization
- ToP: Two-phase framework for constrained multi-objective optimization

Usage
-----
Algorithms are typically accessed through the registry or factory:

    from vamos.engine.algorithm import NSGAII, MOEAD, SPEA2, AGEMOEA, RVEA
    from vamos.engine.algorithm.registry import get_algorithm
    from vamos.engine.algorithm.factory import create_algorithm

For configuration builders:

    from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vamos.engine.algorithm.components.protocol import (
    AlgorithmProtocol,
    ConstraintMode,
    InteractiveAlgorithmProtocol,
    SelectionMethod,
    SurvivalMethod,
)

if TYPE_CHECKING:
    from vamos.engine.algorithm.agemoea import AGEMOEA
    from vamos.engine.algorithm.ccmo import CCMO
    from vamos.engine.algorithm.ctaea import CTAEA
    from vamos.engine.algorithm.ibea import IBEA
    from vamos.engine.algorithm.moead import MOEAD
    from vamos.engine.algorithm.nsgaii import NSGAII
    from vamos.engine.algorithm.nsgaiii import NSGAIII
    from vamos.engine.algorithm.rvea import RVEA
    from vamos.engine.algorithm.smpso import SMPSO
    from vamos.engine.algorithm.smsemoa import SMSEMOA
    from vamos.engine.algorithm.spea2 import SPEA2
    from vamos.engine.algorithm.top import ToP

_ALGORITHM_EXPORTS = {
    "NSGAII": ("vamos.engine.algorithm.nsgaii", "NSGAII"),
    "NSGAIII": ("vamos.engine.algorithm.nsgaiii", "NSGAIII"),
    "MOEAD": ("vamos.engine.algorithm.moead", "MOEAD"),
    "SPEA2": ("vamos.engine.algorithm.spea2", "SPEA2"),
    "SMSEMOA": ("vamos.engine.algorithm.smsemoa", "SMSEMOA"),
    "IBEA": ("vamos.engine.algorithm.ibea", "IBEA"),
    "SMPSO": ("vamos.engine.algorithm.smpso", "SMPSO"),
    "AGEMOEA": ("vamos.engine.algorithm.agemoea", "AGEMOEA"),
    "RVEA": ("vamos.engine.algorithm.rvea", "RVEA"),
    "CCMO": ("vamos.engine.algorithm.ccmo", "CCMO"),
    "CTAEA": ("vamos.engine.algorithm.ctaea", "CTAEA"),
    "ToP": ("vamos.engine.algorithm.top", "ToP"),
}

__all__ = [
    # Algorithms
    "NSGAII",
    "NSGAIII",
    "MOEAD",
    "SPEA2",
    "SMSEMOA",
    "IBEA",
    "SMPSO",
    "AGEMOEA",
    "RVEA",
    "CCMO",
    "CTAEA",
    "ToP",
    # Protocols and enums
    "AlgorithmProtocol",
    "InteractiveAlgorithmProtocol",
    "SelectionMethod",
    "SurvivalMethod",
    "ConstraintMode",
]


def __getattr__(name: str) -> Any:
    target = _ALGORITHM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
