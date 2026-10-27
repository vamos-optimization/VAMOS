"""IBEA-specific state container.

This module provides the state dataclass for IBEA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.results import feasible_result_indices, get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState
from vamos.engine.variation import VariationPipeline


@dataclass
class IBEAState(AlgorithmState):
    """State container for IBEA with ask/tell interface.

    Extends the base AlgorithmState with IBEA-specific fields for
    indicator and fitness tracking.

    Additional Attributes
    ---------------------
    indicator : str
        Quality indicator type ("epsilon" or "hypervolume").
    kappa : float
        Scaling factor for fitness computation.
    fitness : np.ndarray
        Current fitness values.
    pressure : int
        Tournament selection pressure.
    variation : Any
        Variation pipeline for offspring generation.
    """

    indicator: str = "epsilon"
    kappa: float = 0.05
    fitness: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    pressure: int = 2
    variation: VariationPipeline | None = None


def build_ibea_result(
    state: IBEAState,
    hv_reached: bool = False,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build IBEA result dictionary from state.

    Parameters
    ----------
    state : IBEAState
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict
        Result dictionary with X, F, G, population, archive data, and metadata.
        X and F contain only non-dominated solutions when kernel is provided.
    """
    archive_contents = get_external_archive_contents(state)
    if archive_contents is not None and not wants_population_result(state):
        archive_X, archive_F = archive_contents
        result_X = archive_X.copy()
        result_F = archive_F.copy()
        result_G = None
    else:
        should_filter = kernel is not None and not wants_population_result(state)
        if should_filter:
            try:
                selected = feasible_result_indices(state.F, state.G, state.constraint_mode, kernel)
                result_X = state.X[selected].copy()
                result_F = state.F[selected].copy()
                result_G = state.G[selected].copy() if state.G is not None else None
            except (ValueError, IndexError):
                result_X = state.X.copy()
                result_F = state.F.copy()
                result_G = state.G.copy() if state.G is not None else None
        else:
            result_X = state.X.copy()
            result_F = state.F.copy()
            result_G = state.G.copy() if state.G is not None else None

    result = {
        "X": result_X,
        "F": result_F,
        "G": result_G,
        "evaluations": state.n_eval,
        "generation": state.generation,
        "hv_converged": hv_reached,
        "population": {"X": state.X.copy(), "F": state.F.copy()},
    }

    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive_X"] = archive_X.copy()
        result["archive_F"] = archive_F.copy()

    return result


__all__ = [
    "IBEAState",
    "build_ibea_result",
]
