"""NSGA3-specific state container.

This module provides the state dataclass for NSGA-III's ask/tell interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.results import feasible_result_indices, get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState


@dataclass
class NSGAIIIState(AlgorithmState):
    """State container for NSGA-III with ask/tell interface.

    Extends the base AlgorithmState with NSGA-III-specific fields for
    reference directions and variation operators.

    Additional Attributes
    ---------------------
    ref_dirs : np.ndarray[Any, Any]
        Reference directions for diversity maintenance.
    ref_dirs_norm : np.ndarray[Any, Any]
        Normalized reference directions.
    pressure : int
        Tournament selection pressure.
    crossover_fn : Callable
        Crossover operator function.
    mutation_fn : Callable
        Mutation operator function.
    """

    ref_dirs: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    ref_dirs_norm: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    pressure: int = 2
    crossover_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | None = None
    mutation_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | None = None
    result_mode: str = "population"
    ideal_point: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    worst_point: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    extreme_points: np.ndarray[Any, Any] | None = None
    selection_ranks: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)
    selection_crowding: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)


def build_nsgaiii_result(
    state: NSGAIIIState,
    hv_reached: bool = False,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build NSGA-III result dictionary from state.

    Parameters
    ----------
    state : NSGAIIIState
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict
        Result dictionary with X, F, G, reference_directions, population, archive data,
        and metadata. X and F contain only non-dominated solutions when kernel is provided.
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
        "reference_directions": state.ref_dirs.copy(),
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
