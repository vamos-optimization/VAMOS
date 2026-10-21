"""
MOEA/D State container and result building.

This module provides the MOEADState dataclass that holds all mutable state
for the MOEA/D algorithm, similar to NSGAIIState.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.results import feasible_result_indices, get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState

if TYPE_CHECKING:
    pass


@dataclass
class MOEADState(AlgorithmState):
    """
    Mutable state container for MOEA/D algorithm.

    Extends AlgorithmState with MOEA/D-specific fields for weight vectors,
    neighborhoods, aggregation, and subproblem management.

    Attributes
    ----------
    weights : np.ndarray[Any, Any]
        Weight vectors for decomposition, shape (pop_size, n_obj).
    neighbors : np.ndarray[Any, Any]
        Neighborhood indices for each subproblem, shape (pop_size, T).
    ideal : np.ndarray[Any, Any]
        Ideal point (minimum objectives seen), shape (n_obj,).
    aggregator : Callable
        Aggregation function (tchebycheff, weighted_sum, pbi, etc.).
    neighbor_size : int
        Neighborhood size (T parameter).
    delta : float
        Probability of selecting from neighborhood vs. whole population.
    replace_limit : int
        Maximum number of replacements per offspring.
    track_genealogy : bool
        Whether to track genealogy information.
    genealogy_tracker : GenealogyTracker | None
        Tracker for lineage information.
    ids : np.ndarray[Any, Any] | None
        Individual IDs for genealogy tracking.
    """

    # MOEA/D-specific fields
    weights: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    weights_safe: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    weights_unit: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    neighbors: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    ideal: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    aggregator: Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]], np.ndarray[Any, Any]] | None = None
    aggregation_id: int = -1
    aggregation_theta: float = 5.0
    aggregation_rho: float = 0.001
    neighbor_size: int = 20
    delta: float = 0.9
    replace_limit: int = 2
    batch_size: int = 1
    subproblem_order: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([], dtype=int))
    subproblem_cursor: int = 0

    # Cached constraint violation (sum of positive parts), when constraints are enabled.
    cv: np.ndarray[Any, Any] | None = None

    # Variation operators (as callables for flexibility across encodings)
    crossover_fn: Callable[[np.ndarray[Any, Any], np.random.Generator], np.ndarray[Any, Any]] | None = None
    mutation_fn: Callable[[np.ndarray[Any, Any], np.random.Generator], np.ndarray[Any, Any]] | None = None

    # Bounds for integer/real encodings
    xl: np.ndarray[Any, Any] | None = None
    xu: np.ndarray[Any, Any] | None = None

    # Pending offspring tracking for ask/tell
    pending_active_indices: np.ndarray[Any, Any] | None = None
    pending_parent_pairs: np.ndarray[Any, Any] | None = None
    pending_use_neighbors: np.ndarray[Any, Any] | None = None


def build_moead_result(
    state: MOEADState,
    hv_reached: bool = False,
    kernel: Any = None,
) -> dict[str, Any]:
    """
    Build the result dictionary from MOEA/D state.

    Parameters
    ----------
    state : MOEADState
        Current algorithm state.
    hv_reached : bool
        Whether HV threshold was reached.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict[str, Any]
        Result dictionary with X, F, weights, evaluations, population, and optional archive.
        X and F contain only non-dominated solutions when kernel is provided.
    """
    archive_contents = get_external_archive_contents(state)
    if archive_contents is not None and not wants_population_result(state):
        result_X, result_F = archive_contents
        result_G = None
    else:
        should_filter = kernel is not None and not wants_population_result(state)
        if should_filter:
            try:
                selected = feasible_result_indices(state.F, state.G, state.constraint_mode, kernel)
                result_X = state.X[selected]
                result_F = state.F[selected]
                result_G = state.G[selected] if state.G is not None else None
            except (ValueError, IndexError):
                result_X, result_F, result_G = state.X, state.F, state.G
        else:
            result_X, result_F, result_G = state.X, state.F, state.G

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "weights": state.weights,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
        "population": {"X": state.X, "F": state.F},
    }

    if result_G is not None:
        result["G"] = result_G

    if archive_contents is not None:
        arch_X, arch_F = archive_contents
        result["archive"] = {"X": arch_X, "F": arch_F}

    return result


__all__ = [
    "MOEADState",
    "build_moead_result",
]
