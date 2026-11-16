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

from vamos.engine.algorithm.components.results import (
    apply_final_archive_subset,
    feasible_result_indices,
    get_external_archive_constraints,
    get_external_archive_contents,
    wants_population_result,
    with_constraints,
)
from vamos.engine.algorithm.components.state import AlgorithmState
from vamos.engine.archive.factory import update_archive

if TYPE_CHECKING:
    from vamos.engine.algorithm.moead.update_criteria import SubproblemUpdateCriterion

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
    # The live callback's generations: one per pop_size evaluations, as for generational algorithms.
    # live_evals is the evaluation count where the last one ended; checkpoints keep both.
    live_generation: int = 0
    live_evals: int = 0
    # The evaluation count of the last notification: a run that ends past it sends a final one.
    live_sent_evals: int = 0
    batch_size: int = 1
    subproblem_order: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([], dtype=int))
    subproblem_cursor: int = 0

    # Cached constraint violation (sum of positive parts), when constraints are enabled.
    cv: np.ndarray[Any, Any] | None = None
    # The subproblem update criterion; None keeps MOEA/D's own feasibility rules.
    subproblem_update: SubproblemUpdateCriterion | None = None

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

    # What the updates of the external archive leave true of the population (see update_moead_archive).
    archive_covers_population: bool = False
    archive_offered_population: bool = False


# The population (decision vectors, objectives and constraint values) before a step.
ArchiveBaseline = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]


def archive_baseline(state: MOEADState) -> ArchiveBaseline | None:
    """A copy of the population before a step, for :func:`update_moead_archive`; ``None`` without an external archive."""
    if state.archive_manager is None or state.X.dtype.hasobject:
        return None
    return state.X.copy(), state.F.copy(), None if state.G is None else state.G.copy()


def update_moead_archive(state: MOEADState, baseline: ArchiveBaseline | None, *, changed: bool = True) -> None:
    """Offer the population to the external archive after a step, with the result of offering all of it.

    MOEA/D offers its whole population to the archive after every step. This gives the same archive,
    in the same order and with the same truncations, with less work:

    - When the archive covers the population (``covers_last_input`` of the archive, after every
      update since the population was last offered whole), the rows that the step left alone change
      no update. Only the changed rows are offered, in population order, and nothing when none
      changed. A changed row with a non-finite objective or constraint value gets the whole
      population.
    - When no row changed, the last update offered this very population and it left the archive as
      it was (``unchanged_by_last_input``), the update would repeat it exactly: it is skipped.
    - Otherwise the whole population is offered, as before.

    ``changed=False`` tells that the step wrote no row of the population, which spares comparing it
    with ``baseline``.
    """
    manager = state.archive_manager
    if manager is None:
        return
    rows = None if baseline is None else _changed_rows(state, baseline) if changed else np.empty(0, dtype=np.intp)
    if rows is not None and rows.size == 0:
        if state.archive_covers_population or (state.archive_offered_population and manager.unchanged_by_last_input):
            return
    whole = rows is None or not state.archive_covers_population or not _finite_rows(state, rows)
    if whole:
        update_archive(state, state.X, state.F, state.G)
    else:
        assert rows is not None
        update_archive(state, state.X[rows], state.F[rows], None if state.G is None else state.G[rows])
    state.archive_covers_population = manager.covers_last_input
    state.archive_offered_population = whole


def _rows_differ(after: np.ndarray[Any, Any], before: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Rows of ``after`` whose bits differ from those of ``before``."""
    if after.shape != before.shape or after.dtype != before.dtype:
        return np.ones(after.shape[0], dtype=bool)
    rows = after.shape[0]
    after_bytes = np.ascontiguousarray(after).view(np.uint8).reshape(rows, -1)
    before_bytes = np.ascontiguousarray(before).view(np.uint8).reshape(rows, -1)
    return np.asarray((after_bytes != before_bytes).any(axis=1))


def _changed_rows(state: MOEADState, baseline: ArchiveBaseline) -> np.ndarray[Any, Any]:
    """Population rows whose decision vector, objectives or constraint values changed, bit for bit."""
    X_before, F_before, G_before = baseline
    changed = _rows_differ(state.X, X_before) | _rows_differ(state.F, F_before)
    if state.G is not None and G_before is not None:
        changed |= _rows_differ(state.G, G_before)
    elif state.G is not None or G_before is not None:
        changed[:] = True
    return np.flatnonzero(changed)


def _finite_rows(state: MOEADState, rows: np.ndarray[Any, Any]) -> bool:
    if not np.isfinite(state.F[rows]).all():
        return False
    return state.G is None or bool(np.isfinite(state.G[rows]).all())


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
    archive_G = get_external_archive_constraints(state)
    if archive_contents is not None and not wants_population_result(state):
        result_X, result_F = archive_contents
        result_G = archive_G
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
        "population": with_constraints({"X": state.X, "F": state.F}, state.G),
    }

    with_constraints(result, result_G)

    if archive_contents is not None:
        arch_X, arch_F = archive_contents
        result["archive"] = with_constraints({"X": arch_X, "F": arch_F}, archive_G)

    return apply_final_archive_subset(result, state)


__all__ = [
    "ArchiveBaseline",
    "MOEADState",
    "archive_baseline",
    "build_moead_result",
    "update_moead_archive",
]
