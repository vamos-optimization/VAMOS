# algorithm/nsgaii/state.py
"""
State container and result building for NSGA-II.

This module provides the NSGAIIState dataclass and result-building functions,
keeping the main algorithm file focused on the evolutionary loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    MaxMinArchive,
    ReferenceDirectionsArchive,
    SPEA2Archive,
    UnboundedArchive,
)
from vamos.engine.algorithm.components.results import feasible_result_indices, get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.termination import HVTracker
from vamos.engine.hooks.genealogy import GenealogyTracker
from vamos.engine.operators.impl.real import VariationWorkspace
from vamos.engine.variation import VariationPipeline

if TYPE_CHECKING:
    from .steady_state import SteadyStateStep


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class NSGAIIState:
    """Mutable state container for NSGA-II algorithm."""

    # Population
    X: np.ndarray[Any, Any]
    F: np.ndarray[Any, Any]
    G: np.ndarray[Any, Any] | None
    rng: np.random.Generator

    # Variation
    variation: VariationPipeline
    operator_pool: list[VariationPipeline]
    variation_workspace: VariationWorkspace

    # Selection
    sel_method: str = "tournament"
    pressure: int = 2

    # Sizes
    pop_size: int = 100
    offspring_size: int = 100
    replacement_size: int = 1
    incremental_mode: bool = False

    # Constraints
    constraint_mode: str = "feasibility"

    # Archive
    archive_size: int | None = None
    archive_X: np.ndarray[Any, Any] | None = None
    archive_F: np.ndarray[Any, Any] | None = None
    archive_manager: (
        CrowdingDistanceArchive | HypervolumeArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | UnboundedArchive | None
    ) = None
    result_archive: HypervolumeArchive | CrowdingDistanceArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | None = None
    result_mode: str = "non_dominated"

    # Termination
    hv_tracker: HVTracker | None = None

    # Genealogy
    track_genealogy: bool = False
    genealogy_tracker: GenealogyTracker | None = None
    ids: np.ndarray[Any, Any] | None = None

    # Generation tracking
    generation: int = 0
    n_eval: int = 0
    max_evals: int = 0
    step: int = 0
    replacements: int = 0

    # Cached selection metrics (incremental replacement). The compiled steady-state
    # step maintains ranks and crowding only and leaves ``fronts`` as None.
    fronts: list[list[int]] | None = None
    ranks: np.ndarray[Any, Any] | None = None
    crowding: np.ndarray[Any, Any] | None = None
    incremental_enabled: bool = False
    steady_step: SteadyStateStep | None = field(default=None, repr=False)

    # Pending offspring (from ask)
    pending_offspring: np.ndarray[Any, Any] | None = None
    pending_offspring_ids: np.ndarray[Any, Any] | None = None

    # Optional extension hooks
    immigration_manager: Any | None = None
    parent_selection_filter: Any | None = None
    non_breeding_indices: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0, dtype=int))
    live_callback_mode: str = "nd_only"
    generation_callback: Any | None = None
    generation_callback_copy: bool = True

    # HV points function (computed lazily)
    _hv_points_fn: Callable[[], np.ndarray[Any, Any]] | None = field(default=None, repr=False)

    # Unconstrained ranking of F computed at the end of a generation, reused by the next ask().
    _generation_ranking: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = field(default=None, repr=False)

    def hv_points_fn(self) -> np.ndarray[Any, Any]:
        """Get points for hypervolume computation (archive if available, else population)."""
        if self.archive_F is not None and self.archive_F.size > 0:
            return self.archive_F
        return self.F

    def publish_generation_ranking(self, F: np.ndarray[Any, Any], ranks: np.ndarray[Any, Any], crowding: np.ndarray[Any, Any]) -> None:
        """Remember the unconstrained ranking of the current population for the next selection."""
        if F is self.F:
            self._generation_ranking = (np.array(F, copy=True), ranks, crowding)

    def take_generation_ranking(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
        """Return the published ranking when the population is unchanged, consuming it."""
        published = self._generation_ranking
        self._generation_ranking = None
        if published is None:
            return None
        snapshot, ranks, crowding = published
        if snapshot.shape != self.F.shape or not np.array_equal(snapshot, self.F, equal_nan=True):
            return None
        return ranks, crowding


def build_result(
    state: NSGAIIState,
    n_eval: int,
    hv_reached: bool,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build the result dictionary from algorithm state.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state.
    n_eval : int
        Total number of evaluations.
    hv_reached : bool
        Whether HV threshold was reached.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict[str, Any]
        Result dictionary with X, F, evaluations, population, and optional archive.
        X and F contain only non-dominated solutions when kernel is provided.
        Full population is always available in 'population' key.
    """
    archive_contents = get_external_archive_contents(state)
    should_use_archive = archive_contents is not None and not wants_population_result(state)

    if should_use_archive:
        assert archive_contents is not None
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
            except (ValueError, IndexError) as exc:
                _logger().warning("Failed to filter non-dominated solutions: %s", exc)
                result_X, result_F, result_G = state.X, state.F, state.G
        else:
            result_X, result_F, result_G = state.X, state.F, state.G

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": n_eval,
        "hv_reached": hv_reached,
        "population": {"X": state.X, "F": state.F},  # Full population always available
    }
    if result_G is not None:
        result["G"] = result_G

    if archive_contents is not None:
        arch_X, arch_F = archive_contents
        result["archive"] = {"X": arch_X, "F": arch_F}

    return result


def get_archive_contents(state: NSGAIIState) -> dict[str, Any] | None:
    """Extract archive contents from state.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state.

    Returns
    -------
    dict[str, Any] | None
        Archive contents with X and F, or None if no archive.
    """
    archive_contents = get_external_archive_contents(state)
    if archive_contents is None:
        return None
    final_X, final_F = archive_contents
    return {"X": final_X, "F": final_F}


def finalize_genealogy(
    result: dict[str, Any],
    state: NSGAIIState,
    kernel: Any,
) -> None:
    """Add genealogy stats to result if tracking is enabled.

    Parameters
    ----------
    result : dict[str, Any]
        Result dictionary to update.
    state : NSGAIIState
        Current algorithm state.
    kernel : KernelBackend
        Kernel for ranking computation.
    """
    from vamos.engine.hooks.genealogy import generation_contributions, operator_success_stats

    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    try:
        ranks, _ = kernel.nsga2_ranking(state.F)
        nd_mask = ranks == ranks.min(initial=0)
        final_front_ids = state.ids[nd_mask] if state.ids is not None else []
        state.genealogy_tracker.mark_final_front(list(final_front_ids))
        result["genealogy"] = {
            "operator_stats": operator_success_stats(state.genealogy_tracker, list(final_front_ids)),
            "generation_contributions": generation_contributions(state.genealogy_tracker, list(final_front_ids)),
        }
    except (ValueError, IndexError, AttributeError) as exc:
        _logger().warning("Failed to compute genealogy stats: %s", exc)


def compute_selection_metrics(
    kernel: Any,
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    constraint_mode: str,
    precomputed: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute selection metrics (ranks and crowding) with constraint handling.

    Parameters
    ----------
    kernel : KernelBackend
        Kernel for ranking computation.
    F : np.ndarray
        Objective values.
    G : np.ndarray | None
        Constraint values (None if unconstrained).
    constraint_mode : str
        Constraint handling mode.
    precomputed : tuple[np.ndarray, np.ndarray] | None
        Unconstrained ``kernel.nsga2_ranking(F)`` result for this exact ``F``,
        reused instead of ranking the population again.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ranks, crowding) arrays.
    """
    from vamos.foundation.constraints.utils import compute_violation, is_feasible

    if G is not None and constraint_mode != "none":
        cv = compute_violation(G)
        feas = is_feasible(G)
        if feas.any():
            feas_idx = np.nonzero(feas)[0]
            feas_ranks, feas_crowd = kernel.nsga2_ranking(F[feas_idx])
            ranks = np.full(F.shape[0], feas_ranks.max(initial=0) + 1, dtype=int)
            crowding = np.zeros(F.shape[0], dtype=float)
            ranks[feas_idx] = feas_ranks
            crowding[feas_idx] = feas_crowd
            crowding[~feas] = -cv[~feas]
        else:
            ranks = np.zeros(F.shape[0], dtype=int)
            crowding = -cv
        return ranks, crowding
    if precomputed is not None:
        return precomputed
    ranks, crowding = kernel.nsga2_ranking(F)
    return ranks, crowding


def track_offspring_genealogy(
    state: NSGAIIState,
    parent_idx: np.ndarray[Any, Any],
    n_offspring: int,
) -> None:
    """Track genealogy for generated offspring.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state (modified in place).
    parent_idx : np.ndarray
        Indices of parents used.
    n_offspring : int
        Number of offspring generated.
    """
    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    operator_name = f"{state.variation.cross_method}+{state.variation.mut_method}"
    group_size = state.variation.parents_per_group
    children_per_group = state.variation.children_per_group
    parent_groups = parent_idx.reshape(-1, group_size)
    child_ids = []
    gen = state.generation + 1

    for parents in parent_groups:
        parent_ids = state.ids[parents] if state.ids is not None else []
        for _ in range(children_per_group):
            child_ids.append(
                state.genealogy_tracker.new_individual(
                    generation=gen,
                    parents=list(parent_ids),
                    operator_name=operator_name,
                    algorithm_name="nsgaii",
                )
            )
    state.pending_offspring_ids = np.asarray(child_ids[:n_offspring], dtype=int)


def update_archives(
    state: NSGAIIState,
    kernel: Any,
    *,
    X: np.ndarray[Any, Any] | None = None,
    F: np.ndarray[Any, Any] | None = None,
    G: np.ndarray[Any, Any] | None = None,
) -> None:
    """Update external archive state and synchronize cached snapshots.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state (modified in place).
    kernel : KernelBackend
        Kernel backend.
    X : np.ndarray | None
        Candidate decision variables to insert (defaults to state.X).
    F : np.ndarray | None
        Candidate objectives to insert (defaults to state.F).
    G : np.ndarray | None
        Candidate constraints to insert (defaults to state.G).
    """
    del kernel
    X_use = state.X if X is None else X
    F_use = state.F if F is None else F
    G_use = state.G if G is None else G
    if state.archive_manager is not None:
        state.archive_X, state.archive_F = state.archive_manager.update(X_use, F_use, G_use)
        if state.result_archive is state.archive_manager:
            return

    if state.result_archive is not None:
        state.result_archive.update(X_use, F_use, G_use)
        state.archive_X, state.archive_F = state.result_archive.contents()
