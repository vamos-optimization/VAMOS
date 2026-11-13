"""
Hook integration points for live visualization and genealogy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.matching import match_ids_by_tolerance


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


if TYPE_CHECKING:
    from vamos.engine.algorithm.components.state import AlgorithmState
    from vamos.engine.hooks.genealogy import GenealogyTracker
    from vamos.engine.hooks.live_viz import LiveVisualization
    from vamos.foundation.kernel.backend import KernelBackend

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
AnyArray = NDArray[np.generic]


def get_live_viz(
    live_viz: LiveVisualization | None,
) -> LiveVisualization:
    """
    Get live visualization callback, defaulting to no-op.

    Parameters
    ----------
    live_viz : LiveVisualization | None
        User-provided callback or None.

    Returns
    -------
    LiveVisualization
        The callback or a no-op implementation.
    """
    from vamos.engine.hooks.live_viz import NoOpLiveVisualization

    return live_viz or NoOpLiveVisualization()


def notify_generation(
    live_cb: LiveVisualization,
    kernel: KernelBackend,
    generation: int,
    F: FloatArray,
    stats: dict[str, Any] | None = None,
) -> bool:
    """
    Notify live visualization of generation progress with the non-dominated rows of ``F``.

    An exception raised by the callback, or by its ``should_stop``, propagates to the
    caller; only a failure to rank ``F`` falls back to notifying all its rows.

    Parameters
    ----------
    live_cb : LiveVisualization
        Live visualization callback.
    kernel : KernelBackend
        Kernel for computing non-dominated front.
    generation : int
        Current generation number.
    F : np.ndarray
        Current population objectives.
    stats : dict[str, Any] | None
        Optional metrics payload (e.g., evaluation counts).

    Returns
    -------
    bool
        True if the callback requests stopping.
    """
    from vamos.engine.hooks.live_viz import NoOpLiveVisualization

    if isinstance(live_cb, NoOpLiveVisualization):
        return False

    try:
        ranks, _ = kernel.nsga2_ranking(F)
    except (ValueError, IndexError) as exc:
        _logger().debug("Failed to compute non-dominated front for viz: %s", exc)
        front = F
    else:
        front = F[ranks == ranks.min(initial=0)]
    live_cb.on_generation(generation, F=front, stats=stats)
    return live_should_stop(live_cb)


def live_should_stop(live_cb: LiveVisualization) -> bool:
    """Whether the callback's optional ``should_stop`` asks to stop; its exceptions propagate."""
    should_stop = getattr(live_cb, "should_stop", None)
    if not callable(should_stop):
        return False
    return bool(should_stop())


def setup_genealogy(
    pop_size: int,
    F: AnyArray,
    track_genealogy: bool,
    algorithm_name: str = "algorithm",
) -> tuple[GenealogyTracker | None, IntArray | None]:
    """
    Initialize genealogy tracking if enabled.

    Parameters
    ----------
    pop_size : int
        Population size.
    F : np.ndarray
        Initial fitness values.
    track_genealogy : bool
        Whether to enable genealogy tracking.
    algorithm_name : str
        Algorithm name for tracking records.

    Returns
    -------
    tuple[GenealogyTracker | None, np.ndarray | None]
        (genealogy_tracker, individual_ids)
    """
    if not track_genealogy:
        return None, None

    from vamos.engine.hooks.genealogy import DefaultGenealogyTracker

    genealogy_tracker = DefaultGenealogyTracker()
    ids = np.arange(pop_size, dtype=int)
    for i in range(pop_size):
        genealogy_tracker.new_individual(
            generation=0,
            parents=[],
            operator_name=None,
            algorithm_name=algorithm_name,
            fitness=F[i] if F is not None and i < F.shape[0] else None,
        )
    return genealogy_tracker, ids


def track_offspring_genealogy(
    state: AlgorithmState,
    parent_idx: IntArray,
    offspring_count: int,
    operator_name: str = "variation",
    algorithm_name: str = "algorithm",
    *,
    parents_per_group: int | None = None,
    children_per_group: int = 1,
) -> None:
    """
    Record genealogy for new offspring.

    Parameters
    ----------
    state : AlgorithmState
        Algorithm state with genealogy fields.
    parent_idx : np.ndarray
        Indices of parents used to create offspring, one group of
        ``parents_per_group`` after another.
    offspring_count : int
        Number of offspring created.
    operator_name : str
        Name of the variation operator used.
    algorithm_name : str
        Algorithm name for tracking.
    parents_per_group : int | None
        Parents that the variation combines into one group. ``None`` assumes
        one group per child: ``len(parent_idx) // offspring_count`` parents.
    children_per_group : int
        Children that each group produces, in order. SBX turns each pair of
        parents into two children, so both children record both parents.
    """
    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    if parents_per_group is None:
        parents_per_group = max(1, len(parent_idx) // offspring_count)
        children_per_group = 1
    children_per_group = max(1, int(children_per_group))
    pending_ids = []

    for i in range(offspring_count):
        start = (i // children_per_group) * parents_per_group
        end = min(start + parents_per_group, len(parent_idx))
        parent_ids = []
        if state.ids is not None:
            parent_ids = [int(state.ids[p]) for p in parent_idx[start:end] if p < len(state.ids)]

        new_id = state.genealogy_tracker.new_individual(
            generation=state.generation + 1,
            parents=parent_ids,
            operator_name=operator_name,
            algorithm_name=algorithm_name,
            fitness=None,
        )
        pending_ids.append(new_id)

    state.pending_offspring_ids = np.array(pending_ids, dtype=int)


def survivor_genealogy_ids(state: AlgorithmState, survivors: IntArray) -> None:
    """Keep the ids of the survivors of the current population followed by the pending offspring.

    ``survivors`` indexes the parents and offspring stacked in that order, as an
    environmental selection sees them. Without genealogy tracking it does nothing.
    """
    if not state.track_genealogy or state.ids is None:
        return
    pending = state.pending_offspring_ids if state.pending_offspring_ids is not None else np.empty(0, dtype=int)
    combined = np.concatenate([np.asarray(state.ids, dtype=int), np.asarray(pending, dtype=int)])
    state.ids = combined[np.asarray(survivors, dtype=int)]
    state.pending_offspring_ids = None


def finalize_genealogy(
    result: dict[str, Any],
    state: AlgorithmState,
    kernel: KernelBackend,
    F: np.ndarray[Any, Any] | None = None,
) -> None:
    """
    Finalize genealogy tracking and add to result.

    Parameters
    ----------
    result : dict
        Result dictionary to update.
    state : AlgorithmState
        Algorithm state with genealogy tracker.
    kernel : KernelBackend
        Kernel backend for ranking computation.
    F : np.ndarray | None
        Objectives of the rows that ``state.ids`` follows; ``state.F`` by default.
    """
    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    try:
        from vamos.engine.hooks.genealogy import (
            generation_contributions,
            operator_success_stats,
        )

        ranks, _ = kernel.nsga2_ranking(state.F if F is None else F)
        nd_mask = ranks == ranks.min(initial=0)
        final_front_ids = state.ids[nd_mask] if state.ids is not None else []
        state.genealogy_tracker.mark_final_front(list(final_front_ids))

        result["genealogy"] = {
            "operator_stats": operator_success_stats(state.genealogy_tracker, list(final_front_ids)),
            "generation_contributions": generation_contributions(state.genealogy_tracker, list(final_front_ids)),
        }
    except (ValueError, IndexError, AttributeError) as exc:
        logging.getLogger(__name__).warning("Failed to compute genealogy stats: %s", exc)


def match_ids(
    new_X: AnyArray,
    combined_X: AnyArray,
    combined_ids: IntArray,
) -> IntArray:
    """
    Match IDs after survival selection by finding matching rows.

    Parameters
    ----------
    new_X : np.ndarray
        New population after survival selection.
    combined_X : np.ndarray
        Combined parent+offspring population before selection.
    combined_ids : np.ndarray
        IDs corresponding to combined_X.

    Returns
    -------
    np.ndarray
        IDs for the new population.
    """
    return match_ids_by_tolerance(new_X, combined_X, combined_ids)


__all__ = [
    "get_live_viz",
    "notify_generation",
    "live_should_stop",
    "setup_genealogy",
    "track_offspring_genealogy",
    "survivor_genealogy_ids",
    "finalize_genealogy",
    "match_ids",
]
