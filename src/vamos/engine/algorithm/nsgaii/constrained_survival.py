"""NSGA-II with a constraint-handling technique other than cdp (see ``components.constraint_handling``).

As in Evolver's NSGA-II, the technique ranks both the population of the tournaments and the union of
parents and offspring of the replacement, and crowding distance orders each front. Before each
replacement the technique adapts to that union at the run's progress, the evaluations before the
offspring's over the budget, as jMetal's ``ConstraintHandlingReplacement`` does. VAMOS ranks the
population again for the tournaments, as it does with cdp; jMetal reuses the ranks of the replacement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .helpers import compute_crowding, fronts_from_ranks, select_nsga2

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend

    from .state import NSGAIIState


def run_progress(st: NSGAIIState) -> float:
    """The evaluations so far over the budget, in [0, 1]; NaN without a budget."""
    if st.max_evals <= 0:
        return float("nan")
    return min(1.0, st.n_eval / st.max_evals)


def technique_selection_metrics(
    st: NSGAIIState, kernel: KernelBackend
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """The technique's fronts of the population, and the crowding distance within each front."""
    technique = st.constraint_handling
    if technique is None or st.G is None:
        raise RuntimeError("technique_selection_metrics needs a constraint-handling technique and constraint values.")
    ranks = technique.ranks(kernel, st.F, st.G, st.rng)
    crowding = compute_crowding(st.F, fronts_from_ranks(ranks))
    return ranks, np.asarray(crowding, dtype=np.float64)


def technique_survival(
    st: NSGAIIState,
    kernel: KernelBackend,
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any],
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """The rows of the union of parents and offspring that survive: whole fronts, the last cut by crowding."""
    technique = st.constraint_handling
    if technique is None:
        raise RuntimeError("technique_survival needs a constraint-handling technique.")
    technique.update(kernel, F, G, run_progress(st))
    ranks = technique.ranks(kernel, F, G, st.rng)
    fronts = fronts_from_ranks(ranks)
    return select_nsga2(fronts, compute_crowding(F, fronts), st.pop_size)


__all__ = ["run_progress", "technique_selection_metrics", "technique_survival"]
