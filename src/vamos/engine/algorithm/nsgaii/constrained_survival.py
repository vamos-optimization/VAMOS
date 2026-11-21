"""NSGA-II with a constraint-handling technique other than cdp (see ``components.constraint_handling``).

As in Evolver's NSGA-II, the technique ranks both the population of the tournaments and the union of
parents and offspring of the replacement, and crowding distance orders each front. Before each
replacement the technique adapts to that union at the run's progress, the evaluations before the
offspring's over the budget, as jMetal's ``ConstraintHandlingReplacement`` does. VAMOS ranks the
population again for the tournaments, as it does with cdp; jMetal reuses the ranks of the replacement.
VAMOS 1.5.0: NSGA-III's tournament compares the technique's ranks only, without crowding
distance (``nsgaiii.helpers.nsgaiii_parents``).

Development after 1.5.2: a technique with a total order (``total_order``) puts each row in a front of its
own, so NSGA-II skips the crowding distance, which could not change a front of one row; and after each
replacement the technique learns the survivors (``survived``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.constraint_handling import ConstraintHandlingTechnique, adapted_ranks, run_progress

from .helpers import compute_crowding, fronts_from_ranks, select_nsga2

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend

    from .state import NSGAIIState


def technique_selection_metrics(
    technique: ConstraintHandlingTechnique,
    kernel: KernelBackend,
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """The technique's fronts of a population, and the crowding distance within each front.

    A technique with a total order puts each row in a front of its own, which the tournament's ranks
    decide, so its crowding distance is left at 0.
    """
    ranks = technique.ranks(kernel, F, G, rng)
    if technique.total_order:
        return ranks, np.zeros(F.shape[0], dtype=np.float64)
    crowding = compute_crowding(F, fronts_from_ranks(ranks))
    return ranks, np.asarray(crowding, dtype=np.float64)


def technique_survival(
    st: NSGAIIState,
    kernel: KernelBackend,
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any],
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """The rows of the union of parents and offspring that survive: whole fronts, the last cut by crowding.

    A technique with a total order keeps the first ``pop_size`` rows of its order, without crowding
    distance. The technique then learns the survivors (:meth:`ConstraintHandlingTechnique.survived`).
    """
    technique = st.constraint_handling
    if technique is None:
        raise RuntimeError("technique_survival needs a constraint-handling technique.")
    ranks = adapted_ranks(technique, kernel, F, G, st.rng, run_progress(st.n_eval, st.max_evals))
    fronts = fronts_from_ranks(ranks)
    crowding = np.zeros(F.shape[0], dtype=np.float64) if technique.total_order else compute_crowding(F, fronts)
    survivors = select_nsga2(fronts, crowding, st.pop_size)
    technique.survived(F[survivors], G[survivors])
    return survivors


__all__ = ["technique_selection_metrics", "technique_survival"]
