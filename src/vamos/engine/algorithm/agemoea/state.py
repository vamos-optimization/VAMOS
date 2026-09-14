"""AGE-MOEA-specific state container.

This module provides the state dataclass for AGE-MOEA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.results import get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState
from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.engine.variation.pipeline import VariationPipeline


@dataclass
class AGEMOEAState(AlgorithmState):
    """State container for AGE-MOEA with ask/tell interface.

    Extends the base AlgorithmState with AGE-MOEA-specific fields.
    """

    max_evals: int = 0
    variation: VariationPipeline | None = None
    selection_ranks: Any = field(default=None, repr=False, compare=False)
    selection_crowding: Any = field(default=None, repr=False, compare=False)


def _result_population_indices(state: AGEMOEAState, kernel: Any) -> np.ndarray:
    """Return the feasibility-aware non-dominated subset for top-level results."""
    candidates = np.arange(state.F.shape[0], dtype=int)
    if state.G is not None and state.constraint_mode != "none":
        feasible = is_feasible(state.G, n=state.G.shape[0])
        if feasible.any():
            candidates = np.flatnonzero(feasible)
        else:
            violation = compute_violation(state.G, n=state.G.shape[0])
            minimum = float(np.min(violation))
            candidates = np.flatnonzero(violation <= minimum + 1e-12)

    if kernel is None or candidates.size == 0:
        return candidates
    ranks, _ = kernel.nsga2_ranking(state.F[candidates])
    return candidates[np.asarray(ranks) == 0]


def build_agemoea_result(
    state: AGEMOEAState,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build AGE-MOEA result dictionary from state."""
    archive_contents = get_external_archive_contents(state)
    result_G: np.ndarray | None = None
    if archive_contents is not None and not wants_population_result(state):
        archive_X, archive_F = archive_contents
        result_X = archive_X.copy()
        result_F = archive_F.copy()
    elif wants_population_result(state):
        result_X = state.X.copy()
        result_F = state.F.copy()
        result_G = state.G.copy() if state.G is not None else None
    else:
        try:
            selected = _result_population_indices(state, kernel)
            result_X = state.X[selected].copy()
            result_F = state.F[selected].copy()
            result_G = state.G[selected].copy() if state.G is not None else None
        except (ValueError, IndexError):
            result_X = state.X.copy()
            result_F = state.F.copy()
            result_G = state.G.copy() if state.G is not None else None

    population: dict[str, Any] = {"X": state.X.copy(), "F": state.F.copy()}
    if state.G is not None:
        population["G"] = state.G.copy()

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": state.n_eval,
        "generation": state.generation,
        "population": population,
    }
    if result_G is not None:
        result["G"] = result_G
    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive"] = {"X": archive_X, "F": archive_F}
    return result


__all__ = ["AGEMOEAState", "build_agemoea_result"]
