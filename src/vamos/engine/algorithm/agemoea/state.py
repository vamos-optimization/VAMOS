"""AGE-MOEA-specific state container.

This module provides the state dataclass for AGE-MOEA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vamos.engine.algorithm.components.results import get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState

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


def build_agemoea_result(
    state: AGEMOEAState,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build AGE-MOEA result dictionary from state."""
    archive_contents = get_external_archive_contents(state)
    if archive_contents is not None and not wants_population_result(state):
        archive_X, archive_F = archive_contents
        result_X = archive_X.copy()
        result_F = archive_F.copy()
    else:
        if not wants_population_result(state) and kernel is not None:
            try:
                ranks, _ = kernel.nsga2_ranking(state.F)
                nd_mask = ranks == 0
                result_X = state.X[nd_mask].copy()
                result_F = state.F[nd_mask].copy()
            except (ValueError, IndexError):
                result_X = state.X.copy()
                result_F = state.F.copy()
        else:
            result_X = state.X.copy()
            result_F = state.F.copy()

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": state.n_eval,
        "generation": state.generation,
        "population": {"X": state.X.copy(), "F": state.F.copy()},
    }
    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive"] = {"X": archive_X, "F": archive_F}
    return result


__all__ = ["AGEMOEAState", "build_agemoea_result"]
