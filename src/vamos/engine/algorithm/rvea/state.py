"""RVEA-specific state container.

This module provides the state dataclass for RVEA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.results import feasible_result_indices, get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState

if TYPE_CHECKING:
    from vamos.engine.variation.pipeline import VariationPipeline


@dataclass
class RVEAState(AlgorithmState):
    """State container for RVEA with ask/tell interface.

    Extends the base AlgorithmState with RVEA-specific fields for
    reference vectors and adaptation.
    """

    max_evals: int = 0
    max_gen: int = 1
    variation: VariationPipeline | None = None
    ref_dirs: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    V: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    gamma: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    ideal: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    nadir: np.ndarray[Any, Any] | None = None
    alpha: float = 2.0
    adapt_interval: int | None = None
    n_obj: int = 2


def build_rvea_result(
    state: RVEAState,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build RVEA result dictionary from state."""
    archive_contents = get_external_archive_contents(state)
    result_G: np.ndarray[Any, Any] | None = None
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
            selected = feasible_result_indices(state.F, state.G, state.constraint_mode, kernel)
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


__all__ = ["RVEAState", "build_rvea_result"]
