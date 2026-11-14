"""CCMO state container and result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.results import (
    feasible_result_indices,
    get_external_archive_constraints,
    get_external_archive_contents,
    wants_population_result,
    with_constraints,
)
from vamos.engine.algorithm.components.state import AlgorithmState

if TYPE_CHECKING:
    from vamos.engine.variation.pipeline import VariationPipeline


@dataclass
class CCMOState(AlgorithmState):
    """State of CCMO's two populations.

    ``X``, ``F`` and ``G`` are population 1, which solves the problem, and ``fitness`` its CCMO
    fitness; the ``helper_*`` fields are population 2, which solves the problem without
    constraints. After each generation both populations are sorted by fitness, the one each
    solution had in the union it was selected from. ``pending_from_main`` is the number of rows
    of the pending offspring that population 1 created; the rest are population 2's.
    """

    fitness: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)
    helper_X: np.ndarray[Any, Any] | None = None
    helper_F: np.ndarray[Any, Any] | None = None
    helper_G: np.ndarray[Any, Any] | None = None
    helper_fitness: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)
    hv_reached: bool = False
    variation: VariationPipeline | None = None
    differential_evolution: bool = False
    pending_from_main: int = 0


def build_ccmo_result(state: CCMOState, kernel: Any = None) -> dict[str, Any]:
    """Result of CCMO: population 1, its non-dominated feasible rows unless ``result_mode`` is ``population``.

    With an external archive, the result is the archive (unless ``result_mode`` is
    ``population``). ``population`` holds population 1 and ``helper_population`` population 2.
    """
    archive_contents = get_external_archive_contents(state)
    archive_G = get_external_archive_constraints(state)
    if archive_contents is not None and not wants_population_result(state):
        result_X, result_F = (array.copy() for array in archive_contents)
        result_G = archive_G.copy() if archive_G is not None else None
    elif wants_population_result(state):
        result_X, result_F = state.X.copy(), state.F.copy()
        result_G = state.G.copy() if state.G is not None else None
    else:
        selected = feasible_result_indices(state.F, state.G, state.constraint_mode, kernel)
        result_X, result_F = state.X[selected].copy(), state.F[selected].copy()
        result_G = state.G[selected].copy() if state.G is not None else None

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": state.n_eval,
        "generation": state.generation,
        "hv_reached": state.hv_reached,
        "population": with_constraints({"X": state.X.copy(), "F": state.F.copy()}, None if state.G is None else state.G.copy()),
    }
    with_constraints(result, result_G)
    if state.helper_X is not None and state.helper_F is not None:
        helper_G = None if state.helper_G is None else state.helper_G.copy()
        result["helper_population"] = with_constraints({"X": state.helper_X.copy(), "F": state.helper_F.copy()}, helper_G)
    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive"] = with_constraints({"X": archive_X, "F": archive_F}, archive_G)
    return result


__all__ = ["CCMOState", "build_ccmo_result"]
