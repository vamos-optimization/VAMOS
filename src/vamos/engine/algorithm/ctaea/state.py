"""C-TAEA state container and result."""

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
class CTAEAState(AlgorithmState):
    """State of C-TAEA's two archives.

    ``X``, ``F`` and ``G`` are the convergence archive (CA), which is the result, and the
    ``diversity_*`` fields the diversity archive (DA), both in the order of jMetal's lists.
    ``weights`` holds one weight vector per row; their number N is the size of both archives and
    the number of offspring of a generation (``offspring_size``), while ``pop_size`` is the
    requested population size the vectors were generated from.
    """

    weights: np.ndarray[Any, Any] | None = field(default=None, repr=False)
    diversity_X: np.ndarray[Any, Any] | None = None
    diversity_F: np.ndarray[Any, Any] | None = None
    diversity_G: np.ndarray[Any, Any] | None = None
    hv_reached: bool = False
    variation: VariationPipeline | None = None


def build_ctaea_result(state: CTAEAState, kernel: Any = None) -> dict[str, Any]:
    """Result of C-TAEA: the CA, its non-dominated feasible rows unless ``result_mode`` is ``population``.

    With an external archive, the result is the archive (unless ``result_mode`` is
    ``population``). ``population`` holds the whole CA, which is the result of jMetal's port,
    ``diversity_archive`` the DA and ``weights`` the weight vectors, as MOEA/D's result (a
    stored run keeps them as its reference directions).
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
    if state.diversity_X is not None and state.diversity_F is not None:
        diversity_G = None if state.diversity_G is None else state.diversity_G.copy()
        result["diversity_archive"] = with_constraints({"X": state.diversity_X.copy(), "F": state.diversity_F.copy()}, diversity_G)
    if state.weights is not None:
        result["weights"] = state.weights.copy()
    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive"] = with_constraints({"X": archive_X, "F": archive_F}, archive_G)
    return result


__all__ = ["CTAEAState", "build_ctaea_result"]
