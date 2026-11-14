"""ToP state container and result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vamos.engine.algorithm.components.results import (
    feasible_result_indices,
    get_external_archive_constraints,
    get_external_archive_contents,
    wants_population_result,
    with_constraints,
)
from vamos.engine.algorithm.components.state import AlgorithmState

from .phases import DELTA_THRESHOLD, FEASIBLE_RATIO_THRESHOLD, INITIAL_DELTA, PHASE_ONE_EVALUATION_FRACTION

if TYPE_CHECKING:
    from vamos.engine.variation.pipeline import VariationPipeline


@dataclass
class ToPState(AlgorithmState):
    """State of ToP's population and of its switch between the two phases.

    ``X``, ``F`` and ``G`` are the population, in the order of jMetal's list, which decides the
    ties. ``delta`` and ``feasible_ratio`` (``Pf``) are updated after each generation of phase 1,
    and ``phase_two_start`` is the number of evaluations done when the first generation of phase
    2 started, ``None`` before. ``pending_phase`` is the phase (1 or 2) of the pending offspring.
    ``cr`` and ``f`` are the parameters of the differential evolution, and ``variation`` provides
    its repair and mutation.
    """

    delta: float = INITIAL_DELTA
    feasible_ratio: float = 0.0
    phase_two_start: int | None = None
    pending_phase: int | None = None
    cr: float = 1.0
    f: float = 0.5
    delta_threshold: float = DELTA_THRESHOLD
    feasible_ratio_threshold: float = FEASIBLE_RATIO_THRESHOLD
    phase_one_evaluation_fraction: float = PHASE_ONE_EVALUATION_FRACTION
    hv_reached: bool = False
    variation: VariationPipeline | None = None


def build_top_result(state: ToPState, kernel: Any = None) -> dict[str, Any]:
    """Result of ToP: the population, its non-dominated feasible rows unless ``result_mode`` is ``population``.

    With an external archive, the result is the archive (unless ``result_mode`` is
    ``population``). ``population`` holds the whole final population, which is the result of
    jMetal's port and can contain infeasible solutions, and ``phase_two_start`` the evaluations
    done when phase 2 started, or ``None`` if the run did not reach it.
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
        "phase_two_start": state.phase_two_start,
        "population": with_constraints({"X": state.X.copy(), "F": state.F.copy()}, None if state.G is None else state.G.copy()),
    }
    with_constraints(result, result_G)
    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive"] = with_constraints({"X": archive_X, "F": archive_F}, archive_G)
    return result


__all__ = ["ToPState", "build_top_result"]
