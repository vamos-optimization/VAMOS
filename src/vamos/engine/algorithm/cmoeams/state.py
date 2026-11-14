"""CMOEA-MS state container and result."""

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
class CMOEAMSState(AlgorithmState):
    """State of CMOEA-MS's population.

    ``X``, ``F`` and ``G`` are the population, in the order of the union it was selected from, as
    jMetal's list, which decides the ties; ``fitness`` is the fitness each solution had in that
    union (stage A of the initial population at the start), on which the next tournaments are
    decided. ``stage_b`` is whether that fitness is stage B's, and ``stage_b_generations`` the
    number of generations whose fitness was stage B's. ``differential_evolution`` selects the DE
    variation, with ``de_variant``; ``variation`` provides the crossover, repair and mutation.
    """

    fitness: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)
    feasible_ratio_threshold: float = 0.5
    stage_b: bool = False
    stage_b_generations: int = 0
    hv_reached: bool = False
    variation: VariationPipeline | None = None
    differential_evolution: bool = False
    de_variant: str = "rand_1_bin"
    de_cr: float = 1.0
    de_f: float = 0.5


def build_cmoeams_result(state: CMOEAMSState, kernel: Any = None) -> dict[str, Any]:
    """Result of CMOEA-MS: the population, its non-dominated feasible rows unless ``result_mode`` is ``population``.

    With an external archive, the result is the archive (unless ``result_mode`` is
    ``population``). ``population`` holds the whole final population, which is the result of
    jMetal's port and can contain infeasible and dominated solutions; ``stage`` is ``"B"`` when
    its fitness is that of stage B and ``"A"`` otherwise, and ``stage_b_generations`` the number of
    generations in stage B.
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
        "stage": "B" if state.stage_b else "A",
        "stage_b_generations": state.stage_b_generations,
        "population": with_constraints({"X": state.X.copy(), "F": state.F.copy()}, None if state.G is None else state.G.copy()),
    }
    with_constraints(result, result_G)
    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive"] = with_constraints({"X": archive_X, "F": archive_F}, archive_G)
    return result


__all__ = ["CMOEAMSState", "build_cmoeams_result"]
