"""SMPSO state container and result building."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.results import wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState


@dataclass
class SMPSOState(AlgorithmState):
    """State for SMPSO algorithm (ask/tell capable).

    Extends AlgorithmState with PSO-specific fields (velocity, personal bests)
    and the operators needed for the turbulence/mutation step.
    """

    velocity: NDArray[Any] = field(default_factory=lambda: np.array([]))
    pbest_X: NDArray[Any] = field(default_factory=lambda: np.array([]))
    pbest_F: NDArray[Any] = field(default_factory=lambda: np.array([]))
    pbest_G: NDArray[Any] | None = None

    inertia: float = 0.5
    c1: float = 1.5
    c2: float = 1.5
    c1_min: float = 1.5
    c1_max: float = 2.5
    c2_min: float = 1.5
    c2_max: float = 2.5
    r1_min: float = 0.0
    r1_max: float = 1.0
    r2_min: float = 0.0
    r2_max: float = 1.0
    min_weight: float = 0.1
    max_weight: float = 0.1
    change_velocity1: float = -1.0
    change_velocity2: float = -1.0
    mutation_every: int = 6
    vmax: NDArray[Any] = field(default_factory=lambda: np.array([]))
    delta_max: NDArray[Any] = field(default_factory=lambda: np.array([]))
    delta_min: NDArray[Any] = field(default_factory=lambda: np.array([]))

    xl: NDArray[Any] = field(default_factory=lambda: np.array([]))
    xu: NDArray[Any] = field(default_factory=lambda: np.array([]))

    mutation_op: Any = None
    repair_op: Any = None
    archive_crowding: NDArray[Any] | None = field(default=None, repr=False, compare=False)
    pending_particle_indices: NDArray[Any] | None = field(default=None, repr=False, compare=False)


def build_smpso_result(state: SMPSOState, hv_reached: bool = False) -> dict[str, Any]:
    """Build final result dictionary from SMPSO state."""
    pop: dict[str, Any] = {"X": state.X, "F": state.F}
    if state.G is not None and state.constraint_mode != "none":
        pop["G"] = state.G

    # Prefer nondominated leaders archive unless population mode is requested.
    archive_X = state.archive_X
    archive_F = state.archive_F
    if state.archive_manager is not None:
        archive_X, archive_F = state.archive_manager.contents()

    if wants_population_result(state):
        result_X = state.X
        result_F = state.F
        result_G = state.G if state.constraint_mode != "none" else None
    else:
        result_X = archive_X if archive_X is not None and archive_X.size else state.X
        result_F = archive_F if archive_F is not None and archive_F.size else state.F
        result_G = None

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
        "archive": {"X": archive_X, "F": archive_F},
        "population": pop,
    }
    if result_G is not None:
        result["G"] = result_G
    return result


__all__ = ["SMPSOState", "build_smpso_result"]
