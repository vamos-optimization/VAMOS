"""SMPSO state container and result building."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.state import AlgorithmState


@dataclass
class SMPSOState(AlgorithmState):
    """State for SMPSO algorithm (ask/tell capable).

    Extends AlgorithmState with PSO-specific fields (velocity, personal bests)
    and the operators needed for the turbulence/mutation step.
    """

    velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    pbest_X: np.ndarray = field(default_factory=lambda: np.array([]))
    pbest_F: np.ndarray = field(default_factory=lambda: np.array([]))
    pbest_G: np.ndarray | None = None

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
    vmax: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_max: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_min: np.ndarray = field(default_factory=lambda: np.array([]))

    xl: np.ndarray = field(default_factory=lambda: np.array([]))
    xu: np.ndarray = field(default_factory=lambda: np.array([]))

    mutation_op: Any = None
    repair_op: Any = None
    archive_crowding: np.ndarray | None = field(default=None, repr=False, compare=False)
    pending_particle_indices: np.ndarray | None = field(default=None, repr=False, compare=False)


def build_smpso_result(state: SMPSOState, hv_reached: bool = False) -> dict[str, Any]:
    """Build final result dictionary from SMPSO state."""
    pop: dict[str, Any] = {"X": state.X, "F": state.F}
    if state.G is not None and state.constraint_mode != "none":
        pop["G"] = state.G

    # Prefer nondominated leaders archive when available.
    archive_X = state.archive_X
    archive_F = state.archive_F
    if state.archive_manager is not None:
        archive_X, archive_F = state.archive_manager.contents()

    result_X = archive_X if archive_X is not None and archive_X.size else state.X
    result_F = archive_F if archive_F is not None and archive_F.size else state.F

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
        "archive": {"X": archive_X, "F": archive_F},
        "population": pop,
    }
    return result


__all__ = ["SMPSOState", "build_smpso_result"]
