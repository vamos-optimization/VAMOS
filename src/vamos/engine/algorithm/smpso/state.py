"""SMPSO state container and result building."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.state import AlgorithmState

if TYPE_CHECKING:
    from vamos.engine.archive.factory import ArchiveManager


@dataclass
class SMPSOState(AlgorithmState):
    """State for SMPSO algorithm (ask/tell capable).

    Extends AlgorithmState with PSO-specific fields (velocity, personal bests)
    and the operators needed for the turbulence/mutation step.
    """

    velocity: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    pbest_X: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    pbest_F: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    pbest_G: np.ndarray[Any, Any] | None = None

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
    vmax: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    delta_max: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    delta_min: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))

    xl: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    xu: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))

    mutation_op: Any = None
    repair_op: Any = None
    archive_crowding: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)
    pending_particle_indices: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)

    # Optional configured result archive. The inherited archive_* fields remain
    # the intrinsic SMPSO leaders archive used by the PSO dynamics.
    result_archive: ArchiveManager | None = field(default=None, repr=False, compare=False)


def build_smpso_result(state: SMPSOState, hv_reached: bool = False) -> dict[str, Any]:
    """Build final result dictionary from SMPSO state."""
    pop: dict[str, Any] = {"X": state.X, "F": state.F}
    population_G = state.G if state.G is not None and state.constraint_mode != "none" else None
    if population_G is not None:
        pop["G"] = population_G

    # The inherited archive is SMPSO's intrinsic leaders archive and remains
    # visible as ``archive`` for backward compatibility and algorithm tracing.
    archive_X = state.archive_X
    archive_F = state.archive_F
    if state.archive_manager is not None:
        archive_X, archive_F = state.archive_manager.contents()

    external_X: np.ndarray[Any, Any] | None = None
    external_F: np.ndarray[Any, Any] | None = None
    if state.result_archive is not None:
        external_X, external_F = state.result_archive.contents()

    result_G: np.ndarray[Any, Any] | None = None
    mode = str(state.result_mode or "non_dominated").strip().lower()
    if mode == "population":
        result_X = state.X
        result_F = state.F
        result_G = population_G
    elif external_X is not None and external_F is not None and external_F.size:
        result_X = external_X
        result_F = external_F
    elif archive_X is not None and archive_F is not None and archive_F.size:
        result_X = archive_X
        result_F = archive_F
    else:
        result_X = state.X
        result_F = state.F
        result_G = population_G

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
    if state.result_archive is not None:
        result["external_archive"] = {"X": external_X, "F": external_F}
    return result


__all__ = ["SMPSOState", "build_smpso_result"]
