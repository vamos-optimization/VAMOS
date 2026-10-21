"""SMSEMOA-specific state container.

This module provides the state dataclass for SMSEMOA's ask/tell interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.results import feasible_result_indices, get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState


@dataclass
class SMSEMOAState(AlgorithmState):
    """State container for SMSEMOA with ask/tell interface.

    Extends the base AlgorithmState with SMSEMOA-specific fields for
    reference point management and variation operators.

    Additional Attributes
    ---------------------
    ref_point : np.ndarray[Any, Any]
        Current reference point for hypervolume computation.
    ref_offset : float
        Offset for adaptive reference point updates.
    ref_adaptive : bool
        Whether to adaptively update reference point.
    pressure : int
        Tournament selection pressure.
    crossover_fn : Callable
        Crossover operator function.
    mutation_fn : Callable
        Mutation operator function.
    """

    ref_point: np.ndarray[Any, Any] = field(default_factory=lambda: np.array([]))
    ref_offset: float = 0.1
    ref_adaptive: bool = True
    pressure: int = 2
    selection_method: str = "random"
    eliminate_duplicates: bool = False
    crossover_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | None = None
    mutation_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | None = None
    fronts: list[list[int]] | None = None
    ranks: np.ndarray[Any, Any] | None = None
    crowding: np.ndarray[Any, Any] | None = None

    # Workspace buffers (avoid per-iteration allocations in steady-state SMS-EMOA)
    _survival_F: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)
    _survival_G: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)


def build_smsemoa_result(
    state: SMSEMOAState,
    hv_reached: bool = False,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build SMSEMOA result dictionary from state.

    Parameters
    ----------
    state : SMSEMOAState
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict
        Result dictionary with X, F, G, reference_point, population, archive data,
        and metadata. X and F contain only non-dominated solutions when kernel is provided.
    """
    archive_contents = get_external_archive_contents(state)
    if archive_contents is not None and not wants_population_result(state):
        archive_X, archive_F = archive_contents
        result_X = archive_X.copy()
        result_F = archive_F.copy()
        result_G = None
    else:
        should_filter = kernel is not None and not wants_population_result(state)
        if should_filter:
            try:
                selected = feasible_result_indices(state.F, state.G, state.constraint_mode, kernel)
                result_X = state.X[selected].copy()
                result_F = state.F[selected].copy()
                result_G = state.G[selected].copy() if state.G is not None else None
            except (ValueError, IndexError):
                result_X = state.X.copy()
                result_F = state.F.copy()
                result_G = state.G.copy() if state.G is not None else None
        else:
            result_X = state.X.copy()
            result_F = state.F.copy()
            result_G = state.G.copy() if state.G is not None else None

    result = {
        "X": result_X,
        "F": result_F,
        "G": result_G,
        "reference_point": state.ref_point.copy(),
        "evaluations": state.n_eval,
        "generation": state.generation,
        "hv_converged": hv_reached,
        "population": {"X": state.X.copy(), "F": state.F.copy()},
    }

    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive"] = {"X": archive_X.copy(), "F": archive_F.copy()}

    return result
