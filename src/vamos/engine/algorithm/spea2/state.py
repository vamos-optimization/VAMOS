"""SPEA2-specific state container.

Extends AlgorithmState with SPEA2-specific fields for internal archive management.

Note: SPEA2 has two archives:
1. Internal archive (env_X, env_F, env_G): From environmental selection, used for mating
2. External archive (from base AlgorithmState): Optional crowding/hypervolume archive
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.results import feasible_result_indices, get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState


@dataclass
class SPEA2State(AlgorithmState):
    """State for SPEA2 algorithm.

    Extends AlgorithmState with SPEA2-specific fields:
    - env_X, env_F, env_G: Internal SPEA2 archive (from environmental selection)
    - env_archive_size: Target size for environmental selection
    - k_neighbors: k for k-th nearest neighbor distance calculation
    - variation: VariationPipeline for offspring generation
    - xl, xu: Bounds for offspring generation

    The base class archive_X/archive_F/archive_manager are used for the optional
    external archive (crowding/hypervolume based).
    """

    # SPEA2-specific internal archive (from environmental selection)
    env_X: np.ndarray[Any, Any] | None = None
    env_F: np.ndarray[Any, Any] | None = None
    env_G: np.ndarray[Any, Any] | None = None
    env_archive_size: int = 100

    # Algorithm parameters
    k_neighbors: int | None = None

    # Variation operators (callables)
    crossover_fn: Callable[[np.ndarray[Any, Any], np.random.Generator], np.ndarray[Any, Any]] | None = None
    mutation_fn: Callable[[np.ndarray[Any, Any], np.random.Generator], np.ndarray[Any, Any]] | None = None
    xl: np.ndarray[Any, Any] | None = None
    xu: np.ndarray[Any, Any] | None = None
    _fused_offspring: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)
    selection_raw_fitness: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)
    selection_density: np.ndarray[Any, Any] | None = field(default=None, repr=False, compare=False)


def build_spea2_result(state: SPEA2State, hv_reached: bool = False) -> dict[str, Any]:
    """Build final result dictionary from SPEA2 state.

    Parameters
    ----------
    state : SPEA2State
        The algorithm state.
    hv_reached : bool
        Whether HV termination was triggered.

    Returns
    -------
    dict
        Result dictionary with X, F, evaluations, archive, population,
        and optionally G for constrained problems.
    """
    archive_contents = get_external_archive_contents(state)
    if wants_population_result(state):
        result_X = state.X
        result_F = state.F
        result_G = state.G if state.constraint_mode != "none" else None
    elif archive_contents is not None:
        result_X, result_F = archive_contents
        result_G = None
    else:
        assert state.env_X is not None
        assert state.env_F is not None
        if state.env_G is not None and state.constraint_mode != "none":
            # The environmental archive is the result; keep its feasible members when it has any.
            selected = feasible_result_indices(state.env_F, state.env_G, state.constraint_mode)
            result_X, result_F, result_G = state.env_X[selected], state.env_F[selected], state.env_G[selected]
        else:
            result_X, result_F, result_G = state.env_X, state.env_F, None

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
    }
    kernel_profiler = getattr(state, "kernel_profiler", None)
    if kernel_profiler is not None:
        result["kernel_profile"] = kernel_profiler.summary()
    if result_G is not None:
        result["G"] = result_G
    result["archive"] = {"X": state.env_X, "F": state.env_F}
    result["population"] = {"X": state.X, "F": state.F}

    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["external_archive"] = {"X": archive_X, "F": archive_F}

    return result


__all__ = [
    "SPEA2State",
    "build_spea2_result",
]
