"""
Result assembly helpers for engine algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.state import AlgorithmState


def build_result(
    state: AlgorithmState,
    hv_reached: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the result dictionary from algorithm state.

    Parameters
    ----------
    state : AlgorithmState
        Current algorithm state.
    hv_reached : bool
        Whether HV threshold was reached.
    extra : dict[str, Any] | None
        Additional result fields.

    Returns
    -------
    dict[str, Any]
        Result dictionary with X, F, evaluations, and optional archive.
    """
    result: dict[str, Any] = {
        "X": state.X,
        "F": state.F,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
    }

    if state.G is not None:
        result["G"] = state.G

    if state.archive_manager is not None:
        arch_X, arch_F = state.archive_manager.contents()
        result["archive"] = {"X": arch_X, "F": arch_F}
    elif state.archive_X is not None and state.archive_F is not None:
        result["archive"] = {"X": state.archive_X, "F": state.archive_F}

    if extra:
        result.update(extra)

    return result


def get_external_archive_contents(state: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """Return external archive contents regardless of the concrete archive holder."""
    result_archive = getattr(state, "result_archive", None)
    if result_archive is not None:
        archive_X, archive_F = result_archive.contents()
        return archive_X, archive_F

    archive_manager = getattr(state, "archive_manager", None)
    if archive_manager is not None:
        archive_X, archive_F = archive_manager.contents()
        return archive_X, archive_F

    archive = getattr(state, "archive", None)
    if archive is not None:
        archive_X = getattr(archive, "X", None)
        archive_F = getattr(archive, "F", None)
        if archive_X is not None and archive_F is not None:
            return archive_X, archive_F

    archive_X = getattr(state, "archive_X", None)
    archive_F = getattr(state, "archive_F", None)
    if archive_X is not None and archive_F is not None:
        return archive_X, archive_F
    return None


def wants_population_result(state: Any) -> bool:
    mode = str(getattr(state, "result_mode", "non_dominated") or "non_dominated").strip().lower()
    return mode == "population"


def feasible_result_indices(
    F: np.ndarray[Any, Any], G: np.ndarray[Any, Any] | None, constraint_mode: str, kernel: Any = None
) -> NDArray[np.int_]:
    """Rows of a population that form its non-dominated result, feasible solutions first.

    With constraint handling, the candidates are the feasible rows or, when none
    is feasible, the least violated ones; without it, every row. ``kernel``
    keeps only the candidates that no other candidate dominates.
    """
    candidates: NDArray[np.int_] = np.arange(F.shape[0], dtype=int)
    if G is not None and constraint_mode != "none" and G.shape[0]:
        feasible = is_feasible(G, n=G.shape[0])
        if feasible.any():
            candidates = np.flatnonzero(feasible)
        else:
            violation = compute_violation(G, n=G.shape[0])
            candidates = np.flatnonzero(violation <= float(np.min(violation)) + 1e-12)
    if kernel is None or candidates.size == 0:
        return candidates
    ranks, _ = kernel.nsga2_ranking(F[candidates])
    return np.asarray(candidates[np.asarray(ranks) == 0], dtype=int)


__all__ = ["build_result", "feasible_result_indices", "get_external_archive_contents", "wants_population_result"]
