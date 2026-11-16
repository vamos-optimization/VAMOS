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

    with_constraints(result, state.G)

    archive = _external_archive(state)
    if archive is not None:
        result["archive"] = with_constraints({"X": archive[0], "F": archive[1]}, archive[2])

    if extra:
        result.update(extra)

    return result


def _external_archive(
    state: Any,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None] | None:
    """The rows of the external archive and their constraint values, whatever object holds them."""
    for holder in ("result_archive", "archive_manager"):
        archive = getattr(state, holder, None)
        if archive is not None:
            archive_X, archive_F = archive.contents()
            return archive_X, archive_F, archive.constraint_values()

    archive = getattr(state, "archive", None)
    if archive is not None:
        archive_X = getattr(archive, "X", None)
        archive_F = getattr(archive, "F", None)
        if archive_X is not None and archive_F is not None:
            return archive_X, archive_F, getattr(archive, "G", None)

    archive_X = getattr(state, "archive_X", None)
    archive_F = getattr(state, "archive_F", None)
    if archive_X is not None and archive_F is not None:
        return archive_X, archive_F, getattr(state, "archive_G", None)
    return None


def apply_final_archive_subset(result: dict[str, Any], state: Any) -> dict[str, Any]:
    """Report the final subset of the external archive in ``result``, when the archive's configuration selects one.

    With ``ExternalArchiveConfig.final_selection`` other than ``"none"``, the rows ``X``, ``F`` and, when
    the result has them, ``G`` become the archive's rows that the selection keeps (``final_size`` of
    them, or as many as the population without one), and ``result["archive_subset"]`` records the
    selection, the size and the indices of the kept rows in the archive. The archive itself stays whole
    in the result. Nothing changes without an external archive or with ``result_mode="population"``.
    """
    holder = next((getattr(state, name) for name in ("result_archive", "archive_manager") if getattr(state, name, None) is not None), None)
    config = getattr(holder, "external_config", None)
    if config is None or config.final_selection == "none" or wants_population_result(state):
        return result
    archive = _external_archive(state)
    if archive is None:
        return result
    from vamos.engine.archive.final_subset import select_final_subset

    archive_X, archive_F, archive_G = archive
    size = int(state.pop_size) if config.final_size is None else int(config.final_size)
    # The algorithms resolve rng_seed from the run seed; an unresolved None falls back to 0, as setup_archive does.
    seed = 0 if config.rng_seed is None else config.rng_seed
    indices = select_final_subset(np.asarray(archive_F), config.final_selection, size, rng_seed=seed)
    result["X"] = np.asarray(archive_X)[indices].copy()
    result["F"] = np.asarray(archive_F)[indices].copy()
    if result.get("G") is not None and archive_G is not None:
        result["G"] = np.asarray(archive_G)[indices].copy()
    result["archive_subset"] = {"selection": config.final_selection, "size": size, "indices": indices}
    return result


def get_external_archive_contents(state: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """Return external archive contents regardless of the concrete archive holder."""
    archive = _external_archive(state)
    return None if archive is None else (archive[0], archive[1])


def get_external_archive_constraints(state: Any) -> np.ndarray[Any, Any] | None:
    """Constraint values of the rows of :func:`get_external_archive_contents`, or ``None`` without constraints."""
    archive = _external_archive(state)
    return None if archive is None or archive[2] is None else np.asarray(archive[2])


def with_constraints(payload: dict[str, Any], G: np.ndarray[Any, Any] | None) -> dict[str, Any]:
    """``payload`` (rows ``X`` and ``F``) with the constraint values ``G`` of its rows, when there are any."""
    if G is not None:
        payload["G"] = G
    return payload


def wants_population_result(state: Any) -> bool:
    mode = str(getattr(state, "result_mode", "non_dominated") or "non_dominated").strip().lower()
    return mode == "population"


def result_objectives(state: Any) -> np.ndarray[Any, Any]:
    """Objectives of the rows a result built from ``state`` reports, before non-dominated filtering.

    The external archive unless the result mode is ``population``; the whole population in that
    mode; otherwise the feasible rows, or the least violated ones when none is feasible.
    """
    archive = get_external_archive_contents(state)
    if archive is not None and not wants_population_result(state):
        return np.asarray(archive[1])
    F = np.asarray(state.F)
    if wants_population_result(state):
        return F
    return F[feasible_result_indices(F, getattr(state, "G", None), str(getattr(state, "constraint_mode", "none")))]


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


__all__ = [
    "apply_final_archive_subset",
    "build_result",
    "feasible_result_indices",
    "get_external_archive_constraints",
    "get_external_archive_contents",
    "wants_population_result",
    "with_constraints",
]
