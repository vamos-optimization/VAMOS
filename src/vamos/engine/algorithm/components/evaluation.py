"""Evaluation results handed to ``tell``: normalized and validated before any state changes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def extract_evaluation_arrays(
    eval_result: Any,
    constraint_mode: str,
    n_constraints: int | None = None,
    *,
    algorithm: str,
    n_rows: int | None = None,
    n_obj: int | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Return the objective and constraint arrays of an evaluation result.

    ``eval_result`` is an object with ``F`` and an optional ``G``, a mapping
    with ``"F"`` and an optional ``"G"``, or the objective array itself. ``F``
    becomes a 2-D float array. ``G`` is dropped when constraint handling is off
    (``constraint_mode="none"``) or ``n_constraints`` is zero, and is required,
    with ``n_constraints`` columns and one row per row of ``F``, when
    ``n_constraints`` is positive. ``n_rows`` and ``n_obj`` check the shape of
    ``F``. The function changes nothing, so a caller can validate before it
    updates its state; every problem raises ``ValueError`` naming ``algorithm``.
    """
    raw_g: Any = None
    if hasattr(eval_result, "F"):
        raw_f = eval_result.F
        raw_g = getattr(eval_result, "G", None)
    elif isinstance(eval_result, Mapping):
        if "F" not in eval_result:
            raise ValueError(f"{algorithm} evaluation result has no objective values 'F'.")
        raw_f = eval_result["F"]
        raw_g = eval_result.get("G")
    else:
        raw_f = eval_result
    if raw_f is None:
        raise ValueError(f"{algorithm} evaluation result has no objective values F.")

    F = np.asarray(raw_f, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    if F.ndim != 2:
        raise ValueError(f"{algorithm} expected 2-D objective values, got shape {F.shape}.")
    if n_rows is not None and F.shape[0] != n_rows:
        raise ValueError(f"{algorithm} expected {n_rows} objective rows, one per pending decision vector, got {F.shape[0]}.")
    if n_obj is not None and F.shape[1] != n_obj:
        raise ValueError(f"{algorithm} expected {n_obj} objective columns, got {F.shape[1]}.")

    if constraint_mode == "none":
        return F, None
    if n_constraints is not None and n_constraints <= 0:
        return F, None
    if raw_g is None:
        if n_constraints is not None and n_constraints > 0:
            raise ValueError(
                f"{algorithm} requires constraint values G when constraint handling is active "
                f"for a problem declaring {n_constraints} constraint(s)."
            )
        return F, None

    G = np.asarray(raw_g, dtype=float)
    if G.ndim == 1:
        G = G.reshape(-1, 1)
    if G.ndim != 2 or G.shape[0] != F.shape[0]:
        raise ValueError(f"{algorithm} constraint values G must have one row per row of objective values F.")
    if n_constraints is not None and n_constraints > 0 and G.shape[1] != n_constraints:
        raise ValueError(f"{algorithm} expected {n_constraints} constraint column(s) in G, got {G.shape[1]}.")
    return F, G


def combine_constraints(
    current: np.ndarray[Any, Any] | None,
    offspring: np.ndarray[Any, Any] | None,
    constraint_mode: str,
    *,
    algorithm: str,
) -> np.ndarray[Any, Any] | None:
    """Stack the constraint values of a population and its offspring.

    Returns ``None`` when constraint handling is off or neither has values, and
    raises ``ValueError`` when only one of them has values or their column
    counts differ.
    """
    if constraint_mode == "none":
        return None
    if current is None and offspring is None:
        return None
    if current is None or offspring is None:
        raise ValueError(
            f"{algorithm} received inconsistent constraint data across generations. "
            "When constraint handling is active, constrained ask/tell evaluations must provide G."
        )
    if current.shape[1] != offspring.shape[1]:
        raise ValueError(f"{algorithm} constraint column count changed between generations.")
    return np.asarray(np.vstack([current, offspring]), dtype=float)


__all__ = ["combine_constraints", "extract_evaluation_arrays"]
