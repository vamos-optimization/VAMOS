from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.exceptions import EvaluationError


def _validate_output_matrix(
    name: str,
    value: object,
    *,
    expected_shape: tuple[int, int],
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != expected_shape:
        raise EvaluationError(f"Problem evaluation wrote {name} with shape {arr.shape}; expected {expected_shape}.")
    if not np.isfinite(arr).all():
        bad_count = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
        raise EvaluationError(f"Problem evaluation wrote {name} with {bad_count} non-finite value(s).")
    return arr


def evaluate_population_with_constraints(
    problem: Any,
    X: NDArray[np.generic],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """
    Evaluate population and optionally return constraints G if provided by the problem.
    """
    n = int(X.shape[0])
    n_obj = int(problem.n_obj)
    out = {"F": np.full((n, n_obj), np.nan, dtype=float)}
    n_constraints = getattr(problem, "n_constraints", 0)
    if n_constraints and n_constraints > 0:
        out["G"] = np.full((n, int(n_constraints)), np.nan, dtype=float)
    problem.evaluate(X, out)
    F = _validate_output_matrix("F", out.get("F"), expected_shape=(n, n_obj))
    G_raw = out.get("G")
    if n_constraints and n_constraints > 0:
        G = _validate_output_matrix("G", G_raw, expected_shape=(n, int(n_constraints)))
    else:
        G = None
    return F, G


__all__ = ["evaluate_population_with_constraints"]
