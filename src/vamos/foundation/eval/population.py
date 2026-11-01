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
        raise EvaluationError(
            f"Problem evaluation wrote {name} with {bad_count} non-finite value(s). Objectives and constraints must be "
            "finite: give a failed evaluation a large finite penalty or a violated constraint. A NaN can also come "
            "from NaN decision variables, for example produced by invalid operator parameters."
        )
    return arr


def _reject_undeclared_constraints(problem: Any, G: object) -> None:
    """Reject constraint values from a problem that declares no constraints; they would be ignored."""
    if G is None or np.size(np.asarray(G)) == 0:
        return
    raise EvaluationError(
        f"{type(problem).__name__} wrote constraint values G but declares n_constraints=0, so VAMOS would optimize it "
        "as unconstrained. Set n_constraints to the number of constraints (pymoo's n_ieq_constr is not read)."
    )


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
        _reject_undeclared_constraints(problem, G_raw)
        G = None
    return F, G


__all__ = ["evaluate_population_with_constraints"]
