"""Structural checks of a problem object before an optimization run."""

from __future__ import annotations

import numbers

import numpy as np

from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.exceptions import ConfigurationError

_PROBLEM_SUGGESTION = (
    "Pass a registered problem name, a problem class, or an object with integer n_var and n_obj, "
    "bounds xl and xu, and an evaluate(X, out) method, such as a vamos.Problem subclass or make_problem(...)."
)


def validate_problem(problem: object) -> None:
    """Check the ``ProblemProtocol`` contract that an optimization run relies on.

    The checks are structural and never evaluate the problem: ``n_var`` and ``n_obj`` are
    positive integers, ``n_constraints`` (default 0) is a non-negative integer, ``encoding``
    (default ``"real"``) is a known encoding, ``evaluate`` is callable, and ``xl`` and ``xu`` are
    finite numbers, each a scalar or an array of shape ``(n_var,)``, with ``xl <= xu``.

    Raises
    ------
    ConfigurationError
        Naming the first attribute that breaks the contract.
    """
    label = type(problem).__name__
    missing = [name for name in ("n_var", "n_obj", "xl", "xu", "evaluate") if not hasattr(problem, name)]
    if missing:
        raise ConfigurationError(
            f"problem must be a registered name, a problem class, or a problem object; {label} has no {', '.join(missing)}.",
            suggestion=_PROBLEM_SUGGESTION,
            details={"problem_type": label, "missing": missing},
        )
    if not callable(getattr(problem, "evaluate", None)):
        raise ConfigurationError(f"{label}.evaluate must be callable as evaluate(X, out).", suggestion=_PROBLEM_SUGGESTION)
    n_var = _count(problem, "n_var", minimum=1)
    _count(problem, "n_obj", minimum=1)
    _count(problem, "n_constraints", minimum=0, default=0)
    try:
        normalize_encoding(getattr(problem, "encoding", None))
    except (AttributeError, ValueError) as exc:
        raise ConfigurationError(f"{label}.encoding: {exc}") from exc
    xl = np.broadcast_to(_bounds(problem, "xl", n_var), (n_var,))
    xu = np.broadcast_to(_bounds(problem, "xu", n_var), (n_var,))
    inverted = np.flatnonzero(xl > xu)
    if inverted.size:
        index = int(inverted[0])
        raise ConfigurationError(
            f"{label}: xl must not exceed xu (variable {index}: xl={xl[index]}, xu={xu[index]}).",
            details={"problem_type": label, "variable": index},
        )


def _count(problem: object, name: str, *, minimum: int, default: int | None = None) -> int:
    value = getattr(problem, name, default)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < minimum:
        requirement = "a positive integer" if minimum >= 1 else "a non-negative integer"
        raise ConfigurationError(f"{type(problem).__name__}.{name} must be {requirement}; got {value!r}.")
    return int(value)


def _bounds(problem: object, name: str, n_var: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    label = type(problem).__name__
    raw = getattr(problem, name)
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label}.{name} must be numeric; got {raw!r}.") from exc
    if values.ndim != 0 and values.shape != (n_var,):
        raise ConfigurationError(f"{label}.{name} must be a scalar or have shape ({n_var},); got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ConfigurationError(f"{label}.{name} must be finite; got {raw!r}.")
    return values


__all__ = ["validate_problem"]
