from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from vamos.engine.algorithm.components.results import result_objectives
from vamos.foundation.quality_indicators.hypervolume import hypervolume

_FloatArray = npt.NDArray[np.float64]


class HVTracker:
    """Hypervolume target check against the user's fixed reference point.

    Rows that do not weakly dominate the reference point contribute nothing and
    are dropped before the hypervolume is computed, as in moocore, pymoo and
    jMetal. A population with no row inside the reference box has HV 0.
    """

    def __init__(self, hv_config: dict[str, Any] | None, kernel: Any) -> None:
        self.enabled = hv_config is not None
        self.last_value: float | None = None
        if not self.enabled:
            return
        assert hv_config is not None
        self.target = float(hv_config["target_value"])
        self.ref_point: _FloatArray = np.asarray(hv_config["reference_point"], dtype=float)
        if "allow_ref_expand" in hv_config:
            warnings.warn(
                "The 'allow_ref_expand' key of the 'hv' termination criterion is ignored: the target is always "
                "measured against the fixed reference point, and rows outside it are dropped. "
                "The key will be rejected in VAMOS 2.0.",
                FutureWarning,
                stacklevel=2,
            )
        self.evaluator: Callable[[_FloatArray, _FloatArray], float]
        if kernel is not None and kernel.supports_quality_indicator("hypervolume"):
            self.evaluator = kernel.hypervolume
        else:
            self.evaluator = hypervolume

    def reached(self, points: npt.ArrayLike) -> bool:
        if not self.enabled:
            return False
        F = np.asarray(points, dtype=float)
        inside = F[np.all(F <= self.ref_point, axis=1)]
        hv_val = float(self.evaluator(inside, self.ref_point)) if inside.shape[0] else 0.0
        self.last_value = hv_val
        return hv_val >= self.target


TERMINATION_CRITERIA = ("max_evaluations", "n_gen", "hv")


def parse_budget(value: Any, label: str, algorithm_name: str = "algorithm") -> int:
    """Return ``value`` as a positive evaluation or generation count.

    Integers, and floats with an integral value such as ``1e4``, are accepted.
    Booleans, fractional values and counts below 1 raise ``ValueError`` naming
    ``label``: a budget is never truncated.
    """
    if isinstance(value, bool):
        parsed = None
    elif isinstance(value, numbers.Integral):
        parsed = int(value)
    elif isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value).is_integer():
        parsed = int(float(value))
    else:
        parsed = None
    if parsed is None or parsed <= 0:
        raise ValueError(f"{algorithm_name} requires {label} to be a positive integer, got {value!r}.")
    return parsed


def parse_termination(
    termination: tuple[str, Any],
    algorithm_name: str = "algorithm",
    pop_size: int | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """
    Parse termination criterion and return (max_eval, hv_config).

    Every built-in algorithm accepts the same criteria.

    Parameters
    ----------
    termination : tuple[str, Any]
        Termination criterion as (type, value). Supported types:
        - "max_evaluations": value is the total number of evaluations
        - "n_gen": value is a number of generations of ``pop_size`` evaluations,
          the initial population included, so the budget is ``n_gen * pop_size``
        - "hv": value is a dict with ``target_value``, ``reference_point`` and the
          ``max_evaluations`` cap
        Counts must be positive integers; a fractional count is rejected, not truncated.
    algorithm_name : str
        Algorithm name for error messages.
    pop_size : int | None
        Population size, which ``n_gen`` needs.

    Returns
    -------
    tuple[int, dict[str, Any] | None]
        (max_evaluations, hv_config or None)

    Raises
    ------
    ValueError
        If termination type is unsupported, a count is not a positive integer,
        or the HV config is invalid.
    """
    term_type, term_val = termination
    hv_config = None

    if term_type == "max_evaluations":
        max_eval = parse_budget(term_val, "max_evaluations", algorithm_name)
    elif term_type == "n_gen":
        if pop_size is None:
            raise ValueError(f"The 'n_gen' termination of {algorithm_name} needs the population size.")
        max_eval = parse_budget(term_val, "n_gen", algorithm_name) * int(pop_size)
    elif term_type == "hv":
        if not isinstance(term_val, Mapping):
            raise ValueError(f"HV-based termination for {algorithm_name} takes a mapping with target_value and reference_point.")
        hv_config = dict(term_val)
        if hv_config.get("max_evaluations") is None:
            raise ValueError(f"HV-based termination for {algorithm_name} requires a positive max_evaluations value.")
        max_eval = parse_budget(hv_config["max_evaluations"], "max_evaluations", f"HV-based termination for {algorithm_name}")
    else:
        raise ValueError(
            f"Unsupported termination criterion {term_type!r} for {algorithm_name}; use one of {', '.join(TERMINATION_CRITERIA)}."
        )
    return max_eval, hv_config


def hv_target_reached(state: Any) -> bool:
    """Whether the state's HV tracker, when an ``("hv", ...)`` termination enables it, has reached its target.

    The target is measured on the rows that the result reports (``result_objectives``), so an
    infeasible row the result drops, or an archive that a population result ignores, cannot stop
    the run early.
    """
    tracker = getattr(state, "hv_tracker", None)
    return bool(tracker is not None and tracker.enabled and tracker.reached(result_objectives(state)))


def validate_initial_budget(max_evaluations: int, pop_size: int, algorithm_name: str) -> None:
    """Ensure the budget can cover the initial population."""
    if max_evaluations < pop_size:
        raise ValueError(
            f"{algorithm_name} requires max_evaluations >= pop_size because the initial population "
            f"consumes pop_size evaluations (max_evaluations={max_evaluations}, pop_size={pop_size})."
        )


def remaining_evaluations(n_eval: int, max_evaluations: int, algorithm_name: str) -> int:
    """Return remaining evaluation budget or fail if ask() is called after termination."""
    remaining = int(max_evaluations) - int(n_eval)
    if remaining <= 0:
        raise RuntimeError(f"{algorithm_name} has no remaining evaluation budget; call result() instead of ask().")
    return remaining


def capped_offspring_size(n_eval: int, max_evaluations: int, requested: int, algorithm_name: str) -> int:
    """Cap an offspring/batch request to the remaining evaluation budget."""
    return min(int(requested), remaining_evaluations(n_eval, max_evaluations, algorithm_name))


__all__ = [
    "TERMINATION_CRITERIA",
    "HVTracker",
    "capped_offspring_size",
    "hv_target_reached",
    "parse_budget",
    "parse_termination",
    "remaining_evaluations",
    "validate_initial_budget",
]
