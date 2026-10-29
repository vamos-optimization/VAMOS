from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

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


def parse_termination(
    termination: tuple[str, Any],
    algorithm_name: str = "algorithm",
) -> tuple[int, dict[str, Any] | None]:
    """
    Parse termination criterion and return (max_eval, hv_config).

    Parameters
    ----------
    termination : tuple[str, Any]
        Termination criterion as (type, value). Supported types:
        - "max_evaluations": value is the max number of evaluations
        - "hv": value is a dict with hypervolume config
    algorithm_name : str
        Algorithm name for error messages.

    Returns
    -------
    tuple[int, dict[str, Any] | None]
        (max_evaluations, hv_config or None)

    Raises
    ------
    ValueError
        If termination type is unsupported or HV config is invalid.
    """
    term_type, term_val = termination
    hv_config = None

    if term_type == "max_evaluations":
        max_eval = int(term_val)
    elif term_type == "hv":
        hv_config = dict(term_val)
        max_eval = int(hv_config.get("max_evaluations", 0))
        if max_eval <= 0:
            raise ValueError(f"HV-based termination for {algorithm_name} requires a positive max_evaluations value.")
    else:
        raise ValueError(f"Unsupported termination criterion '{term_type}' for {algorithm_name}.")

    if max_eval <= 0:
        raise ValueError(f"{algorithm_name} requires max_evaluations to be positive.")
    return max_eval, hv_config


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


__all__ = ["HVTracker", "parse_termination", "validate_initial_budget", "remaining_evaluations", "capped_offspring_size"]
