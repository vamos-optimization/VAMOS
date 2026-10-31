"""
Shared utility functions for algorithm implementations.

Centralizes common operations like bounds resolution, termination
validation, and other helpers used across multiple algorithms.
Probability expressions are parsed by ``vamos.engine.variation.helpers``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.encoding import EncodingLike, normalize_encoding

if TYPE_CHECKING:
    from vamos.foundation.problem.types import ProblemProtocol


def resolve_bounds_array(
    problem: ProblemProtocol,
    encoding: EncodingLike = "real",
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    """
    Resolve problem bounds to numpy arrays with correct dtype.

    Parameters
    ----------
    problem : ProblemProtocol
        Problem instance with xl, xu, and n_var attributes.
    encoding : str
        Problem encoding type. "integer" uses int dtype, others use float.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (xl, xu) lower and upper bounds as 1D arrays of shape (n_var,).

    Examples
    --------
    >>> problem = ZDT1(n_var=30)
    >>> xl, xu = resolve_bounds_array(problem)
    >>> xl.shape
    (30,)
    """
    normalized = normalize_encoding(encoding)
    bounds_dtype = int if normalized == "integer" else float
    n_var = problem.n_var

    xl = np.asarray(problem.xl, dtype=bounds_dtype)
    xu = np.asarray(problem.xu, dtype=bounds_dtype)

    # Handle scalar bounds
    if xl.ndim == 0:
        xl = np.full(n_var, xl, dtype=bounds_dtype)
    if xu.ndim == 0:
        xu = np.full(n_var, xu, dtype=bounds_dtype)

    return xl, xu


def validate_termination(
    termination: tuple[str, Any],
    supported_types: tuple[str, ...] = ("max_evaluations",),
) -> tuple[str, Any]:
    """
    Validate termination criterion.

    Parameters
    ----------
    termination : tuple[str, Any]
        Termination criterion as (type, value).
    supported_types : tuple[str, ...]
        Supported termination types for the algorithm.

    Returns
    -------
    tuple[str, Any]
        Validated (type, value) tuple.

    Raises
    ------
    ValueError
        If termination type is not supported.
    """
    term_type, term_val = termination
    if term_type not in supported_types:
        raise ValueError(f"Unsupported termination type '{term_type}'. Supported: {', '.join(supported_types)}")
    return term_type, term_val


def parse_operator_config(
    config: tuple[str, dict[str, Any]] | dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Parse operator configuration in various formats.

    Parameters
    ----------
    config : tuple or dict
        Operator config as either:
        - tuple: (method_name, params_dict)
        - dict: {"method": name, **params}

    Returns
    -------
    tuple[str, dict[str, Any]]
        (method_name, params_dict) normalized format.
    """
    if isinstance(config, tuple):
        method, params = config
        return str(method).lower(), dict(params) if params else {}

    if isinstance(config, dict):
        config = dict(config)
        method = str(config.pop("method", config.pop("name", ""))).lower()
        return method, config

    raise ValueError(f"Invalid operator config format: {type(config)}")


def variation_operator_label(config: dict[str, Any], default: str = "variation") -> str:
    """Build a compact ``crossover+mutation`` label from algorithm config."""

    def _method_name(value: Any) -> str:
        if isinstance(value, tuple) and value:
            return str(value[0]).strip().lower()
        if isinstance(value, dict):
            method = value.get("method", value.get("name", value.get("type")))
            if method is not None:
                return str(method).strip().lower()
            return ""
        if isinstance(value, str):
            return value.strip().lower()
        return ""

    cross_name = _method_name(config.get("crossover"))
    mut_name = _method_name(config.get("mutation"))
    if cross_name and mut_name:
        return f"{cross_name}+{mut_name}"
    if cross_name:
        return cross_name
    if mut_name:
        return mut_name
    return default


def compute_ideal_nadir(F: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute ideal and nadir points from objective values.

    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n_solutions, n_obj).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ideal, nadir) points, each of shape (n_obj,).
    """
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    return ideal, nadir


def normalize_objectives(
    F: NDArray[np.float64],
    ideal: NDArray[np.float64] | None = None,
    nadir: NDArray[np.float64] | None = None,
    eps: float = 1e-10,
) -> NDArray[np.float64]:
    """
    Normalize objective values to [0, 1] range.

    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n_solutions, n_obj).
    ideal : np.ndarray | None
        Ideal point. If None, computed from F.
    nadir : np.ndarray | None
        Nadir point. If None, computed from F.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized objectives in [0, 1].
    """
    if ideal is None or nadir is None:
        ideal_computed, nadir_computed = compute_ideal_nadir(F)
        ideal = ideal if ideal is not None else ideal_computed
        nadir = nadir if nadir is not None else nadir_computed

    range_vals = nadir - ideal
    range_vals = np.where(range_vals < eps, eps, range_vals)
    return np.asarray((F - ideal) / range_vals, dtype=float)


__all__ = [
    "resolve_bounds_array",
    "validate_termination",
    "parse_operator_config",
    "variation_operator_label",
    "compute_ideal_nadir",
    "normalize_objectives",
]
