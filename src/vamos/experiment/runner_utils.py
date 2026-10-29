"""
Helper utilities shared by runner components.
"""

from __future__ import annotations

import os

import numpy as np

from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.exceptions import BoundsError, ConfigurationError, ProblemDimensionError
from vamos.foundation.problem.registry import ProblemSelection
from vamos.foundation.problem.types import ProblemProtocol


def validate_problem(problem: ProblemProtocol) -> None:
    if problem.n_var <= 0 or problem.n_obj <= 0:
        raise ProblemDimensionError(
            "Problem must have positive n_var and n_obj.",
            n_var=problem.n_var,
            n_obj=problem.n_obj,
        )
    # Registry problems such as RE, RWA and ZCAT do not declare n_constraints; treat them as unconstrained, as optimize() does.
    if (getattr(problem, "n_constraints", 0) or 0) < 0:
        raise ProblemDimensionError("n_constraints must be >= 0.", n_var=problem.n_var, n_obj=problem.n_obj)
    xl = np.asarray(problem.xl)
    xu = np.asarray(problem.xu)
    if xl.ndim > 1 or xu.ndim > 1:
        raise BoundsError("Problem bounds must be scalars or 1D arrays.")
    if xl.ndim == 1 and xl.shape[0] != problem.n_var:
        raise BoundsError("Lower bounds length must match n_var.")
    if xu.ndim == 1 and xu.shape[0] != problem.n_var:
        raise BoundsError("Upper bounds length must match n_var.")
    if np.any(xl > xu):
        raise BoundsError("Lower bounds must not exceed upper bounds.")
    try:
        encoding = normalize_encoding(problem.encoding)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if encoding == "mixed":
        if not hasattr(problem, "mixed_spec"):
            raise ConfigurationError("Mixed-encoding problems must define 'mixed_spec'.")
        spec = problem.mixed_spec
        spec_keys = set(spec.keys())
        if not {"perm_idx", "real_idx", "int_idx", "cat_idx"} & spec_keys:
            raise ConfigurationError("mixed_spec must define at least one of perm_idx, real_idx, int_idx, or cat_idx.")
        if "real_idx" in spec_keys and not {"real_lower", "real_upper"} <= spec_keys:
            raise ConfigurationError("mixed_spec missing required fields: real_lower, real_upper.")
        if "int_idx" in spec_keys and not {"int_lower", "int_upper"} <= spec_keys:
            raise ConfigurationError("mixed_spec missing required fields: int_lower, int_upper.")
        if "cat_idx" in spec_keys and "cat_cardinality" not in spec_keys:
            raise ConfigurationError("mixed_spec missing required field: cat_cardinality.")


def problem_output_dir(selection: ProblemSelection, config: ExperimentConfig) -> str:
    spec = getattr(selection, "spec", None)
    label = getattr(spec, "label", None) or getattr(spec, "key", "unknown")
    safe = str(label).replace(" ", "_").upper()
    return os.path.join(config.output_root, f"{safe}")


def run_output_dir(selection: ProblemSelection, algorithm_name: str, engine_name: str, seed: int, config: ExperimentConfig) -> str:
    base = problem_output_dir(selection, config)
    return os.path.join(
        base,
        algorithm_name.lower(),
        engine_name.lower(),
        f"seed_{seed}",
    )


__all__ = ["validate_problem", "problem_output_dir", "run_output_dir"]
