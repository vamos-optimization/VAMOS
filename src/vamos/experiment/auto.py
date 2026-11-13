"""
Auto-optimization helpers for the unified optimize entry point.

These utilities derive the quick defaults of ``optimize`` (algorithm,
population size, max evaluations) from lightweight problem metadata. They
suit a first run, not a converged front: 3000 evaluations of a population of
200 on ZDT1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.problem.registry import make_problem_selection


def _resolve_problem(
    problem: Any,
    *,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve problem from string, class, or instance."""
    if isinstance(problem, str):
        kwargs = dict(problem_kwargs or {})
        if n_var is not None:
            kwargs["n_var"] = n_var
        if n_obj is not None:
            kwargs["n_obj"] = n_obj
        selection = make_problem_selection(problem, **kwargs)
        return selection.instantiate()
    if isinstance(problem, type):
        kwargs = dict(problem_kwargs or {})
        if n_var is not None:
            kwargs["n_var"] = n_var
        if n_obj is not None:
            kwargs["n_obj"] = n_obj
        if kwargs:
            return problem(**kwargs)
        return problem()
    return problem


def _select_algorithm(n_obj: int, encoding: str) -> str:
    """Select algorithm based on problem characteristics."""
    encoding = normalize_encoding(encoding)
    if n_obj == 1:
        raise ConfigurationError(
            "algorithm='auto' only supports multi-objective problems with n_obj >= 2.",
            suggestion="Select an explicit algorithm or provide a problem with at least 2 objectives.",
            details={"n_obj": n_obj, "encoding": encoding},
        )
    elif n_obj == 2:
        return "nsgaii"
    elif n_obj >= 3:
        return "nsgaiii"
    return "nsgaii"


def suggest_algorithm(
    problem: Any | None = None,
    *,
    n_var: int | None = None,
    n_obj: int | None = None,
    encoding: str | None = None,
    problem_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Suggest an algorithm without running optimization."""
    if problem is None:
        if n_obj is None:
            raise ConfigurationError("suggest_algorithm() requires n_obj when problem is not provided.")
        return _select_algorithm(n_obj, encoding or "real")

    problem_instance = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, problem_kwargs=problem_kwargs)
    inferred_n_obj = getattr(problem_instance, "n_obj", None)
    if inferred_n_obj is None:
        if n_obj is None:
            raise ConfigurationError("suggest_algorithm() could not infer n_obj from problem; provide n_obj explicitly.")
        inferred_n_obj = n_obj
    inferred_encoding = getattr(problem_instance, "encoding", encoding or "real")
    return _select_algorithm(inferred_n_obj, inferred_encoding)


def _compute_pop_size(n_var: int, n_obj: int) -> int:
    """Compute reasonable population size."""
    base = max(50, 10 * n_var)
    # For many-objective, prefer larger populations
    if n_obj >= 3:
        base = max(base, 100)
    return min(base, 200)


def _compute_max_evaluations(n_var: int, n_obj: int) -> int:
    """Compute reasonable max evaluations."""
    base = max(1000, 100 * n_var)
    if n_obj >= 3:
        base = max(base, 5000)
    return min(base, 50000)
