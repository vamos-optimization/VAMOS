from __future__ import annotations

import numpy as np
import pytest

from vamos.experiment.runner_utils import validate_problem
from vamos.foundation.exceptions import BoundsError, ConfigurationError, ProblemDimensionError
from vamos.foundation.problem.base import Problem


class _InvalidDimsProblem(Problem):
    n_var = 0
    n_obj = 2
    xl = np.zeros(0)
    xu = np.zeros(0)

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.zeros((X.shape[0], self.n_obj))


class _BadBoundsProblem(Problem):
    n_var = 2
    n_obj = 2
    xl = np.array([1.0, 0.0])
    xu = np.array([0.0, 1.0])

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.zeros((X.shape[0], self.n_obj))


class _MixedMissingSpecProblem(Problem):
    n_var = 3
    n_obj = 2
    xl = np.zeros(3)
    xu = np.ones(3)
    encoding = "mixed"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.zeros((X.shape[0], self.n_obj))


def test_validate_problem_accepts_reference_problem(zdt1_problem: Problem) -> None:
    validate_problem(zdt1_problem)


def test_validate_problem_rejects_invalid_dimensions() -> None:
    with pytest.raises(ProblemDimensionError, match="positive n_var and n_obj"):
        validate_problem(_InvalidDimsProblem())


def test_validate_problem_rejects_invalid_bounds() -> None:
    with pytest.raises(BoundsError, match="Lower bounds must not exceed upper bounds"):
        validate_problem(_BadBoundsProblem())


def test_validate_problem_requires_mixed_spec() -> None:
    with pytest.raises(ConfigurationError, match="mixed_spec"):
        validate_problem(_MixedMissingSpecProblem())


def test_validate_problem_accepts_every_registry_problem() -> None:
    from vamos.foundation.problem.registry import available_problem_names, make_problem_selection

    rejected: dict[str, str] = {}
    for name in available_problem_names():
        try:
            problem = make_problem_selection(name).instantiate()
        except ImportError:
            continue  # optional dependency (for example scikit-learn) is not installed
        try:
            validate_problem(problem)
        except Exception as exc:  # noqa: BLE001 - report every rejected problem at once
            rejected[name] = f"{type(exc).__name__}: {exc}"
    assert rejected == {}


def test_validate_problem_rejects_negative_constraint_count(zdt1_problem: Problem) -> None:
    zdt1_problem.n_constraints = -1
    with pytest.raises(ProblemDimensionError, match="n_constraints"):
        validate_problem(zdt1_problem)
