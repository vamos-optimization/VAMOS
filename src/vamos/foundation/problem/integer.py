from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem


def _validate_bounds(n_var: int, lower: np.ndarray[Any, Any], upper: np.ndarray[Any, Any]) -> None:
    if lower.shape != upper.shape or lower.shape[0] != n_var:
        raise ValueError("lower and upper bounds must be 1D arrays of length n_var.")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds.")


class IntegerResourceAllocationProblem(Problem):
    """
    Allocate integer resources to tasks.
    Objective 1: minimize cost.
    Objective 2: maximize diminishing returns (as negative utility).
    """

    def __init__(self, n_var: int = 20, max_per_task: int = 10) -> None:
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        self.n_var = int(n_var)
        self.n_obj = 2
        self.encoding = "integer"
        self.xl = np.zeros(self.n_var, dtype=int)
        self.xu = np.full(self.n_var, max(1, int(max_per_task)), dtype=int)

        rng = np.random.default_rng(321)
        self.task_cost = rng.uniform(0.5, 2.0, size=self.n_var)
        self.task_reward = rng.uniform(1.0, 3.0, size=self.n_var)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        X_int = np.clip(np.rint(X), self.xl, self.xu).astype(int, copy=False)
        cost = X_int @ self.task_cost
        reward = self.task_reward * np.sqrt(X_int)
        total_reward = reward.sum(axis=1)
        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = cost
        F[:, 1] = -total_reward

    def describe(self) -> dict[str, float | int]:
        upper_bounds = np.asarray(self.xu, dtype=int)
        return {
            "n_var": self.n_var,
            "avg_cost": float(self.task_cost.mean()),
            "avg_reward": float(self.task_reward.mean()),
            "max_per_task": int(upper_bounds[0]),
        }


class IntegerJobAssignmentProblem(Problem):
    """
    Assign job types (integer labels) to positions with penalties.
    Objective 1: minimize mismatch cost to preferred types.
    Objective 2: minimize diversity penalty (encourage spread across types).
    """

    def __init__(self, n_positions: int = 30, n_job_types: int = 5) -> None:
        if n_positions <= 0 or n_job_types <= 1:
            raise ValueError("n_positions must be positive and n_job_types > 1.")
        self.n_var = int(n_positions)
        self.n_obj = 2
        self.encoding = "integer"
        self.n_job_types = int(n_job_types)
        self.xl = np.zeros(self.n_var, dtype=int)
        self.xu = np.full(self.n_var, self.n_job_types - 1, dtype=int)

        rng = np.random.default_rng(99)
        self.preferences = rng.integers(0, self.n_job_types, size=self.n_var, dtype=int)
        self.mismatch_penalty = rng.uniform(0.5, 2.5, size=self.n_var)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        X_int = np.clip(np.rint(X), self.xl, self.xu).astype(int, copy=False)
        mismatch = (X_int != self.preferences).astype(float)
        mismatch_cost = mismatch @ self.mismatch_penalty

        # Diversity: penalize dominance of a single job type.
        counts = np.stack([(X_int == t).sum(axis=1) for t in range(self.n_job_types)], axis=1)
        max_share = counts.max(axis=1) / float(self.n_var)
        diversity_penalty = max_share

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = mismatch_cost
        F[:, 1] = diversity_penalty

    def describe(self) -> dict[str, float | int]:
        return {
            "n_var": self.n_var,
            "n_job_types": self.n_job_types,
            "avg_mismatch_penalty": float(self.mismatch_penalty.mean()),
        }


__all__ = [
    "IntegerResourceAllocationProblem",
    "IntegerJobAssignmentProblem",
]
