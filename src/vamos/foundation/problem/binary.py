from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem


def _as_bits(X: np.ndarray[Any, Any], n_var: int) -> np.ndarray[Any, Any]:
    bits = np.asarray(X)
    if bits.ndim != 2 or bits.shape[1] != n_var:
        raise ValueError(f"Expected decision matrix of shape (N, {n_var}).")
    return (bits > 0.5).astype(np.int8, copy=False)


class BinaryFeatureSelectionProblem(Problem):
    """
    Synthetic feature-selection style benchmark.
    Objective 1: maximize predictive utility (minimize negative gain).
    Objective 2: minimize cumulative feature cost.
    """

    def __init__(self, n_var: int = 50) -> None:
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = 0
        self.xu = 1
        self.encoding = "binary"

        rng = np.random.default_rng(12345)
        self.utility = np.abs(rng.normal(loc=1.0, scale=0.5, size=self.n_var)) + 0.05
        self.cost = rng.uniform(0.2, 1.5, size=self.n_var)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        bits = _as_bits(X, self.n_var)
        gain = bits @ self.utility
        complexity = bits @ self.cost
        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = -gain
        F[:, 1] = complexity

    def describe(self) -> dict[str, float | int]:
        return {
            "n_var": self.n_var,
            "utility_mean": float(self.utility.mean()),
            "cost_mean": float(self.cost.mean()),
        }


class BinaryKnapsackProblem(Problem):
    """
    Multi-objective knapsack surrogate.
    Objective 1: stay close to a target capacity (absolute deviation).
    Objective 2: maximize item value (minimize negative value).
    """

    def __init__(self, n_var: int = 50, capacity_ratio: float = 0.4) -> None:
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = 0
        self.xu = 1
        self.encoding = "binary"

        rng = np.random.default_rng(2023)
        self.weights = rng.uniform(1.0, 10.0, size=self.n_var)
        self.values = rng.uniform(1.0, 5.0, size=self.n_var)
        self.capacity = float(capacity_ratio) * float(self.weights.sum())

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        bits = _as_bits(X, self.n_var)
        total_weight = bits @ self.weights
        total_value = bits @ self.values
        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = np.abs(total_weight - self.capacity)
        F[:, 1] = -total_value

    def describe(self) -> dict[str, float | int]:
        return {
            "n_var": self.n_var,
            "capacity": self.capacity,
            "avg_weight": float(self.weights.mean()),
            "avg_value": float(self.values.mean()),
        }


class BinaryQUBOProblem(Problem):
    """
    Quadratic unconstrained binary optimization (QUBO) surrogate.
    Objective 1: minimize quadratic energy x^T Q x + b^T x.
    Objective 2: maximize the number of selected bits (minimize negative count).
    """

    def __init__(self, n_var: int = 30) -> None:
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = 0
        self.xu = 1
        self.encoding = "binary"

        rng = np.random.default_rng(7)
        base = rng.normal(scale=0.5, size=(self.n_var, self.n_var))
        sym = 0.5 * (base + base.T)
        diag_boost = np.diag(np.abs(rng.normal(scale=0.2, size=self.n_var)))
        self.Q = sym + diag_boost
        self.bias = rng.normal(scale=0.2, size=self.n_var)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        bits = _as_bits(X, self.n_var)
        quad = np.einsum("bi,ij,bj->b", bits, self.Q, bits)
        linear = bits @ self.bias
        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = quad + linear
        F[:, 1] = -bits.sum(axis=1)

    def describe(self) -> dict[str, float | int]:
        return {
            "n_var": self.n_var,
            "mean_bias": float(self.bias.mean()),
            "mean_qubo_entry": float(self.Q.mean()),
        }


__all__ = [
    "BinaryFeatureSelectionProblem",
    "BinaryKnapsackProblem",
    "BinaryQUBOProblem",
]
