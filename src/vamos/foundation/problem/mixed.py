from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem


def _split_counts(n_var: int) -> tuple[int, int, int]:
    # Ensure at least one variable of each type.
    base = max(3, n_var)
    n_real = max(1, base // 3)
    n_int = max(1, base // 3)
    n_cat = base - n_real - n_int
    if n_cat <= 0:
        n_cat = 1
        n_real = max(1, n_real - 1)
    return n_real, n_int, n_cat


class MixedDesignProblem(Problem):
    """
    Synthetic mixed-encoding benchmark with real, integer, and categorical variables.

    Objective 1: weighted squared error to target real/int values + category mismatch penalties.
    Objective 2: linear cost combining ints and category-specific costs.
    """

    def __init__(self, n_var: int = 9) -> None:
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        n_real, n_int, n_cat = _split_counts(n_var)
        self.n_real = n_real
        self.n_int = n_int
        self.n_cat = n_cat
        self.n_var = n_real + n_int + n_cat
        self.n_obj = 2
        self.encoding = "mixed"

        # Indices
        real_idx = np.arange(0, n_real, dtype=int)
        int_idx = np.arange(n_real, n_real + n_int, dtype=int)
        cat_idx = np.arange(n_real + n_int, self.n_var, dtype=int)

        rng = np.random.default_rng(123)
        real_lower = np.zeros(n_real)
        real_upper = np.ones(n_real)
        int_lower = np.zeros(n_int, dtype=int)
        int_upper = rng.integers(3, 8, size=n_int, dtype=int)
        cat_cardinality = rng.integers(3, 7, size=n_cat, dtype=int)

        self.real_target = rng.uniform(0.2, 0.8, size=n_real)
        self.int_target = (int_upper // 2) + rng.integers(0, 2, size=n_int)
        self.cat_target = np.array([rng.integers(0, c) for c in cat_cardinality], dtype=int)
        self.cat_cost = rng.uniform(0.5, 1.5, size=n_cat)

        # Aggregate bounds for compatibility.
        xl = np.concatenate([real_lower, int_lower, np.zeros(n_cat, dtype=float)])
        xu = np.concatenate([real_upper, int_upper.astype(float), cat_cardinality.astype(float) - 1.0])
        self.xl = xl
        self.xu = xu

        self.mixed_spec = {
            "real_idx": real_idx,
            "int_idx": int_idx,
            "cat_idx": cat_idx,
            "real_lower": real_lower,
            "real_upper": real_upper,
            "int_lower": int_lower,
            "int_upper": int_upper,
            "cat_cardinality": cat_cardinality,
        }
        self._validate_mixed_spec(self.mixed_spec)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")

        real_idx = self.mixed_spec["real_idx"]
        int_idx = self.mixed_spec["int_idx"]
        cat_idx = self.mixed_spec["cat_idx"]

        # Cast and clip to type domains.
        X_real = np.asarray(X[:, real_idx], dtype=float)
        X_int = np.clip(np.rint(X[:, int_idx]), self.mixed_spec["int_lower"], self.mixed_spec["int_upper"]).astype(int)
        X_cat = np.mod(np.rint(X[:, cat_idx]), self.mixed_spec["cat_cardinality"]).astype(int)

        # Objective 1: deviation from targets.
        real_err = np.square(X_real - self.real_target).sum(axis=1)
        int_err = np.square(X_int - self.int_target).sum(axis=1)
        cat_mismatch = (X_cat != self.cat_target).sum(axis=1)
        obj1 = real_err + int_err + cat_mismatch

        # Objective 2: cost (ints weighted, category-specific cost).
        int_cost = (X_int * (1.0 + 0.2 * self.int_target)).sum(axis=1)
        cat_cost = (X_cat * self.cat_cost).sum(axis=1)
        obj2 = int_cost + cat_cost

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = obj1
        F[:, 1] = obj2

    def describe(self) -> dict[str, int | float]:
        return {
            "n_var": self.n_var,
            "n_real": self.n_real,
            "n_int": self.n_int,
            "n_cat": self.n_cat,
        }

    @staticmethod
    def _validate_mixed_spec(spec: dict[str, np.ndarray[Any, Any]]) -> None:
        required = (
            "real_idx",
            "int_idx",
            "cat_idx",
            "real_lower",
            "real_upper",
            "int_lower",
            "int_upper",
            "cat_cardinality",
        )
        for key in required:
            if key not in spec:
                raise ValueError(f"mixed_spec missing required field '{key}'.")
        real_idx = np.asarray(spec["real_idx"], dtype=int)
        int_idx = np.asarray(spec["int_idx"], dtype=int)
        cat_idx = np.asarray(spec["cat_idx"], dtype=int)
        perm_idx = np.asarray(spec.get("perm_idx", []), dtype=int)
        if real_idx.ndim != 1 or int_idx.ndim != 1 or cat_idx.ndim != 1 or perm_idx.ndim != 1:
            raise ValueError("mixed_spec indices must be 1D arrays.")
        all_idx = np.concatenate([real_idx, int_idx, cat_idx, perm_idx])
        if np.unique(all_idx).size != all_idx.size:
            raise ValueError("mixed_spec indices must be non-overlapping.")
        if all_idx.size == 0:
            raise ValueError("mixed_spec must contain at least one decision variable.")
        if np.any(all_idx < 0):
            raise ValueError("mixed_spec indices must be non-negative.")
        max_idx = int(np.max(all_idx))
        if max_idx + 1 != all_idx.size:
            raise ValueError("mixed_spec indices must cover variables densely from 0..n_var-1.")
        n_real = real_idx.size
        n_int = int_idx.size
        n_cat = cat_idx.size
        real_lower = np.asarray(spec["real_lower"], dtype=float)
        real_upper = np.asarray(spec["real_upper"], dtype=float)
        if real_lower.shape[0] != n_real or real_upper.shape[0] != n_real:
            raise ValueError("real_lower/real_upper lengths must match real_idx size.")
        if np.any(real_lower > real_upper):
            raise ValueError("real_lower must be <= real_upper elementwise.")
        int_lower = np.asarray(spec["int_lower"], dtype=int)
        int_upper = np.asarray(spec["int_upper"], dtype=int)
        if int_lower.shape[0] != n_int or int_upper.shape[0] != n_int:
            raise ValueError("int_lower/int_upper lengths must match int_idx size.")
        if np.any(int_lower > int_upper):
            raise ValueError("int_lower must be <= int_upper elementwise.")
        cat_card = np.asarray(spec["cat_cardinality"], dtype=int)
        if cat_card.shape[0] != n_cat:
            raise ValueError("cat_cardinality length must match cat_idx size.")
        if np.any(cat_card <= 0):
            raise ValueError("cat_cardinality values must be positive.")


__all__ = ["MixedDesignProblem"]
