from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem


class ZDT3Problem(Problem):
    """ZDT3 benchmark with a disconnected Pareto front."""

    def __init__(self, n_var: int) -> None:
        if n_var < 2:
            raise ValueError("ZDT3 requires at least two decision variables.")
        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = 0.0
        self.xu = 1.0

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected input shape (N, {self.n_var}), got {X.shape}.")
        f1 = X[:, 0]
        if self.n_var > 1:
            g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        else:
            g = np.ones(X.shape[0])
        ratio = np.divide(f1, g, out=np.zeros_like(f1), where=g != 0.0)
        f2 = g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1))
        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = f1
        F[:, 1] = f2
