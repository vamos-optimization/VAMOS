from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem


class ZDT4Problem(Problem):
    """ZDT4 benchmark introducing multimodality through shifted bounds."""

    def __init__(self, n_var: int) -> None:
        if n_var < 2:
            raise ValueError("ZDT4 requires at least two decision variables.")
        self.n_var = int(n_var)
        self.n_obj = 2
        lower = np.full(self.n_var, -5.0)
        upper = np.full(self.n_var, 5.0)
        lower[0] = 0.0
        upper[0] = 1.0
        self.xl = lower
        self.xu = upper

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected input shape (N, {self.n_var}), got {X.shape}.")
        f1 = X[:, 0]
        if self.n_var > 1:
            x_tail = X[:, 1:]
            g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(x_tail**2 - 10.0 * np.cos(4.0 * np.pi * x_tail), axis=1)
        else:
            g = np.ones(X.shape[0])
        ratio = np.divide(f1, g, out=np.zeros_like(f1), where=g != 0.0)
        f2 = g * (1.0 - np.sqrt(ratio))
        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = f1
        F[:, 1] = f2
