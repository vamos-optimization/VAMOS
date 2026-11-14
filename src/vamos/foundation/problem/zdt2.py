from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem


class ZDT2Problem(Problem):
    """
    Classic bi-objective benchmark with a concave Pareto front.
    Shares structure with ZDT1 but uses a quadratic term in the second objective.
    """

    def __init__(self, n_var: int) -> None:
        if int(n_var) < 2:
            raise ValueError("ZDT2 requires at least two decision variables.")
        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = 0.0
        self.xu = 1.0

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected input shape (N, {self.n_var}), got {X.shape}.")
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = f1
        F[:, 1] = f2
