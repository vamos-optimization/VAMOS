# problem/zdt1.py
from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem


class ZDT1Problem(Problem):
    def __init__(self, n_var: int) -> None:
        if int(n_var) < 2:
            raise ValueError("ZDT1 requires at least two decision variables.")
        self.n_var = int(n_var)
        self.n_obj = 2
        # Bounds (identical for all decision variables in this problem)
        self.xl = 0.0
        self.xu = 1.0

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        """
        X: array shape (N, n_var)
        out["F"]: array shape (N, 2)
        """
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = f1
        F[:, 1] = f2
