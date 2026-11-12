from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output

from .tanabe_ishibuchi_utils import sum_violations_gte0


class RE31Problem:
    """
    Tanabe & Ishibuchi (2020) RE31 / RE3-3-1: Two bar truss design.
    """

    def __init__(self) -> None:
        self.n_var = 3
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.array([1e-5, 1e-5, 1.0], dtype=float)
        self.xu = np.array([100.0, 100.0, 3.0], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        root1 = np.sqrt(16.0 + x3 * x3)
        root2 = np.sqrt(1.0 + x3 * x3)
        f1 = x1 * root1 + x2 * root2
        f2 = (20.0 * root1) / (x1 * x3)

        g1 = 0.1 - f1
        g2 = 100000.0 - f2
        g3 = 100000.0 - ((80.0 * root2) / (x3 * x2))
        f3 = sum_violations_gte0(g1, g2, g3)

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3


class RE32Problem:
    """
    Tanabe & Ishibuchi (2020) RE32 / RE3-4-2: Welded beam design.
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.array([0.125, 0.1, 0.1, 0.125], dtype=float)
        self.xu = np.array([5.0, 10.0, 10.0, 5.0], dtype=float)

        self._P = 6000.0
        self._L = 14.0
        self._E = 30.0e6
        self._G = 12.0e6
        self._tau_max = 13600.0
        self._sigma_max = 30000.0

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        P = self._P
        L = self._L
        E = self._E
        G = self._G

        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        f2 = (4.0 * P * (L**3)) / (E * x4 * x3 * x3 * x3)

        M = P * (L + (x2 / 2.0))
        R = np.sqrt((x2 * x2) / 4.0 + np.square((x1 + x3) / 2.0))
        J = 2.0 * np.sqrt(2.0) * x1 * x2 * ((x2 * x2) / 12.0 + np.square((x1 + x3) / 2.0))

        tau_dd = (M * R) / J
        tau_d = P / (np.sqrt(2.0) * x1 * x2)
        tau = np.sqrt((tau_d * tau_d) + ((2.0 * tau_d * tau_dd * x2) / (2.0 * R)) + (tau_dd * tau_dd))

        sigma = (6.0 * P * L) / (x4 * x3 * x3)

        tmp1 = 4.013 * E * np.sqrt((x3 * x3 * (x4**6)) / 36.0) / (L * L)
        tmp2 = (x3 / (2.0 * L)) * np.sqrt(E / (4.0 * G))
        PC = tmp1 * (1.0 - tmp2)

        g1 = self._tau_max - tau
        g2 = self._sigma_max - sigma
        g3 = x4 - x1
        g4 = PC - P
        f3 = sum_violations_gte0(g1, g2, g3, g4)

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3


class RE33Problem:
    """
    Tanabe & Ishibuchi (2020) RE33 / RE3-4-3: Disc brake design.
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.array([55.0, 75.0, 1000.0, 11.0], dtype=float)
        self.xu = np.array([80.0, 110.0, 3000.0, 20.0], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        diff2 = x2 * x2 - x1 * x1
        diff3 = x2 * x2 * x2 - x1 * x1 * x1
        f1 = 4.9e-5 * diff2 * (x4 - 1.0)
        f2 = (9.82e6 * diff2) / (x3 * x4 * diff3)

        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * diff2))
        g3 = 1.0 - (2.22e-3 * x3 * diff3) / np.power(diff2, 2.0)
        g4 = (2.66e-2 * x3 * x4 * diff3) / diff2 - 900.0
        f3 = sum_violations_gte0(g1, g2, g3, g4)

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3


class RE34Problem:
    """
    Tanabe & Ishibuchi (2020) RE34 / RE3-5-4: Vehicle crashworthiness design.
    """

    def __init__(self) -> None:
        self.n_var = 5
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.full(self.n_var, 1.0, dtype=float)
        self.xu = np.full(self.n_var, 3.0, dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        f1 = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
        f2 = (
            6.5856
            + (1.15 * x1)
            - (1.0427 * x2)
            + (0.9738 * x3)
            + (0.8364 * x4)
            - (0.3695 * x1 * x4)
            + (0.0861 * x1 * x5)
            + (0.3628 * x2 * x4)
            - (0.1106 * x1 * x1)
            - (0.3437 * x3 * x3)
            + (0.1764 * x4 * x4)
        )
        f3 = (
            -0.0551
            + (0.0181 * x1)
            + (0.1024 * x2)
            + (0.0421 * x3)
            - (0.0073 * x1 * x2)
            + (0.024 * x2 * x3)
            - (0.0118 * x2 * x4)
            - (0.0204 * x3 * x4)
            - (0.008 * x3 * x5)
            - (0.0241 * x2 * x2)
            + (0.0109 * x4 * x4)
        )

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3


__all__ = ["RE31Problem", "RE32Problem", "RE33Problem", "RE34Problem"]
