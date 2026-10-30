from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .tanabe_ishibuchi_utils import sum_violations_gte0


class RE21Problem:
    """
    Tanabe & Ishibuchi (2020) RE21 / RE2-4-1: Four bar truss design.

    Objectives (minimize):
      f1: structural volume surrogate
      f2: joint displacement surrogate
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 2
        self.encoding = "continuous"

        F = 10.0
        sigma = 10.0
        a = F / sigma
        self.xl = np.array([a, np.sqrt(2.0) * a, np.sqrt(2.0) * a, a], dtype=float)
        self.xu = np.full(self.n_var, 3.0 * a, dtype=float)

        self._F = F
        self._E = 2.0e5
        self._L = 200.0

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        L = self._L
        Fconst = self._F
        E = self._E

        f1 = L * ((2.0 * x1) + np.sqrt(2.0) * x2 + np.sqrt(np.maximum(x3, 0.0)) + x4)
        f2 = ((Fconst * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))

        Fout = out["F"]
        Fout[:, 0] = f1
        Fout[:, 1] = f2


class RE22Problem:
    """
    Tanabe & Ishibuchi (2020) RE22 / RE2-3-2: Reinforced concrete beam design.

    Mixed variable type: x1 is discrete (reinforcement area), x2,x3 are continuous.

    x1 is a categorical index into the 76 reinforcement areas below. Tanabe and Ishibuchi's ``reproblem.py``
    (and jMetal) search ``x1`` in ``[0.2, 15]`` and round it to the nearest of 77 listed values; their list reads
    "3,10", that is 3 and 10, where VAMOS reads 3.10. The value 10 never reaches the front, and 3.10 adds a short
    segment of it, so the two fronts differ by at most 6e-4 after normalization.

    Objectives (minimize):
      f1: total cost
      f2: sum of constraint violations (g_i(x) >= 0 feasible)
    """

    _X1_VALUES = np.array(
        [
            0.20,
            0.31,
            0.40,
            0.44,
            0.60,
            0.62,
            0.79,
            0.80,
            0.88,
            0.93,
            1.00,
            1.20,
            1.24,
            1.32,
            1.40,
            1.55,
            1.58,
            1.60,
            1.76,
            1.80,
            1.86,
            2.00,
            2.17,
            2.20,
            2.37,
            2.40,
            2.48,
            2.60,
            2.64,
            2.79,
            2.80,
            3.00,
            3.08,
            3.10,
            3.16,
            3.41,
            3.52,
            3.60,
            3.72,
            3.95,
            3.96,
            4.00,
            4.03,
            4.20,
            4.34,
            4.40,
            4.65,
            4.74,
            4.80,
            4.84,
            5.00,
            5.28,
            5.40,
            5.53,
            5.72,
            6.00,
            6.16,
            6.32,
            6.60,
            7.11,
            7.20,
            7.80,
            7.90,
            8.00,
            8.40,
            8.69,
            9.00,
            9.48,
            10.27,
            11.00,
            11.06,
            11.85,
            12.00,
            13.00,
            14.00,
            15.00,
        ],
        dtype=float,
    )

    def __init__(self) -> None:
        self.n_var = 3
        self.n_obj = 2
        self.encoding = "mixed"

        # x0 is categorical index into _X1_VALUES.
        self.xl = np.array([0.0, 0.0, 0.0], dtype=float)
        self.xu = np.array([float(self._X1_VALUES.size - 1), 20.0, 40.0], dtype=float)

        self.mixed_spec: dict[str, NDArray[np.generic]] = {
            "real_idx": np.array([1, 2], dtype=int),
            "int_idx": np.array([], dtype=int),
            "cat_idx": np.array([0], dtype=int),
            "real_lower": np.array([0.0, 0.0], dtype=float),
            "real_upper": np.array([20.0, 40.0], dtype=float),
            "int_lower": np.array([], dtype=int),
            "int_upper": np.array([], dtype=int),
            "cat_cardinality": np.array([self._X1_VALUES.size], dtype=int),
        }

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        idx = np.clip(np.rint(X[:, 0]), 0, self._X1_VALUES.size - 1).astype(int)
        x1 = self._X1_VALUES[idx]
        x2 = X[:, 1]
        x3 = X[:, 2]

        f1 = (29.4 * x1) + (0.6 * x2 * x3)

        denom = np.maximum(x2, 1e-12)
        g1 = (x1 * x3) - 7.735 * ((x1 * x1) / denom) - 180.0
        g2 = 4.0 - (x3 / denom)
        f2 = sum_violations_gte0(g1, g2)

        Fout = out["F"]
        Fout[:, 0] = f1
        Fout[:, 1] = f2


class RE23Problem:
    """
    Tanabe & Ishibuchi (2020) RE23 / RE2-4-3: Pressure vessel design.

    Mixed variable type: x1,x2 are discrete thickness steps (multiples of 0.0625),
    x3,x4 are continuous.
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 2
        self.encoding = "mixed"

        # x0,x1 are integer step counts in [1, 100] scaled by 0.0625 in evaluation.
        self.xl = np.array([1.0, 1.0, 10.0, 10.0], dtype=float)
        self.xu = np.array([100.0, 100.0, 200.0, 240.0], dtype=float)

        self.mixed_spec: dict[str, NDArray[np.generic]] = {
            "real_idx": np.array([2, 3], dtype=int),
            "int_idx": np.array([0, 1], dtype=int),
            "cat_idx": np.array([], dtype=int),
            "real_lower": np.array([10.0, 10.0], dtype=float),
            "real_upper": np.array([200.0, 240.0], dtype=float),
            "int_lower": np.array([1, 1], dtype=int),
            "int_upper": np.array([100, 100], dtype=int),
            "cat_cardinality": np.array([], dtype=int),
        }

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        steps1 = np.clip(np.rint(X[:, 0]), 1, 100).astype(int)
        steps2 = np.clip(np.rint(X[:, 1]), 1, 100).astype(int)
        x1 = 0.0625 * steps1.astype(float)
        x2 = 0.0625 * steps2.astype(float)
        x3 = X[:, 2]
        x4 = X[:, 3]

        f1 = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)

        g1 = x1 - (0.0193 * x3)
        g2 = x2 - (0.00954 * x3)
        g3 = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000.0
        f2 = sum_violations_gte0(g1, g2, g3)

        Fout = out["F"]
        Fout[:, 0] = f1
        Fout[:, 1] = f2


class RE24Problem:
    """
    Tanabe & Ishibuchi (2020) RE24 / RE2-2-4: Hatch cover design.

    Bounds ``x1`` in ``[0.5, 4]`` and ``x2`` in ``[0.5, 50]``, as in the authors' ``reproblem.py`` and jMetal.
    """

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.encoding = "continuous"
        self.xl = np.array([0.5, 0.5], dtype=float)
        self.xu = np.array([4.0, 50.0], dtype=float)

        self._E = 700000.0
        self._sigma_b_max = 700.0
        self._tau_max = 450.0
        self._delta_max = 1.5

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        x1 = X[:, 0]
        x2 = X[:, 1]
        E = self._E

        f1 = x1 + 120.0 * x2

        sigma_k = E * x1 * x1 / 100.0
        sigma_b = 4500.0 / (x1 * x2)
        tau = 1800.0 / x2
        delta = (56.2e4) / (E * x1 * x2 * x2)

        g1 = 1.0 - (sigma_b / self._sigma_b_max)
        g2 = 1.0 - (tau / self._tau_max)
        g3 = 1.0 - (delta / self._delta_max)
        g4 = 1.0 - (sigma_b / sigma_k)
        f2 = sum_violations_gte0(g1, g2, g3, g4)

        Fout = out["F"]
        Fout[:, 0] = f1
        Fout[:, 1] = f2


class RE25Problem:
    """
    Tanabe & Ishibuchi (2020) RE25 / RE2-3-5: Coil compression spring design.

    Mixed variable type: x1 is integer, x3 is discrete.

    x3 is a categorical index into the 19 wire diameters from 0.092 to 0.5. The authors' ``reproblem.py`` (and
    jMetal) list 42 diameters from 0.009, but bound x3 to ``[0.09, 0.5]`` and round it to the nearest listed
    value, so only these 19 are reachable; their approximated front was computed that way.
    """

    _X3_VALUES = np.array(
        [
            0.092,
            0.105,
            0.12,
            0.135,
            0.148,
            0.162,
            0.177,
            0.192,
            0.207,
            0.225,
            0.244,
            0.263,
            0.283,
            0.307,
            0.331,
            0.362,
            0.394,
            0.4375,
            0.5,
        ],
        dtype=float,
    )

    def __init__(self) -> None:
        self.n_var = 3
        self.n_obj = 2
        self.encoding = "mixed"

        # x0: integer coil count in [1,70]
        # x1: diameter in [0.6,3]
        # x2: categorical index into _X3_VALUES
        self.xl = np.array([1.0, 0.6, 0.0], dtype=float)
        self.xu = np.array([70.0, 3.0, float(self._X3_VALUES.size - 1)], dtype=float)
        self.mixed_spec: dict[str, NDArray[np.generic]] = {
            "real_idx": np.array([1], dtype=int),
            "int_idx": np.array([0], dtype=int),
            "cat_idx": np.array([2], dtype=int),
            "real_lower": np.array([0.6], dtype=float),
            "real_upper": np.array([3.0], dtype=float),
            "int_lower": np.array([1], dtype=int),
            "int_upper": np.array([70], dtype=int),
            "cat_cardinality": np.array([self._X3_VALUES.size], dtype=int),
        }

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        x1 = np.clip(np.rint(X[:, 0]), 1, 70).astype(int).astype(float)
        x2 = X[:, 1]
        idx = np.clip(np.rint(X[:, 2]), 0, self._X3_VALUES.size - 1).astype(int)
        x3 = self._X3_VALUES[idx]

        f1 = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2.0)) / 4.0

        C = x2 / x3
        Cf = ((4.0 * C - 1.0) / (4.0 * C - 4.0)) + (0.615 * x3 / x2)

        Fmax = 1000.0
        S = 189000.0
        G = 11.5e6
        K = (G * (x3**4)) / (8.0 * x1 * (x2**3))
        lmax = 14.0
        lf = (Fmax / K) + 1.05 * (x1 + 2.0) * x3
        Fp = 300.0
        sigma_p = Fp / K
        sigma_pm = 6.0
        sigma_w = 1.25

        g1 = -((8.0 * Cf * Fmax * x2) / (np.pi * (x3**3))) + S
        g2 = -lf + lmax
        g3 = -3.0 + (x2 / x3)
        g4 = -sigma_p + sigma_pm
        g5 = -sigma_p - ((Fmax - Fp) / K) - 1.05 * (x1 + 2.0) * x3 + lf
        g6 = sigma_w - ((Fmax - Fp) / K)
        f2 = sum_violations_gte0(g1, g2, g3, g4, g5, g6)

        Fout = out["F"]
        Fout[:, 0] = f1
        Fout[:, 1] = f2


__all__ = ["RE21Problem", "RE22Problem", "RE23Problem", "RE24Problem", "RE25Problem"]
