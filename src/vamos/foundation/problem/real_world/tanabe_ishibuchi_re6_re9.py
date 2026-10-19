from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .tanabe_ishibuchi_utils import sum_violations_gte0


class RE61Problem:
    """
    Tanabe & Ishibuchi (2020) RE61 / RE6-3-1: Water resource planning.
    """

    def __init__(self) -> None:
        self.n_var = 3
        self.n_obj = 6
        self.encoding = "continuous"
        self.xl = np.array([0.01, 0.01, 0.01], dtype=float)
        self.xu = np.array([0.45, 0.10, 0.10], dtype=float)

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

        f1 = 106780.37 * (x2 + x3) + 61704.67
        f2 = 3000.0 * x1
        f3 = 305700.0 * 2289.0 * x2 / np.power(0.06 * 2289.0, 0.65)
        f4 = 250.0 * 2289.0 * np.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        f5 = 25.0 * (1.39 / (x1 * x2) + 4940.0 * x3 - 80.0)

        g1 = 1.0 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)
        g2 = 1.0 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)
        g3 = 50000.0 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)
        g4 = 16000.0 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)
        g5 = 10000.0 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)
        g6 = 2000.0 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54)
        g7 = 550.0 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48)
        f6 = sum_violations_gte0(g1, g2, g3, g4, g5, g6, g7)

        Fout = out["F"]
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3
        Fout[:, 3] = f4
        Fout[:, 4] = f5
        Fout[:, 5] = f6


class RE91Problem:
    """
    Tanabe & Ishibuchi (2020) RE91 / RE9-7-1: Car cab design.

    Notes:
      - The original formulation includes stochastic parameters (x8..x11). For
        reproducible deterministic behavior, set ``stochastic=False`` (default) to
        evaluate at the mean values used by the reference implementation.
    """

    def __init__(self, *, stochastic: bool = False, seed: int = 123) -> None:
        self.n_var = 7
        self.n_obj = 9
        self.encoding = "continuous"
        self.xl = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], dtype=float)
        self.xu = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2], dtype=float)

        self._stochastic = bool(stochastic)
        self._rng = np.random.default_rng(int(seed))

    def _sample_stochastic(self, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if not self._stochastic:
            x8: NDArray[np.float64] = np.full(n, 0.345, dtype=float)
            x9: NDArray[np.float64] = np.full(n, 0.192, dtype=float)
            x10: NDArray[np.float64] = np.zeros(n, dtype=float)
            x11: NDArray[np.float64] = np.zeros(n, dtype=float)
            return x8, x9, x10, x11

        x8 = 0.006 * self._rng.normal(size=n) + 0.345
        x9 = 0.006 * self._rng.normal(size=n) + 0.192
        x10 = 10.0 * self._rng.normal(size=n)
        x11 = 10.0 * self._rng.normal(size=n)
        return x8, x9, x10, x11

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        x1, x2, x3, x4, x5, x6, x7 = (X[:, i] for i in range(7))
        x8, x9, x10, x11 = self._sample_stochastic(X.shape[0])

        f1 = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.75 * x5 + 0.00001 * x6 + 2.73 * x7

        f2 = np.maximum(0.0, (1.16 - 0.3717 * x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10) / 1.0)
        f3 = np.maximum(
            0.0,
            (
                0.261
                - 0.0159 * x1 * x2
                - 0.188 * x1 * x8
                - 0.019 * x2 * x7
                + 0.0144 * x3 * x5
                + 0.87570001 * x5 * x10
                + 0.08045 * x6 * x9
                + 0.00139 * x8 * x11
                + 0.00001575 * x10 * x11
            )
            / 0.32,
        )
        f4 = np.maximum(
            0.0,
            (
                0.214
                + 0.00817 * x5
                - 0.131 * x1 * x8
                - 0.0704 * x1 * x9
                + 0.03099 * x2 * x6
                - 0.018 * x2 * x7
                + 0.0208 * x3 * x8
                + 0.121 * x3 * x9
                - 0.00364 * x5 * x6
                + 0.0007715 * x5 * x10
                - 0.0005354 * x6 * x10
                + 0.00121 * x8 * x11
                + 0.00184 * x9 * x10
                - 0.018 * x2 * x2
            )
            / 0.32,
        )
        f5 = np.maximum(
            0.0,
            (0.74 - 0.61 * x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 * x2) / 0.32,
        )

        urd = 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.77 * x7 * x8 + 0.32 * x9 * x10
        mrd = 33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11.0 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22.0 * x8 * x9
        lrd = 46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10
        avg_def = (urd + mrd + lrd) / 3.0
        f6 = np.maximum(0.0, avg_def / 32.0)

        f7 = np.maximum(
            0.0,
            (4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 * x11) / 4.0,
        )

        f8 = np.maximum(
            0.0,
            (10.58 - 0.674 * x1 * x2 - 1.95 * x2 * x8 + 0.02054 * x3 * x10 - 0.0198 * x4 * x10 + 0.028 * x6 * x10) / 9.9,
        )
        f9 = np.maximum(
            0.0,
            (16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 * x11) / 15.7,
        )

        Fout = out["F"]
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3
        Fout[:, 3] = f4
        Fout[:, 4] = f5
        Fout[:, 5] = f6
        Fout[:, 6] = f7
        Fout[:, 7] = f8
        Fout[:, 8] = f9


__all__ = ["RE61Problem", "RE91Problem"]
