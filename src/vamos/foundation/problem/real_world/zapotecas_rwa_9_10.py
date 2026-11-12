from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output


class RWA9Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA9: Ultra-wideband antenna design.

    Minimization formulation (paper, Eq. 10):
      f1(x) =  VP
      f2(x) = -VWi
      f3(x) = -VWL
      f4(x) = -FF
      f5(x) =  PG
    """

    def __init__(self) -> None:
        self.n_var = 10
        self.n_obj = 5
        self.encoding = "continuous"
        self.xl = np.array([5.0, 10.0, 5.0, 6.0, 3.0, 11.5, 17.5, 2.0, 17.5, 2.0], dtype=float)
        self.xu = np.array([7.0, 12.0, 6.0, 7.0, 4.0, 12.5, 22.5, 3.0, 22.5, 3.0], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        a1, a2, b1, b2, d1, d2, l1, l2, w1, w2 = (X[:, i] for i in range(10))

        w1n = (w1 - 20.0) / 0.5
        l1n = (l1 - 20.0) / 2.5
        a1n = a1 - 6.0
        b1n = (b1 - 5.5) / 0.5
        a2n = a2 - 11.0
        w2n = (w2 - 2.5) / 0.5
        l2n = (l2 - 2.5) / 0.5
        b2n = (b2 - 6.5) / 0.5
        d2n = (d2 - 12.0) / 0.5

        vp = (
            502.94
            - 27.18 * w1n
            + 43.08 * l1n
            + 47.75 * a1n
            + 32.25 * b1n
            + 31.67 * a2n
            - 36.19 * w1n * w2n
            - 39.44 * w1n * a1n
            + 57.45 * a1n * b1n
        )
        vwi = 130.53 + 45.97 * l1n - 52.93 * w1n - 78.93 * a1n + 79.22 * a2n + 47.23 * w1n * a1n - 40.61 * w1n * a2n - 50.62 * a1n * a2n
        vwl = 203.16 - 42.75 * w1n + 56.67 * a1n + 19.88 * b1n - 12.89 * a2n - 35.09 * a1n * b1n - 22.91 * b1n * a2n
        ff = 0.76 - 0.06 * l1n + 0.03 * l2n + 0.02 * a2n - 0.02 * b2n - 0.03 * d2n + 0.03 * l1n * w1n - 0.02 * l1n * l2n + 0.02 * l1n * b2n

        # Note: the appendix uses (a2 - 6.0) and (b2 - 5.5)/0.5 in the last interaction term.
        a2m6 = a2 - 6.0
        b2m55 = (b2 - 5.5) / 0.5
        pg = 1.08 - 0.12 * l1n - 0.26 * w1n - 0.05 * a2n - 0.12 * b2n + 0.08 * a1n * b2n + 0.07 * a2m6 * b2m55

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = vp
        F[:, 1] = -vwi
        F[:, 2] = -vwl
        F[:, 3] = -ff
        F[:, 4] = pg


class RWA10Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA10: Water and oil repellent fabric development.

    Minimization formulation (paper, Eq. 11):
      f1(x) = -WCA
      f2(x) = -OCA
      f3(x) = -AP
      f4(x) = -CRA
      f5(x) =  Stiff
      f6(x) = -Tear
      f7(x) = -Tensile
    """

    def __init__(self) -> None:
        self.n_var = 3
        self.n_obj = 7
        self.encoding = "continuous"
        self.xl = np.array([10.0, 10.0, 150.0], dtype=float)
        self.xu = np.array([50.0, 50.0, 170.0], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        ocpc, kfel, ctemp = (X[:, i] for i in range(3))

        wca = -1331.04 + 1.99 * ocpc + 0.33 * kfel + 17.12 * ctemp - 0.02 * ocpc * ocpc - 0.05 * ctemp * ctemp
        oca = -4231.14 + 4.27 * ocpc + 1.50 * kfel + 52.30 * ctemp - 0.04 * ocpc * kfel - 0.04 * ocpc * ocpc - 0.16 * ctemp * ctemp
        ap = (
            1766.80
            - 32.32 * ocpc
            - 24.56 * kfel
            - 10.48 * ctemp
            + 0.24 * ocpc * ctemp
            + 0.19 * kfel * ctemp
            - 0.06 * ocpc * ocpc
            - 0.10 * kfel * kfel
        )
        cra = -2342.13 - 1.556 * ocpc + 0.77 * kfel + 31.14 * ctemp + 0.03 * ocpc * ocpc - 0.10 * ctemp * ctemp
        stiff = 9.34 + 0.02 * ocpc - 0.03 * kfel - 0.03 * ctemp - 0.001 * ocpc * kfel + 0.0009 * kfel * kfel
        tear = 1954.71 + 14.246 * ocpc + 5.00 * kfel - 4.30 * ctemp - 0.22 * ocpc * ocpc - 0.33 * kfel * kfel
        tensile = 828.16 + 3.55 * ocpc + 73.65 * kfel + 10.80 * ctemp - 0.56 * kfel * ctemp + 0.20 * kfel * kfel

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = -wca
        F[:, 1] = -oca
        F[:, 2] = -ap
        F[:, 3] = -cra
        F[:, 4] = stiff
        F[:, 5] = -tear
        F[:, 6] = -tensile


__all__ = ["RWA9Problem", "RWA10Problem"]
