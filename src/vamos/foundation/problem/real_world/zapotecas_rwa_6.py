from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output


class RWA6Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA6: Milling parameters for ultrahigh-strength steel.

    Minimization formulation (paper, Eq. 7):
      f1(x) =  Ft
      f2(x) =  Ra
      f3(x) = -MRR

    Notes:
      - The paper defines MRR with constants ``z`` (tooth count) and ``d`` (tool diameter),
        but does not provide numerical values in the appendix. Since ``z/d`` is a positive
        constant factor, it does not affect dominance relations. We default to z=1, d=1.
    """

    def __init__(self, *, z: float = 1.0, d: float = 1.0) -> None:
        if z <= 0.0:
            raise ValueError("z must be positive.")
        if d <= 0.0:
            raise ValueError("d must be positive.")

        self.n_var = 4
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.array([12.56, 0.02, 1.0, 0.5], dtype=float)
        self.xu = np.array([25.12, 0.06, 5.0, 2.0], dtype=float)
        self._z = float(z)
        self._d = float(d)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        vc, fz, ap, ae = (X[:, i] for i in range(4))

        ft = (
            -54.3
            - 1.18 * vc
            - 2429.0 * fz
            + 104.2 * ap
            + 129.0 * ae
            - 18.9 * vc * fz
            - 0.209 * vc * ap
            - 0.673 * vc * ae
            + 265.0 * fz * ap
            + 1209.0 * fz * ae
            + 22.76 * ap * ae
            + 0.066 * vc * vc
            + 32117.0 * fz * fz
            - 16.98 * ap * ap
            - 47.6 * ae * ae
        )
        ra = (
            0.227
            - 0.0072 * vc
            + 1.89 * fz
            - 0.0203 * ap
            + 0.3075 * ae
            - 0.198 * vc * fz
            - 0.000955 * vc * ap
            - 0.00656 * vc * ae
            + 0.209 * fz * ap
            + 0.783 * fz * ae
            + 0.02275 * ap * ae
            + 0.000355 * vc * vc
            + 35.0 * fz * fz
            + 0.00037 * ap * ap
            - 0.0791 * ae * ae
        )
        mrr = (1000.0 * vc * fz * self._z * ap * ae) / (np.pi * self._d)

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = ft
        F[:, 1] = ra
        F[:, 2] = -mrr


__all__ = ["RWA6Problem"]
