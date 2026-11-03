from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem


class LZ09Problem(Problem):
    """
    Li & Zhang (LZ09) benchmark family as described in:
    H. Li and Q. Zhang, "Multiobjective optimization problems with complicated Pareto sets,
    MOEA/D and NSGA-II", IEEE TEC 2009.
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        *,
        ptype: int,
        dtype: int,
        ltype: int,
    ) -> None:
        if n_var < 3:
            raise ValueError(f"LZ09 problems need at least 3 variables (their distance terms divide by n_var - 2); got n_var={n_var}.")
        if n_obj not in (2, 3):
            raise ValueError("LZ09 problems support 2 or 3 objectives.")
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.ptype = int(ptype)
        self.dtype = int(dtype)
        self.ltype = int(ltype)
        # All LZ09 definitions operate in [0, 1]
        self.xl = 0.0
        self.xu = 1.0

    # --- Helper functions translated from the reference implementations (vectorized) ---
    def _ps_func2(self, x: np.ndarray[Any, Any], t1: np.ndarray[Any, Any], dim: int, css: int) -> np.ndarray[Any, Any]:
        dim = dim + 1
        xy = 2.0 * (x - 0.5)
        if self.ltype == 21:
            beta = xy - np.power(t1, 0.5 * (self.n_var + 3.0 * dim - 8.0) / (self.n_var - 2.0))
        elif self.ltype == 22:
            theta = 6.0 * np.pi * t1 + dim * np.pi / self.n_var
            beta = xy - np.sin(theta)
        elif self.ltype == 23:
            theta = 6.0 * np.pi * t1 + dim * np.pi / self.n_var
            ra = 0.8 * t1
            if css == 1:
                beta = xy - ra * np.cos(theta)
            else:
                beta = xy - ra * np.sin(theta)
        elif self.ltype == 24:
            theta = 6.0 * np.pi * t1 + dim * np.pi / self.n_var
            ra = 0.8 * t1
            if css == 1:
                beta = xy - ra * np.cos(theta / 3.0)
            else:
                beta = xy - ra * np.sin(theta)
        elif self.ltype == 25:
            rho = 0.8
            phi = np.pi * t1
            theta = 6.0 * np.pi * t1 + dim * np.pi / self.n_var
            if css == 1:
                beta = xy - rho * np.sin(phi) * np.sin(theta)
            elif css == 2:
                beta = xy - rho * np.sin(phi) * np.cos(theta)
            else:
                beta = xy - rho * np.cos(phi)
        elif self.ltype == 26:
            theta = 6.0 * np.pi * t1 + dim * np.pi / self.n_var
            ra = 0.3 * t1 * (t1 * np.cos(4.0 * theta) + 2.0)
            if css == 1:
                beta = xy - ra * np.cos(theta)
            else:
                beta = xy - ra * np.sin(theta)
        else:  # pragma: no cover - guarded by constructor choices
            raise ValueError(f"Unsupported ltype '{self.ltype}' for 2D LZ09 PS function.")
        return np.asarray(beta, dtype=float)

    def _ps_func3(self, x: np.ndarray[Any, Any], t1: np.ndarray[Any, Any], t2: np.ndarray[Any, Any], dim: int) -> np.ndarray[Any, Any]:
        dim = dim + 1
        xy = 4.0 * (x - 0.5)
        if self.ltype == 31:
            rate = 1.0 * dim / self.n_var
            beta = xy - 4.0 * (t1 * t1 * rate + t2 * (1.0 - rate)) + 2.0
        elif self.ltype == 32:
            theta = 2.0 * np.pi * t1 + dim * np.pi / self.n_var
            beta = xy - 2.0 * t2 * np.sin(theta)
        else:  # pragma: no cover - guarded by constructor choices
            raise ValueError(f"Unsupported ltype '{self.ltype}' for 3D LZ09 PS function.")
        return np.asarray(beta, dtype=float)

    def _alpha_2d(self, x0: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if self.ptype == 21:
            return x0, 1.0 - np.sqrt(x0)
        if self.ptype == 22:
            return x0, 1.0 - x0 * x0
        if self.ptype == 23:
            alpha0 = x0
            alpha1 = 1.0 - np.sqrt(alpha0) - alpha0 * np.sin(10.0 * alpha0 * alpha0 * np.pi)
            return alpha0, alpha1
        if self.ptype == 24:
            return x0, 1.0 - x0 - 0.05 * np.sin(4.0 * np.pi * x0)
        raise ValueError(f"Unsupported ptype '{self.ptype}' for 2D alpha.")

    def _alpha_3d(
        self, x0: np.ndarray[Any, Any], x1: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if self.ptype == 31:
            return (
                np.cos(x0 * np.pi / 2.0) * np.cos(x1 * np.pi / 2.0),
                np.cos(x0 * np.pi / 2.0) * np.sin(x1 * np.pi / 2.0),
                np.sin(x0 * np.pi / 2.0),
            )
        if self.ptype == 32:
            return (
                1.0 - np.cos(x0 * np.pi / 2.0) * np.cos(x1 * np.pi / 2.0),
                1.0 - np.cos(x0 * np.pi / 2.0) * np.sin(x1 * np.pi / 2.0),
                1.0 - np.sin(x0 * np.pi / 2.0),
            )
        if self.ptype == 33:
            return (
                x0,
                x1,
                3.0 - (np.sin(3.0 * np.pi * x0) + np.sin(3.0 * np.pi * x1) - 2.0 * (x0 + x1)),
            )
        if self.ptype == 34:
            return (x0 - x1, x0 * (1.0 - x1), 1.0 - x0)
        raise ValueError(f"Unsupported ptype '{self.ptype}' for 3D alpha.")

    def _beta_func(self, vals: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        # vals shape: (N, dim)
        dim = vals.shape[1]
        if dim == 0:
            return np.zeros(vals.shape[0])
        if self.dtype == 1:
            beta = 2.0 * np.sum(vals * vals, axis=1) / dim
        elif self.dtype == 2:
            weights = np.sqrt(np.arange(1, dim + 1, dtype=float))
            beta = 2.0 * np.sum(weights * vals * vals, axis=1) / dim
        elif self.dtype == 3:
            xx = 2.0 * vals
            beta = 2.0 * np.sum(xx * xx - np.cos(4.0 * np.pi * xx) + 1.0, axis=1) / dim
        elif self.dtype == 4:
            xx = 2.0 * vals
            weights = np.sqrt(np.arange(1, dim + 1, dtype=float))
            prod = np.prod(np.cos(10.0 * np.pi * xx / weights), axis=1)
            beta = 2.0 * (np.sum(xx * xx, axis=1) - 2.0 * prod + 2.0) / dim
        else:  # pragma: no cover - guarded by constructor choices
            raise ValueError(f"Unsupported dtype '{self.dtype}' for beta function.")
        return np.asarray(beta, dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        n_points = X.shape[0]
        F = out.get("F")
        if F is None or F.shape != (n_points, self.n_obj):
            F = np.empty((n_points, self.n_obj), dtype=float)
            out["F"] = F

        if self.n_obj == 2:
            aa_2d: list[np.ndarray[Any, Any]] = []
            bb_2d: list[np.ndarray[Any, Any]] = []
            if self.ltype in (21, 22, 23, 24, 26):
                for idx in range(1, self.n_var):
                    xj = X[:, idx]
                    if idx % 2 == 0:
                        aa_2d.append(self._ps_func2(xj, X[:, 0], idx, css=1))
                    else:
                        bb_2d.append(self._ps_func2(xj, X[:, 0], idx, css=2))
            elif self.ltype == 25:
                for idx in range(1, self.n_var):
                    xj = X[:, idx]
                    if idx % 3 == 0:
                        aa_2d.append(self._ps_func2(xj, X[:, 0], idx, css=1))
                    elif idx % 3 == 1:
                        bb_2d.append(self._ps_func2(xj, X[:, 0], idx, css=2))
                    else:
                        css = 3
                        target = aa_2d if idx % 2 == 0 else bb_2d
                        target.append(self._ps_func2(xj, X[:, 0], idx, css=css))
            else:  # pragma: no cover
                raise ValueError(f"Unsupported ltype '{self.ltype}' for 2-objective LZ09.")

            aa_arr = np.column_stack(aa_2d) if aa_2d else np.zeros((n_points, 0))
            bb_arr = np.column_stack(bb_2d) if bb_2d else np.zeros((n_points, 0))
            g = self._beta_func(aa_arr)
            h = self._beta_func(bb_arr)
            alpha0, alpha1 = self._alpha_2d(X[:, 0])
            F[:, 0] = alpha0 + h
            F[:, 1] = alpha1 + g
        else:
            aa_3d: list[np.ndarray[Any, Any]] = []
            bb_3d: list[np.ndarray[Any, Any]] = []
            cc_3d: list[np.ndarray[Any, Any]] = []
            for idx in range(2, self.n_var):
                xj = X[:, idx]
                res = self._ps_func3(xj, X[:, 0], X[:, 1], idx)
                if idx % 3 == 0:
                    aa_3d.append(res)
                elif idx % 3 == 1:
                    bb_3d.append(res)
                else:
                    cc_3d.append(res)

            aa_arr = np.column_stack(aa_3d) if aa_3d else np.zeros((n_points, 0))
            bb_arr = np.column_stack(bb_3d) if bb_3d else np.zeros((n_points, 0))
            cc_arr = np.column_stack(cc_3d) if cc_3d else np.zeros((n_points, 0))
            g = self._beta_func(aa_arr)
            h = self._beta_func(bb_arr)
            e = self._beta_func(cc_arr)
            alpha0, alpha1, alpha2 = self._alpha_3d(X[:, 0], X[:, 1])
            F[:, 0] = alpha0 + h
            F[:, 1] = alpha1 + g
            F[:, 2] = alpha2 + e


class LZ09_F1(LZ09Problem):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=2, dtype=1, ltype=21, ptype=21)


class LZ09_F2(LZ09Problem):
    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, n_obj=2, dtype=1, ltype=22, ptype=21)


class LZ09_F3(LZ09Problem):
    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, n_obj=2, dtype=1, ltype=23, ptype=21)


class LZ09_F4(LZ09Problem):
    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, n_obj=2, dtype=1, ltype=24, ptype=21)


class LZ09_F5(LZ09Problem):
    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, n_obj=2, dtype=1, ltype=26, ptype=21)


class LZ09_F6(LZ09Problem):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=3, dtype=1, ltype=32, ptype=31)


class LZ09_F7(LZ09Problem):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=2, dtype=3, ltype=21, ptype=21)


class LZ09_F8(LZ09Problem):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=2, dtype=4, ltype=21, ptype=21)


class LZ09_F9(LZ09Problem):
    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, n_obj=2, dtype=1, ltype=22, ptype=22)


__all__ = [
    "LZ09Problem",
    "LZ09_F1",
    "LZ09_F2",
    "LZ09_F3",
    "LZ09_F4",
    "LZ09_F5",
    "LZ09_F6",
    "LZ09_F7",
    "LZ09_F8",
    "LZ09_F9",
    "LZ09F1Problem",
    "LZ09F2Problem",
    "LZ09F3Problem",
    "LZ09F4Problem",
    "LZ09F5Problem",
    "LZ09F6Problem",
    "LZ09F7Problem",
    "LZ09F8Problem",
    "LZ09F9Problem",
    "has_lz09",
]


# Aliases matching the registry naming conventions
class LZ09F1Problem(LZ09_F1):
    pass


class LZ09F2Problem(LZ09_F2):
    pass


class LZ09F3Problem(LZ09_F3):
    pass


class LZ09F4Problem(LZ09_F4):
    pass


class LZ09F5Problem(LZ09_F5):
    pass


class LZ09F6Problem(LZ09_F6):
    pass


class LZ09F7Problem(LZ09_F7):
    pass


class LZ09F8Problem(LZ09_F8):
    pass


class LZ09F9Problem(LZ09_F9):
    pass


def has_lz09() -> bool:
    """Always true since this implementation is built-in (no extra deps)."""
    return True
