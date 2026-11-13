from __future__ import annotations

import math
from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem


def _as_float_vector(values: object) -> np.ndarray[Any, Any]:
    return np.asarray(values, dtype=float)


def _safe_divide(num: np.ndarray[Any, Any], den: np.ndarray[Any, Any], *, eps: float = 1e-12) -> np.ndarray[Any, Any]:
    num_arr = np.asarray(num, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    out = np.zeros_like(num_arr, dtype=float)
    np.divide(num_arr, den_arr, out=out, where=np.abs(den_arr) > eps)
    return out


def _safe_sqrt(expr: np.ndarray[Any, Any] | float) -> np.ndarray[Any, Any]:
    return _as_float_vector(np.sqrt(np.maximum(np.asarray(expr, dtype=float), 0.0)))


def _safe_arcsin(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return _as_float_vector(np.arcsin(np.clip(np.asarray(x, dtype=float), -1.0, 1.0)))


def _prepare_output(
    X: np.ndarray[Any, Any],
    out: dict[str, np.ndarray[Any, Any]],
    *,
    n_var: int,
    n_obj: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[1] != n_var:
        raise ValueError(f"Expected decision matrix of shape (N, {n_var}).")

    n = X_arr.shape[0]
    F = out.get("F")
    if F is None or F.shape != (n, n_obj):
        F = np.empty((n, n_obj), dtype=float)
        out["F"] = F
    return X_arr, F


def _dtlz_g1(X_m: np.ndarray[Any, Any], k: int) -> np.ndarray[Any, Any]:
    return _as_float_vector(100.0 * (k + np.sum(np.square(X_m - 0.5) - np.cos(20.0 * np.pi * (X_m - 0.5)), axis=1)))


def _dtlz_g2(X_m: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return _as_float_vector(np.sum(np.square(X_m - 0.5), axis=1))


def _dtlz1_objectives(X_head: np.ndarray[Any, Any], g: np.ndarray[Any, Any], n_obj: int) -> np.ndarray[Any, Any]:
    n = X_head.shape[0]
    F = np.empty((n, n_obj), dtype=float)
    for i in range(n_obj):
        f = 0.5 * (1.0 + g)
        head = n_obj - i - 1
        if head > 0:
            f *= np.prod(X_head[:, :head], axis=1)
        if i > 0:
            f *= 1.0 - X_head[:, head]
        F[:, i] = f
    return F


def _dtlz23_objectives(X_head: np.ndarray[Any, Any], g: np.ndarray[Any, Any], n_obj: int) -> np.ndarray[Any, Any]:
    n = X_head.shape[0]
    F = np.empty((n, n_obj), dtype=float)
    for i in range(n_obj):
        f = 1.0 + g
        head = n_obj - i - 1
        if head > 0:
            f *= np.prod(np.cos(X_head[:, :head] * np.pi / 2.0), axis=1)
        if i > 0:
            f *= np.sin(X_head[:, head] * np.pi / 2.0)
        F[:, i] = f
    return F


def _constraint_c1_linear(F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return _as_float_vector(-(1.0 - F[:, -1] / 0.6 - np.sum(F[:, :-1] / 0.5, axis=1)))


def _constraint_c1_spherical(F: np.ndarray[Any, Any], r: float) -> np.ndarray[Any, Any]:
    radius = np.sum(F * F, axis=1)
    return _as_float_vector(-(radius - 16.0) * (radius - r * r))


def _constraint_c2(F: np.ndarray[Any, Any], r: float) -> np.ndarray[Any, Any]:
    n_obj = F.shape[1]
    sum_sq = np.sum(F * F, axis=1)

    v1 = np.full(F.shape[0], np.inf, dtype=float)
    for i in range(n_obj):
        temp = np.square(F[:, i] - 1.0) + (sum_sq - F[:, i] * F[:, i]) - r * r
        v1 = np.minimum(v1, temp)

    a = 1.0 / np.sqrt(float(n_obj))
    v2 = np.sum(np.square(F - a), axis=1) - r * r
    return _as_float_vector(np.minimum(v1, v2))


def _constraint_c3_linear(F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    n_obj = F.shape[1]
    g_terms = []
    total = np.sum(F, axis=1)
    for i in range(n_obj):
        g_terms.append(1.0 - F[:, i] / 0.5 - (total - F[:, i]))
    return np.column_stack(g_terms)


def _constraint_dc1(X: np.ndarray[Any, Any], a: float, b: float) -> np.ndarray[Any, Any]:
    return b - np.cos(a * np.pi * X[:, 0])


def _constraint_dc2(g: np.ndarray[Any, Any], a: float, b: float) -> np.ndarray[Any, Any]:
    return np.column_stack([b - np.cos(a * np.pi * g), b - np.exp(-g)])


def _constraint_dc3(X: np.ndarray[Any, Any], g: np.ndarray[Any, Any], a: float, b: float) -> np.ndarray[Any, Any]:
    return np.column_stack([b - np.cos(a * np.pi * g), b - np.cos(a * np.pi * X)])


def _constraint_c3_spherical(F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    squared = np.sum(F * F, axis=1, keepdims=True)
    return np.asarray(-(F * F / 4.0 + squared - F * F - 1.0))


# DC-DTLZ's constants in the authors' implementation (Li et al., 2019): a, b, and the factor that turns the
# objectives' g into the g of the constraints (DC2 divides it by 100; DC1's constraint does not use g).
DC_DTLZ_AUTHORS = {"dc1": (5.0, 0.95, 1.0), "dc2": (3.0, 0.9, 0.01), "dc3": (5.0, 0.5, 1.0)}


class CDTLZProblem(Problem):
    def __init__(self, variant: str, *, n_var: int = 12, n_obj: int = 3, r: float | None = None) -> None:
        key = str(variant).lower()
        if key not in {"c1dtlz1", "c1dtlz3", "c2dtlz2", "c3dtlz1", "c3dtlz4"}:
            raise ValueError(f"Unknown C-DTLZ variant '{variant}'.")
        if n_var < n_obj:
            raise ValueError("C-DTLZ requires n_var >= n_obj, one distance variable at least.")

        self.variant = key
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.n_constraints = self.n_obj if key.startswith("c3") else 1
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)

        if key == "c1dtlz3":
            if r is None:
                if self.n_obj < 5:
                    r = 9.0
                elif self.n_obj <= 12:
                    r = 12.5
                else:
                    r = 15.0
        elif key == "c2dtlz2" and r is None:
            if self.n_obj == 2:
                r = 0.2
            elif self.n_obj == 3:
                r = 0.4
            else:
                r = 0.5
        self._r = float(r) if r is not None else None

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X_arr, F_out = _prepare_output(X, out, n_var=self.n_var, n_obj=self.n_obj)
        X_head = X_arr[:, : self.n_obj - 1]
        X_m = X_arr[:, self.n_obj - 1 :]
        k = self.n_var - self.n_obj + 1

        if self.variant in {"c1dtlz1", "c3dtlz1"}:
            g = _dtlz_g1(X_m, k)
            F = _dtlz1_objectives(X_head, g, self.n_obj)
        elif self.variant == "c1dtlz3":
            g = _dtlz_g1(X_m, k)
            F = _dtlz23_objectives(X_head, g, self.n_obj)
        elif self.variant == "c2dtlz2":
            g = _dtlz_g2(X_m)
            F = _dtlz23_objectives(X_head, g, self.n_obj)
        elif self.variant == "c3dtlz4":
            # DTLZ4: DTLZ2's distance and its position variables raised to alpha = 100.
            g = _dtlz_g2(X_m)
            F = _dtlz23_objectives(X_head**100.0, g, self.n_obj)
        else:  # pragma: no cover - guarded by constructor
            raise RuntimeError("Unexpected C-DTLZ variant.")

        if self.variant == "c1dtlz1":
            G = _constraint_c1_linear(F)[:, None]
        elif self.variant == "c1dtlz3":
            G = _constraint_c1_spherical(F, self._r if self._r is not None else 9.0)[:, None]
        elif self.variant == "c2dtlz2":
            G = _constraint_c2(F, self._r if self._r is not None else 0.4)[:, None]
        elif self.variant == "c3dtlz4":
            G = _constraint_c3_spherical(F)
        else:  # c3dtlz1
            G = _constraint_c3_linear(F)

        F_out[:] = F
        out["G"] = G


class DCDTLZProblem(Problem):
    """DC-DTLZ (Li et al., 2019): DTLZ1 and DTLZ3 with constraints that cut their fronts into pieces.

    ``a`` and ``b`` shape the constraints, ``g_factor`` multiplies DTLZ1's distance ``g`` in the objectives,
    and ``constraint_g_scale`` turns that ``g`` into the one the constraints use. The defaults are the
    authors': ``a = 5``, ``b = 0.95`` in DC1; ``a = 3``, ``b = 0.9`` and ``g / 100`` in DC2; ``a = 5``,
    ``b = 0.5`` in DC3; ``g_factor = 100``. jMetal, PlatEMO and STEP use ``a = 3``, ``b = 0.5`` and
    ``constraint_g_scale = 1`` in all six.
    """

    def __init__(
        self,
        variant: str,
        *,
        n_var: int = 12,
        n_obj: int = 3,
        a: float | None = None,
        b: float | None = None,
        g_factor: float = 100.0,
        constraint_g_scale: float | None = None,
    ) -> None:
        key = str(variant).lower()
        if key not in {"dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3", "dc3dtlz1", "dc3dtlz3"}:
            raise ValueError(f"Unknown DC-DTLZ variant '{variant}'.")
        if n_var < n_obj - 1:
            raise ValueError("DC-DTLZ requires n_var >= n_obj - 1, one position variable per objective but the last.")
        default_a, default_b, default_scale = DC_DTLZ_AUTHORS[key[:3]]
        self.a = float(default_a if a is None else a)
        self.b = float(default_b if b is None else b)
        self.g_factor = float(g_factor)
        self.constraint_g_scale = float(default_scale if constraint_g_scale is None else constraint_g_scale)
        if not self.a > 0.0:
            raise ValueError(f"DC-DTLZ's a must be positive, got {self.a}.")
        if not -1.0 <= self.b <= 1.0:
            raise ValueError(f"DC-DTLZ's b must be in [-1, 1], got {self.b}.")
        if not self.g_factor > 0.0 or not self.constraint_g_scale > 0.0:
            raise ValueError("DC-DTLZ's g_factor and constraint_g_scale must be positive.")

        self.variant = key
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        if key.startswith("dc1"):
            self.n_constraints = 1
        elif key.startswith("dc2"):
            self.n_constraints = 2
        else:
            self.n_constraints = self.n_obj
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X_arr, F_out = _prepare_output(X, out, n_var=self.n_var, n_obj=self.n_obj)
        X_head = X_arr[:, : self.n_obj - 1]
        X_m = X_arr[:, self.n_obj - 1 :]
        k = self.n_var - self.n_obj + 1

        g = _dtlz_g1(X_m, k) * (self.g_factor / 100.0)
        if self.variant.endswith("dtlz1"):
            F = _dtlz1_objectives(X_head, g, self.n_obj)
        else:
            F = _dtlz23_objectives(X_head, g, self.n_obj)

        g_constraint = g * self.constraint_g_scale
        if self.variant.startswith("dc1"):
            G = _constraint_dc1(X_arr, self.a, self.b)[:, None]
        elif self.variant.startswith("dc2"):
            G = _constraint_dc2(g_constraint, self.a, self.b)
        else:
            G = _constraint_dc3(X_head, g_constraint, self.a, self.b)

        F_out[:] = F
        out["G"] = G


class MWProblem(Problem):
    def __init__(self, variant: str, *, n_var: int = 15, n_obj: int | None = None) -> None:
        key = str(variant).lower()
        if not key.startswith("mw"):
            raise ValueError(f"Unknown MW variant '{variant}'.")
        try:
            idx = int(key[2:])
        except ValueError as exc:
            raise ValueError(f"Unknown MW variant '{variant}'.") from exc
        if idx < 1 or idx > 14:
            raise ValueError(f"Unknown MW variant '{variant}'.")

        self.variant = key
        self.index = idx

        if idx in {4, 8, 14}:
            resolved_n_obj = int(n_obj) if n_obj is not None else 3
        else:
            resolved_n_obj = 2
        if resolved_n_obj < 2:
            raise ValueError("MW requires at least two objectives.")

        self.n_var = int(n_var)
        self.n_obj = int(resolved_n_obj)

        self.n_constraints = {
            1: 1,
            2: 1,
            3: 2,
            4: 1,
            5: 3,
            6: 1,
            7: 2,
            8: 1,
            9: 1,
            10: 3,
            11: 4,
            12: 2,
            13: 2,
            14: 1,
        }[idx]

        xl = 0.0
        if idx == 6:
            xu = 1.1
        elif idx == 11:
            xu = math.sqrt(2.0)
        elif idx in {13, 14}:
            xu = 1.5
        else:
            xu = 1.0

        self.xl = np.full(self.n_var, xl, dtype=float)
        self.xu = np.full(self.n_var, xu, dtype=float)

    @staticmethod
    def _la1(A: float, B: float, C: float, D: float, theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return _as_float_vector(A * np.power(np.sin(B * np.pi * np.power(theta, C)), D))

    @staticmethod
    def _la2(A: float, B: float, C: float, D: float, theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return _as_float_vector(A * np.power(np.sin(B * np.power(theta, C)), D))

    @staticmethod
    def _la3(A: float, B: float, C: float, D: float, theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return _as_float_vector(A * np.power(np.cos(B * np.power(theta, C)), D))

    def _g1(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        d = self.n_var
        n = d - self.n_obj
        i = np.arange(self.n_obj - 1, d, dtype=float)
        z = np.power(X[:, self.n_obj - 1 :], n)
        exp_term = 1.0 - np.exp(-10.0 * np.square(z - 0.5 - i / (2.0 * d)))
        return _as_float_vector(1.0 + np.sum(exp_term, axis=1))

    def _g2(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        d = self.n_var
        i = np.arange(self.n_obj - 1, d, dtype=float)
        z = 1.0 - np.exp(-10.0 * np.square(X[:, self.n_obj - 1 :] - i / d))
        contrib = (0.1 / d) * z * z + 1.5 - 1.5 * np.cos(2.0 * np.pi * z)
        return _as_float_vector(1.0 + np.sum(contrib, axis=1))

    def _g3(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        contrib = 2.0 * np.square(X[:, self.n_obj - 1 :] + np.square(X[:, self.n_obj - 2 : -1] - 0.5) - 1.0)
        return _as_float_vector(1.0 + np.sum(contrib, axis=1))

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X_arr, F_out = _prepare_output(X, out, n_var=self.n_var, n_obj=self.n_obj)
        idx = self.index

        if idx == 1:
            g = self._g1(X_arr)
            f0 = X_arr[:, 0]
            f1 = g * (1.0 - 0.85 * f0 / g)
            g0 = f0 + f1 - 1.0 - self._la1(0.5, 2.0, 1.0, 8.0, np.sqrt(2.0) * (f1 - f0))
            F = np.column_stack([f0, f1])
            G = g0[:, None]
        elif idx == 2:
            g = self._g2(X_arr)
            f0 = X_arr[:, 0]
            f1 = g * (1.0 - f0 / g)
            g0 = f0 + f1 - 1.0 - self._la1(0.5, 3.0, 1.0, 8.0, np.sqrt(2.0) * (f1 - f0))
            F = np.column_stack([f0, f1])
            G = g0[:, None]
        elif idx == 3:
            g = self._g3(X_arr)
            f0 = X_arr[:, 0]
            f1 = g * (1.0 - f0 / g)
            diff = np.sqrt(2.0) * (f1 - f0)
            g0 = f0 + f1 - 1.05 - self._la1(0.45, 0.75, 1.0, 6.0, diff)
            g1 = 0.85 - f0 - f1 + self._la1(0.3, 0.75, 1.0, 2.0, diff)
            F = np.column_stack([f0, f1])
            G = np.column_stack([g0, g1])
        elif idx == 4:
            g = self._g1(X_arr)
            F = g[:, None] * np.ones((X_arr.shape[0], self.n_obj), dtype=float)
            F[:, 1:] *= X_arr[:, (self.n_obj - 2) :: -1]
            F[:, :-1] *= np.flip(np.cumprod(1.0 - X_arr[:, : self.n_obj - 1], axis=1), axis=1)
            g0 = F.sum(axis=1) - 1.0 - self._la1(0.4, 2.5, 1.0, 8.0, F[:, -1] - np.sum(F[:, :-1], axis=1))
            G = g0[:, None]
        elif idx == 5:
            g = self._g1(X_arr)
            f0 = g * X_arr[:, 0]
            ratio = _safe_divide(f0, g)
            f1 = g * _safe_sqrt(1.0 - np.square(ratio))
            atan = np.arctan2(f1, f0)
            g0 = np.square(f0) + np.square(f1) - np.square(1.7 - self._la2(0.2, 2.0, 1.0, 1.0, atan))
            t = 0.5 * np.pi - 2.0 * np.abs(atan - 0.25 * np.pi)
            g1 = np.square(1.0 + self._la2(0.5, 6.0, 3.0, 1.0, t)) - np.square(f0) - np.square(f1)
            g2 = np.square(1.0 - self._la2(0.45, 6.0, 3.0, 1.0, t)) - np.square(f0) - np.square(f1)
            F = np.column_stack([f0, f1])
            G = np.column_stack([g0, g1, g2])
        elif idx == 6:
            g = self._g2(X_arr)
            f0 = g * X_arr[:, 0]
            ratio = _safe_divide(f0, g)
            f1 = g * _safe_sqrt(1.1 * 1.1 - np.square(ratio))
            atan = np.arctan2(f1, f0)
            g0 = (
                np.square(f0 / (1.0 + self._la3(0.15, 6.0, 4.0, 10.0, atan)))
                + np.square(f1 / (1.0 + self._la3(0.75, 6.0, 4.0, 10.0, atan)))
                - 1.0
            )
            F = np.column_stack([f0, f1])
            G = g0[:, None]
        elif idx == 7:
            g = self._g3(X_arr)
            f0 = g * X_arr[:, 0]
            ratio = _safe_divide(f0, g)
            f1 = g * _safe_sqrt(1.0 - np.square(ratio))
            atan = np.arctan2(f1, f0)
            g0 = np.square(f0) + np.square(f1) - np.square(1.2 + np.abs(self._la2(0.4, 4.0, 1.0, 16.0, atan)))
            g1 = np.square(1.15 - self._la2(0.2, 4.0, 1.0, 8.0, atan)) - np.square(f0) - np.square(f1)
            F = np.column_stack([f0, f1])
            G = np.column_stack([g0, g1])
        elif idx == 8:
            g = self._g2(X_arr)
            F = g[:, None] * np.ones((X_arr.shape[0], self.n_obj), dtype=float)
            F[:, 1:] *= np.sin(0.5 * np.pi * X_arr[:, (self.n_obj - 2) :: -1])
            cos_terms = np.cos(0.5 * np.pi * X_arr[:, : self.n_obj - 1])
            F[:, :-1] *= np.flip(np.cumprod(cos_terms, axis=1), axis=1)
            f_sq = np.sum(F * F, axis=1)
            denom = _safe_sqrt(f_sq)
            angle = _safe_arcsin(_safe_divide(F[:, -1], denom))
            g0 = f_sq - np.square(1.25 - self._la2(0.5, 6.0, 1.0, 2.0, angle))
            G = g0[:, None]
        elif idx == 9:
            g = self._g1(X_arr)
            f0 = g * X_arr[:, 0]
            f1 = g * (1.0 - np.power(f0 / g, 0.6))
            t1 = (1.0 - 0.64 * f0 * f0 - f1) * (1.0 - 0.36 * f0 * f0 - f1)
            t2 = (1.35 * 1.35 - np.square(f0 + 0.35) - f1) * (1.15 * 1.15 - np.square(f0 + 0.15) - f1)
            g0 = np.minimum(t1, t2)
            F = np.column_stack([f0, f1])
            G = g0[:, None]
        elif idx == 10:
            g = self._g2(X_arr)
            f0 = g * np.power(X_arr[:, 0], self.n_var)
            f1 = g * (1.0 - np.square(f0 / g))
            g0 = -(2.0 - 4.0 * f0 * f0 - f1) * (2.0 - 8.0 * f0 * f0 - f1)
            g1 = (2.0 - 2.0 * f0 * f0 - f1) * (2.0 - 16.0 * f0 * f0 - f1)
            g2 = (1.0 - f0 * f0 - f1) * (1.2 - 1.2 * f0 * f0 - f1)
            F = np.column_stack([f0, f1])
            G = np.column_stack([g0, g1, g2])
        elif idx == 11:
            g = self._g3(X_arr)
            f0 = g * X_arr[:, 0]
            ratio = _safe_divide(f0, g)
            f1 = g * _safe_sqrt(2.0 - np.square(ratio))
            g0 = -(3.0 - f0 * f0 - f1) * (3.0 - 2.0 * f0 * f0 - f1)
            g1 = (3.0 - 0.625 * f0 * f0 - f1) * (3.0 - 7.0 * f0 * f0 - f1)
            g2 = -(1.62 - 0.18 * f0 * f0 - f1) * (1.125 - 0.125 * f0 * f0 - f1)
            g3 = (2.07 - 0.23 * f0 * f0 - f1) * (0.63 - 0.07 * f0 * f0 - f1)
            F = np.column_stack([f0, f1])
            G = np.column_stack([g0, g1, g2, g3])
        elif idx == 12:
            g = self._g1(X_arr)
            f0 = g * X_arr[:, 0]
            f1 = g * (0.85 - 0.8 * (f0 / g) - 0.08 * np.abs(np.sin(3.2 * np.pi * (f0 / g))))
            g0 = -(1.0 - 0.625 * f0 - f1 + 0.08 * np.sin(2.0 * np.pi * (f1 - f0 / 1.6))) * (
                1.4 - 0.875 * f0 - f1 + 0.08 * np.sin(2.0 * np.pi * (f1 / 1.4 - f0 / 1.6))
            )
            g1 = (1.0 - 0.8 * f0 - f1 + 0.08 * np.sin(2.0 * np.pi * (f1 - f0 / 1.5))) * (
                1.8 - 1.125 * f0 - f1 + 0.08 * np.sin(2.0 * np.pi * (f1 / 1.8 - f0 / 1.6))
            )
            F = np.column_stack([f0, f1])
            G = np.column_stack([g0, g1])
        elif idx == 13:
            g = self._g2(X_arr)
            f0 = g * X_arr[:, 0]
            f1 = g * (5.0 - np.exp(f0 / g) - np.abs(0.5 * np.sin(3.0 * np.pi * f0 / g)))
            g0 = -(5.0 - (1.0 + f0 + 0.5 * f0 * f0) - 0.5 * np.sin(3.0 * np.pi * f0) - f1) * (
                5.0 - (1.0 + 0.7 * f0) - 0.5 * np.sin(3.0 * np.pi * f0) - f1
            )
            g1 = (5.0 - np.exp(f0) - 0.5 * np.sin(3.0 * np.pi * f0) - f1) * (5.0 - (1.0 + 0.4 * f0) - 0.5 * np.sin(3.0 * np.pi * f0) - f1)
            F = np.column_stack([f0, f1])
            G = np.column_stack([g0, g1])
        elif idx == 14:
            g = self._g3(X_arr)
            F = np.zeros((X_arr.shape[0], self.n_obj), dtype=float)
            F[:, :-1] = X_arr[:, : self.n_obj - 1]
            la = self._la1(1.5, 1.1, 2.0, 1.0, F[:, :-1])
            inter = np.sum(6.0 - np.exp(F[:, :-1]) - la, axis=1)
            F[:, -1] = g / (self.n_obj - 1) * inter
            alpha = 6.1 - 1.0 - F[:, :-1] - 0.5 * np.square(F[:, :-1]) - la
            g0 = F[:, -1] - (1.0 / (self.n_obj - 1)) * np.sum(alpha, axis=1)
            G = g0[:, None]
        else:  # pragma: no cover - guarded by constructor
            raise RuntimeError("Unexpected MW variant.")

        F_out[:] = F
        out["G"] = G


def make_constrained_many_problem(name: str, *, n_var: int, n_obj: int | None) -> object:
    key = str(name).lower()
    if key in {"c1dtlz1", "c1dtlz3", "c2dtlz2", "c3dtlz1", "c3dtlz4"}:
        return CDTLZProblem(key, n_var=n_var, n_obj=(n_obj if n_obj is not None else 3))
    if key in {"dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3", "dc3dtlz1", "dc3dtlz3"}:
        return DCDTLZProblem(key, n_var=n_var, n_obj=(n_obj if n_obj is not None else 3))
    if key.startswith("mw"):
        return MWProblem(key, n_var=n_var, n_obj=n_obj)
    raise KeyError(f"Unknown constrained benchmark '{name}'.")


__all__ = ["DC_DTLZ_AUTHORS", "CDTLZProblem", "DCDTLZProblem", "MWProblem", "make_constrained_many_problem"]
