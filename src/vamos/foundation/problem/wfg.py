"""
NumPy implementation of the WFG1-WFG9 benchmark suite.
Based on the standard WFG toolkit definitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem


def _correct_to_01(x: np.ndarray[Any, Any], epsilon: float = 1.0e-10) -> np.ndarray[Any, Any]:
    x = np.asarray(x, dtype=float).copy()
    x[np.logical_and(x < 0.0, x >= -epsilon)] = 0.0
    x[np.logical_and(x > 1.0, x <= 1.0 + epsilon)] = 1.0
    return x


# -----------------------------------------------------------------------------
# Transformations
# -----------------------------------------------------------------------------


def _transformation_shift_linear(value: np.ndarray[Any, Any], shift: float = 0.35) -> np.ndarray[Any, Any]:
    return _correct_to_01(np.fabs(value - shift) / np.fabs(np.floor(shift - value) + shift))


def _transformation_shift_deceptive(y: np.ndarray[Any, Any], a: float = 0.35, b: float = 0.005, c: float = 0.05) -> np.ndarray[Any, Any]:
    tmp1 = np.floor(y - a + b) * (1.0 - c + (a - b) / b) / (a - b)
    tmp2 = np.floor(a + b - y) * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b)
    ret = 1.0 + (np.fabs(y - a) - b) * (tmp1 + tmp2 + 1.0 / b)
    return _correct_to_01(ret)


def _transformation_shift_multi_modal(y: np.ndarray[Any, Any], a: float, b: float, c: float) -> np.ndarray[Any, Any]:
    tmp1 = np.fabs(y - c) / (2.0 * (np.floor(c - y) + c))
    tmp2 = (4.0 * a + 2.0) * np.pi * (0.5 - tmp1)
    ret = (1.0 + np.cos(tmp2) + 4.0 * b * np.power(tmp1, 2.0)) / (b + 2.0)
    return _correct_to_01(ret)


def _transformation_bias_flat(y: np.ndarray[Any, Any], a: float, b: float, c: float) -> np.ndarray[Any, Any]:
    ret = a + np.minimum(0.0, np.floor(y - b)) * (a * (b - y) / b) - np.minimum(0.0, np.floor(c - y)) * ((1.0 - a) * (y - c) / (1.0 - c))
    return _correct_to_01(ret)


def _transformation_bias_poly(y: np.ndarray[Any, Any], alpha: float) -> np.ndarray[Any, Any]:
    return _correct_to_01(y**alpha)


def _transformation_param_dependent(
    y: np.ndarray[Any, Any], y_deg: np.ndarray[Any, Any], a: float = 0.98 / 49.98, b: float = 0.02, c: float = 50.0
) -> np.ndarray[Any, Any]:
    aux = a - (1.0 - 2.0 * y_deg) * np.fabs(np.floor(0.5 - y_deg) + a)
    ret = np.power(y, b + (c - b) * aux)
    return _correct_to_01(ret)


def _transformation_param_deceptive(y: np.ndarray[Any, Any], a: float = 0.35, b: float = 0.001, c: float = 0.05) -> np.ndarray[Any, Any]:
    tmp1 = np.floor(y - a + b) * (1.0 - c + (a - b) / b) / (a - b)
    tmp2 = np.floor(a + b - y) * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b)
    ret = 1.0 + (np.fabs(y - a) - b) * (tmp1 + tmp2 + 1.0 / b)
    return _correct_to_01(ret)


# -----------------------------------------------------------------------------
# Reductions
# -----------------------------------------------------------------------------


def _reduction_weighted_sum(y: np.ndarray[Any, Any], w: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return _correct_to_01(np.dot(y, w) / w.sum())


def _reduction_weighted_sum_uniform(y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return _correct_to_01(y.mean(axis=1))


def _reduction_non_sep(y: np.ndarray[Any, Any], a: int) -> np.ndarray[Any, Any]:
    n, m = y.shape
    val = np.ceil(a / 2.0)
    num = np.zeros(n, dtype=float)
    for j in range(m):
        num += y[:, j]
        for k in range(a - 1):
            num += np.fabs(y[:, j] - y[:, (1 + j + k) % m])
    denom = m * val * (1.0 + 2.0 * a - 2.0 * val) / a
    return _correct_to_01(num / denom)


# -----------------------------------------------------------------------------
# Shapes
# -----------------------------------------------------------------------------


def _shape_concave(x: np.ndarray[Any, Any], m: int) -> np.ndarray[Any, Any]:
    m_dim = x.shape[1]
    if m == 1:
        ret = np.prod(np.sin(0.5 * x[:, :m_dim] * np.pi), axis=1)
    elif 1 < m <= m_dim:
        ret = np.prod(np.sin(0.5 * x[:, : m_dim - m + 1] * np.pi), axis=1)
        ret *= np.cos(0.5 * x[:, m_dim - m + 1] * np.pi)
    else:
        ret = np.cos(0.5 * x[:, 0] * np.pi)
    return _correct_to_01(ret)


def _shape_convex(x: np.ndarray[Any, Any], m: int) -> np.ndarray[Any, Any]:
    m_dim = x.shape[1]
    if m == 1:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, :m_dim] * np.pi), axis=1)
    elif 1 < m <= m_dim:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, : m_dim - m + 1] * np.pi), axis=1)
        ret *= 1.0 - np.sin(0.5 * x[:, m_dim - m + 1] * np.pi)
    else:
        ret = 1.0 - np.sin(0.5 * x[:, 0] * np.pi)
    return _correct_to_01(ret)


def _shape_linear(x: np.ndarray[Any, Any], m: int) -> np.ndarray[Any, Any]:
    m_dim = x.shape[1]
    if m == 1:
        ret = np.prod(x, axis=1)
    elif 1 < m <= m_dim:
        ret = np.prod(x[:, : m_dim - m + 1], axis=1)
        ret *= 1.0 - x[:, m_dim - m + 1]
    else:
        ret = 1.0 - x[:, 0]
    return _correct_to_01(ret)


def _shape_mixed(x: np.ndarray[Any, Any], a: float = 5.0, alpha: float = 1.0) -> np.ndarray[Any, Any]:
    aux = 2.0 * a * np.pi
    ret = np.power(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha)
    return _correct_to_01(ret)


def _shape_disconnected(x: np.ndarray[Any, Any], alpha: float = 1.0, beta: float = 1.0, a: float = 5.0) -> np.ndarray[Any, Any]:
    aux = np.cos(a * np.pi * x**beta)
    return _correct_to_01(1.0 - x**alpha * aux**2)


def _validate_wfg2_wfg3(l: int) -> None:
    if l % 2 != 0:
        raise ValueError("In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.")


# -----------------------------------------------------------------------------
# WFG Problems
# -----------------------------------------------------------------------------


class WFGProblem(Problem):
    def __init__(self, name: str, n_var: int, n_obj: int, k: int | None = None, l: int | None = None) -> None:
        self.name = name
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = 2.0 * np.arange(1, self.n_var + 1, dtype=float)
        self.S = np.arange(2, 2 * self.n_obj + 1, 2, dtype=float)
        self.A = np.ones(self.n_obj - 1, dtype=float)

        if k is None:
            k = 4 if self.n_obj == 2 else 2 * (self.n_obj - 1)
        if l is None:
            l = self.n_var - k

        self.k = int(k)
        self.l = int(l)
        self._validate()

    def _validate(self) -> None:
        if self.n_obj < 2:
            raise ValueError("WFG problems must have two or more objectives.")
        if self.k % (self.n_obj - 1) != 0:
            raise ValueError("Position parameter (k) must be divisible by number of objectives minus one.")
        if self.k < 4:
            raise ValueError("Position parameter (k) must be greater or equal than 4.")
        if self.l < 1:
            raise ValueError(
                f"WFG problems need at least one distance-related variable (l = n_var - k >= 1): "
                f"with k={self.k}, n_var must be at least {self.k + 1}; got n_var={self.n_var}."
            )
        if (self.k + self.l) < self.n_obj:
            raise ValueError("Sum of distance and position parameters must be >= number of objectives.")
        if (self.k + self.l) != self.n_var:
            raise ValueError("n_var must equal k + l.")

    def _post(self, t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        x = []
        for i in range(t.shape[1] - 1):
            x.append(np.maximum(t[:, -1], self.A[i]) * (t[:, i] - 0.5) + 0.5)
        x.append(t[:, -1])
        return np.column_stack(x)

    def _calculate(self, x: np.ndarray[Any, Any], h: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        return np.asarray(x[:, -1][:, None] + self.S * np.column_stack(h), dtype=float)

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        raise NotImplementedError

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        x = np.asarray(X, dtype=float)
        f = self._evaluate(x)
        if "F" in out and out["F"] is not None:
            out["F"][:] = f
        else:
            out["F"] = f


class WFG1Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg1", n_var, n_obj, k, l)

    @staticmethod
    def _t1(x: np.ndarray[Any, Any], n: int, k: int) -> np.ndarray[Any, Any]:
        x[:, k:n] = _transformation_shift_linear(x[:, k:n], 0.35)
        return x

    @staticmethod
    def _t2(x: np.ndarray[Any, Any], n: int, k: int) -> np.ndarray[Any, Any]:
        x[:, k:n] = _transformation_bias_flat(x[:, k:n], 0.8, 0.75, 0.85)
        return x

    @staticmethod
    def _t3(x: np.ndarray[Any, Any], n: int) -> np.ndarray[Any, Any]:
        x[:, :n] = _transformation_bias_poly(x[:, :n], 0.02)
        return x

    @staticmethod
    def _t4(x: np.ndarray[Any, Any], m: int, n: int, k: int) -> np.ndarray[Any, Any]:
        w = np.arange(2, 2 * n + 1, 2, dtype=float)
        gap = k // (m - 1)
        t = []
        for idx in range(1, m):
            head = (idx - 1) * gap
            tail = idx * gap
            t.append(_reduction_weighted_sum(x[:, head:tail], w[head:tail]))
        t.append(_reduction_weighted_sum(x[:, k:n], w[k:n]))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        y = self._t1(y, self.n_var, self.k)
        y = self._t2(y, self.n_var, self.k)
        y = self._t3(y, self.n_var)
        y = self._t4(y, self.n_obj, self.n_var, self.k)
        y = self._post(y)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_mixed(y[:, 0], alpha=1.0, a=5.0))
        return self._calculate(y, h)


class WFG2Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg2", n_var, n_obj, k, l)
        _validate_wfg2_wfg3(self.l)

    @staticmethod
    def _t2(x: np.ndarray[Any, Any], n: int, k: int) -> np.ndarray[Any, Any]:
        y = [x[:, i] for i in range(k)]
        l = n - k
        ind_non_sep = k + l // 2
        i = k + 1
        while i <= ind_non_sep:
            head = k + 2 * (i - k) - 2
            tail = k + 2 * (i - k)
            y.append(_reduction_non_sep(x[:, head:tail], 2))
            i += 1
        return np.column_stack(y)

    @staticmethod
    def _t3(x: np.ndarray[Any, Any], m: int, n: int, k: int) -> np.ndarray[Any, Any]:
        ind_r_sum = k + (n - k) // 2
        gap = k // (m - 1)
        t = [_reduction_weighted_sum_uniform(x[:, (idx - 1) * gap : (idx * gap)]) for idx in range(1, m)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:ind_r_sum]))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        y = WFG1Problem._t1(y, self.n_var, self.k)
        y = self._t2(y, self.n_var, self.k)
        y = self._t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_disconnected(y[:, 0], alpha=1.0, beta=1.0, a=5.0))
        return self._calculate(y, h)


class WFG3Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg3", n_var, n_obj, k, l)
        _validate_wfg2_wfg3(self.l)
        self.A[1:] = 0.0

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        y = WFG1Problem._t1(y, self.n_var, self.k)
        y = WFG2Problem._t2(y, self.n_var, self.k)
        y = WFG2Problem._t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y)

        h = [_shape_linear(y[:, :-1], m + 1) for m in range(self.n_obj)]
        return self._calculate(y, h)


class WFG4Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg4", n_var, n_obj, k, l)

    @staticmethod
    def _t1(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return _transformation_shift_multi_modal(x, 30.0, 10.0, 0.35)

    @staticmethod
    def _t2(x: np.ndarray[Any, Any], m: int, k: int) -> np.ndarray[Any, Any]:
        gap = k // (m - 1)
        t = [_reduction_weighted_sum_uniform(x[:, (idx - 1) * gap : (idx * gap)]) for idx in range(1, m)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:]))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        y = self._t1(y)
        y = self._t2(y, self.n_obj, self.k)
        y = self._post(y)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]
        return self._calculate(y, h)


class WFG5Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg5", n_var, n_obj, k, l)

    @staticmethod
    def _t1(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return _transformation_param_deceptive(x, a=0.35, b=0.001, c=0.05)

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        y = self._t1(y)
        y = WFG4Problem._t2(y, self.n_obj, self.k)
        y = self._post(y)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]
        return self._calculate(y, h)


class WFG6Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg6", n_var, n_obj, k, l)

    @staticmethod
    def _t2(x: np.ndarray[Any, Any], m: int, n: int, k: int) -> np.ndarray[Any, Any]:
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (idx - 1) * gap : (idx * gap)], gap) for idx in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        y = WFG1Problem._t1(y, self.n_var, self.k)
        y = self._t2(y, self.n_obj, self.n_var, self.k)
        y = self._post(y)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]
        return self._calculate(y, h)


class WFG7Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg7", n_var, n_obj, k, l)

    @staticmethod
    def _t1(x: np.ndarray[Any, Any], k: int) -> np.ndarray[Any, Any]:
        for i in range(k):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1 :])
            x[:, i] = _transformation_param_dependent(x[:, i], aux)
        return x

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        y = self._t1(y, self.k)
        y = WFG1Problem._t1(y, self.n_var, self.k)
        y = WFG4Problem._t2(y, self.n_obj, self.k)
        y = self._post(y)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]
        return self._calculate(y, h)


class WFG8Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg8", n_var, n_obj, k, l)

    @staticmethod
    def _t1(x: np.ndarray[Any, Any], n: int, k: int) -> np.ndarray[Any, Any]:
        ret = []
        for i in range(k, n):
            aux = _reduction_weighted_sum_uniform(x[:, :i])
            ret.append(_transformation_param_dependent(x[:, i], aux, a=0.98 / 49.98, b=0.02, c=50.0))
        return np.column_stack(ret)

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        if self.k < self.n_var:
            y[:, self.k : self.n_var] = self._t1(y, self.n_var, self.k)
        y = WFG1Problem._t1(y, self.n_var, self.k)
        y = WFG4Problem._t2(y, self.n_obj, self.k)
        y = self._post(y)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]
        return self._calculate(y, h)


class WFG9Problem(WFGProblem):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None) -> None:
        super().__init__("wfg9", n_var, n_obj, k, l)

    @staticmethod
    def _t1(x: np.ndarray[Any, Any], n: int) -> np.ndarray[Any, Any]:
        ret = []
        for i in range(0, n - 1):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1 :])
            ret.append(_transformation_param_dependent(x[:, i], aux))
        return np.column_stack(ret)

    @staticmethod
    def _t2(x: np.ndarray[Any, Any], n: int, k: int) -> np.ndarray[Any, Any]:
        a = [_transformation_shift_deceptive(x[:, i], 0.35, 0.001, 0.05) for i in range(k)]
        b = [_transformation_shift_multi_modal(x[:, i], 30.0, 95.0, 0.35) for i in range(k, n)]
        return np.column_stack(a + b)

    @staticmethod
    def _t3(x: np.ndarray[Any, Any], m: int, n: int, k: int) -> np.ndarray[Any, Any]:
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (idx - 1) * gap : (idx * gap)], gap) for idx in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        if self.n_var > 1:
            y[:, : self.n_var - 1] = self._t1(y, self.n_var)
        y = self._t2(y, self.n_var, self.k)
        y = self._t3(y, self.n_obj, self.n_var, self.k)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]
        return self._calculate(y, h)


__all__ = [
    "WFGProblem",
    "WFG1Problem",
    "WFG2Problem",
    "WFG3Problem",
    "WFG4Problem",
    "WFG5Problem",
    "WFG6Problem",
    "WFG7Problem",
    "WFG8Problem",
    "WFG9Problem",
]
