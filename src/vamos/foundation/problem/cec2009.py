from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem


class _CEC2009Base(Problem):
    def __init__(
        self,
        *,
        n_var: int,
        n_obj: int,
        n_var_min: int,
        lower_rest: float,
        upper_rest: float,
    ) -> None:
        if n_var < n_var_min:
            raise ValueError(f"CEC2009 problem requires at least {n_var_min} decision variables.")
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)

        xl = np.full(self.n_var, float(lower_rest), dtype=float)
        xu = np.full(self.n_var, float(upper_rest), dtype=float)
        head = max(1, self.n_obj - 1)
        xl[:head] = 0.0
        xu[:head] = 1.0
        self.xl = xl
        self.xu = xu

    def _prepare(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        n = X.shape[0]
        F = out.get("F")
        if F is None or F.shape != (n, self.n_obj):
            F = np.empty((n, self.n_obj), dtype=float)
            out["F"] = F
        return X, F

    def _split_even_odd(self, *, start_j: int = 2) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        j = np.arange(start_j, self.n_var + 1)
        even_mask = j % 2 == 0
        odd_mask = ~even_mask
        return j, even_mask, odd_mask

    def _split_mod3(
        self, *, start_j: int = 3
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        j = np.arange(start_j, self.n_var + 1)
        mod1 = j % 3 == 1
        mod2 = j % 3 == 2
        mod0 = j % 3 == 0
        return j, mod1, mod2, mod0


class _CEC2009UF2ObjBase(_CEC2009Base):
    def __init__(self, n_var: int = 30, *, lower_rest: float = -1.0, upper_rest: float = 1.0) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_var_min=3, lower_rest=lower_rest, upper_rest=upper_rest)


class _CEC2009UF3ObjBase(_CEC2009Base):
    def __init__(self, n_var: int = 30, *, lower_rest: float = -2.0, upper_rest: float = 2.0) -> None:
        super().__init__(n_var=n_var, n_obj=3, n_var_min=3, lower_rest=lower_rest, upper_rest=upper_rest)


class CEC2009_UF1(_CEC2009UF2ObjBase):
    """UF1 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        angles = 6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var
        y = rest - np.sin(angles)
        y2 = y * y

        sum_odd = np.sum(y2[:, odd_mask], axis=1)
        sum_even = np.sum(y2[:, even_mask], axis=1)
        count_odd = max(1, int(odd_mask.sum()))
        count_even = max(1, int(even_mask.sum()))

        F[:, 0] = x0 + 2.0 * sum_odd / count_odd
        F[:, 1] = 1.0 - np.sqrt(x0) + 2.0 * sum_even / count_even


class CEC2009_UF2(_CEC2009UF2ObjBase):
    """UF2 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        base = 0.3 * x0[:, None] * (x0[:, None] * np.cos(24.0 * np.pi * x0[:, None] + 4.0 * j * np.pi / self.n_var) + 2.0)
        angles = 6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var
        sin_term = np.sin(angles)
        cos_term = np.cos(angles)

        y_even = rest[:, even_mask] - base[:, even_mask] * sin_term[:, even_mask]
        y_odd = rest[:, odd_mask] - base[:, odd_mask] * cos_term[:, odd_mask]

        sum_even = np.sum(y_even * y_even, axis=1)
        sum_odd = np.sum(y_odd * y_odd, axis=1)
        count_even = max(1, int(even_mask.sum()))
        count_odd = max(1, int(odd_mask.sum()))

        F[:, 0] = x0 + 2.0 * sum_odd / count_odd
        F[:, 1] = 1.0 - np.sqrt(x0) + 2.0 * sum_even / count_even


class CEC2009_UF3(_CEC2009UF2ObjBase):
    """UF3 from the CEC2009 competition."""

    def __init__(self, n_var: int = 30, *, lower_rest: float = 0.0, upper_rest: float = 1.0) -> None:
        # Unlike UF1 and UF2, UF3 searches [0, 1]^n.
        super().__init__(n_var=n_var, lower_rest=lower_rest, upper_rest=upper_rest)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        power = 0.5 * (1.0 + 3.0 * (j - 2.0) / (self.n_var - 2.0))
        y = rest - np.power(x0[:, None], power)
        p = np.cos(20.0 * y * np.pi / np.sqrt(j))

        y_even = y[:, even_mask]
        y_odd = y[:, odd_mask]
        p_even = p[:, even_mask]
        p_odd = p[:, odd_mask]

        sum_even = np.sum(y_even * y_even, axis=1)
        sum_odd = np.sum(y_odd * y_odd, axis=1)
        prod_even = np.prod(p_even, axis=1)
        prod_odd = np.prod(p_odd, axis=1)

        count_even = max(1, int(even_mask.sum()))
        count_odd = max(1, int(odd_mask.sum()))

        F[:, 0] = x0 + 2.0 * (4.0 * sum_odd - 2.0 * prod_odd + 2.0) / count_odd
        F[:, 1] = 1.0 - np.sqrt(x0) + 2.0 * (4.0 * sum_even - 2.0 * prod_even + 2.0) / count_even


class CEC2009_UF4(_CEC2009UF2ObjBase):
    """UF4 from the CEC2009 competition."""

    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var, lower_rest=-2.0, upper_rest=2.0)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        y = rest - np.sin(6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var)
        h = np.abs(y) / (1.0 + np.exp(2.0 * np.abs(y)))

        sum_odd = np.sum(h[:, odd_mask], axis=1)
        sum_even = np.sum(h[:, even_mask], axis=1)
        count_odd = max(1, int(odd_mask.sum()))
        count_even = max(1, int(even_mask.sum()))

        F[:, 0] = x0 + 2.0 * sum_odd / count_odd
        F[:, 1] = 1.0 - x0 * x0 + 2.0 * sum_even / count_even


class CEC2009_UF5(_CEC2009UF2ObjBase):
    """UF5 from the CEC2009 competition."""

    def __init__(self, n_var: int = 30, *, N: int = 10, epsilon: float = 0.1) -> None:
        super().__init__(n_var=n_var, lower_rest=-1.0, upper_rest=1.0)
        self._N = int(N)
        self._epsilon = float(epsilon)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        y = rest - np.sin(6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var)
        h = 2.0 * y * y - np.cos(4.0 * np.pi * y) + 1.0

        sum_odd = np.sum(h[:, odd_mask], axis=1)
        sum_even = np.sum(h[:, even_mask], axis=1)
        count_odd = max(1, int(odd_mask.sum()))
        count_even = max(1, int(even_mask.sum()))
        hump = (0.5 / self._N + self._epsilon) * np.abs(np.sin(2.0 * self._N * np.pi * x0))

        F[:, 0] = x0 + hump + 2.0 * sum_odd / count_odd
        F[:, 1] = 1.0 - x0 + hump + 2.0 * sum_even / count_even


class CEC2009_UF6(_CEC2009UF2ObjBase):
    """UF6 from the CEC2009 competition."""

    def __init__(self, n_var: int = 30, *, N: int = 2, epsilon: float = 0.1) -> None:
        super().__init__(n_var=n_var, lower_rest=-1.0, upper_rest=1.0)
        self._N = int(N)
        self._epsilon = float(epsilon)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        y = rest - np.sin(6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var)
        p = np.cos(20.0 * y * np.pi / np.sqrt(j))

        y2 = y * y
        sum_odd = np.sum(y2[:, odd_mask], axis=1)
        sum_even = np.sum(y2[:, even_mask], axis=1)
        prod_odd = np.prod(p[:, odd_mask], axis=1)
        prod_even = np.prod(p[:, even_mask], axis=1)
        count_odd = max(1, int(odd_mask.sum()))
        count_even = max(1, int(even_mask.sum()))

        hump = 2.0 * (0.5 / self._N + self._epsilon) * np.sin(2.0 * self._N * np.pi * x0)
        hump = np.maximum(0.0, hump)

        F[:, 0] = x0 + hump + 2.0 * (4.0 * sum_odd - 2.0 * prod_odd + 2.0) / count_odd
        F[:, 1] = 1.0 - x0 + hump + 2.0 * (4.0 * sum_even - 2.0 * prod_even + 2.0) / count_even


class CEC2009_UF7(_CEC2009UF2ObjBase):
    """UF7 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        y = rest - np.sin(6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var)
        y2 = y * y

        sum_odd = np.sum(y2[:, odd_mask], axis=1)
        sum_even = np.sum(y2[:, even_mask], axis=1)
        count_odd = max(1, int(odd_mask.sum()))
        count_even = max(1, int(even_mask.sum()))
        root = np.power(x0, 0.2)

        F[:, 0] = root + 2.0 * sum_odd / count_odd
        F[:, 1] = 1.0 - root + 2.0 * sum_even / count_even


class CEC2009_UF8(_CEC2009UF3ObjBase):
    """UF8 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        x1 = X[:, 1]
        rest = X[:, 2:]
        j, mod1, mod2, mod0 = self._split_mod3()
        phase = 2.0 * np.pi * x0[:, None] + j * np.pi / self.n_var
        y = rest - 2.0 * x1[:, None] * np.sin(phase)
        y2 = y * y

        sum1 = np.sum(y2[:, mod1], axis=1)
        sum2 = np.sum(y2[:, mod2], axis=1)
        sum3 = np.sum(y2[:, mod0], axis=1)
        count1 = max(1, int(mod1.sum()))
        count2 = max(1, int(mod2.sum()))
        count3 = max(1, int(mod0.sum()))

        F[:, 0] = np.cos(0.5 * np.pi * x0) * np.cos(0.5 * np.pi * x1) + 2.0 * sum1 / count1
        F[:, 1] = np.cos(0.5 * np.pi * x0) * np.sin(0.5 * np.pi * x1) + 2.0 * sum2 / count2
        F[:, 2] = np.sin(0.5 * np.pi * x0) + 2.0 * sum3 / count3


class CEC2009_UF9(_CEC2009UF3ObjBase):
    """UF9 from the CEC2009 competition.

    Like UF8, UF9 adds the squared distance ``y_j**2`` of the CEC2009 report.
    """

    def __init__(self, n_var: int = 30, *, epsilon: float = 0.1) -> None:
        super().__init__(n_var=n_var, lower_rest=-2.0, upper_rest=2.0)
        self._epsilon = float(epsilon)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        x1 = X[:, 1]
        rest = X[:, 2:]
        j, mod1, mod2, mod0 = self._split_mod3()
        phase = 2.0 * np.pi * x0[:, None] + j * np.pi / self.n_var
        y = rest - 2.0 * x1[:, None] * np.sin(phase)
        y2 = y * y

        sum1 = np.sum(y2[:, mod1], axis=1)
        sum2 = np.sum(y2[:, mod2], axis=1)
        sum3 = np.sum(y2[:, mod0], axis=1)
        count1 = max(1, int(mod1.sum()))
        count2 = max(1, int(mod2.sum()))
        count3 = max(1, int(mod0.sum()))

        front_mod = (1.0 + self._epsilon) * (1.0 - 4.0 * np.square(2.0 * x0 - 1.0))
        front_mod = np.maximum(0.0, front_mod)

        F[:, 0] = 0.5 * (front_mod + 2.0 * x0) * x1 + 2.0 * sum1 / count1
        F[:, 1] = 0.5 * (front_mod - 2.0 * x0 + 2.0) * x1 + 2.0 * sum2 / count2
        F[:, 2] = 1.0 - x1 + 2.0 * sum3 / count3


class CEC2009_UF10(_CEC2009UF3ObjBase):
    """UF10 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        x1 = X[:, 1]
        rest = X[:, 2:]
        j, mod1, mod2, mod0 = self._split_mod3()
        phase = 2.0 * np.pi * x0[:, None] + j * np.pi / self.n_var
        y = rest - 2.0 * x1[:, None] * np.sin(phase)
        h = 4.0 * y * y - np.cos(8.0 * np.pi * y) + 1.0

        sum1 = np.sum(h[:, mod1], axis=1)
        sum2 = np.sum(h[:, mod2], axis=1)
        sum3 = np.sum(h[:, mod0], axis=1)
        count1 = max(1, int(mod1.sum()))
        count2 = max(1, int(mod2.sum()))
        count3 = max(1, int(mod0.sum()))

        F[:, 0] = np.cos(0.5 * np.pi * x0) * np.cos(0.5 * np.pi * x1) + 2.0 * sum1 / count1
        F[:, 1] = np.cos(0.5 * np.pi * x0) * np.sin(0.5 * np.pi * x1) + 2.0 * sum2 / count2
        F[:, 2] = np.sin(0.5 * np.pi * x0) + 2.0 * sum3 / count3


class CEC2009_CF1(_CEC2009UF2ObjBase):
    """Constrained CF1 from the CEC2009 competition, on ``[0, 1]^n``.

    CEC2009 states the constraint as ``c(x) >= 0``; ``out["G"]`` holds ``-c(x)`` so
    that, as everywhere in VAMOS, ``G <= 0`` means feasible.
    """

    n_constraints = 1

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, lower_rest=0.0, upper_rest=1.0)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        power = 0.5 * (1.0 + 3.0 * (j - 2.0) / (self.n_var - 2.0))
        y = rest - np.power(x0[:, None], power)

        y_odd = y[:, odd_mask]
        y_even = y[:, even_mask]
        sum_odd = np.sum(y_odd * y_odd, axis=1)
        sum_even = np.sum(y_even * y_even, axis=1)
        count_odd = max(1, int(odd_mask.sum()))
        count_even = max(1, int(even_mask.sum()))

        f0 = x0 + 2.0 * sum_odd / count_odd
        f1 = 1.0 - x0 + 2.0 * sum_even / count_even
        constraint = f1 + f0 - np.abs(np.sin(10.0 * np.pi * (f0 - f1 + 1.0))) - 1.0

        F[:, 0] = f0
        F[:, 1] = f1
        out["G"] = -constraint[:, None]


__all__ = [
    "CEC2009_UF1",
    "CEC2009_UF2",
    "CEC2009_UF3",
    "CEC2009_UF4",
    "CEC2009_UF5",
    "CEC2009_UF6",
    "CEC2009_UF7",
    "CEC2009_UF8",
    "CEC2009_UF9",
    "CEC2009_UF10",
    "CEC2009_CF1",
]
