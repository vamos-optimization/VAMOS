from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.problem.base import Problem


def _as_float_vector(values: object) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float)


def _sphere(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return _as_float_vector(np.sum(np.square(x), axis=1))


def _schwefel(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return _as_float_vector(np.max(np.abs(x), axis=1))


def _rastrigin(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return _as_float_vector(np.sum(np.square(x) - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1))


def _rosenbrock(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        return np.zeros(x.shape[0], dtype=float)
    a = x[:, :-1]
    b = x[:, 1:]
    return _as_float_vector(np.sum(100.0 * np.square(np.square(a) - b) + np.square(a - 1.0), axis=1))


def _ackley(x: NDArray[np.float64]) -> NDArray[np.float64]:
    d = x.shape[1]
    if d == 0:
        return np.zeros(x.shape[0], dtype=float)
    sum_sq = np.sum(np.square(x), axis=1)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x), axis=1)
    return _as_float_vector(20.0 - 20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + np.e)


def _griewank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    d = x.shape[1]
    if d == 0:
        return np.zeros(x.shape[0], dtype=float)
    sum_term = np.sum(np.square(x) / 4000.0, axis=1)
    denom = np.sqrt(np.arange(1, d + 1, dtype=float))
    prod_term = np.prod(np.cos(x / denom), axis=1)
    return _as_float_vector(sum_term - prod_term + 1.0)


class _LSMOPBase(Problem):
    """
    Vectorized implementation of the LSMOP benchmark family.

    Definitions follow Cheng, Jin and Olhofer (2017), "Test problems for large-scale multiobjective and
    many-objective optimization", IEEE Transactions on Cybernetics 47(12), as implemented in PlatEMO
    (``LSMOP1.m``-``LSMOP9.m``). LSMOP1-4 use the linear linkage ``(1 + i/n) * x_i - 10 * x_1`` and LSMOP5-9 the
    nonlinear ``(1 + cos(0.5 * pi * i/n)) * x_i - 10 * x_1``, for ``i >= M``. jMetal differs in two places:
    it uses the nonlinear linkage in all nine problems, and LSMOP9 does not divide each group's ``g`` by
    ``sublen``.
    """

    def __init__(
        self,
        *,
        n_var: int = 300,
        n_obj: int = 3,
        nk: int = 5,
        odd_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        even_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        if n_obj < 2:
            raise ValueError("LSMOP requires at least two objectives.")
        if n_var <= n_obj:
            raise ValueError("LSMOP requires n_var > n_obj.")
        if nk <= 0:
            raise ValueError("LSMOP requires nk > 0.")

        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.nk = int(nk)

        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.full(self.n_var, 10.0, dtype=float)
        self.xu[: self.n_obj - 1] = 1.0

        self._odd_func = odd_func
        self._even_func = even_func

        self.sub_len, self.len = self._build_group_layout(self.n_var, self.n_obj, self.nk)
        if np.any(self.sub_len <= 0):
            raise ValueError("Invalid LSMOP partitioning (some subcomponents are empty). Increase n_var, reduce n_obj, or reduce nk.")

    @staticmethod
    def _build_group_layout(n_var: int, n_obj: int, nk: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        c = 3.8 * 0.1 * (1.0 - 0.1)
        c_values = [c]
        total = c
        for _ in range(n_obj - 1):
            c = 3.8 * c * (1.0 - c)
            c_values.append(c)
            total += c

        budget = n_var - n_obj + 1
        sub_len = np.array([int(np.floor(ci / total * budget / nk)) for ci in c_values], dtype=int)
        cum = np.cumsum(sub_len * nk)
        offsets = np.concatenate(([0], cum))
        return sub_len, offsets

    def _prepare(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        n = X.shape[0]
        F = out.get("F")
        if F is None or F.shape != (n, self.n_obj):
            F = np.empty((n, self.n_obj), dtype=float)
            out["F"] = F
        return X, F

    def _linkage_scale(self, ratio: NDArray[np.float64]) -> NDArray[np.float64]:
        """Nonlinear linkage factor of LSMOP5-9, for ``ratio = i/n`` with ``i = M, ..., n``."""
        return _as_float_vector(1.0 + np.cos(ratio * np.pi / 2.0))

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        Y = np.array(X, copy=True, dtype=float)
        idx = np.arange(self.n_obj, self.n_var + 1, dtype=float)
        scale = self._linkage_scale(idx / self.n_var)
        Y[:, self.n_obj - 1 :] = scale[None, :] * Y[:, self.n_obj - 1 :] - 10.0 * Y[:, [0]]
        return Y

    def _accumulate_g(self, Y: NDArray[np.float64]) -> NDArray[np.float64]:
        n = Y.shape[0]
        G = np.zeros((n, self.n_obj), dtype=float)

        for obj_idx in range(self.n_obj):
            block_width = int(self.sub_len[obj_idx])
            base = int(self.len[obj_idx] + self.n_obj - 1)
            func = self._odd_func if (obj_idx + 1) % 2 == 1 else self._even_func
            for sub_idx in range(self.nk):
                start = base + sub_idx * block_width
                end = start + block_width
                G[:, obj_idx] += func(Y[:, start:end])

        return G

    def _normalize_g(self, G: NDArray[np.float64]) -> NDArray[np.float64]:
        denom = self.sub_len.astype(float) * float(self.nk)
        return G / denom[None, :]

    def _evaluate_f(self, X: NDArray[np.float64], G: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        X, F = self._prepare(X, out)
        Y = self._transform(X)
        G = self._accumulate_g(Y)
        F_val = self._evaluate_f(X, G)
        F[:] = F_val


class _LSMOP1To4Base(_LSMOPBase):
    def _linkage_scale(self, ratio: NDArray[np.float64]) -> NDArray[np.float64]:
        """Linear linkage factor of LSMOP1-4, for ``ratio = i/n`` with ``i = M, ..., n``."""
        return _as_float_vector(1.0 + ratio)

    def _evaluate_f(self, X: NDArray[np.float64], G: NDArray[np.float64]) -> NDArray[np.float64]:
        G_norm = self._normalize_g(G)
        n = X.shape[0]

        left: NDArray[np.float64] = np.ones((n, self.n_obj), dtype=float)
        left[:, 1:] = X[:, : self.n_obj - 1]
        left = _as_float_vector(np.flip(np.cumprod(left, axis=1), axis=1))

        right = np.ones((n, self.n_obj), dtype=float)
        right[:, 1:] = 1.0 - np.flip(X[:, : self.n_obj - 1], axis=1)

        operand = left * right
        return _as_float_vector((1.0 + G_norm) * operand)


class _LSMOP5To8Base(_LSMOPBase):
    def _evaluate_f(self, X: NDArray[np.float64], G: NDArray[np.float64]) -> NDArray[np.float64]:
        G_norm = self._normalize_g(G)
        n = X.shape[0]

        left: NDArray[np.float64] = np.ones((n, self.n_obj), dtype=float)
        left[:, 1:] = np.cos(X[:, : self.n_obj - 1] * np.pi / 2.0)
        left = _as_float_vector(np.flip(np.cumprod(left, axis=1), axis=1))

        right = np.ones((n, self.n_obj), dtype=float)
        right[:, 1:] = np.sin(np.flip(X[:, : self.n_obj - 1], axis=1) * np.pi / 2.0)

        operand = left * right
        shifted = np.concatenate((G_norm[:, 1:], np.zeros((n, 1), dtype=float)), axis=1)
        return _as_float_vector((1.0 + G_norm + shifted) * operand)


class LSMOP1(_LSMOP1To4Base):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_sphere, even_func=_sphere)


class LSMOP2(_LSMOP1To4Base):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_griewank, even_func=_schwefel)


class LSMOP3(_LSMOP1To4Base):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_rastrigin, even_func=_rosenbrock)


class LSMOP4(_LSMOP1To4Base):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_ackley, even_func=_griewank)


class LSMOP5(_LSMOP5To8Base):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_sphere, even_func=_sphere)


class LSMOP6(_LSMOP5To8Base):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_rosenbrock, even_func=_schwefel)


class LSMOP7(_LSMOP5To8Base):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_ackley, even_func=_rosenbrock)


class LSMOP8(_LSMOP5To8Base):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_griewank, even_func=_sphere)


class LSMOP9(_LSMOPBase):
    def __init__(self, n_var: int = 300, n_obj: int = 3, nk: int = 5) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, nk=nk, odd_func=_sphere, even_func=_ackley)

    def _evaluate_f(self, X: NDArray[np.float64], G: NDArray[np.float64]) -> NDArray[np.float64]:
        n = X.shape[0]
        coef_g = 1.0 + np.sum(self._normalize_g(G), axis=1)

        F = np.empty((n, self.n_obj), dtype=float)
        F[:, : self.n_obj - 1] = X[:, : self.n_obj - 1]

        if self.n_obj > 1:
            head = F[:, : self.n_obj - 1]
            denom = (1.0 + coef_g)[:, None]
            sum_term = np.sum((head / denom) * (1.0 + np.sin(3.0 * np.pi * head)), axis=1)
        else:  # pragma: no cover - guarded by constructor
            sum_term = np.zeros(n, dtype=float)

        F[:, self.n_obj - 1] = (1.0 + coef_g) * (self.n_obj - sum_term)
        return F


__all__ = [
    "LSMOP1",
    "LSMOP2",
    "LSMOP3",
    "LSMOP4",
    "LSMOP5",
    "LSMOP6",
    "LSMOP7",
    "LSMOP8",
    "LSMOP9",
]
