import warnings
from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem


class DTLZBase(Problem):
    def __init__(self, n_var: int, n_obj: int) -> None:
        if n_var < n_obj:
            raise ValueError(
                f"DTLZ problems need n_var >= n_obj (at least one distance variable, k = n_var - n_obj + 1 >= 1); "
                f"got n_var={n_var}, n_obj={n_obj}."
            )
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = 0.0
        self.xu = 1.0

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        raise NotImplementedError

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        self._evaluate(X, out)


class DTLZ1Problem(DTLZBase):
    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 4  # Standard k=5
        super().__init__(n_var, n_obj)

        # Validate standard configuration
        k = n_var - n_obj + 1
        if k != 5:
            warnings.warn(
                f"Non-standard DTLZ1 configuration: k={k} (standard k=5). Consider using n_var={n_obj + 4} for standard benchmark.",
                UserWarning,
                stacklevel=2,
            )

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        g = 100.0 * (
            self.n_var
            - self.n_obj
            + 1
            + np.sum(
                (X[:, self.n_obj - 1 :] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, self.n_obj - 1 :] - 0.5)),
                axis=1,
            )
        )
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= X[:, j]
            if i > 0:
                idx = self.n_obj - i - 1
                f *= 1.0 - X[:, idx]
            F[:, i] = f

        F_res = 0.5 * (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ2Problem(DTLZBase):
    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 9  # Standard k=10
        super().__init__(n_var, n_obj)

        # Validate standard configuration
        k = n_var - n_obj + 1
        if k != 10:
            warnings.warn(
                f"Non-standard DTLZ2 configuration: k={k} (standard k=10). Consider using n_var={n_obj + 9} for standard benchmark.",
                UserWarning,
                stacklevel=2,
            )

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        g = np.sum((X[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= np.cos(X[:, j] * np.pi / 2.0)
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin(X[:, idx] * np.pi / 2.0)
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ3Problem(DTLZBase):
    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 9  # Standard k=10
        super().__init__(n_var, n_obj)

        # Validate standard configuration
        k = n_var - n_obj + 1
        if k != 10:
            warnings.warn(
                f"Non-standard DTLZ3 configuration: k={k} (standard k=10). Consider using n_var={n_obj + 9} for standard benchmark.",
                UserWarning,
                stacklevel=2,
            )

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        g = 100.0 * (
            self.n_var
            - self.n_obj
            + 1
            + np.sum(
                (X[:, self.n_obj - 1 :] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, self.n_obj - 1 :] - 0.5)),
                axis=1,
            )
        )
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= np.cos(X[:, j] * np.pi / 2.0)
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin(X[:, idx] * np.pi / 2.0)
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ4Problem(DTLZBase):
    def __init__(self, n_var: int | None = None, n_obj: int = 3, alpha: float = 100.0) -> None:
        if n_var is None:
            n_var = n_obj + 9  # Standard k=10
        super().__init__(n_var, n_obj)
        self.alpha = alpha

        # Validate standard configuration
        k = n_var - n_obj + 1
        if k != 10:
            warnings.warn(
                f"Non-standard DTLZ4 configuration: k={k} (standard k=10). Consider using n_var={n_obj + 9} for standard benchmark.",
                UserWarning,
                stacklevel=2,
            )

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        g = np.sum((X[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        alpha = self.alpha
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= np.cos((X[:, j] ** alpha) * np.pi / 2.0)
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin((X[:, idx] ** alpha) * np.pi / 2.0)
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ5Problem(DTLZBase):
    """DTLZ5: Degenerate Pareto front with theta transformation."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 9  # Standard k=10
        super().__init__(n_var, n_obj)

        # Validate standard configuration
        k = n_var - n_obj + 1
        if k != 10:
            warnings.warn(
                f"Non-standard DTLZ5 configuration: k={k} (standard k=10). Consider using n_var={n_obj + 9} for standard benchmark.",
                UserWarning,
                stacklevel=2,
            )

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        g = np.sum((X[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)

        theta = np.empty((X.shape[0], self.n_obj - 1), dtype=float)
        theta[:, 0] = X[:, 0] * np.pi / 2.0
        if self.n_obj > 2:
            t = (np.pi / (4.0 * (1.0 + g)))[:, None]
            theta[:, 1:] = t * (1.0 + 2.0 * g[:, None] * X[:, 1 : self.n_obj - 1])

        F = np.ones((X.shape[0], self.n_obj), dtype=float)
        for i in range(self.n_obj):
            f = np.ones(X.shape[0], dtype=float)
            for j in range(self.n_obj - i - 1):
                f *= np.cos(theta[:, j])
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin(theta[:, idx])
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ6Problem(DTLZBase):
    """DTLZ6: Degenerate Pareto front with biased distance function."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 9  # Standard k=10
        super().__init__(n_var, n_obj)

        # Validate standard configuration
        k = n_var - n_obj + 1
        if k != 10:
            warnings.warn(
                f"Non-standard DTLZ6 configuration: k={k} (standard k=10). Consider using n_var={n_obj + 9} for standard benchmark.",
                UserWarning,
                stacklevel=2,
            )

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        g = np.sum(np.power(X[:, self.n_obj - 1 :], 0.1), axis=1)

        theta = np.empty((X.shape[0], self.n_obj - 1), dtype=float)
        theta[:, 0] = X[:, 0] * np.pi / 2.0
        if self.n_obj > 2:
            t = (np.pi / (4.0 * (1.0 + g)))[:, None]
            theta[:, 1:] = t * (1.0 + 2.0 * g[:, None] * X[:, 1 : self.n_obj - 1])

        F = np.ones((X.shape[0], self.n_obj), dtype=float)
        for i in range(self.n_obj):
            f = np.ones(X.shape[0], dtype=float)
            for j in range(self.n_obj - i - 1):
                f *= np.cos(theta[:, j])
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin(theta[:, idx])
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ7Problem(DTLZBase):
    """DTLZ7: Problem with disconnected Pareto-optimal regions."""

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 19  # Standard k=20
        super().__init__(n_var, n_obj)

        # Validate standard configuration
        k = n_var - n_obj + 1
        if k != 20:
            warnings.warn(
                f"Non-standard DTLZ7 configuration: k={k} (standard k=20). Consider using n_var={n_obj + 19} for standard benchmark.",
                UserWarning,
                stacklevel=2,
            )

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        # g function uses the last k variables
        g = 1.0 + (9.0 / (self.n_var - self.n_obj + 1)) * np.sum(X[:, self.n_obj - 1 :], axis=1)

        # First M-1 objectives are just x_i
        F = np.zeros((X.shape[0], self.n_obj))
        for i in range(self.n_obj - 1):
            F[:, i] = X[:, i]

        # Last objective: h function
        h = self.n_obj - np.sum(
            (F[:, : self.n_obj - 1] / (1.0 + g[:, None])) * (1.0 + np.sin(3.0 * np.pi * F[:, : self.n_obj - 1])),
            axis=1,
        )
        F[:, self.n_obj - 1] = (1.0 + g) * h

        if "F" in out and out["F"] is not None:
            out["F"][:] = F
        else:
            out["F"] = F


def _position_shape(head: np.ndarray[Any, Any], tail: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """DTLZ's position shape, as PlatEMO writes it: ``fliplr(cumprod([1, head])) .* [1, fliplr(tail)]``.

    ``f_1 = h_1 ... h_{M-1}``, ``f_i = h_1 ... h_{M-i} t_{M-i+1}`` and ``f_M = t_1``: DTLZ1 takes ``h = x`` and
    ``t = 1 - x``, DTLZ2 ``h = cos(x pi / 2)`` and ``t = sin(x pi / 2)``, for the M - 1 position variables.
    """
    ones = np.ones((head.shape[0], 1))
    return np.asarray(np.cumprod(np.hstack([ones, head]), axis=1)[:, ::-1] * np.hstack([ones, tail[:, ::-1]]))


def _warn_non_standard_k(name: str, n_var: int, n_obj: int, standard: int) -> None:
    k = n_var - n_obj + 1
    if k != standard:
        warnings.warn(
            f"Non-standard {name} configuration: k={k} (standard k={standard}). "
            f"Consider using n_var={n_obj + standard - 1} for standard benchmark.",
            UserWarning,
            stacklevel=3,
        )


class IDTLZ1Problem(DTLZBase):
    """Inverted DTLZ1 (Jain and Deb, 2014), as PlatEMO's ``IDTLZ1``.

    VAMOS 1.5.0: ``f = (1 + g) / 2 - (1 + g) / 2 * s``, where ``g`` is DTLZ1's distance function
    and ``s`` its position shape (DTLZ1 is ``f = (1 + g) / 2 * s``). With ``g = 0`` the front is DTLZ1's
    simplex turned over, ``f = (1 - w) / 2`` for ``w`` on the unit simplex. ``n_var`` defaults to
    ``n_obj + 4`` (k = 5).
    """

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 4
        super().__init__(n_var, n_obj)
        _warn_non_standard_k("IDTLZ1", n_var, n_obj, 5)

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        m = self.n_obj
        distance = X[:, m - 1 :] - 0.5
        g = 100.0 * (self.n_var - m + 1 + np.sum(distance**2 - np.cos(20.0 * np.pi * distance), axis=1))
        position = X[:, : m - 1]
        scale = (1.0 + g)[:, None]
        F = scale / 2.0 - 0.5 * scale * _position_shape(position, 1.0 - position)
        if "F" in out and out["F"] is not None:
            out["F"][:] = F
        else:
            out["F"] = F


class IDTLZ2Problem(DTLZBase):
    """Inverted DTLZ2 (Jain and Deb, 2014), as PlatEMO's ``IDTLZ2``.

    VAMOS 1.5.0: ``f = (1 + g) - (1 + g) * s``, where ``g = sum (x_i - 0.5)^2`` over the distance
    variables and ``s`` is DTLZ2's position shape (DTLZ2 is ``f = (1 + g) * s``). With ``g = 0`` the front is
    the unit sphere's positive orthant turned over, ``f = 1 - u`` for ``u`` on it. ``n_var`` defaults to
    ``n_obj + 9`` (k = 10).
    """

    def __init__(self, n_var: int | None = None, n_obj: int = 3) -> None:
        if n_var is None:
            n_var = n_obj + 9
        super().__init__(n_var, n_obj)
        _warn_non_standard_k("IDTLZ2", n_var, n_obj, 10)

    def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        m = self.n_obj
        g = np.sum((X[:, m - 1 :] - 0.5) ** 2, axis=1)
        angle = X[:, : m - 1] * np.pi / 2.0
        scale = (1.0 + g)[:, None]
        F = scale - scale * _position_shape(np.cos(angle), np.sin(angle))
        if "F" in out and out["F"] is not None:
            out["F"][:] = F
        else:
            out["F"] = F
