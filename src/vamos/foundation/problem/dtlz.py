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
