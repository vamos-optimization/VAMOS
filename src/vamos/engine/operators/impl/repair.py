from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class GradientRepair:
    """
    Simple gradient-based repair: takes a step against constraint violation gradients.

    ``constraint_fun(X)`` returns constraint values of shape (N, n_constraints) with
    the ``g <= 0`` convention, and ``constraint_grad(X)`` their gradients of shape
    (N, n_constraints, n_var). Only the violations ``max(g, 0)`` count, so a
    satisfied constraint neither offsets a violated one nor moves the point.
    """

    def __init__(
        self,
        constraint_fun: Callable[[NDArray[np.float64]], ArrayLike],
        constraint_grad: Callable[[NDArray[np.float64]], ArrayLike],
        step_size: float = 0.1,
        max_iters: int = 3,
    ) -> None:
        self.constraint_fun = constraint_fun
        self.constraint_grad = constraint_grad
        self.step_size = float(step_size)
        self.max_iters = int(max_iters)

    def __call__(
        self,
        X: ArrayLike,
        xl: ArrayLike,
        xu: ArrayLike,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        X_new = np.array(X, dtype=float, copy=True)
        for _ in range(self.max_iters):
            violations = np.maximum(np.asarray(self.constraint_fun(X_new), dtype=float), 0.0)  # (N, m)
            grads = np.asarray(self.constraint_grad(X_new), dtype=float)  # (N, m, n_var)
            total_violation = violations.sum(axis=1, keepdims=True)
            if np.all(total_violation <= 1e-12):
                break
            # Aggregate gradients weighted by violation magnitude
            direction = (violations[..., None] * grads).sum(axis=1)
            X_new = X_new - self.step_size * direction
            X_new = np.clip(X_new, xl, xu)
        return X_new


__all__ = ["GradientRepair"]
