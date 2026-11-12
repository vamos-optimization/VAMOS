"""ZXH-CF1-16: the constrained problems of Zhou, Xiang and He.

Y. Zhou, Y. Xiang and X. He, "Constrained multiobjective optimization: test problem construction and
performance evaluations", IEEE Transactions on Evolutionary Computation 25(1), 172-186 (2021),
doi:10.1109/TEVC.2020.3011829. The definitions follow jMetal's implementation
(``org.uma.jmetal.problem.multiobjective.cf``, classes ``CF1`` to ``CF16``, by Yi Xiang), which PlatEMO's
``ZXH_CF1`` to ``ZXH_CF16`` also implement.

Each problem combines a distance function, a front shape and a type of constraints. With M objectives,
``S = x1^2 + ... + xM^2`` and the angles ``theta_i = 2 / pi atan(sqrt(x_{i+1}^2 + ... + xM^2) / x_i)``,
i = 1, ..., M - 1, the objectives are ``(1 + T) f(theta)`` with ``T = (1 - S)^2 + h``:

- ``h``, of the variables after the first M, is a sphere, Rosenbrock, Ackley or Griewank function with
  its minimum 0 where they are all 0.2;
- ``f`` is a linear, concave, convex or mixed front shape;
- every problem has the constraint ``S + h <= 1``, and the ones with a radius ``r > 0`` also ``S + h >= r``;
- the first k angles (k = M - 1 up to three objectives, M // 2 up to eight, then 3) are free (type A),
  in [1/4, 3/4] (type B), outside (1/4, 3/4) (type C), or outside (1/10, 2/5) and (7/10, 4/5) (type D).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem

# Distance function and front shape of each problem. CF1-4 are of type A, CF5-8 of type B, CF9-12 of
# type C and CF13-16 of type D; in each group the radius r is 0, 1/4, 1/2 and 3/4.
_VARIANTS = {
    1: ("sphere", "linear"),
    2: ("rosenbrock", "concave"),
    3: ("ackley", "convex"),
    4: ("griewank", "mixed"),
    5: ("rosenbrock", "convex"),
    6: ("sphere", "mixed"),
    7: ("griewank", "linear"),
    8: ("ackley", "concave"),
    9: ("ackley", "mixed"),
    10: ("griewank", "convex"),
    11: ("sphere", "concave"),
    12: ("rosenbrock", "linear"),
    13: ("griewank", "concave"),
    14: ("ackley", "linear"),
    15: ("rosenbrock", "mixed"),
    16: ("sphere", "convex"),
}
_TYPES = ("A", "B", "C", "D")
# Where every distance variable sits on the Pareto set.
_OPTIMUM = 0.2
# Type B keeps each constrained angle in [alpha, beta]; type C keeps it out of (alpha, beta).
_ALPHA = 0.25
_BETA = 0.75
# Type D keeps each constrained angle out of these two intervals.
_D_INTERVALS = ((0.1, 0.4), (0.7, 0.8))
# The variables are in [1e-10, 1 - 1e-10], so that no angle divides by zero.
_MARGIN = 1e-10


def constrained_angles(n_obj: int) -> int:
    """The number k of angles that the constraints of types B, C and D restrict."""
    if n_obj <= 3:
        return n_obj - 1
    if n_obj <= 8:
        return n_obj // 2
    return 3


class ZXHCFProblem(Problem):
    """ZXH-CF``variant`` (1-16) with ``n_var`` variables in ``[1e-10, 1 - 1e-10]`` and ``n_obj`` objectives.

    Constraints follow VAMOS's convention, ``g <= 0`` feasible. They come first ``S + h - 1`` and, when
    ``r > 0``, ``r - (S + h)``, which are jMetal's values with the opposite sign. Then, for each of the
    first k angles:

    - type B: ``theta - 3/4`` and ``1/4 - theta``, as in jMetal;
    - type C: ``min(theta - 1/4, 3/4 - theta)``, positive only inside (1/4, 3/4);
    - type D: ``min(theta - 1/10, 2/5 - theta)`` and ``min(theta - 7/10, 4/5 - theta)``.

    A constraint of type C or D excludes an interval, so it is not a pair of constraints that must both
    hold. jMetal stores the two bounds of each interval, ``<= 0`` feasible, and sets the overall violation
    and the number of violated constraints itself: an interval adds the smaller of its two values when both
    are positive. VAMOS stores that smaller value, which gives jMetal's feasibility and overall violation
    with one column per interval, as PlatEMO does. jMetal counts two violated constraints for an interval of
    type C and three for one of type D; VAMOS counts the column once.
    """

    def __init__(self, variant: int, *, n_var: int = 13, n_obj: int = 3) -> None:
        number = int(variant)
        if number not in _VARIANTS:
            raise ValueError(f"Unknown ZXH-CF variant {variant!r}; the variants are 1 to 16.")
        n_obj = int(n_obj)
        n_var = int(n_var)
        distance, shape = _VARIANTS[number]
        if n_obj < 2:
            raise ValueError(f"ZXH-CF{number} needs at least 2 objectives, got {n_obj}.")
        # The Ackley function averages over the variables after the first M, so it needs one (jMetal checks it).
        minimum = n_obj + 1 if distance == "ackley" else n_obj
        if n_var < minimum:
            raise ValueError(f"ZXH-CF{number} with {n_obj} objectives needs at least {minimum} variables, got {n_var}.")
        self.variant = number
        self.n_var = n_var
        self.n_obj = n_obj
        self.distance = distance
        self.shape = shape
        self.constraint_type = _TYPES[(number - 1) // 4]
        self.radius = ((number - 1) % 4) / 4.0
        self.k = constrained_angles(n_obj)
        per_angle = {"A": 0, "B": 2, "C": 1, "D": 2}[self.constraint_type]
        self.n_constraints = (2 if self.radius > 0.0 else 1) + per_angle * self.k
        self.xl = np.full(self.n_var, _MARGIN, dtype=float)
        self.xu = np.full(self.n_var, 1.0 - _MARGIN, dtype=float)

    def _h(self, Z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """The distance function of the variables after the first M; 0 where they are all 0.2."""
        d = Z - _OPTIMUM
        if self.distance == "sphere":
            return np.asarray(np.sum(d * d, axis=1))
        if self.distance == "rosenbrock":
            head, tail = d[:, :-1], d[:, 1:]
            return np.asarray(np.sum(100.0 * (head * head - tail) ** 2 + head * head, axis=1))
        if self.distance == "ackley":
            count = Z.shape[1]
            mean_square = np.sum(d * d, axis=1) / count
            mean_cosine = np.sum(np.cos(2.0 * np.pi * d), axis=1) / count
            return np.asarray(20.0 - 20.0 * np.exp(-0.2 * np.sqrt(mean_square)) - np.exp(mean_cosine) + np.exp(1.0))
        # Griewank, scaled by 5.
        index = np.arange(1, Z.shape[1] + 1)
        product = np.prod(np.cos(10.0 * np.pi * d / np.sqrt(index)), axis=1)
        return np.asarray(5.0 * (1.0 + np.sum(d * d, axis=1) - product))

    def _front_shape(self, theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """The objectives at ``T = 0``: f_i is the product of the earlier factors and the own term."""
        n = theta.shape[0]
        if self.shape == "linear":
            own, carried = 1.0 - theta, theta
        else:
            own, carried = np.cos(np.pi / 2.0 * theta), np.sin(np.pi / 2.0 * theta)
        prefix = np.concatenate([np.ones((n, 1)), np.cumprod(carried, axis=1)], axis=1)
        f = prefix * np.concatenate([own, np.ones((n, 1))], axis=1)
        if self.shape in ("convex", "mixed"):
            f = 1.0 - f
        if self.shape == "mixed":
            segments = 2.0
            f[:, 0] = theta[:, 0] - np.cos(2.0 * np.pi * segments * theta[:, 0] + np.pi / 2.0) / (2.0 * segments * np.pi)
        return np.asarray(f)

    def _angle_constraints(self, theta: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
        columns: list[np.ndarray[Any, Any]] = []
        for i in range(self.k):
            t = theta[:, i]
            if self.constraint_type == "B":
                columns.extend([t - _BETA, -(t - _ALPHA)])
            elif self.constraint_type == "C":
                columns.append(np.minimum(t - _ALPHA, -(t - _BETA)))
            elif self.constraint_type == "D":
                columns.extend(np.minimum(t - low, -(t - high)) for low, high in _D_INTERVALS)
        return columns

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        m = self.n_obj
        position = X[:, :m]
        # S[:, i] = x_i^2 + ... + x_M^2, accumulated from the last variable as jMetal does.
        S = np.cumsum((position * position)[:, ::-1], axis=1)[:, ::-1]
        theta = 2.0 / np.pi * np.arctan(np.sqrt(S[:, 1:]) / position[:, :-1])
        h = self._h(X[:, m:])
        T = (1.0 - S[:, 0]) * (1.0 - S[:, 0]) + h
        out["F"] = (1.0 + T)[:, None] * self._front_shape(theta)

        radius = S[:, 0] + h
        columns = [radius - 1.0]
        if self.radius > 0.0:
            columns.append(-(radius - self.radius))
        columns.extend(self._angle_constraints(theta))
        out["G"] = np.column_stack(columns)


__all__ = ["ZXHCFProblem", "constrained_angles"]
