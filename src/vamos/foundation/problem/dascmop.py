"""DAS-CMOP1-9: constrained problems with tunable difficulty.

Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb and E. Goodman, "Difficulty adjustable and scalable
constrained multiobjective test problem toolkit", Evolutionary Computation 28(3), 339-378 (2020). Three
parameters in [0, 1] set the difficulty: ``eta`` (diversity: the share of the front cut by the first
constraint), ``zeta`` (feasibility: the width of the feasible band of the distance function) and ``gamma``
(convergence: the radius of the infeasible regions on the front). The definitions follow jMetal's
implementation (``org.uma.jmetal.problem.multiobjective.dascmop``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem

# The default difficulty triplets (eta, zeta, gamma) of jMetal.
DEFAULT_DIFFICULTY = {
    1: (0.0, 0.5, 0.5),
    2: (0.0, 0.5, 0.5),
    3: (0.5, 0.5, 0.5),
    4: (0.5, 0.5, 0.5),
    5: (0.5, 0.5, 0.5),
    6: (0.5, 0.5, 0.5),
    7: (0.5, 0.5, 0.5),
    8: (0.5, 0.5, 0.5),
    9: (0.5, 0.5, 0.5),
}

_A = 20.0
_D = 0.5
_NO_UPPER_BOUND = 1.0e30
_EQUALITY_TOLERANCE = 1.0e-4
# Centres of the rotated ellipses of the two-objective problems, and their axes.
_P = (0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0)
_Q = (1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5)
_SQUARED_A = 0.3
_SQUARED_B = 1.2
_THETA = -0.25 * math.pi
# Centres of the spheres of the three-objective problems.
_CENTERS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (3.0**-0.5, 3.0**-0.5, 3.0**-0.5))


class DASCMOPProblem(Problem):
    """DAS-CMOP``variant`` (1-9) with ``n_var`` variables in [0, 1] and the difficulty ``(eta, zeta, gamma)``.

    DAS-CMOP1-6 have two objectives and 11 constraints, DAS-CMOP7-9 three objectives and 7 constraints.
    Constraints follow VAMOS's convention, ``g <= 0`` feasible: they are jMetal's values with the opposite
    sign. ``zeta = 1`` turns the second constraint into an equality, met within 1e-4.
    """

    def __init__(
        self,
        variant: int,
        *,
        n_var: int = 30,
        eta: float | None = None,
        zeta: float | None = None,
        gamma: float | None = None,
    ) -> None:
        number = int(variant)
        if number not in DEFAULT_DIFFICULTY:
            raise ValueError(f"Unknown DAS-CMOP variant {variant!r}; the variants are 1 to 9.")
        n_obj = 3 if number >= 7 else 2
        if int(n_var) < n_obj:
            raise ValueError(f"DAS-CMOP{number} needs at least as many variables as objectives ({n_obj}), got {n_var}.")
        defaults = DEFAULT_DIFFICULTY[number]
        difficulty = tuple(float(d if v is None else v) for v, d in zip((eta, zeta, gamma), defaults))
        for name, value in zip(("eta", "zeta", "gamma"), difficulty):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"DAS-CMOP's {name} must be in [0, 1], got {value}.")
        self.variant = number
        self.eta, self.zeta, self.gamma = difficulty
        self.n_var = int(n_var)
        self.n_obj = n_obj
        self.n_constraints = 11 if n_obj == 2 else 7
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)
        self._b = 2.0 * self.eta - 1.0
        self._d = 0.0 if self.zeta == 0.0 else _D
        self._e = _NO_UPPER_BOUND if self.zeta == 0.0 else _D - math.log(self.zeta)
        self._r = 0.5 * self.gamma

    def _distance(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        k = self.variant
        if k <= 3:
            # Every variable, the first included, against sin(pi x1 / 2).
            return np.asarray(np.sum((X - np.sin(0.5 * np.pi * X[:, :1])) ** 2, axis=1))
        if k <= 8:
            position = 1 if k <= 6 else 2
            y = X[:, position:] - 0.5
            return np.asarray((self.n_var - position) + np.sum(y * y - np.cos(20.0 * np.pi * y), axis=1))
        j = np.arange(3, self.n_var + 1)
        target = np.cos(0.25 * j / self.n_var * np.pi * (X[:, :1] + X[:, 1:2]))
        return np.asarray(np.sum((X[:, 2:] - target) ** 2, axis=1))

    def _shape(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        k = self.variant
        x0 = X[:, 0]
        if k <= 6:
            if k in {1, 4}:
                f2 = 1.0 - x0 * x0
            else:
                f2 = 1.0 - np.sqrt(x0)
                if k in {3, 6}:
                    f2 = f2 + 0.5 * np.abs(np.sin(5.0 * np.pi * x0))
            return np.column_stack([x0, f2])
        x1 = X[:, 1]
        if k == 7:
            return np.column_stack([x0 * x1, x1 * (1.0 - x0), 1.0 - x1])
        t0, t1 = 0.5 * np.pi * x0, 0.5 * np.pi * x1
        return np.column_stack([np.cos(t0) * np.cos(t1), np.cos(t0) * np.sin(t1), np.sin(t0)])

    def _type_ii(self, g: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.zeta == 1.0:
            return np.asarray(_EQUALITY_TOLERANCE - np.abs(g - self._e))
        return np.asarray((self._e - g) * (g - self._d))

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        g = self._distance(X)
        F = self._shape(X) + g[:, None]
        C = np.empty((X.shape[0], self.n_constraints), dtype=float)
        C[:, 0] = np.sin(_A * np.pi * X[:, 0]) - self._b
        if self.n_obj == 2:
            C[:, 1] = self._type_ii(g)
            dx, dy = F[:, :1] - np.asarray(_P), F[:, 1:2] - np.asarray(_Q)
            u = dx * math.cos(_THETA) - dy * math.sin(_THETA)
            v = dx * math.sin(_THETA) + dy * math.cos(_THETA)
            C[:, 2:] = u * u / _SQUARED_A + v * v / _SQUARED_B - self._r
        else:
            C[:, 1] = np.cos(_A * np.pi * X[:, 1]) - self._b
            C[:, 2] = self._type_ii(g)
            C[:, 3:] = np.sum((F[:, None, :] - np.asarray(_CENTERS)[None, :, :]) ** 2, axis=2) - self._r * self._r
        out["F"] = F
        out["G"] = -C


__all__ = ["DEFAULT_DIFFICULTY", "DASCMOPProblem"]
