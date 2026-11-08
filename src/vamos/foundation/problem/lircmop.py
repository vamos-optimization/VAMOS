"""LIR-CMOP1-14: constrained problems with large infeasible regions.

Z. Fan, W. Li, X. Cai, H. Huang, Y. Fang, Y. You, J. Mo, C. Wei and E. Goodman, "An improved epsilon
constraint-handling method in MOEA/D for CMOPs with large infeasible regions", Soft Computing 23,
12491-12510 (2019). The definitions follow jMetal's implementation (``org.uma.jmetal.problem.multiobjective
.lircmop``), which follows the paper since jMetal pull request #522.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem

# Number of constraints of each problem.
_CONSTRAINTS = {1: 2, 2: 2, 3: 3, 4: 3, 5: 2, 6: 2, 7: 3, 8: 3, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 3}

# Rotated ellipses (x offset, y offset, a, b) of LIR-CMOP5-8, rotated by -pi/4 with r = 0.1.
_ELLIPSES = {
    5: ((1.6, 1.6, 2.0, 4.0), (2.5, 2.5, 2.0, 8.0)),
    6: ((1.8, 1.8, 2.0, 8.0), (2.8, 2.8, 2.0, 8.0)),
    7: ((1.2, 1.2, 2.0, 6.0), (2.25, 2.25, 2.5, 12.0), (3.5, 3.5, 2.5, 10.0)),
    8: ((1.2, 1.2, 2.0, 6.0), (2.25, 2.25, 2.5, 12.0), (3.5, 3.5, 2.5, 10.0)),
}

# LIR-CMOP9-12: the offset of the wave constraint and the ellipse (x offset, y offset, a, b).
_WAVES = {
    9: (2.0, (1.4, 1.4, 1.5, 6.0)),
    10: (1.0, (1.1, 1.2, 2.0, 4.0)),
    11: (2.1, (1.2, 1.2, 1.5, 5.0)),
    12: (2.5, (1.6, 1.6, 1.5, 6.0)),
}

# LIR-CMOP13 and 14: pairs of radii (outer, inner) of the infeasible spherical shells.
_SHELLS = {13: ((3.0, 2.0), (1.9, 1.8)), 14: ((3.0, 2.0), (1.9, 1.8), (1.75, 1.6))}


def _ellipse(
    f1: np.ndarray[Any, Any], f2: np.ndarray[Any, Any], ellipse: tuple[float, float, float, float], theta: float
) -> np.ndarray[Any, Any]:
    """Positive outside the ellipse of axes ``a`` and ``b`` centred at the offsets and rotated by ``theta``."""
    dx, dy = f1 - ellipse[0], f2 - ellipse[1]
    u = (dx * np.cos(theta) - dy * np.sin(theta)) / ellipse[2]
    v = (dx * np.sin(theta) + dy * np.cos(theta)) / ellipse[3]
    return np.asarray(u * u + v * v - 0.1)


class LIRCMOPProblem(Problem):
    """LIR-CMOP``variant`` (1-14) with ``n_var`` variables in [0, 1].

    LIR-CMOP1-12 have two objectives and LIR-CMOP13 and 14 three. Constraints follow VAMOS's convention,
    ``g <= 0`` feasible: they are jMetal's values with the opposite sign.
    """

    def __init__(self, variant: int, *, n_var: int = 30) -> None:
        number = int(variant)
        if number not in _CONSTRAINTS:
            raise ValueError(f"Unknown LIR-CMOP variant {variant!r}; the variants are 1 to 14.")
        n_obj = 3 if number >= 13 else 2
        if int(n_var) < n_obj:
            raise ValueError(f"LIR-CMOP{number} needs at least {n_obj} variables, got {n_var}.")
        self.variant = number
        self.n_var = int(n_var)
        self.n_obj = n_obj
        self.n_constraints = _CONSTRAINTS[number]
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)

    def _distances(self, X: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """g1 over the variables 3, 5, ... and g2 over 2, 4, ... (one-based), as each group defines them."""
        x0 = X[:, :1]
        odd = np.arange(2, self.n_var, 2)
        even = np.arange(1, self.n_var, 2)
        if self.variant <= 4:
            g1 = np.sum((X[:, odd] - np.sin(0.5 * np.pi * x0)) ** 2, axis=1)
            g2 = np.sum((X[:, even] - np.cos(0.5 * np.pi * x0)) ** 2, axis=1)
        else:
            g1 = np.sum((X[:, odd] - np.sin(0.5 * (odd + 1) / self.n_var * np.pi * x0)) ** 2, axis=1)
            g2 = np.sum((X[:, even] - np.cos(0.5 * (even + 1) / self.n_var * np.pi * x0)) ** 2, axis=1)
        return g1, g2

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        k = self.variant
        x0 = X[:, 0]
        concave = k in {1, 3, 6, 8, 9, 12}  # f2 uses 1 - x^2 rather than 1 - sqrt(x)
        shape = 1.0 - x0 * x0 if concave else 1.0 - np.sqrt(x0)
        if k >= 13:
            g = np.sum(10.0 * (X[:, 2:] - 0.5) ** 2, axis=1)
            radius = 1.7057 + g
            t0, t1 = 0.5 * np.pi * x0, 0.5 * np.pi * X[:, 1]
            F = np.column_stack([radius * np.cos(t0) * np.cos(t1), radius * np.cos(t0) * np.sin(t1), radius * np.sin(t0)])
            squared = np.sum(F * F, axis=1)
            C = np.column_stack([(squared - outer**2) * (squared - inner**2) for outer, inner in _SHELLS[k]])
        else:
            g1, g2 = self._distances(X)
            if k <= 4:
                F = np.column_stack([x0 + g1, shape + g2])
                columns = [(0.51 - g1) * (g1 - 0.5), (0.51 - g2) * (g2 - 0.5)]
                if k >= 3:
                    columns.append(np.sin(20.0 * np.pi * x0) - 0.5)
                C = np.column_stack(columns)
            elif k <= 8:
                F = np.column_stack([x0 + 10.0 * g1 + 0.7057, shape + 10.0 * g2 + 0.7057])
                C = np.column_stack([_ellipse(F[:, 0], F[:, 1], e, -0.25 * np.pi) for e in _ELLIPSES[k]])
            else:
                F = np.column_stack([1.7057 * x0 * (10.0 * g1 + 1.0), 1.7057 * shape * (10.0 * g2 + 1.0)])
                offset, ellipse = _WAVES[k]
                alpha = 0.25 * np.pi
                along = F[:, 0] * np.cos(alpha) - F[:, 1] * np.sin(alpha)
                wave = F[:, 0] * np.sin(alpha) + F[:, 1] * np.cos(alpha) - np.sin(4.0 * np.pi * along) - offset
                C = np.column_stack([wave, _ellipse(F[:, 0], F[:, 1], ellipse, -0.25 * np.pi)])
        out["F"] = F
        out["G"] = -C


__all__ = ["LIRCMOPProblem"]
