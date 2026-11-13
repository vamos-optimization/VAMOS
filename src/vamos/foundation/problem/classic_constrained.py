"""Classic constrained multiobjective problems, as jMetal defines them.

- Binh and Korn's second problem (Binh2), 2 variables, 2 constraints.
- ConstrEx (Deb, Multi-Objective Optimization Using Evolutionary Algorithms, 2001), 2 variables, 2 constraints.
- Srinivas and Deb (1994), 2 variables, 2 constraints.
- Tanaka et al. (1995), 2 variables, 2 constraints.
- Osyczka and Kundu (1995), 6 variables, 6 constraints.
- Golinski's speed reducer (Kurpati, Azarm and Wu, 2002), 7 variables, 11 constraints.
- The water resource planning problem (Ray, Tai and Seow, 2001), 3 variables, 5 objectives, 7 constraints.
- The car side impact problem (Jain and Deb, 2014), 7 variables, 3 objectives, 10 constraints.

The definitions, bounds and constraint scaling are those of jMetal's ``org.uma.jmetal.problem.multiobjective``
classes of the same names. Constraints follow VAMOS's convention, ``g <= 0`` feasible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem

Array = np.ndarray[Any, Any]


def _binh2(x: Array) -> tuple[list[Array], list[Array]]:
    f = [4.0 * x[0] ** 2 + 4.0 * x[1] ** 2, (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2]
    c = [-((x[0] - 5.0) ** 2) - x[1] ** 2 + 25.0, (x[0] - 8.0) ** 2 + (x[1] + 3.0) ** 2 - 7.7]
    return f, c


def _constrex(x: Array) -> tuple[list[Array], list[Array]]:
    f = [x[0], (1.0 + x[1]) / x[0]]
    c = [x[1] + 9.0 * x[0] - 6.0, -x[1] + 9.0 * x[0] - 1.0]
    return f, c


def _srinivas(x: Array) -> tuple[list[Array], list[Array]]:
    f = [2.0 + (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2, 9.0 * x[0] - (x[1] - 1.0) ** 2]
    c = [1.0 - (x[0] ** 2 + x[1] ** 2) / 225.0, (3.0 * x[1] - x[0]) / 10.0 - 1.0]
    return f, c


def _tanaka(x: Array) -> tuple[list[Array], list[Array]]:
    f = [x[0], x[1]]
    c = [
        x[0] ** 2 + x[1] ** 2 - 1.0 - 0.1 * np.cos(16.0 * np.arctan(x[0] / x[1])),
        -2.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.5),
    ]
    return f, c


def _osyczka2(x: Array) -> tuple[list[Array], list[Array]]:
    x1, x2, x3, x4, x5, x6 = x
    f = [
        -(25.0 * (x1 - 2.0) ** 2 + (x2 - 2.0) ** 2 + (x3 - 1.0) ** 2 + (x4 - 4.0) ** 2 + (x5 - 1.0) ** 2),
        x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2,
    ]
    c = [
        (x1 + x2) / 2.0 - 1.0,
        (6.0 - x1 - x2) / 6.0,
        (2.0 - x2 + x1) / 2.0,
        (2.0 - x1 + 3.0 * x2) / 2.0,
        (4.0 - (x3 - 3.0) ** 2 - x4) / 4.0,
        ((x5 - 3.0) ** 2 + x6 - 4.0) / 4.0,
    ]
    return f, c


def _golinski(x: Array) -> tuple[list[Array], list[Array]]:
    x1, x2, x3, x4, x5, x6, x7 = x
    f1 = (
        0.7854 * x1 * x2 * x2 * ((10.0 * x3 * x3) / 3.0 + 14.933 * x3 - 43.0934)
        - 1.508 * x1 * (x6 * x6 + x7 * x7)
        + 7.477 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    )
    aux = 745.0 * x4 / (x2 * x3)
    f2 = np.sqrt(aux * aux + 1.69e7) / (0.1 * x6**3)
    a = 745.0 * x5 / (x2 * x3)
    c = [
        -(1.0 / (x1 * x2 * x2 * x3) - 1.0 / 27.0),
        -(1.0 / (x1 * x2 * x2 * x3 * x3) - 1.0 / 397.5),
        -(x4**3 / (x2 * x3 * x3 * x6**4) - 1.0 / 1.93),
        -(x5**3 / (x2 * x3 * x7**4) - 1.0 / 1.93),
        -(x2 * x3 - 40.0),
        -(x1 / x2 - 12.0),
        -(5.0 - x1 / x2),
        -(1.9 - x4 + 1.5 * x6),
        -(1.9 - x5 + 1.1 * x7),
        -(f2 - 1300.0),
        -(np.sqrt(a * a + 1.575e8) / (0.1 * x7**3) - 1100.0),
    ]
    return [f1, f2], c


def _water(x: Array) -> tuple[list[Array], list[Array]]:
    x0, x1, x2 = x
    ratio = 1.0 / (x0 * x1)
    f = [
        106780.37 * (x1 + x2) + 61704.67,
        3000.0 * x0,
        305700.0 * 2289.0 * x1 / (0.06 * 2289.0) ** 0.65,
        250.0 * 2289.0 * np.exp(-39.75 * x1 + 9.9 * x2 + 2.74),
        25.0 * (1.39 * ratio + 4940.0 * x2 - 80.0),
    ]
    c = [
        1.0 - (0.00139 * ratio + 4.94 * x2 - 0.08),
        1.0 - (0.000306 * ratio + 1.082 * x2 - 0.0986),
        50000.0 - (12.307 * ratio + 49408.24 * x2 + 4051.02),
        16000.0 - (2.098 * ratio + 8046.33 * x2 - 696.71),
        10000.0 - (2.138 * ratio + 7883.39 * x2 - 705.04),
        2000.0 - (0.417 * x0 * x1 + 1721.26 * x2 - 136.54),
        550.0 - (0.164 * ratio + 631.13 * x2 - 54.48),
    ]
    return f, c


def _car_side_impact(x: Array) -> tuple[list[Array], list[Array]]:
    x1, x2, x3, x4, x5, x6, x7 = x
    force = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
    vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
    vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
    f = [
        1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7,
        force,
        0.5 * (vmbp + vfd),
    ]
    g1 = 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3
    g2 = 0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6
    g3 = (
        0.214
        + 0.00817 * x5
        - 0.045195 * x1
        - 0.0135168 * x1
        + 0.03099 * x2 * x6
        - 0.018 * x2 * x7
        + 0.007176 * x3
        + 0.023232 * x3
        - 0.00364 * x5 * x6
        - 0.018 * x2 * x2
    )
    g4 = 0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2
    g5 = 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7
    g6 = 33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728
    g7 = 46.36 - 9.9 * x2 - 4.4505 * x1
    c = [1.0 - g1, 0.32 - g2, 0.32 - g3, 0.32 - g4, 32.0 - g5, 32.0 - g6, 32.0 - g7, 4.0 - force, 9.9 - vmbp, 15.7 - vfd]
    return f, c


# key: (function, lower bounds, upper bounds, number of objectives, number of constraints)
CLASSICS: dict[str, tuple[Callable[[Array], tuple[list[Array], list[Array]]], tuple[float, ...], tuple[float, ...], int, int]] = {
    "binh2": (_binh2, (0.0, 0.0), (5.0, 3.0), 2, 2),
    "constrex": (_constrex, (0.1, 0.0), (1.0, 5.0), 2, 2),
    "srinivas": (_srinivas, (-20.0, -20.0), (20.0, 20.0), 2, 2),
    "tanaka": (_tanaka, (1e-4, 1e-4), (np.pi, np.pi), 2, 2),
    "osyczka2": (_osyczka2, (0.0, 0.0, 1.0, 0.0, 1.0, 0.0), (10.0, 10.0, 5.0, 6.0, 5.0, 10.0), 2, 6),
    "golinski": (_golinski, (2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0), (3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5), 2, 11),
    "water": (_water, (0.01, 0.01, 0.01), (0.45, 0.10, 0.10), 5, 7),
    "car_side_impact": (
        _car_side_impact,
        (0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4),
        (1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2),
        3,
        10,
    ),
}


class ClassicConstrainedProblem(Problem):
    """One of :data:`CLASSICS`, with jMetal's bounds and constraint scaling."""

    def __init__(self, key: str) -> None:
        name = str(key).lower()
        if name not in CLASSICS:
            raise ValueError(f"Unknown classic constrained problem {key!r}; the problems are: {', '.join(CLASSICS)}.")
        function, lower, upper, n_obj, n_constraints = CLASSICS[name]
        self.key = name
        self._function = function
        self.n_var = len(lower)
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.xl = np.asarray(lower, dtype=float)
        self.xu = np.asarray(upper, dtype=float)

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        f, c = self._function(X.T)
        out["F"] = np.column_stack(f)
        out["G"] = -np.column_stack(c)


__all__ = ["CLASSICS", "ClassicConstrainedProblem"]
