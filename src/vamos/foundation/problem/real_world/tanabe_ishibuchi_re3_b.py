from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output

from .tanabe_ishibuchi_utils import sum_violations_gte0

Array = np.ndarray[Any, Any]


def speed_reducer(x: Array) -> tuple[list[Array], list[Array]]:
    """Objectives and constraints (``>= 0`` feasible) of the speed reducer design (RE35 and CRE24).

    The rows of ``x`` are the variables; the number of teeth ``x[2]`` must already be an integer.
    """
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]

    f1 = (
        0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934)
        - 1.508 * x1 * (x6 * x6 + x7 * x7)
        + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
        + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    )

    tmp = np.power((745.0 * x4) / (x2 * x3), 2.0) + 1.69e7
    f2 = np.sqrt(tmp) / (0.1 * x6 * x6 * x6)

    g1 = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
    g2 = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
    g3 = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
    g4 = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
    g5 = -(x2 * x3) + 40.0
    g6 = -(x1 / x2) + 12.0
    g7 = -5.0 + (x1 / x2)
    g8 = -1.9 + x4 - 1.5 * x6
    g9 = -1.9 + x5 - 1.1 * x7
    g10 = -f2 + 1300.0
    tmp2 = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575e8
    g11 = -np.sqrt(tmp2) / (0.1 * x7 * x7 * x7) + 1100.0
    return [f1, f2], [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]


def gear_train(x: Array) -> tuple[list[Array], list[Array]]:
    """Objectives and constraint (``>= 0`` feasible) of the gear train design (RE36 and CRE25).

    The rows of ``x`` are the four numbers of teeth, already integers.
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    f1 = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
    f2 = np.maximum.reduce([x1, x2, x3, x4])
    g1 = 0.5 - (f1 / 6.931)
    return [f1, f2], [g1]


class RE35Problem:
    """
    Tanabe & Ishibuchi (2020) RE35 / RE3-7-5: Speed reducer design.
    """

    def __init__(self) -> None:
        self.n_var = 7
        self.n_obj = 3
        self.encoding = "mixed"

        self.xl = np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0], dtype=float)
        self.xu = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5], dtype=float)
        self.mixed_spec: dict[str, Array] = {
            "real_idx": np.array([0, 1, 3, 4, 5, 6], dtype=int),
            "int_idx": np.array([2], dtype=int),
            "cat_idx": np.array([], dtype=int),
            "real_lower": np.array([2.6, 0.7, 7.3, 7.3, 2.9, 5.0], dtype=float),
            "real_upper": np.array([3.6, 0.8, 8.3, 8.3, 3.9, 5.5], dtype=float),
            "int_lower": np.array([17], dtype=int),
            "int_upper": np.array([28], dtype=int),
            "cat_cardinality": np.array([], dtype=int),
        }

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        x = X.T.copy()
        x[2] = np.clip(np.rint(X[:, 2]), 17, 28).astype(int).astype(float)
        (f1, f2), g = speed_reducer(x)
        f3 = sum_violations_gte0(*g)

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3


class RE36Problem:
    """
    Tanabe & Ishibuchi (2020) RE36 / RE3-4-6: Gear train design.
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 3
        self.encoding = "integer"
        self.xl = np.full(self.n_var, 12, dtype=int)
        self.xu = np.full(self.n_var, 60, dtype=int)

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        X_int = np.clip(np.rint(X), self.xl, self.xu).astype(int, copy=False)
        (f1, f2), (g1,) = gear_train(X_int.T.astype(float))
        f3 = np.maximum(-g1, 0.0)

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3


class RE37Problem:
    """
    Tanabe & Ishibuchi (2020) RE37 / RE3-4-7: Rocket injector design.
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        f1 = (
            0.692
            + (0.477 * x1)
            - (0.687 * x2)
            - (0.080 * x3)
            - (0.0650 * x4)
            - (0.167 * x1 * x1)
            - (0.0129 * x2 * x1)
            + (0.0796 * x2 * x2)
            - (0.0634 * x3 * x1)
            - (0.0257 * x3 * x2)
            + (0.0877 * x3 * x3)
            - (0.0521 * x4 * x1)
            + (0.00156 * x4 * x2)
            + (0.00198 * x4 * x3)
            + (0.0184 * x4 * x4)
        )

        f2 = (
            0.153
            - (0.322 * x1)
            + (0.396 * x2)
            + (0.424 * x3)
            + (0.0226 * x4)
            + (0.175 * x1 * x1)
            + (0.0185 * x2 * x1)
            - (0.0701 * x2 * x2)
            - (0.251 * x3 * x1)
            + (0.179 * x3 * x2)
            + (0.0150 * x3 * x3)
            + (0.0134 * x4 * x1)
            + (0.0296 * x4 * x2)
            + (0.0752 * x4 * x3)
            + (0.0192 * x4 * x4)
        )

        f3 = (
            0.370
            - (0.205 * x1)
            + (0.0307 * x2)
            + (0.108 * x3)
            + (1.019 * x4)
            - (0.135 * x1 * x1)
            + (0.0141 * x2 * x1)
            + (0.0998 * x2 * x2)
            + (0.208 * x3 * x1)
            - (0.0301 * x3 * x2)
            - (0.226 * x3 * x3)
            + (0.353 * x4 * x1)
            - (0.0497 * x4 * x3)
            - (0.423 * x4 * x4)
            + (0.202 * x2 * x1 * x1)
            - (0.281 * x3 * x1 * x1)
            - (0.342 * x2 * x2 * x1)
            - (0.245 * x2 * x2 * x3)
            + (0.281 * x3 * x3 * x2)
            - (0.184 * x4 * x4 * x1)
            - (0.281 * x2 * x1 * x3)
        )

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3


__all__ = ["RE35Problem", "RE36Problem", "RE37Problem", "gear_train", "speed_reducer"]
