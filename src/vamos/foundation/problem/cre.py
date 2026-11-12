"""CRE21-CRE51: Tanabe and Ishibuchi's real-world problems with explicit constraints.

R. Tanabe and H. Ishibuchi, "An easy-to-use real-world multi-objective optimization problem suite", Applied
Soft Computing 89, 106078 (2020), https://doi.org/10.1016/j.asoc.2020.106078.

The definitions are those of jMetal's ``org.uma.jmetal.problem.multiobjective.cre`` classes at commit
``1e4b83cae6``, which follow the authors' reference code since jMetal pull request #513 (CRE21, CRE25 and
CRE31). The RE problems of the same suite add the sum of the constraint violations as an objective; the
CRE problems keep the constraints, and share their formulas with them:

- CRE21 (two-bar truss), CRE22 (welded beam), CRE23 (disc brake), CRE24 (speed reducer) and CRE25 (gear
  train) are RE31, RE32, RE33, RE35 and RE36 with their constraints;
- CRE31 (car side impact) is VAMOS's ``car_side_impact`` and CRE51 (water resource planning) its ``water``;
- CRE32 (conceptual marine design) is RE42 without RE42's guards against negative powers and divisions by
  zero, which jMetal does not have.

CRE24 rounds its number of teeth and CRE25 its four numbers of teeth to the nearest integer, ties to even,
inside the evaluation, as jMetal does (``Math.rint``); the variables stay real. Constraints follow VAMOS's
convention, ``g <= 0`` feasible: they are jMetal's values with the opposite sign.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem
from vamos.foundation.problem.classic_constrained import CLASSICS
from vamos.foundation.problem.real_world.tanabe_ishibuchi_re3_a import disc_brake, two_bar_truss, welded_beam
from vamos.foundation.problem.real_world.tanabe_ishibuchi_re3_b import gear_train, speed_reducer

Array = np.ndarray[Any, Any]
Formula = Callable[[Array], tuple[list[Array], list[Array]]]


def _cre24(x: Array) -> tuple[list[Array], list[Array]]:
    rounded = np.array(x, dtype=float)
    rounded[2] = np.rint(rounded[2])
    return speed_reducer(rounded)


def _cre25(x: Array) -> tuple[list[Array], list[Array]]:
    return gear_train(np.rint(x))


def _cre32(x: Array) -> tuple[list[Array], list[Array]]:
    length, beam, depth, draught, speed, block = x[0], x[1], x[2], x[3], x[4], x[5]
    displacement = 1.025 * length * beam * draught * block
    froude = 0.5144 * speed / np.power(9.8065 * length, 0.5)
    a = (4977.06 * block * block) - (8105.61 * block) + 4456.51
    b = (-10847.2 * block * block) + (12817.0 * block) - 6960.32
    power = (np.power(displacement, 2.0 / 3.0) * np.power(speed, 3.0)) / (a + (b * froude))
    outfit_weight = 1.0 * np.power(length, 0.8) * np.power(beam, 0.6) * np.power(depth, 0.3) * np.power(block, 0.1)
    steel_weight = 0.034 * np.power(length, 1.7) * np.power(beam, 0.7) * np.power(depth, 0.4) * np.power(block, 0.5)
    machinery_weight = 0.17 * np.power(power, 0.9)
    light_ship_weight = steel_weight + outfit_weight + machinery_weight
    ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (2400.0 * np.power(power, 0.8)))
    capital_costs = 0.2 * ship_cost
    dwt = displacement - light_ship_weight
    running_costs = 40000.0 * np.power(dwt, 0.3)
    sea_days = (5000.0 / 24.0) * speed
    daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
    fuel_cost = 1.05 * daily_consumption * sea_days * 100.0
    port_cost = 6.3 * np.power(dwt, 0.8)
    fuel_carried = daily_consumption * (sea_days + 5.0)
    miscellaneous_dwt = 2.0 * np.power(dwt, 0.5)
    cargo_dwt = dwt - fuel_carried - miscellaneous_dwt
    port_days = 2.0 * ((cargo_dwt / 8000.0) + 0.5)
    round_trips = 350.0 / (sea_days + port_days)
    voyage_costs = (fuel_cost + port_cost) * round_trips
    annual_costs = capital_costs + running_costs + voyage_costs
    annual_cargo = cargo_dwt * round_trips
    f = [annual_costs / annual_cargo, light_ship_weight, -annual_cargo]
    kb = 0.53 * draught
    bmt = ((0.085 * block - 0.002) * beam * beam) / (draught * block)
    kg = 1.0 + 0.52 * depth
    c = [
        (length / beam) - 6.0,
        -(length / depth) + 15.0,
        -(length / draught) + 19.0,
        0.45 * np.power(dwt, 0.31) - draught,
        0.7 * depth + 0.7 - draught,
        500000.0 - dwt,
        dwt - 3000.0,
        0.32 - froude,
        (kb + bmt - kg) - (0.07 * beam),
    ]
    return f, c


# key: (formula, lower bounds, upper bounds, number of objectives, number of constraints)
CRE: dict[str, tuple[Formula, tuple[float, ...], tuple[float, ...], int, int]] = {
    "cre21": (two_bar_truss, (1e-5, 1e-5, 1.0), (100.0, 100.0, 3.0), 2, 3),
    "cre22": (welded_beam, (0.125, 0.1, 0.1, 0.125), (5.0, 10.0, 10.0, 5.0), 2, 4),
    "cre23": (disc_brake, (55.0, 75.0, 1000.0, 11.0), (80.0, 110.0, 3000.0, 20.0), 2, 4),
    "cre24": (_cre24, (2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0), (3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5), 2, 11),
    "cre25": (_cre25, (12.0, 12.0, 12.0, 12.0), (60.0, 60.0, 60.0, 60.0), 2, 1),
    "cre31": (CLASSICS["car_side_impact"][0], (0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4), (1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2), 3, 10),
    "cre32": (_cre32, (150.0, 20.0, 13.0, 10.0, 14.0, 0.63), (274.32, 32.31, 25.0, 11.71, 18.0, 0.75), 3, 9),
    "cre51": (CLASSICS["water"][0], (0.01, 0.01, 0.01), (0.45, 0.10, 0.10), 5, 7),
}


class CREProblem(Problem):
    """One of :data:`CRE`, with jMetal's bounds; ``G`` is jMetal's constraints with the opposite sign."""

    def __init__(self, key: str) -> None:
        name = str(key).lower()
        if name not in CRE:
            raise ValueError(f"Unknown CRE problem {key!r}; the problems are: {', '.join(CRE)}.")
        formula, lower, upper, n_obj, n_constraints = CRE[name]
        self.key = name
        self._formula = formula
        self.n_var = len(lower)
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.xl = np.asarray(lower, dtype=float)
        self.xu = np.asarray(upper, dtype=float)

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        f, c = self._formula(X.T)
        out["F"] = np.column_stack(f)
        out["G"] = -np.column_stack(c)


__all__ = ["CRE", "CREProblem"]
