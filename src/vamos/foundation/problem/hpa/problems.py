"""The multi-objective HPA problems: Namura's human-powered aircraft design benchmark.

N. Namura, "Single and Multi-Objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design",
EMO 2025, LNCS 15513, https://doi.org/10.1007/978-981-96-3506-1_14 (arXiv:2312.08953), with the definitions of the
authors' reference code (https://github.com/Nobuo-Namura/hpa, commit ``3524115``, ``hpa/problem.py``;
BSD-3-Clause-Clear, copyright 2023 Fujitsu Limited).

A problem ``HPAMNL-l`` has ``M`` objectives, ``N`` constraints (0 for the penalized version), index ``L`` and
difficulty level ``l``: 0, 1 or 2, each design space containing the one below. The wing has ``n_div`` segments, 4
in the paper. The variables are normalized to [0, 1], as in the reference code. Every constrained problem
``HPAMNL`` has a penalized version ``HPAM0L`` (the paper's eq. 5): each objective ``f_i`` becomes
``f_i + v_i * sum_j w_j * max(0, g_j)``, with the paper's weights (Table 1). The single-objective problems
(HPA101-HPA103 and HPA131-HPA143) are not included.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem

from .design import MAX_DIHEDRAL_ANGLE_AT_ROOT, WingConfig
from .plies import optimize_plies
from .wing import Wing

Array = np.ndarray[Any, Any]

#: The objectives (all minimized) and their penalty weights ``v_i`` (Table 1).
OBJECTIVES: dict[str, tuple[Callable[[Wing], float], float]] = {
    "power": (lambda w: w.power, 10.0),  # f1, required power [W]
    "drag": (lambda w: w.drag, 1.0),  # f2 [N]
    "velocity": (lambda w: -w.v_inf, 1.0),  # f3, minus the cruise speed [m/s]
    "deflection": (lambda w: max(np.abs(w.wing_tip_deflection), np.abs(w.zerolift_tip_deflection)), 0.5),  # f4 [m]
    "twist": (lambda w: w.max_twist, 0.1),  # f5, twist at the wing tip [deg]
    "efficiency": (lambda w: -w.wing_efficiency, 0.1),  # f6, minus the span efficiency
    "empty_weight": (lambda w: w.empty_weight, 1.0),  # f7 [kg]
    "span": (lambda w: w.span, 1.0),  # f8 [m]
    "root_aoa": (lambda w: w.y_aoa[0], 1.0),  # f9, angle of attack at the root [deg]
    "wire_tension": (lambda w: w.wire_tension, 200.0),  # f10 [N]
    "payload": (lambda w: -w.payload, 10.0),  # f11, minus the payload [kg]
}

#: The constraints, ``g <= 0`` feasible, and their penalty weights ``w_j`` (Table 1), in the reference code's order.
CONSTRAINTS: dict[str, tuple[Callable[[Wing], float], float]] = {
    "strain": (lambda w: w.strain_constraint, 10.0),  # g1, the CFRP strain
    "zerolift": (lambda w: -w.zerolift_tip_deflection, 2.0),  # g3, the tip above the ground when parked
    "deflection": (lambda w: w.deflection_constraint, 2.0),  # g2, the dihedral angle at the tip in flight
    "power": (lambda w: w.power_constraint, 0.1),  # g4, at most 400 W
    "speed": (lambda w: w.speed_constraint, 10.0),  # g5, at least 7.3 m/s
}

_ALL = ("strain", "zerolift", "deflection", "power")
_STRUCTURE = ("strain", "zerolift", "deflection")
_NO_WIRE = ("strain", "speed")


@dataclass(frozen=True)
class HPAType:
    """A problem of Table 2: its wing (``max_plys``, wire, dihedral and payload variables), objectives and constraints."""

    max_plys: int
    wire: bool
    dihedral: bool
    payload: bool
    objectives: tuple[str, ...]
    constraints: tuple[str, ...]
    root_dihedral: float = MAX_DIHEDRAL_ANGLE_AT_ROOT

    def config(self, level: int, n_div: int = 4) -> WingConfig:
        return WingConfig(n_div, self.max_plys, level, self.wire, self.dihedral, self.payload, self.root_dihedral)


#: The penalized problems by index ``M0L``; the constrained one of each is ``MNL``.
HPA_TYPES: dict[str, HPAType] = {
    "201": HPAType(10, True, True, False, ("power", "velocity"), _ALL),
    "202": HPAType(20, False, False, False, ("power", "deflection"), _NO_WIRE, root_dihedral=0.0),
    "203": HPAType(20, True, True, True, ("drag", "payload"), _STRUCTURE),
    "204": HPAType(20, True, True, False, ("velocity", "twist"), _ALL),
    "205": HPAType(20, True, True, False, ("velocity", "wire_tension"), _ALL),
    "301": HPAType(10, True, True, False, ("power", "velocity", "efficiency"), _ALL),
    "302": HPAType(20, False, False, False, ("power", "deflection", "twist"), _NO_WIRE, root_dihedral=0.0),
    "303": HPAType(20, True, True, True, ("drag", "root_aoa", "payload"), _STRUCTURE),
    "304": HPAType(20, True, True, False, ("drag", "velocity", "twist"), _ALL),
    "305": HPAType(20, True, True, False, ("power", "velocity", "wire_tension"), _ALL),
    "401": HPAType(20, True, True, False, ("drag", "velocity", "twist", "empty_weight"), _ALL),
    "402": HPAType(20, False, False, False, ("power", "deflection", "twist", "span"), _NO_WIRE, root_dihedral=0.0),
    "403": HPAType(20, True, True, False, ("velocity", "efficiency", "empty_weight", "wire_tension"), _ALL),
    "501": HPAType(20, True, True, False, ("drag", "velocity", "twist", "efficiency", "empty_weight"), _ALL),
    "502": HPAType(20, True, True, True, ("power", "velocity", "root_aoa", "wire_tension", "payload"), _ALL),
    "601": HPAType(20, True, True, False, ("power", "velocity", "twist", "efficiency", "empty_weight", "span"), _ALL),
    "901": HPAType(
        20,
        True,
        True,
        True,
        ("power", "velocity", "twist", "efficiency", "empty_weight", "span", "root_aoa", "wire_tension", "payload"),
        _ALL,
    ),
}


def constrained_index(index: str) -> str:
    """The index ``MNL`` of the constrained version of the penalized problem ``M0L``."""
    return f"{index[0]}{len(HPA_TYPES[index].constraints)}{index[2]}"


#: Every problem by index: the penalized ones (``M0L``) and the constrained ones (``MNL``), with whether it is the latter.
HPA_INDICES: dict[str, tuple[str, bool]] = {
    **{index: (index, False) for index in HPA_TYPES},
    **{constrained_index(index): (index, True) for index in HPA_TYPES},
}


def analyze(config: WingConfig, x: Array) -> Wing:
    """The wing of the normalized variables ``x``, analyzed as ``HPADesigner._evaluate`` does."""
    wing = Wing(config.wing(x), config.n_div, config.level)
    wing.interpolate()
    wing.lifting_line()
    wing.refine_grid()
    wing.compute_wing_weight()
    wing.compute_local_lift()
    wing.stiffness()
    wing.zerolift_deflection()
    wing.flight_deflection()
    wing.twist()
    if config.level < 2:
        optimize_plies(wing)
    wing.compute_constraints()
    wing.compute_power()
    return wing


class HPAProblem(Problem):
    """``HPA<index>-<level>``, such as ``HPA241-0`` (constrained) or ``HPA201-0`` (penalized), with ``n_div`` segments."""

    def __init__(self, index: str, level: int, *, n_div: int = 4) -> None:
        key = str(index).lower().removeprefix("hpa")
        if key not in HPA_INDICES:
            raise ValueError(f"Unknown HPA problem {index!r}; the problems are: {', '.join(sorted(HPA_INDICES))}.")
        if level not in (0, 1, 2):
            raise ValueError(f"The level of an HPA problem is 0, 1 or 2; got {level!r}.")
        if not isinstance(n_div, int) or isinstance(n_div, bool) or n_div < 1:
            raise ValueError(f"n_div must be a positive integer; got {n_div!r}.")
        base, constrained = HPA_INDICES[key]
        self.index = key
        self.level = level
        self.constrained = constrained
        self.type = HPA_TYPES[base]
        self.config = self.type.config(level, n_div)
        self.n_var = self.config.n_var
        self.n_obj = len(self.type.objectives)
        self.n_constraints = len(self.type.constraints) if constrained else 0
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def _row(self, x: Array) -> tuple[list[float], list[float]]:
        wing = analyze(self.config, x)
        objectives = [OBJECTIVES[name] for name in self.type.objectives]
        constraints = [CONSTRAINTS[name] for name in self.type.constraints]
        if self.constrained:
            return [value(wing) for value, _ in objectives], [value(wing) for value, _ in constraints]
        penalty: float | None = None
        for value, weight in constraints:
            term = weight * max(0.0, value(wing))
            penalty = term if penalty is None else penalty + term
        assert penalty is not None
        return [value(wing) + weight * penalty for value, weight in objectives], []

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        rows = [self._row(x) for x in X]
        out["F"] = np.array([f for f, _ in rows], dtype=float).reshape(X.shape[0], self.n_obj)
        if self.constrained:
            out["G"] = np.array([g for _, g in rows], dtype=float).reshape(X.shape[0], self.n_constraints)


__all__ = ["CONSTRAINTS", "HPAProblem", "HPAType", "HPA_INDICES", "HPA_TYPES", "OBJECTIVES", "analyze", "constrained_index"]
