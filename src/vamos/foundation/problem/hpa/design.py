"""The design variables of an HPA problem and the wing they define.

This is ``HPADesigner.__init__`` and ``HPADesigner._x2param`` of Namura's reference code
(https://github.com/Nobuo-Namura/hpa, commit ``3524115``, ``hpa/designer.py``; BSD-3-Clause-Clear, copyright 2023
Fujitsu Limited), ported without pandas. The numbers and the order of the operations are the reference code's, so
that the evaluations agree with it to the last bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]

from .airfoils import AIRFOILS, ANGLE_OF_ATTACK, LIFT_SLOPE, PROFILE_DRAG, REYNOLDS, THICKNESS, ZERO_LIFT_ANGLE

Array = np.ndarray[Any, Any]

GRAVITY = 9.8
#: Thickness of a CFRP ply [m].
PLY_THICKNESS = 0.125e-3
#: Minimum increment of the width of a flange laminate [m].
DELTA_WIDTH = 2e-3
#: Largest tension of the flying wires [N].
WIRE_MAX_TENSION = 135.0 * GRAVITY
#: Largest dihedral angle at the wing root [deg], and the root's dihedral when it is not a variable.
MAX_DIHEDRAL_ANGLE_AT_ROOT = 4.0
#: Mass of the pilot [kg], and the largest payload.
PILOT_WEIGHT = 60.0


class IndexTable:
    """A property of the airfoils at a (fractional) airfoil index: ``scipy.interpolate.interp1d(index, values)``.

    For floats, ``interp1d`` interpolates with ``numpy.interp``, after the same conversion of its argument, and
    raises the same error out of the range of the indices; this is that, without its overhead.
    """

    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = np.asarray(values, dtype=float)
        self.index = np.arange(len(values))

    def __call__(self, x: Any) -> Array:
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.inexact):
            x = x.astype(float)
        below, above = x < self.index[0], x > self.index[-1]
        if below.any():
            value = x.flat[int(np.argmax(below))]
            raise ValueError(f"A value ({value}) in x_new is below the interpolation range's minimum value ({self.index[0]}).")
        if above.any():
            value = x.flat[int(np.argmax(above))]
            raise ValueError(f"A value ({value}) in x_new is above the interpolation range's maximum value ({self.index[-1]}).")
        return np.asarray(np.interp(x, self.index, self.values))


@dataclass(frozen=True)
class AirfoilModels:
    """The airfoil properties as functions of a (fractional) airfoil index, as the reference code interpolates them.

    ``grids`` (row ``d``: the ``sizes[d]`` points of dimension ``d``) and ``table`` are those of ``profile_drag``,
    for the compiled kernel.
    """

    lift_slope: IndexTable
    zero_lift_angle: IndexTable
    thickness: IndexTable
    profile_drag: Any
    re_min: float
    re_max: float
    aoa_min: float
    aoa_max: float
    grids: Array
    sizes: Array
    table: Array


@lru_cache(maxsize=1)
def airfoil_models() -> AirfoilModels:
    index = np.arange(len(AIRFOILS))
    reynolds = np.asarray(REYNOLDS, dtype=float)
    angles = np.asarray(ANGLE_OF_ATTACK, dtype=float)
    profile_drag = RegularGridInterpolator((index, reynolds, angles), np.array(PROFILE_DRAG, dtype=float), method="linear")
    grids = np.zeros((3, max(len(grid) for grid in profile_drag.grid)))
    for d, grid in enumerate(profile_drag.grid):
        grids[d, : len(grid)] = grid
    return AirfoilModels(
        lift_slope=IndexTable(LIFT_SLOPE),
        zero_lift_angle=IndexTable(ZERO_LIFT_ANGLE),
        thickness=IndexTable(THICKNESS),
        profile_drag=profile_drag,
        re_min=float(reynolds.min()),
        re_max=float(reynolds.max()),
        aoa_min=float(angles.min()),
        aoa_max=float(angles.max()),
        grids=grids,
        sizes=np.array([len(grid) for grid in profile_drag.grid], dtype=np.int64),
        table=np.ascontiguousarray(profile_drag.values, dtype=float),
    )


@dataclass(frozen=True)
class WingDesign:
    """A wing, as ``_x2param`` returns it: the sections, the beam and the stacking sequence of its CFRP plies."""

    span: float
    y_div: Array
    y_chord: Array
    y_aoa: Array
    y_diameter: Array
    y_airfoil: Array
    y_wire: float
    wire_tension: float
    dihedral_angle_at_root: float
    payload: float
    ply_wing: list[Array]
    strain_ratios: Array


def _base_plies(segment: int, n_div: int) -> Array:
    """The four plies around the whole pipe: [axis angle, half width angle, root, tip, thickness] each."""
    angle = 30 if segment == int(n_div / 2) else 45
    t = PLY_THICKNESS
    return np.array([[90, 180, 0, 1, t], [angle, 180, 0, 1, t], [-angle, 180, 0, 1, t], [90, 180, 0, 1, t]])


def _linspace(start: Array, stop: Array, num: int) -> Array:
    """``numpy.linspace(start[i], stop[i], num)`` in row ``i``, computed as NumPy computes each."""
    div = num - 1
    delta = stop - start
    step = delta / div
    rows = np.arange(0, num, dtype=float)[None, :] * step[:, None]
    zero = step == 0  # NumPy's handling of denormal steps
    if zero.any():
        rows[zero] = np.arange(0, num, dtype=float)[None, :] / div * delta[zero][:, None]
    rows += start[:, None]
    rows[:, -1] = stop
    return rows


def _plies(n_div: int, m: int) -> Array:
    """The plies of every segment: its four around the whole pipe, then ``m`` flanges of axis, root and tip 0."""
    return _plies_template(n_div, m).copy()


@cache
def _plies_template(n_div: int, m: int) -> Array:
    plies = np.zeros((n_div, 4 + m, 5))
    plies[:, :4] = [_base_plies(i, n_div) for i in range(n_div)]
    plies[:, 4:, 4] = PLY_THICKNESS
    plies.flags.writeable = False
    return plies


def _rounded_sigmoid(x: Array, a: float = 50.0, b: float = 0.5) -> Array:
    # Exact 0 and 1 are needed at the root and tip of a pipe: four digits are its machining accuracy.
    return np.asarray(np.round(1.0 / (1.0 + np.exp(-a * (x - b))), 4))


@dataclass(frozen=True)
class WingConfig:
    """The design variables of an HPA problem: ``HPADesigner(n_div, max_plys, level, AIRFOIL, WIRE, DIHEDRAL, PAYLOAD)``.

    ``root_dihedral`` is the dihedral angle at the root when it is not a variable: 4 degrees, or 0 for the
    problems without wire and dihedral variables that minimize the deflection (HPA202, HPA302, HPA402 and their
    constrained versions).
    """

    n_div: int = 4
    max_plys: int = 10
    level: int = 0
    wire: bool = True
    dihedral: bool = False
    payload: bool = False
    root_dihedral: float = MAX_DIHEDRAL_ANGLE_AT_ROOT

    @property
    def airfoil(self) -> bool:
        return self.level > 0

    @property
    def n_var(self) -> int:
        n, m = self.n_div, self.max_plys
        count = {0: n * 3 + 4, 1: n * 6 + 2, 2: n * (5 + 2 * m) + 2}[self.level]
        count += int(self.wire) + int(self.dihedral) + int(self.payload)
        return count + (n + 1 if self.airfoil else 0)

    def _extra_index(self, name: str) -> int:
        n, m = self.n_div, self.max_plys
        index = {0: n * 3 + 4, 1: n * 6 + 2, 2: n * (5 + 2 * m) + 2}[self.level]
        for flag, label in ((self.wire, "wire"), (self.dihedral, "dihedral"), (self.payload, "payload")):
            if label == name:
                return index
            index += int(flag)
        raise ValueError(name)

    def bounds(self) -> tuple[Array, Array]:
        """The lower and upper bounds of the design variables, before their normalization to [0, 1]."""
        n, m = self.n_div, self.max_plys
        lower, upper = np.zeros(self.n_var), np.ones(self.n_var)
        lower[0:n], upper[0:n] = 8 / n, 20 / n  # segment lengths [m]
        lower[n : n * 2 + 1], upper[n : n * 2 + 1] = 0.4, 1.5  # chords [m]
        lower[n * 2 + 1 : n * 2 + 2], upper[n * 2 + 1 : n * 2 + 2] = 4.0, 7.0  # angle of attack at the root [deg]
        lower[n * 2 + 2 : n * 3 + 2], upper[n * 2 + 2 : n * 3 + 2] = -8.0 / n, 0.0  # washout [deg]
        if self.level == 0:
            lower[n * 3 + 2 : n * 3 + 4], upper[n * 3 + 2 : n * 3 + 4] = 0.01, 1.0  # strain reduction factors
        if self.level > 0:
            lower[n * 3 + 2 : n * 4 + 2], upper[n * 3 + 2 : n * 4 + 2] = 0.8, 1.0  # beam diameter over the largest
        if self.level == 1:
            lower[n * 4 + 2 : n * 6 + 2], upper[n * 4 + 2 : n * 6 + 2] = 0.01, 1.0  # strain reduction factors
        elif self.level == 2:
            lower[n * 4 + 2 : n * (4 + m) + 2], upper[n * 4 + 2 : n * (4 + m) + 2] = 0.0, 1.0  # flange lengths
            lower[n * (4 + m) + 2 : n * (4 + m * 2) + 2] = DELTA_WIDTH  # flange width increments [m]
            upper[n * (4 + m) + 2 : n * (4 + m * 2) + 2] = 260e-3
            lower[n * (4 + m * 2) + 2 : n * (5 + m * 2) + 2] = 0.0  # spanwise reference positions
            upper[n * (4 + m * 2) + 2 : n * (5 + m * 2) + 2] = 1.0
        if self.wire:
            lower[self._extra_index("wire")], upper[self._extra_index("wire")] = 0.0, WIRE_MAX_TENSION
        if self.dihedral:
            lower[self._extra_index("dihedral")] = 0.0
            upper[self._extra_index("dihedral")] = MAX_DIHEDRAL_ANGLE_AT_ROOT
        if self.payload:
            lower[self._extra_index("payload")], upper[self._extra_index("payload")] = 0.0, PILOT_WEIGHT
        if self.airfoil:
            lower[-n - 1 :], upper[-n - 1 :] = 0, len(AIRFOILS) - 1
        return lower, upper

    def baseline_airfoils(self) -> Array:
        """The airfoils of level 0, where they are not variables."""
        n = self.n_div
        index = {name: i for i, name in enumerate(AIRFOILS)}
        airfoils = np.full(n + 1, index["DAE31"])
        if n < 3:
            airfoils[0:n] = index["DAE21"]
        elif n == 3:
            airfoils[0:2] = index["DAE11"]
            airfoils[2] = index["DAE21"]
        else:
            airfoils[0:2] = index["DAE11"]
            airfoils[2] = index["DAE21"]
            airfoils[-1] = index["DAE41"]
        return airfoils

    def wing(self, normalized: Array) -> WingDesign:
        """The wing of normalized design variables in [0, 1] (``_x2param`` with ``NORMALIZED=True``)."""
        n, m = self.n_div, self.max_plys
        lower, extent = _scale(self)
        x = lower + extent * np.asarray(normalized, dtype=float)
        span = x[0:n].sum() * 2
        y_div = np.hstack([0.0, x[0:n]]).cumsum() / (span * 0.5)
        y_div = np.where(y_div < 1.0, y_div, 1.0)
        y_chord = x[n : n * 2 + 1]
        y_aoa = x[n * 2 + 1 : n * 3 + 2].cumsum()
        y_airfoil = x[-n - 1 :] if self.airfoil else _baseline_airfoils(self).copy()
        y_wire = 0.95 * y_div[int(n / 2)] if n > 1 else 0.5 * y_div[1]
        wire_tension = x[self._extra_index("wire")] if self.wire else 0.0
        dihedral = x[self._extra_index("dihedral")] if self.dihedral else self.root_dihedral
        payload = x[self._extra_index("payload")] if self.payload else 0.0
        strain_ratios = np.ones([2, n])
        thickness = airfoil_models().thickness
        # The thinnest section of each segment, on 100 stations.
        airfoils = _linspace(y_airfoil[:-1], y_airfoil[1:], 100)
        if not np.all((airfoils >= thickness.index[0]) & (airfoils <= thickness.index[-1])):
            for row in airfoils:  # the error of the first segment out of the airfoils
                thickness(row)
        largest = (_linspace(y_chord[:-1], y_chord[1:], 100) * thickness(airfoils)).min(axis=1)
        if self.level == 0:
            y_diameter = np.hstack([largest[0], largest])
            strain_ratios = np.tile(x[n * 3 + 2 : n * 3 + 4], [n, 1]).T
        else:
            y_diameter = x[n * 3 + 2 : n * 4 + 2] * largest
            y_diameter = np.hstack([y_diameter[0], y_diameter])
        if self.level < 2:
            phis = 30 + np.rad2deg((DELTA_WIDTH * np.arange(m))[None, :] / (0.5 * y_diameter[1:])[:, None])
            phis = np.clip(phis, 0, 179)
            plies = _plies(n, m)
            plies[:, 4:, 1] = phis
            plies[:, 4:, 3] = 1.0
            if self.level == 1:
                strain_ratios = x[n * 4 + 2 : n * 6 + 2].reshape(2, -1)
        else:
            length = x[n * 4 + 2 : n * (4 + m) + 2].reshape(n, m)
            length = 1 - np.clip(length.cumsum(axis=1), 0, 1)
            fixed = _rounded_sigmoid(x[n * (4 + m * 2) + 2 : n * (5 + m * 2) + 2].reshape(n, 1))
            tip = fixed + length * (1 - fixed)
            root = fixed * (1 - length)
            widths = x[n * (4 + m) + 2 : n * (4 + m * 2) + 2].reshape(n, m).copy()
            widths[:, 0] -= DELTA_WIDTH
            phis = 30 + np.rad2deg(widths / (0.5 * y_diameter[1:])[:, None]).cumsum(axis=1)
            phis = np.clip(phis, 0, 179)
            plies = _plies(n, m)
            plies[:, 4:, 1] = phis
            plies[:, 4:, 2] = root
            plies[:, 4:, 3] = tip
        return WingDesign(
            span=float(span),
            y_div=y_div,
            y_chord=y_chord,
            y_aoa=y_aoa,
            y_diameter=y_diameter,
            y_airfoil=y_airfoil,
            y_wire=float(y_wire),
            wire_tension=float(wire_tension),
            dihedral_angle_at_root=float(dihedral),
            payload=float(payload),
            ply_wing=list(plies),
            strain_ratios=strain_ratios,
        )


@cache
def _scale(config: WingConfig) -> tuple[Array, Array]:
    """The lower bounds of the variables and the extents of their ranges, for :meth:`WingConfig.wing`."""
    lower, upper = config.bounds()
    extent = upper - lower
    lower.flags.writeable = False
    extent.flags.writeable = False
    return lower, extent


@cache
def _baseline_airfoils(config: WingConfig) -> Array:
    airfoils = config.baseline_airfoils()
    airfoils.flags.writeable = False
    return airfoils


__all__ = ["AirfoilModels", "IndexTable", "WingConfig", "WingDesign", "airfoil_models"]
