"""The aerodynamic and structural analysis of an HPA wing.

This is ``HPADesigner._evaluate`` of Namura's reference code (https://github.com/Nobuo-Namura/hpa, commit
``3524115``, ``hpa/designer.py``; BSD-3-Clause-Clear, copyright 2023 Fujitsu Limited), in its benchmark mode
(``FINE_MODE=False``), ported without pandas: the reference code's table interpolations are ``numpy.interp``,
which pandas uses for them, so the numbers and the order of the operations are the reference code's. The analysis
solves the lifting-line equations on a coarse spanwise grid, refines the grid for the beam-column analysis of the
CFRP main spar, at flight (with the flying wires) and on the ground, and, below level 2, sizes the flange plies.

With Numba, the kernels of :mod:`._kernels` run the loops of the analysis, and what the plies do not change (the
grid, the geometry of the spar, the stiffness of its plies around the whole pipe, the weight of its ribs, skin,
joints and wire) is computed once per wing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg as linalg  # type: ignore[import-untyped]

from ._kernels import BLOCK, kernels
from ._tables import entire_moduli, flange_modulus, lifting_line_grid, moduli, structural_grid
from .design import GRAVITY, PILOT_WEIGHT, WIRE_MAX_TENSION, WingDesign, airfoil_models

Array = np.ndarray[Any, Any]

RHO = 0.1174 * GRAVITY  # air density at 30 degrees Celsius, the worst flight condition
VISC_MU = 1.615e-5 * RHO
MAX_POWER = 400.0  # [W]
V_MIN = 7.3  # [m/s]
LOAD_FACTOR = 1.5
SAFETY_FACTOR = 2.0
WATER_WEIGHT = 4.0  # [kg]
DENSITY_RIB_SKIN = 0.195  # [kg/m^2]
MAX_DIHEDRAL_ANGLE_AT_TIP = 8.0  # [deg]
DRIVETRAIN_EFFICIENCY = 0.85 * 0.95  # propeller * drivetrain
CALIBRATION_FACTOR = 1.1
CDS_BODY = 0.09  # [m^2]
CD_TAIL = 0.009
HORIZONTAL_TAIL_VOLUME = 0.415
HORIZONTAL_TAIL_ARM = 5.3  # [m]
VERTICAL_TAIL_VOLUME = 0.0137
VERTICAL_TAIL_ARM = 6.1  # [m]
BASE_WIRE_AREA = 2e-6  # [m^2]
WIRE_DENSITY = 0.017 / BASE_WIRE_AREA  # [kg/m^3]
WIRE_CD = 1.0
WIRE_JOINT_WEIGHT = 0.25  # [kg]
Z_WIRE = 1.85  # [m]
WIRE_ITERATIONS = 5
ALLOWABLE_STRAIN = 0.0027 / (LOAD_FACTOR * SAFETY_FACTOR)
DENSITY_CFRP = 1.35e3  # [kg/m^3]
COEF_REAR_SPAR = 1.2  # weight of the main spar (1), the rear spar (0.1) and its arm (0.1)


def moment(y: Array, F: Array) -> Array:
    """The bending moment of the loads ``F`` at the points ``y``, from the tip."""
    y, F = y[::-1], F[::-1]
    return np.asarray(np.hstack([0.0, np.cumsum((np.cumsum(F)[:-1] + 0.5 * F[1:]) * np.abs(np.diff(y)))])[::-1])


def moment_with_axial_force(y: Array, F: Array, T: Array, EI: Array) -> tuple[Array, Array]:
    """The bending moment and shear force of beam-column theory, with the axial force ``T``.

    The recurrence runs on Python floats, whose operations are the IEEE operations of NumPy's scalars in the
    reference code, in the same order: the same values, a few times faster.
    """
    n = len(y)
    L = np.abs(np.diff(y)).tolist()
    Fs, Ts, EIs = np.asarray(F, dtype=float).tolist(), np.asarray(T, dtype=float).tolist(), np.asarray(EI, dtype=float).tolist()
    M, S = [0.0] * n, [0.0] * n
    for i in range(n - 2, -1, -1):
        TL_EI = Ts[i] * (0.5 * L[i]) ** 2 / EIs[i]
        S[i] = (Fs[i] + (1 - TL_EI) * S[i + 1] - Ts[i] * L[i] * M[i + 1] / EIs[i]) / (1 + TL_EI)
        M[i] = (0.5 * L[i] * Fs[i] + (1 - TL_EI) * M[i + 1] + L[i] * S[i + 1]) / (1 + TL_EI)
    return np.array(M), np.array(S)


def _integrate(values: Array, y0: Array, span: float) -> Array:
    return np.asarray(np.hstack([0.0, 0.5 * (values[1:] + values[:-1]) * np.diff(y0) * span * 0.5]).cumsum())


class Wing:
    """The state of an analysis: the reference designer's attributes, set as ``_evaluate`` computes them."""

    def __init__(self, design: WingDesign, n_div: int, level: int) -> None:
        self.level = level
        self.n_div = n_div
        self.n_aero = max(50, n_div * 5)
        self.n_struc = 6 * self.n_aero + 1
        self.span = design.span
        self.y_div = design.y_div
        self.y_chord = design.y_chord
        self.y_aoa = design.y_aoa
        self.y_diameter = design.y_diameter
        self.y_airfoil = design.y_airfoil
        self.y_wire = design.y_wire
        self.wire_tension = design.wire_tension
        self.dihedral_angle_at_root = design.dihedral_angle_at_root
        self.payload = design.payload
        # The plies of every segment, one after the other; ply_wing[i] is a view of segment i's.
        self._plies = np.vstack(design.ply_wing).astype(float, copy=False)
        ends = np.cumsum([0] + [len(plies) for plies in design.ply_wing]).tolist()
        self.ply_wing = [self._plies[start:end] for start, end in zip(ends[:-1], ends[1:])]
        self.strain_ratios = design.strain_ratios
        self.max_plys = self.ply_wing[0].shape[0] - 4
        # The kernels sum the plies of a segment as NumPy does, in one block of its pairwise summation.
        self._kernels = kernels() if max(len(plies) for plies in self.ply_wing) <= BLOCK else None
        self._segment = np.repeat(np.arange(len(self.ply_wing)), [len(plies) for plies in self.ply_wing])
        self._entire: tuple[Array, Array, Array] | None = None
        self._flanges: tuple[Array, ...] = ()

    def interpolate(self) -> None:
        """The sections interpolated on the lifting-line grid, from the root (``theta = pi / 2``) to the tip."""
        self.theta_aero, self.y0_aero = lifting_line_grid(self.n_aero)[:2]
        self.chord_aero = np.interp(self.y0_aero, self.y_div, self.y_chord)
        self.aoa_aero = np.interp(self.y0_aero, self.y_div, self.y_aoa)
        self.airfoil_aero = np.interp(self.y0_aero, self.y_div, self.y_airfoil)

    def lifting_line(self) -> None:
        models = airfoil_models()
        cl_slope = models.lift_slope(self.airfoil_aero)
        _, _, m, ms, sin_theta, sin_thetas, sin_ms_thetas = lifting_line_grid(self.n_aero)
        mu = cl_slope[:-1] * self.chord_aero[:-1] * 0.25 / self.span
        absolute_aoa = np.deg2rad(self.aoa_aero - models.zero_lift_angle(self.airfoil_aero))
        # The kernels sum a row as NumPy does, in one block of its pairwise summation.
        compiled = self._kernels if self.n_aero <= BLOCK else None
        if compiled is not None:
            A = np.empty((self.n_aero, self.n_aero))
            compiled.lifting_line(mu, ms, sin_thetas, sin_ms_thetas, A)
        else:
            A = (ms * np.tile(mu, [self.n_aero, 1]).T + sin_thetas) * sin_ms_thetas
        b = mu * absolute_aoa[:-1] * sin_theta
        xa = linalg.lu_solve(linalg.lu_factor(A), b)
        self.wing_area = np.sum((self.y_chord[:-1] + self.y_chord[1:]) * np.diff(self.y_div) * 0.5) * self.span
        self.aspect_ratio = self.span**2 / self.wing_area
        delta = np.sum(m[1:] * ((xa[1:] / xa[0]) ** 2))
        self.wing_efficiency = 1 / (1 + delta)
        self.CL = np.pi * self.aspect_ratio * xa[0]
        self.CDi = self.CL**2 / (np.pi * self.wing_efficiency * self.aspect_ratio)
        if compiled is not None:
            induced_aoa = np.empty(self.n_aero + 1)
            compiled.induced_angles(xa, ms, sin_ms_thetas, sin_thetas, induced_aoa)
            induced_aoa[-1] = absolute_aoa[-1]
        else:
            induced_aoa = np.sum(ms * np.tile(xa, [self.n_aero, 1]) * sin_ms_thetas / sin_thetas, axis=1)
            induced_aoa = np.hstack([induced_aoa, absolute_aoa[-1]])
        self.local_cl_aero = cl_slope * (absolute_aoa - induced_aoa)

    def refine_grid(self) -> None:
        self.y0 = structural_grid(self.n_struc)
        self.chord = np.interp(self.y0, self.y0_aero, self.chord_aero)
        self.aoa = np.interp(self.y0, self.y0_aero, self.aoa_aero)
        self.local_cl = np.interp(self.y0, self.y0_aero, self.local_cl_aero)
        diameter = np.full(len(self.y0), self.y_diameter[-1])
        for i in range(len(self.y_div) - 1, 0, -1):
            diameter[self.y0 < self.y_div[i]] = self.y_diameter[i]
        self.diameter = diameter
        self.y = self.y0 * self.span * 0.5
        # The steps of the spar and the squares of their halves, for the beam-column recurrence; the squares are
        # Python's, the C library's pow, as the reference code's NumPy scalars compute them.
        self._steps = np.abs(np.diff(self.y0 * self.span * 0.5))
        self._half_steps_squared = np.array([(0.5 * step) ** 2 for step in self._steps.tolist()])
        self.dihedral_position = 0.5 * self.span * self.y0 * np.sin(np.deg2rad(self.dihedral_angle_at_root))
        if self._kernels is not None:
            self._prepare_kernels()

    def _prepare_kernels(self) -> None:
        """What the kernels take and the plies do not change, computed as the NumPy code computes it."""
        self._half_diameter = 0.5 * self.diameter
        self._cubed_half_diameter = self._half_diameter**3
        joints = [int(np.argmax(np.where(self.y0 < y_joint, self.y0, 0))) for y_joint in self.y_div[1:-1]]
        self._joints = np.array(joints, dtype=np.int64)
        self._joint_weights = np.array([0.5 * self.diameter[k + 1] * 2.293 + 3.373e-2 for k in joints], dtype=float)
        ribs_and_skin = DENSITY_RIB_SKIN * np.diff(self.y0) * 0.5 * self.span * 0.5 * (self.chord[1:] + self.chord[:-1])
        self._ribs_and_skin = np.hstack([ribs_and_skin, 0])
        self._i_wire = int(np.argmin(np.abs(self.y0 - self.y_wire)))
        self._below_wire = self.y0 < self.y_wire
        self._widths = np.diff(self.y0) * self.span * 0.5
        self._lift_chords = self.local_cl[1:] * self.chord[1:] + self.local_cl[:-1] * self.chord[:-1]
        self._wire_weight = self._wire()
        # The steps of the spar and the squares of their halves (and a 0), y0, the dihedral position and the diameter.
        self._geometry = np.zeros((5, self.n_struc))
        self._geometry[0, :-1], self._geometry[1, :-1] = self._steps, self._half_steps_squared
        self._geometry[2], self._geometry[3], self._geometry[4] = self.y0, self.dihedral_position, self.diameter

    def _wire(self) -> float:
        """The wire's diameter and length, and its weight."""
        wire_ratio = self.wire_tension / WIRE_MAX_TENSION
        wire_area = BASE_WIRE_AREA * wire_ratio
        self.wire_diameter = 2 * np.sqrt(wire_area / np.pi)
        self.wire_length = np.sqrt((self.y_wire * self.span * 0.5) ** 2 + Z_WIRE**2)
        return float(self.wire_length * wire_area * WIRE_DENSITY + WIRE_JOINT_WEIGHT * (wire_ratio**1.5))

    def compute_wing_weight(self) -> None:
        self.beam_weight = np.zeros(len(self.ply_wing))
        self.wing_weight = np.zeros(self.n_struc)
        if self._kernels is not None:
            self._kernels.wing_weight(
                self.y0, self._half_diameter, self._segment, self.y_div, self._plies, np.deg2rad(self._plies[:, 1]),
                self.span, DENSITY_CFRP, COEF_REAR_SPAR, self._joints, self._joint_weights, self._ribs_and_skin,
                self._i_wire, self._wire_weight, self.beam_weight, self.wing_weight,
            )  # fmt: skip
            return
        for i, plies in enumerate(self.ply_wing):
            y0_start, y0_end = self.y_div[i], self.y_div[i + 1]
            for ply in plies:
                phi = np.deg2rad(ply[1])
                inside = (self.y0 >= ply[2] * (y0_end - y0_start) + y0_start) & (self.y0 <= ply[3] * (y0_end - y0_start) + y0_start)
                R = np.where(inside, 0.5 * self.diameter, 0)
                L = np.where(inside, np.hstack([np.diff(self.y0), 0]), 0)
                ply_weight = DENSITY_CFRP * 2 * phi * R.max() * ply[4] * 0.5 * self.span * (ply[3] - ply[2]) * (y0_end - y0_start)
                self.beam_weight[i] += ply_weight
                self.wing_weight += DENSITY_CFRP * 2 * phi * R * ply[4] * 0.5 * self.span * L * COEF_REAR_SPAR
        for y_joint in self.y_div[1:-1]:
            k = int(np.argmax(np.where(self.y0 < y_joint, self.y0, 0)))
            self.wing_weight[k] += 0.5 * self.diameter[k + 1] * 2.293 + 3.373e-2  # empirical estimation of a joint
        ribs_and_skin = DENSITY_RIB_SKIN * np.diff(self.y0) * 0.5 * self.span * 0.5 * (self.chord[1:] + self.chord[:-1])
        self.wing_weight += np.hstack([ribs_and_skin, 0])
        # The wire, simplified: its actual weight and length depend on the deflected wing.
        i_wire = np.argmin(np.abs(self.y0 - self.y_wire))
        self.wing_weight[i_wire] += self._wire()

    def compute_local_lift(self) -> None:
        self.body_tail_weight = 0.007 * (self.span - 15.0) ** 2.0 + 14.0 if self.span > 15.0 else 14.0  # empirical
        self.empty_weight = self.wing_weight.sum() * 2 + self.body_tail_weight
        self.weight = PILOT_WEIGHT + WATER_WEIGHT + self.empty_weight + self.payload
        self.v_inf = np.sqrt(2 * self.weight * GRAVITY / (RHO * self.wing_area * self.CL))
        if self._kernels is not None:
            self.local_lift = np.empty(self.n_struc)
            self._kernels.local_lift(0.5 * RHO * (self.v_inf**2), self._widths, self._lift_chords, self.local_lift)
            return
        lift = (
            0.5
            * RHO
            * (self.v_inf**2)
            * (np.diff(self.y0) * self.span * 0.5)
            * 0.5
            * (self.local_cl[1:] * self.chord[1:] + self.local_cl[:-1] * self.chord[:-1])
        )
        self.local_lift = np.hstack([lift, 0.0])  # [N]

    def _entire_plies(self, i: int, plies: Array) -> tuple[Array, Array, Array]:
        """What the plies of segment ``i`` around the whole pipe, which must be symmetric, add to EI, EA and GIp."""
        y0_start, y0_end = self.y_div[i], self.y_div[i + 1]
        sum_ply = np.zeros(self.n_struc)
        Exx, Eyy, Exy, Gxy = np.zeros(self.n_struc), np.zeros(self.n_struc), np.zeros(self.n_struc), np.zeros(self.n_struc)
        for ply in plies[plies[:, 1] >= 180]:
            inside = (self.y0 >= ply[2] * (y0_end - y0_start) + y0_start) & (self.y0 < ply[3] * (y0_end - y0_start) + y0_start)
            mask = ply[4] * np.where(inside, 1, 0)
            sum_ply += mask
            factors = entire_moduli(float(ply[0]))
            Exx += mask * factors[0]
            Eyy += mask * factors[1]
            Exy += mask * factors[2]
            Gxy += mask * factors[3]
        _sum_ply = np.where(sum_ply > 0, sum_ply, np.inf)
        Exx, Eyy, Exy, Gxy = Exx / _sum_ply, Eyy / _sum_ply, Exy / _sum_ply, Gxy / _sum_ply
        Ex = Exx - Exy**2 / np.where(sum_ply > 0, Eyy, np.inf)
        R = np.where((self.y0 >= y0_start) & (self.y0 < y0_end), 0.5 * self.diameter, 0)
        Iy = np.pi * sum_ply * R**3
        return Ex * Iy, Ex * 2 * np.pi * R * sum_ply, Gxy * Iy * 2

    def stiffness(self) -> None:
        self.Ex_ply = np.zeros([self.n_div, self.max_plys])
        self.EI_ply = np.zeros([self.n_struc, self.max_plys])
        self.EA_ply = np.zeros([self.n_struc, self.max_plys])
        self.EI = np.zeros(self.n_struc)
        self.EA = np.zeros(self.n_struc)
        nui, ETL, Ex0 = moduli()
        if self._kernels is not None:
            self._stiffness_kernel(nui, ETL, Ex0)
            return
        self.GIp = np.zeros(self.n_struc)
        for i, plies in enumerate(self.ply_wing):
            y0_start, y0_end = self.y_div[i], self.y_div[i + 1]
            EI, EA, GIp = self._entire_plies(i, plies)
            self.EI += EI
            self.EA += EA
            self.GIp += GIp
            # The flange plies, around part of the pipe.
            for j, ply in enumerate(plies[plies[:, 1] < 180]):
                axis = np.deg2rad(ply[0])
                phi = np.deg2rad(ply[1])
                Ex_j = flange_modulus(axis, Ex0, nui, ETL)
                inside = (self.y0 >= ply[2] * (y0_end - y0_start) + y0_start) & (self.y0 < ply[3] * (y0_end - y0_start) + y0_start)
                R = np.where(inside, 0.5 * self.diameter, 0)
                Iy = (phi + np.sin(phi)) * ply[4] * R**3
                EA_j = Ex_j * 2 * phi * R * ply[4]
                self.EI += Ex_j * Iy
                self.EA += EA_j
                self.Ex_ply[i, j] = Ex_j
                self.EI_ply[:, j] += Ex_j * Iy
                self.EA_ply[:, j] += EA_j
        self.EI[-1] = self.EI[-2]
        self.EA[-1] = self.EA[-2]
        self.GIp[-1] = self.GIp[-2]

    def _stiffness_kernel(self, nui: float, ETL: float, Ex0: float) -> None:
        """:meth:`stiffness` with the compiled kernel; the plies around the whole pipe never change."""
        assert self._kernels is not None
        plies = self._plies
        if self._entire is None:
            # The plies around the whole pipe, and what does not change of the flanges: their segment and index
            # in it, modulus, thickness, and the start and length of their segment.
            rows = np.flatnonzero(plies[:, 1] >= 180)
            entire_EI, entire_EA, GIp = np.empty((self.n_div, self.n_struc)), np.empty((self.n_div, self.n_struc)), np.zeros(self.n_struc)
            self._kernels.entire_plies(
                self.y0, self._half_diameter, self._cubed_half_diameter, self.y_div, self._segment[rows], plies[rows],
                np.array([entire_moduli(float(axis)) for axis in plies[rows, 0]], dtype=float).reshape(-1, 4), np.pi,
                entire_EI, entire_EA, GIp,
            )  # fmt: skip
            self._entire = (entire_EI, entire_EA, GIp)
            rows = np.flatnonzero(plies[:, 1] < 180)
            segment = self._segment[rows]
            axes = np.deg2rad(plies[rows, 0])
            Ex = np.array([flange_modulus(axis, Ex0, nui, ETL) for axis in axes], dtype=float) if axes.any() else np.full(rows.size, Ex0)
            start = self.y_div[segment]
            flange = np.arange(rows.size) - np.searchsorted(segment, segment)
            self._flanges = (rows, segment, flange, Ex, plies[rows, 4], start, self.y_div[segment + 1] - start)
        entire_EI, entire_EA, self.GIp = self._entire
        rows, segment, flange, Ex, thickness, start, length = self._flanges
        flanges = plies[rows]
        phi = np.deg2rad(flanges[:, 1])
        self._kernels.stiffness(
            self.y0, self._half_diameter, self._cubed_half_diameter, entire_EI, entire_EA, segment, flange,
            flanges[:, 2] * length + start, flanges[:, 3] * length + start, Ex, (phi + np.sin(phi)) * thickness,
            Ex * 2 * phi, thickness, self.EI, self.EA, self.EI_ply, self.EA_ply,
        )  # fmt: skip
        self.Ex_ply[segment, flange] = Ex

    def zerolift_deflection(self) -> None:
        """The wing on the ground, under its own weight."""
        if self._kernels is not None:
            n = self.n_struc
            self.moment_zerolift, self.M_EI_zerolift = np.empty(n), np.empty(n)
            self.deflection_angle_zerolift, self.deflection_zerolift = np.empty(n), np.empty(n)
            self.total_deflection_zerolift, self.strain_zerolift = np.empty(n), np.empty(n)
            self._kernels.zerolift(
                self.wing_weight, GRAVITY, self._steps, self.EI, self.y0, self.span, self.dihedral_position, self.diameter,
                self.moment_zerolift, self.M_EI_zerolift, self.deflection_angle_zerolift, self.deflection_zerolift,
                self.total_deflection_zerolift, self.strain_zerolift,
            )  # fmt: skip
            return
        self.moment_zerolift = moment(self.y0 * self.span * 0.5, -self.wing_weight * GRAVITY)
        self.M_EI_zerolift = self.moment_zerolift / self.EI
        self.deflection_angle_zerolift = self._integrate(self.M_EI_zerolift)
        self.deflection_zerolift = self._integrate(self.deflection_angle_zerolift)
        self.total_deflection_zerolift = self.dihedral_position + self.deflection_zerolift
        self.strain_zerolift = self.M_EI_zerolift * self.diameter * 0.5

    def _wire_forces(self, z_wire_deflection: Any) -> tuple[Any, Any]:
        """The axial force of the flying wire and its load on the spar, at the deflection ``z_wire_deflection``."""
        wire_angle = np.arctan((Z_WIRE + z_wire_deflection) / (self.y_wire * self.span * 0.5)) if self.y_wire > 0 else np.pi * 0.5
        dihedral_angle_at_wire = np.arctan(z_wire_deflection / (self.y_wire * self.span * 0.5))
        axial_force = self.wire_tension * np.cos(wire_angle) / np.cos(dihedral_angle_at_wire)
        wire_load = axial_force * np.sin(wire_angle - dihedral_angle_at_wire) / np.cos(wire_angle)
        return axial_force, wire_load

    def flight_deflection(self) -> None:
        """The wing in flight, with the axial force of the flying wires iterated with the deflection."""
        if self._kernels is not None:
            self._flight_kernel()
            return
        i_wire = np.argmin(np.abs(self.y0 - self.y_wire))
        if self.wire_tension > 0:
            z_wire_deflection = 0.0
            for _ in range(WIRE_ITERATIONS):
                axial_force, wire_load = self._wire_forces(z_wire_deflection)
                self.wire_load = np.zeros(self.n_struc)
                self.wire_load[i_wire] = wire_load
                self.axial_force = np.where(self.y0 < self.y_wire, axial_force, 0)
                self.total_load = self.local_lift - self.wire_load - self.wing_weight * GRAVITY
                self.moment = moment_with_axial_force(self.y0 * self.span * 0.5, self.total_load, self.axial_force, self.EI)[0]
                self._deflect()
                z_wire_deflection = self.total_deflection[i_wire]
        else:
            self.wire_load = np.zeros(self.n_struc)
            self.axial_force = np.zeros(self.n_struc)
            self.total_load = self.local_lift - self.wing_weight * GRAVITY
            self.moment = moment(self.y0 * self.span * 0.5, self.total_load)
            self._deflect()
        self.strain = self.M_EI * self.diameter * 0.5 + self.axial_force / self.EA

    def _flight_kernel(self) -> None:
        """:meth:`flight_deflection` with the compiled kernel."""
        assert self._kernels is not None
        out = np.empty((9, self.n_struc))
        wire = self.wire_tension > 0
        iterations = WIRE_ITERATIONS if wire else 1
        z_wire_deflection: Any = 0.0
        for iteration in range(iterations):
            axial_force, wire_load = self._wire_forces(z_wire_deflection) if wire else (0.0, 0.0)
            z_wire_deflection = self._kernels.flight(
                wire, iteration == iterations - 1, self.local_lift, self.wing_weight, GRAVITY, self._i_wire, wire_load,
                self._below_wire, axial_force, self._geometry, self.EI, self.EA, self.span, out,
            )  # fmt: skip
        self.axial_force, self.total_load, self.moment, _, self.M_EI = out[0], out[1], out[2], out[3], out[4]
        self.deflection_angle, self.deflection, self.total_deflection, self.strain = out[5], out[6], out[7], out[8]
        self.wire_load = np.zeros(self.n_struc)
        if wire:
            self.wire_load[self._i_wire] = wire_load

    def _integrate(self, values: Array) -> Array:
        return _integrate(values, self.y0, self.span)

    def _deflect(self) -> None:
        self.M_EI = self.moment / self.EI
        self.deflection_angle = self._integrate(self.M_EI)
        self.deflection = self._integrate(self.deflection_angle)
        self.total_deflection = self.dihedral_position + self.deflection

    def twist(self) -> None:
        """The reference code's simple estimate: the center of pressure moves 1% of the chord."""
        delta_center_of_pressure = 0.01
        torque = self.local_lift[:-1] * 0.5 * (self.chord[1:] + self.chord[:-1]) * delta_center_of_pressure
        torque = torque[::-1].cumsum()[::-1]
        self.twist_angle = np.rad2deg(np.hstack([0.0, torque / (0.5 * (self.GIp[1:] + self.GIp[:-1])) * np.diff(self.y)]).cumsum())

    def update(self, stiffness: bool = False) -> None:
        """Weight, lift and deflections again, after a change of the plies."""
        self.compute_wing_weight()
        self.compute_local_lift()
        if stiffness:
            self.stiffness()
        self.zerolift_deflection()
        self.flight_deflection()

    def compute_constraints(self) -> None:
        self.max_strain = max(np.abs(self.strain).max(), np.abs(self.strain_zerolift).max())
        self.wing_tip_deflection = self.total_deflection[-1]
        self.zerolift_tip_deflection = self.total_deflection_zerolift[-1]
        self.strain_constraint = self.max_strain / ALLOWABLE_STRAIN - 1
        allowable_deflection = 0.5 * self.span * np.sin(np.deg2rad(MAX_DIHEDRAL_ANGLE_AT_TIP))
        self.deflection_constraint = self.wing_tip_deflection - allowable_deflection
        self.max_twist = np.abs(self.twist_angle).max()
        self.speed_constraint = 1 - (self.v_inf / V_MIN) ** 3

    def compute_power(self) -> None:
        """Drag and power on the lifting-line grid (``_compute_power_aero``)."""
        models = airfoil_models()
        re_aero = self.chord_aero * self.v_inf * RHO / VISC_MU
        coordinates = [
            self.airfoil_aero,
            np.clip(re_aero, models.re_min, models.re_max),
            np.clip(self.aoa_aero, models.aoa_min, models.aoa_max),
        ]
        if self._kernels is not None:
            cd0_aero = np.empty(self.airfoil_aero.size)
            out = self._kernels.profile_drag(*coordinates, models.grids, models.sizes, models.table, cd0_aero)
            if out >= 0:
                raise ValueError(f"One of the requested xi is out of bounds in dimension {out}")
        else:
            cd0_aero = models.profile_drag(np.vstack(coordinates).T)
        # The reference code weights both ends of an interval by the chord of its outer end. The products keep its
        # order, which rounding depends on.
        distance = np.abs(np.diff(self.y0_aero))
        CD0 = (
            np.sum(0.5 * (cd0_aero[1:] * self.chord_aero[1:] + cd0_aero[:-1] * self.chord_aero[1:]) * distance * self.span * 0.5)
            * 2
            / self.wing_area
        )
        widths = distance * self.span * 0.5
        mac = 2.0 / self.wing_area * np.sum(widths * 0.5 * (self.chord_aero[1:] ** 2 + self.chord_aero[:-1] ** 2))
        horizontal_tail_area = self.wing_area * mac * HORIZONTAL_TAIL_VOLUME / HORIZONTAL_TAIL_ARM
        vertical_tail_area = self.wing_area * self.span * VERTICAL_TAIL_VOLUME / VERTICAL_TAIL_ARM
        parasite = CDS_BODY + CD0 * self.wing_area + CD_TAIL * (horizontal_tail_area + vertical_tail_area)
        parasite += WIRE_CD * self.wire_length * self.wire_diameter
        self.drag = (CALIBRATION_FACTOR * parasite + self.CDi * self.wing_area) * 0.5 * RHO * self.v_inf**2
        self.power = self.drag * self.v_inf / DRIVETRAIN_EFFICIENCY
        self.power_constraint = self.power - MAX_POWER


__all__ = ["Wing", "moment", "moment_with_axial_force"]
