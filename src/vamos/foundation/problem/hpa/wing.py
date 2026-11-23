"""The aerodynamic and structural analysis of an HPA wing.

This is ``HPADesigner._evaluate`` of Namura's reference code (https://github.com/Nobuo-Namura/hpa, commit
``3524115``, ``hpa/designer.py``; BSD-3-Clause-Clear, copyright 2023 Fujitsu Limited), in its benchmark mode
(``FINE_MODE=False``), ported without pandas: the reference code's table interpolations are ``numpy.interp``,
which pandas uses for them, so the numbers and the order of the operations are the reference code's. The analysis
solves the lifting-line equations on a coarse spanwise grid, refines the grid for the beam-column analysis of the
CFRP main spar, at flight (with the flying wires) and on the ground, and, below level 2, sizes the flange plies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg as linalg  # type: ignore[import-untyped]

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
E_L, E_T, G_LT = 170e9, 6.0e9, 3.3e9  # [Pa]
NU_L, NU_T = 0.28, 0.01
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
        self.ply_wing = [plies.copy() for plies in design.ply_wing]
        self.strain_ratios = design.strain_ratios
        self.max_plys = self.ply_wing[0].shape[0] - 4

    def interpolate(self) -> None:
        """The sections interpolated on the lifting-line grid, from the root (``theta = pi / 2``) to the tip."""
        theta = 0.5 * np.pi * np.linspace(0, 1, self.n_aero + 1)
        y = np.cos(theta)
        order = np.argsort(y, kind="stable")
        self.theta_aero = theta[order]
        self.y0_aero = y[order]
        self.chord_aero = np.interp(self.y0_aero, self.y_div, self.y_chord)
        self.aoa_aero = np.interp(self.y0_aero, self.y_div, self.y_aoa)
        self.airfoil_aero = np.interp(self.y0_aero, self.y_div, self.y_airfoil)

    def lifting_line(self) -> None:
        models = airfoil_models()
        cl_slope = models.lift_slope(self.airfoil_aero)
        theta = self.theta_aero[:-1]
        m = np.arange(1, 2 * self.n_aero, 2)
        ms = np.tile(m, [self.n_aero, 1])
        mu = cl_slope[:-1] * self.chord_aero[:-1] * 0.25 / self.span
        mus = np.tile(mu, [self.n_aero, 1]).T
        thetas = np.tile(theta, [self.n_aero, 1]).T
        absolute_aoa = np.deg2rad(self.aoa_aero - models.zero_lift_angle(self.airfoil_aero))
        A = (ms * mus + np.sin(thetas)) * np.sin(ms * thetas)
        b = mu * absolute_aoa[:-1] * np.sin(theta)
        xa = linalg.lu_solve(linalg.lu_factor(A), b)
        self.wing_area = np.sum((self.y_chord[:-1] + self.y_chord[1:]) * np.diff(self.y_div) * 0.5) * self.span
        self.aspect_ratio = self.span**2 / self.wing_area
        delta = np.sum(m[1:] * ((xa[1:] / xa[0]) ** 2))
        self.wing_efficiency = 1 / (1 + delta)
        self.CL = np.pi * self.aspect_ratio * xa[0]
        self.CDi = self.CL**2 / (np.pi * self.wing_efficiency * self.aspect_ratio)
        induced_aoa = np.sum(ms * np.tile(xa, [self.n_aero, 1]) * np.sin(ms * thetas) / np.sin(thetas), axis=1)
        induced_aoa = np.hstack([induced_aoa, absolute_aoa[-1]])
        self.local_cl_aero = cl_slope * (absolute_aoa - induced_aoa)

    def refine_grid(self) -> None:
        self.y0 = np.linspace(0, 1, self.n_struc)
        self.chord = np.interp(self.y0, self.y0_aero, self.chord_aero)
        self.aoa = np.interp(self.y0, self.y0_aero, self.aoa_aero)
        self.local_cl = np.interp(self.y0, self.y0_aero, self.local_cl_aero)
        diameter = np.full(len(self.y0), self.y_diameter[-1])
        for i in range(len(self.y_div) - 1, 0, -1):
            diameter[self.y0 < self.y_div[i]] = self.y_diameter[i]
        self.diameter = diameter
        self.y = self.y0 * self.span * 0.5
        self.dihedral_position = 0.5 * self.span * self.y0 * np.sin(np.deg2rad(self.dihedral_angle_at_root))

    def compute_wing_weight(self) -> None:
        self.beam_weight = np.zeros(len(self.ply_wing))
        self.wing_weight = np.zeros(self.n_struc)
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
        wire_ratio = self.wire_tension / WIRE_MAX_TENSION
        wire_area = BASE_WIRE_AREA * wire_ratio
        self.wire_diameter = 2 * np.sqrt(wire_area / np.pi)
        self.wire_length = np.sqrt((self.y_wire * self.span * 0.5) ** 2 + Z_WIRE**2)
        wire_weight = self.wire_length * wire_area * WIRE_DENSITY + WIRE_JOINT_WEIGHT * (wire_ratio**1.5)
        self.wing_weight[i_wire] += wire_weight

    def compute_local_lift(self) -> None:
        self.body_tail_weight = 0.007 * (self.span - 15.0) ** 2.0 + 14.0 if self.span > 15.0 else 14.0  # empirical
        self.empty_weight = self.wing_weight.sum() * 2 + self.body_tail_weight
        self.weight = PILOT_WEIGHT + WATER_WEIGHT + self.empty_weight + self.payload
        self.v_inf = np.sqrt(2 * self.weight * GRAVITY / (RHO * self.wing_area * self.CL))
        lift = (
            0.5
            * RHO
            * (self.v_inf**2)
            * (np.diff(self.y0) * self.span * 0.5)
            * 0.5
            * (self.local_cl[1:] * self.chord[1:] + self.local_cl[:-1] * self.chord[:-1])
        )
        self.local_lift = np.hstack([lift, 0.0])  # [N]

    def stiffness(self) -> None:
        self.Ex_ply = np.zeros([self.n_div, self.max_plys])
        self.EI_ply = np.zeros([self.n_struc, self.max_plys])
        self.EA_ply = np.zeros([self.n_struc, self.max_plys])
        self.EI = np.zeros(self.n_struc)
        self.EA = np.zeros(self.n_struc)
        self.GIp = np.zeros(self.n_struc)
        nui = 1 / (1 - NU_L * NU_T)
        ETL = E_T * NU_L * nui + 2 * G_LT
        Exx0 = E_L * nui
        Eyy0 = E_T * nui
        Exy0 = E_T * NU_L * nui
        Ex0 = Exx0 - Exy0**2 / Eyy0
        for i, plies in enumerate(self.ply_wing):
            y0_start, y0_end = self.y_div[i], self.y_div[i + 1]
            # The plies around the whole pipe, which must be symmetric.
            sum_ply = np.zeros(self.n_struc)
            Exx, Eyy, Exy, Gxy = np.zeros(self.n_struc), np.zeros(self.n_struc), np.zeros(self.n_struc), np.zeros(self.n_struc)
            for ply in plies[plies[:, 1] >= 180]:
                axis = np.deg2rad(ply[0])
                inside = (self.y0 >= ply[2] * (y0_end - y0_start) + y0_start) & (self.y0 < ply[3] * (y0_end - y0_start) + y0_start)
                mask = ply[4] * np.where(inside, 1, 0)
                sum_ply += mask
                sin, cos = np.sin(axis), np.cos(axis)
                Exx += mask * (E_T * nui * sin**4 + 2 * ETL * (sin * cos) ** 2 + E_L * nui * cos**4)
                Eyy += mask * (E_L * nui * sin**4 + 2 * ETL * (sin * cos) ** 2 + E_T * nui * cos**4)
                Exy += mask * (E_T * NU_L * nui * (sin**4 + cos**4) + ((E_L + E_T) * nui - 4 * G_LT) * (sin * cos) ** 2)
                Gxy += mask * (0.25 * (E_L + E_T * (1 - 2 * NU_L)) * nui * np.sin(2 * axis) ** 2 + G_LT * np.cos(2 * axis) ** 2)
            _sum_ply = np.where(sum_ply > 0, sum_ply, np.inf)
            Exx, Eyy, Exy, Gxy = Exx / _sum_ply, Eyy / _sum_ply, Exy / _sum_ply, Gxy / _sum_ply
            Ex = Exx - Exy**2 / np.where(sum_ply > 0, Eyy, np.inf)
            R = np.where((self.y0 >= y0_start) & (self.y0 < y0_end), 0.5 * self.diameter, 0)
            Iy = np.pi * sum_ply * R**3
            self.EI += Ex * Iy
            self.EA += Ex * 2 * np.pi * R * sum_ply
            self.GIp += Gxy * Iy * 2
            # The flange plies, around part of the pipe.
            for j, ply in enumerate(plies[plies[:, 1] < 180]):
                axis = np.deg2rad(ply[0])
                phi = np.deg2rad(ply[1])
                if axis == 0:
                    Ex_j = Ex0
                else:
                    sin, cos = np.sin(axis), np.cos(axis)
                    Exx_j = E_T * nui * sin**4 + 2 * ETL * (sin * cos) ** 2 + E_L * nui * cos**4
                    Eyy_j = E_L * nui * sin**4 + 2 * ETL * (sin * cos) ** 2 + E_T * nui * cos**4
                    Exy_j = E_T * NU_L * nui * (sin**4 + cos**4) + ((E_L + E_T) * nui - 4 * G_LT) * (sin * cos) ** 2
                    Ex_j = Exx_j - Exy_j**2 / Eyy_j
                inside = (self.y0 >= ply[2] * (y0_end - y0_start) + y0_start) & (self.y0 < ply[3] * (y0_end - y0_start) + y0_start)
                R = np.where(inside, 0.5 * self.diameter, 0)
                Iy = (phi + np.sin(phi)) * ply[4] * R**3
                EA = Ex_j * 2 * phi * R * ply[4]
                self.EI += Ex_j * Iy
                self.EA += EA
                self.Ex_ply[i, j] = Ex_j
                self.EI_ply[:, j] += Ex_j * Iy
                self.EA_ply[:, j] += EA
        self.EI[-1] = self.EI[-2]
        self.EA[-1] = self.EA[-2]
        self.GIp[-1] = self.GIp[-2]

    def zerolift_deflection(self) -> None:
        """The wing on the ground, under its own weight."""
        self.moment_zerolift = moment(self.y0 * self.span * 0.5, -self.wing_weight * GRAVITY)
        self.M_EI_zerolift = self.moment_zerolift / self.EI
        self.deflection_angle_zerolift = _integrate(self.M_EI_zerolift, self.y0, self.span)
        self.deflection_zerolift = _integrate(self.deflection_angle_zerolift, self.y0, self.span)
        self.total_deflection_zerolift = self.dihedral_position + self.deflection_zerolift
        self.strain_zerolift = self.M_EI_zerolift * self.diameter * 0.5

    def flight_deflection(self) -> None:
        """The wing in flight, with the axial force of the flying wires iterated with the deflection."""
        i_wire = np.argmin(np.abs(self.y0 - self.y_wire))
        if self.wire_tension > 0:
            z_wire_deflection = 0.0
            for _ in range(WIRE_ITERATIONS):
                wire_angle = np.arctan((Z_WIRE + z_wire_deflection) / (self.y_wire * self.span * 0.5)) if self.y_wire > 0 else np.pi * 0.5
                dihedral_angle_at_wire = np.arctan(z_wire_deflection / (self.y_wire * self.span * 0.5))
                axial_force = self.wire_tension * np.cos(wire_angle) / np.cos(dihedral_angle_at_wire)
                wire_load = axial_force * np.sin(wire_angle - dihedral_angle_at_wire) / np.cos(wire_angle)
                self.wire_load = np.zeros(self.n_struc)
                self.wire_load[i_wire] = wire_load
                self.axial_force = np.where(self.y0 < self.y_wire, axial_force, 0)
                self.total_load = self.local_lift - self.wire_load - self.wing_weight * GRAVITY
                self.moment, _ = moment_with_axial_force(self.y0 * self.span * 0.5, self.total_load, self.axial_force, self.EI)
                self._deflect()
                z_wire_deflection = self.total_deflection[i_wire]
        else:
            self.wire_load = np.zeros(self.n_struc)
            self.axial_force = np.zeros(self.n_struc)
            self.total_load = self.local_lift - self.wing_weight * GRAVITY
            self.moment = moment(self.y0 * self.span * 0.5, self.total_load)
            self._deflect()
        self.strain = self.M_EI * self.diameter * 0.5 + self.axial_force / self.EA

    def _deflect(self) -> None:
        self.M_EI = self.moment / self.EI
        self.deflection_angle = _integrate(self.M_EI, self.y0, self.span)
        self.deflection = _integrate(self.deflection_angle, self.y0, self.span)
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
        points = np.vstack(
            [self.airfoil_aero, np.clip(re_aero, models.re_min, models.re_max), np.clip(self.aoa_aero, models.aoa_min, models.aoa_max)]
        ).T
        cd0_aero = models.profile_drag(points)
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
