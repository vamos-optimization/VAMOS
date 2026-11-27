"""The kernels of the analysis of the wing (:mod:`.wing`), under the rules of :mod:`._kernels`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._kernel_helpers import block_sum, count_at_most, count_below, first_at_least

Array = np.ndarray[Any, Any]


def integrate(values: Array, y0: Array, span: float, out: Array, count: int) -> None:
    """``out[:count]`` = that of the cumulative trapezoidal integral of :func:`.wing._integrate`, summed as
    NumPy's ``cumsum``."""
    out[0] = 0.0
    for k in range(1, count):
        out[k] = out[k - 1] + 0.5 * (values[k] + values[k - 1]) * (y0[k] - y0[k - 1]) * span * 0.5


def moment(F: Array, steps: Array, out: Array) -> None:
    """``out`` = :func:`.wing.moment` of the loads ``F``; ``steps`` is ``abs(diff(y))``, which its reversal negates."""
    n = F.size
    out[n - 1] = 0.0
    load = 0.0
    total = 0.0
    for i in range(n - 1):  # the reversed arrays of wing.moment: F[::-1][i] is F[n - 1 - i]
        load = F[n - 1] if i == 0 else load + F[n - 1 - i]
        term = (load + 0.5 * F[n - 2 - i]) * steps[n - 2 - i]
        total = term if i == 0 else total + term
        out[n - 2 - i] = total


def beam_column(L: Array, H: Array, F: Array, T: Array, EI: Array, M: Array, S: Array) -> None:
    """Fill ``M`` and ``S`` with the beam-column recurrence of :func:`.wing.moment_with_axial_force`.

    ``H[i]`` is ``(0.5 * L[i]) ** 2``, computed by the caller as the reference code computes it.
    """
    n = L.size
    M[n] = 0.0
    S[n] = 0.0
    for i in range(n - 1, -1, -1):
        TL_EI = T[i] * H[i] / EI[i]
        S[i] = (F[i] + (1.0 - TL_EI) * S[i + 1] - T[i] * L[i] * M[i + 1] / EI[i]) / (1.0 + TL_EI)
        M[i] = (0.5 * L[i] * F[i] + (1.0 - TL_EI) * M[i + 1] + L[i] * S[i + 1]) / (1.0 + TL_EI)


def wing_weight(
    y0: Array,
    half_diameter: Array,
    segment: Array,
    y_div: Array,
    plies: Array,
    phi: Array,
    span: float,
    density: float,
    coef: float,
    joints: Array,
    joint_weights: Array,
    ribs_and_skin: Array,
    i_wire: int,
    wire_weight: float,
    beam_weight: Array,
    weight: Array,
) -> None:
    """``beam_weight`` and ``weight`` (zeros) = those of :meth:`.wing.Wing.compute_wing_weight`.

    ``plies`` has one row per ply, ``[axis, half width angle, root, tip, thickness]``, of segment ``segment``;
    ``phi`` is the half width angle in radians. Then come the joints (grid index and weight), the ribs and skin of
    each grid interval (and 0 at the tip), and the wire.
    """
    n = y0.size
    for p in range(plies.shape[0]):
        i = segment[p]
        start, end = y_div[i], y_div[i + 1]
        root, tip, thickness = plies[p, 2], plies[p, 3], plies[p, 4]
        low = root * (end - start) + start
        high = tip * (end - start) + start
        largest = 0.0
        for k in range(first_at_least(y0, low), count_at_most(y0, high)):
            r = half_diameter[k]
            if r > largest:
                largest = r
            length = y0[k + 1] - y0[k] if k < n - 1 else 0.0
            weight[k] += density * 2.0 * phi[p] * r * thickness * 0.5 * span * length * coef
        beam_weight[i] += density * 2.0 * phi[p] * largest * thickness * 0.5 * span * (tip - root) * (end - start)
    for q in range(joints.size):
        weight[joints[q]] += joint_weights[q]
    for k in range(n):
        weight[k] += ribs_and_skin[k]
    weight[i_wire] += wire_weight


def local_lift(scale: float, widths: Array, lift_chords: Array, out: Array) -> None:
    """``out`` = the lift of :meth:`.wing.Wing.compute_local_lift`; ``scale`` is ``0.5 * RHO * v_inf ** 2``."""
    n = widths.size
    for k in range(n):
        out[k] = scale * widths[k] * 0.5 * lift_chords[k]
    out[n] = 0.0


def zerolift(
    weight: Array,
    gravity: float,
    steps: Array,
    EI: Array,
    y0: Array,
    span: float,
    dihedral: Array,
    diameter: Array,
    moment_out: Array,
    M_EI: Array,
    angle: Array,
    deflection: Array,
    total: Array,
    strain: Array,
) -> None:
    """The arrays of :meth:`.wing.Wing.zerolift_deflection`, the wing on the ground under its own ``weight``."""
    n = weight.size
    F = np.empty(n)
    for k in range(n):
        F[k] = -weight[k] * gravity
    moment(F, steps, moment_out)
    for k in range(n):
        M_EI[k] = moment_out[k] / EI[k]
    integrate(M_EI, y0, span, angle, n)
    integrate(angle, y0, span, deflection, n)
    for k in range(n):
        total[k] = dihedral[k] + deflection[k]
        strain[k] = M_EI[k] * diameter[k] * 0.5


def flight(
    wire: bool,
    last: bool,
    lift: Array,
    weight: Array,
    gravity: float,
    i_wire: int,
    wire_load: float,
    below_wire: Array,
    axial_force: float,
    geometry: Array,
    EI: Array,
    EA: Array,
    span: float,
    out: Array,
) -> float:
    """One deflection of :meth:`.wing.Wing.flight_deflection`: the deflection at the wire.

    With the flying wire, the axial force ``axial_force`` acts inboard of it and its load ``wire_load`` at
    ``i_wire``; without (``wire`` false), the moment is :func:`.wing.moment`'s. The rows of ``geometry`` are the
    steps of the spar and the squares of their halves (and a 0), ``y0``, the dihedral position and the diameter;
    those of ``out`` are the axial force, the load, the moment, the shear force, ``M / EI``, the deflection angle,
    the deflection, the total deflection and the strain. Before the ``last`` deflection, the deflections are
    integrated up to the wire, and the strain is not computed.
    """
    n = lift.size
    steps, half_steps_squared, y0 = geometry[0, : n - 1], geometry[1, : n - 1], geometry[2]
    dihedral, diameter = geometry[3], geometry[4]
    T, load, moment_out, shear, M_EI = out[0], out[1], out[2], out[3], out[4]
    angle, deflection, total, strain = out[5], out[6], out[7], out[8]
    for k in range(n):
        if wire:
            T[k] = axial_force if below_wire[k] else 0.0
            load[k] = lift[k] - (wire_load if k == i_wire else 0.0) - weight[k] * gravity
        else:
            T[k] = 0.0
            load[k] = lift[k] - weight[k] * gravity
    if wire:
        beam_column(steps, half_steps_squared, load, T, EI, moment_out, shear)
    else:
        moment(load, steps, moment_out)
    count = n if last else i_wire + 1
    for k in range(count):
        M_EI[k] = moment_out[k] / EI[k]
    integrate(M_EI, y0, span, angle, count)
    integrate(angle, y0, span, deflection, count)
    for k in range(count):
        total[k] = dihedral[k] + deflection[k]
    if last:
        for k in range(n):
            strain[k] = M_EI[k] * diameter[k] * 0.5 + T[k] / EA[k]
    return float(total[i_wire])


def lifting_line(mu: Array, ms: Array, sin_thetas: Array, sin_ms_thetas: Array, A: Array) -> None:
    """``A`` = the matrix of :meth:`.wing.Wing.lifting_line`, ``(ms * mus + sin(thetas)) * sin(ms * thetas)``."""
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            A[r, c] = (ms[r, c] * mu[r] + sin_thetas[r, c]) * sin_ms_thetas[r, c]


def induced_angles(xa: Array, ms: Array, sin_ms_thetas: Array, sin_thetas: Array, out: Array) -> None:
    """``out[:-1]`` = the induced angles of :meth:`.wing.Wing.lifting_line`, each row summed as NumPy sums it."""
    n = xa.size
    row = np.empty(n)
    for r in range(n):
        for c in range(n):
            row[c] = ms[r, c] * xa[c] * sin_ms_thetas[r, c] / sin_thetas[r, c]
        out[r] = block_sum(row, n)


def entire_plies(
    y0: Array,
    half_diameter: Array,
    cubed_half_diameter: Array,
    y_div: Array,
    segment: Array,
    plies: Array,
    moduli: Array,
    pi: float,
    entire_EI: Array,
    entire_EA: Array,
    GIp: Array,
) -> None:
    """What the plies around the whole pipe add to EI, EA (``entire_EI[i]`` for segment ``i``) and ``GIp`` (zeros).

    ``plies`` are those rows of the plies of segment ``segment`` (:meth:`.wing.Wing._entire_plies`), and ``moduli``
    their ``Exx``, ``Eyy``, ``Exy`` and ``Gxy`` factors, computed by NumPy.
    """
    n = y0.size
    low = np.empty(plies.shape[0])
    high = np.empty(plies.shape[0])
    for i in range(entire_EI.shape[0]):
        start, end = y_div[i], y_div[i + 1]
        first, last = plies.shape[0], 0  # the rows of segment i, which are consecutive
        for q in range(plies.shape[0]):
            if segment[q] == i:
                first, last = min(first, q), q + 1
                low[q] = plies[q, 2] * (end - start) + start
                high[q] = plies[q, 3] * (end - start) + start
        for k in range(n):
            sum_ply, Exx, Eyy, Exy, Gxy = 0.0, 0.0, 0.0, 0.0, 0.0
            for q in range(first, last):
                mask = plies[q, 4] * (1.0 if y0[k] >= low[q] and y0[k] < high[q] else 0.0)
                sum_ply += mask
                Exx += mask * moduli[q, 0]
                Eyy += mask * moduli[q, 1]
                Exy += mask * moduli[q, 2]
                Gxy += mask * moduli[q, 3]
            divisor = sum_ply if sum_ply > 0 else np.inf
            Exx, Eyy, Exy, Gxy = Exx / divisor, Eyy / divisor, Exy / divisor, Gxy / divisor
            Ex = Exx - Exy * Exy / (Eyy if sum_ply > 0 else np.inf)  # NumPy squares an array by multiplying
            pipe = y0[k] >= start and y0[k] < end
            R = half_diameter[k] if pipe else 0.0
            Iy = pi * sum_ply * (cubed_half_diameter[k] if pipe else 0.0)
            entire_EI[i, k] = Ex * Iy
            entire_EA[i, k] = Ex * 2.0 * pi * R * sum_ply
            GIp[k] += Gxy * Iy * 2.0
    GIp[n - 1] = GIp[n - 2]


def stiffness(
    y0: Array,
    half_diameter: Array,
    cubed_half_diameter: Array,
    entire_EI: Array,
    entire_EA: Array,
    segment: Array,
    flange: Array,
    low: Array,
    high: Array,
    Ex: Array,
    inertia: Array,
    area: Array,
    thickness: Array,
    EI: Array,
    EA: Array,
    EI_ply: Array,
    EA_ply: Array,
) -> None:
    """``EI``, ``EA``, ``EI_ply`` and ``EA_ply`` (zeros) = those of :meth:`.wing.Wing.stiffness`.

    Segment by segment, the stiffness ``entire_EI[i]`` and ``entire_EA[i]`` of its plies around the whole pipe, then
    of its flanges between ``low`` and ``high``: ``inertia`` is ``(phi + sin(phi)) * thickness`` and ``area`` is
    ``Ex * 2 * phi``, as the reference code multiplies them; ``cubed_half_diameter`` is ``half_diameter ** 3``
    computed by NumPy.
    """
    n = y0.size
    p = 0
    for i in range(entire_EI.shape[0]):
        for k in range(n):
            EI[k] += entire_EI[i, k]
            EA[k] += entire_EA[i, k]
        while p < segment.size and segment[p] == i:
            j = flange[p]
            for k in range(first_at_least(y0, low[p]), count_below(y0, high[p])):
                Iy = inertia[p] * cubed_half_diameter[k]
                EA_k = area[p] * half_diameter[k] * thickness[p]
                EI[k] += Ex[p] * Iy
                EA[k] += EA_k
                EI_ply[k, j] += Ex[p] * Iy
                EA_ply[k, j] += EA_k
            p += 1
    EI[n - 1] = EI[n - 2]
    EA[n - 1] = EA[n - 2]


def grid_interval(grid: Array, value: float) -> int:
    """The interval ``i`` of an ascending grid with ``grid[i] <= value < grid[i + 1]``, the last one for its end.

    This is SciPy's ``find_interval_ascending`` for a value in the grid's range, as ``RegularGridInterpolator``
    finds it.
    """
    m = grid.size
    if value == grid[m - 1]:
        return m - 2
    low, high = 0, m - 2
    while low < high:
        mid = (low + high + 1) // 2
        if grid[mid] <= value:
            low = mid
        else:
            high = mid - 1
    return low


def profile_drag(airfoil: Array, reynolds: Array, angle: Array, grids: Array, sizes: Array, table: Array, out: Array) -> int:
    """``out`` = the linear interpolation of ``RegularGridInterpolator((index, REYNOLDS, ANGLE_OF_ATTACK), table)``.

    ``grids[d, :sizes[d]]`` is the grid of dimension ``d``. As SciPy does, each corner of a point's cell adds, in
    the order of ``itertools.product``, its value times the product of its weights, ``1 - t`` at the lower end of
    a dimension and ``t`` at the upper one. It returns the first dimension out of its grid, or -1 when none is.
    """
    n = airfoil.size
    for d in range(3):
        values = airfoil if d == 0 else reynolds if d == 1 else angle
        for j in range(n):
            if not (grids[d, 0] <= values[j] and values[j] <= grids[d, sizes[d] - 1]):
                return d
    index = np.empty(3, np.int64)
    c = np.empty(3, np.int64)
    t = np.empty(3)
    for j in range(n):
        for d in range(3):
            value = airfoil[j] if d == 0 else reynolds[j] if d == 1 else angle[j]
            i = grid_interval(grids[d, : sizes[d]], value)
            index[d] = i
            t[d] = (value - grids[d, i]) / (grids[d, i + 1] - grids[d, i])
        total = 0.0
        for corner in range(8):
            weight = 1.0
            for d in range(3):
                upper = (corner >> (2 - d)) & 1
                weight = weight * (t[d] if upper else 1.0 - t[d])
                c[d] = index[d] + upper
            total = total + table[c[0], c[1], c[2]] * weight
        out[j] = total
    return -1
