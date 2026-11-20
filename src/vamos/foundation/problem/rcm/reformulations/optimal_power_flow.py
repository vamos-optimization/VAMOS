"""RCM40-RCM46 reformulated: the generator powers, decoded by a Newton power flow.

The 26 equalities of the optimal power flow problems are the power balance of buses 2-14 of the IEEE
14-bus system, whose unknowns are the voltages ``x1..x26`` (bus 1 is the slack, ``V1 = 1``). The genotype
is the eight controls ``x27..x34``, the active and reactive powers of the generators of buses 2, 3, 6 and
8, in ``[0, 1]``; the voltages solve the power flow by Newton's method in rectangular coordinates from a
flat start (``V = 1``), at most 30 iterations, until ``max |h| <= 1e-13``. A row is dropped when its
voltages exceed 50 in magnitude. The decoding fails when ``max |h| > 1e-10`` (failure measure ``1e3``) or
a voltage component leaves ``[-1, 1]`` (failure measure: the sum of the excesses ``|x_i| - 1``). The
flat-start solution is the operating, high-voltage one; its voltages leave ``[-1, 1]`` for most controls,
so only a small region of the controls decodes successfully, and the failure measure leads the search
there. The decoding is the same for the seven problems; only the objectives differ.
"""

from __future__ import annotations

import functools

import numpy as np

from vamos.foundation.problem.rcm._common import Array
from vamos.foundation.problem.rcm.power_system import _OPF_GENERATORS, _network
from vamos.foundation.problem.rcm.reformulations._base import NOT_CONVERGED, RCMReformulation, ordered_product_sum

H_TOLERANCE = 1e-10  # the largest |h| of a decoded design (the problems allow 1e-4)
NEWTON_TOLERANCE = 1e-13
MAX_ITERATIONS = 30
LOST = 50.0  # a row whose voltages exceed this magnitude has diverged


@functools.cache
def _admittance() -> tuple[Array, Array, Array, Array]:
    """Read-only: the admittance ``Y = G + jB`` of the network, the conjugate of its block of buses 2-14, and
    the active and reactive demands."""
    g, b, p, q = _network("OPF")
    Y = g + 1j * b
    arrays = (Y, np.conj(Y)[1:, 1:], p, q)
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _injections(U: Array) -> tuple[Array, Array]:
    """The active and reactive power generated at each bus (rows x 14) by the controls (rows x 8)."""
    p_generated = np.zeros((U.shape[0], 14))
    q_generated = np.zeros((U.shape[0], 14))
    p_generated[:, list(_OPF_GENERATORS)] = U[:, :4]
    q_generated[:, list(_OPF_GENERATORS)] = U[:, 4:]
    return p_generated, q_generated


def _mismatch(Z: Array, p_generated: Array, q_generated: Array) -> tuple[Array, Array, Array]:
    """The 26 power mismatches h, the bus voltages and the currents ``I = Y V``, summed bus by bus."""
    Y, _, p_load, q_load = _admittance()
    V = np.ones((Z.shape[0], 14), dtype=complex)
    V[:, 1:] = Z[:, 0:13] + 1j * Z[:, 13:26]
    # I = Y V, adding the buses' terms from the first to the last
    current = ordered_product_sum(V[:, None, :], Y[None, :, :])
    power = V * np.conj(current)
    delta_p = power.real - p_generated + p_load
    delta_q = power.imag - q_generated + q_load
    return np.concatenate([delta_p[:, 1:], delta_q[:, 1:]], axis=1), V, current


def _jacobian(V: Array, current: Array) -> Array:
    """``d h / d z`` with ``z = (Re V2..V14, Im V2..V14)``: ``dS/dVr = diag(conj I) + diag(V) conj(Y)``."""
    _, conj_block, _, _ = _admittance()
    diagonal = np.arange(13)
    A = V[:, 1:, None] * conj_block[None, :, :]
    conj_i = np.conj(current[:, 1:])
    d_real = A.copy()
    d_real[:, diagonal, diagonal] += conj_i
    d_imag = -1j * A
    d_imag[:, diagonal, diagonal] += 1j * conj_i
    J = np.empty((V.shape[0], 26, 26))
    J[:, :13, :13] = d_real.real
    J[:, :13, 13:] = d_imag.real
    J[:, 13:, :13] = d_real.imag
    J[:, 13:, 13:] = d_imag.imag
    return J


def _solve(J: Array, h: Array) -> Array:
    try:
        return np.asarray(np.linalg.solve(J, h[..., None])[..., 0])
    except np.linalg.LinAlgError:  # a singular Jacobian: least squares, row by row
        return np.array([np.linalg.lstsq(Jk, hk, rcond=None)[0] for Jk, hk in zip(J, h)])


def flat_start() -> Array:
    """The flat start of the power flow, ``V = 1`` at every bus: the real parts, then the imaginary parts."""
    return np.concatenate([np.ones(13), np.zeros(13)])


def power_flow(U: Array) -> tuple[Array, Array]:
    """The flat-start Newton power flow of the controls (rows x 8): the voltages (rows x 26) and max |h|."""
    n = U.shape[0]
    p_generated, q_generated = _injections(U)
    Z = np.tile(flat_start(), (n, 1))
    residual = np.full(n, np.inf)
    active = np.arange(n)
    for _ in range(MAX_ITERATIONS + 1):
        if active.size == 0:
            break
        # Every row still moving: a slice, which spares the copies of indexing by rows.
        rows: slice | Array = slice(None) if active.size == n else active
        h, V, current = _mismatch(Z[rows], p_generated[rows], q_generated[rows])
        r = np.abs(h).max(axis=1)
        residual[rows] = r
        going = r > NEWTON_TOLERANCE
        if not going.all():
            active, h, V, current = active[going], h[going], V[going], current[going]
            if active.size == 0:
                break
            rows = active
        Z[rows] -= _solve(_jacobian(V, current), h)
        moved = Z[rows]
        with np.errstate(invalid="ignore"):
            lost = ~np.isfinite(moved).all(axis=1) | (np.abs(moved).max(axis=1) > LOST)
        if lost.any():
            residual[active[lost]] = np.inf
            active = active[~lost]
    if active.size:  # not below NEWTON_TOLERANCE after MAX_ITERATIONS: the last residual
        residual[active] = np.abs(_mismatch(Z[active], p_generated[active], q_generated[active])[0]).max(axis=1)
    residual[~np.isfinite(Z).all(axis=1)] = np.inf
    return Z, residual


def decode_opf(U: Array) -> tuple[Array, Array]:
    """The controls (rows x 8) -> designs x1..x34 and the failure measure (positive where it failed)."""
    U = np.clip(U, 0.0, 1.0)
    with np.errstate(invalid="ignore", over="ignore"):
        Z, residual = power_flow(U)
    converged = residual <= H_TOLERANCE
    Z = np.where(converged[:, None], Z, flat_start())
    excess = np.maximum(np.abs(Z) - 1.0, 0.0).sum(axis=1)
    return np.concatenate([Z, U], axis=1), np.where(converged, excess, NOT_CONVERGED)


class OptimalPowerFlowReformulation(RCMReformulation):
    """RCM40-RCM46 searched through the eight generator powers, the voltages decoded by a flat-start power flow."""

    def __init__(self, number: int) -> None:
        if not 40 <= number <= 46:
            raise ValueError(f"No optimal power flow reformulation of RCM{number}; they are RCM40 to RCM46.")
        super().__init__(number, np.zeros(8), np.ones(8), 0)

    def _solve(self, U: Array) -> tuple[Array, Array]:
        return decode_opf(U)


__all__ = ["H_TOLERANCE", "OptimalPowerFlowReformulation", "decode_opf", "flat_start", "power_flow"]
