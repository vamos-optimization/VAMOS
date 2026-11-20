"""RCM36-RCM39 reformulated: the generator sizes, decoded by a Newton power flow.

The 24 equalities of the distributed generation sizing problems are the power-flow equations of buses
4-15 of a 15-node network, whose unknowns are the voltages ``x1..x24``. The genotype is the four sizes
``x25..x28`` of the generators of buses 7 and 15, in ``[0, 1]``; the voltages solve the power flow by
Newton's method from a flat start (phase voltages ``1, exp(j 4 pi / 3), exp(j 2 pi / 3)``) with the
analytical Jacobian, at most 40 iterations. A row stops at ``max |h| <= 1e-12``, or below ``1e-11`` once a
step no longer halves the mismatch, or when it diverges. The decoding fails when ``max |h| > 1e-10``
(failure measure ``1e3``) or a voltage component leaves ``[-1, 1]`` (failure measure: the largest
``|x_i|`` minus 1). The decoding is the same for the four problems; only the objectives differ.
"""

from __future__ import annotations

import functools

import numpy as np

from vamos.foundation.problem.rcm._common import Array
from vamos.foundation.problem.rcm.power_system import _network
from vamos.foundation.problem.rcm.reformulations._base import NOT_CONVERGED, RCMReformulation, ordered_product_sum

H_TOLERANCE = 1e-10  # the largest |h| of a decoded design (the problems allow 1e-4)
NEWTON_TOLERANCE = 1e-12
MAX_ITERATIONS = 40
# Load-bus indices (0-based, among buses 4-15) of the generators of buses 7 and 15.
_GENERATORS = [3, 11]


@functools.cache
def _network_data() -> tuple[Array, Array, Array, Array, Array]:
    """Read-only: the slack voltages ``1, exp(j 4 pi / 3), exp(j 2 pi / 3)``, the rows of ``Y = G + jB`` of
    buses 4-15, the conjugate of their block of buses 4-15, and the active and reactive loads."""
    g, b, p, q = _network("DG")
    Y = g + 1j * b
    arrays = (np.array([1.0, np.exp(1j * 4 * np.pi / 3), np.exp(1j * 2 * np.pi / 3)]), Y[3:15], np.conj(Y[3:15, 3:15]), p, q)
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _currents(V: Array) -> Array:
    """The currents into buses 4-15, ``I = Y V`` with the slack voltages, adding the buses' terms in order."""
    slack, rows, _, _, _ = _network_data()
    full = np.concatenate([np.broadcast_to(slack, (V.shape[0], 3)), V], axis=1)
    return ordered_product_sum(full[:, None, :], rows[None, :, :])


def _mismatch(V: Array, P: Array, Q: Array) -> tuple[Array, Array]:
    """The 24 power mismatches h at the load-bus voltages, and the currents."""
    current = _currents(V)
    power = V * np.conj(current)
    return np.concatenate([power.real - P, power.imag - Q], axis=1), current


def _newton_step(V: Array, P: Array, Q: Array) -> tuple[Array, Array]:
    """The mismatch h (rows x 24) and the Newton step (rows x 12, complex) at the load-bus voltages V."""
    _, _, conj_block, _, _ = _network_data()
    h, current = _mismatch(V, P, Q)
    # dS_i/dRe(V_k) = d_ik conj(I_i) + V_i conj(Y_ik); dS_i/dIm(V_k) = j (d_ik conj(I_i) - V_i conj(Y_ik))
    diagonal = np.arange(12)
    A = V[:, :, None] * conj_block[None, :, :]
    d_real = A.copy()
    d_real[:, diagonal, diagonal] += np.conj(current)
    d_imag = -1j * A
    d_imag[:, diagonal, diagonal] += 1j * np.conj(current)
    J = np.concatenate([np.concatenate([d_real.real, d_imag.real], axis=2), np.concatenate([d_real.imag, d_imag.imag], axis=2)], axis=1)
    try:
        step = np.linalg.solve(J, h[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:  # a singular Jacobian: solve the rows one by one
        step = np.full(h.shape, np.nan)
        for row in range(h.shape[0]):
            try:
                step[row] = np.linalg.solve(J[row], h[row])
            except np.linalg.LinAlgError:
                step[row] = np.nan  # a singular row stops: its step is not finite
    return h, step[:, :12] + 1j * step[:, 12:]


def decode_dg(U: Array) -> tuple[Array, Array]:
    """The generator sizes (rows x 4) -> designs x1..x28 and the failure measure (positive where it failed)."""
    U = np.clip(U, 0.0, 1.0)
    n = U.shape[0]
    slack, _, _, p_load, q_load = _network_data()
    P = np.tile(p_load[3:15], (n, 1))
    Q = np.tile(q_load[3:15], (n, 1))
    P[:, _GENERATORS] += U[:, 0:2]  # delP = Psp - Pdg - P = 0
    Q[:, _GENERATORS] += U[:, 2:4]
    flat = np.tile(np.concatenate([slack] * 4), (n, 1))
    V = flat.copy()
    active = np.arange(n)
    previous = np.full(n, np.inf)
    for _ in range(MAX_ITERATIONS):
        if active.size == 0:
            break
        h, step = _newton_step(V[active], P[active], Q[active])
        with np.errstate(invalid="ignore"):
            error = np.abs(h).max(axis=1)
            diverged = ~np.isfinite(error) | ~np.isfinite(step).all(axis=1) | (np.abs(V[active]).max(axis=1) > 10.0)
            converged = (error <= NEWTON_TOLERANCE) | ((error <= 1e-11) & (error >= 0.5 * previous[active]))
        stop = diverged | converged
        going = active[~stop]
        V[going] -= step[~stop]
        previous[active] = error
        active = going
    finite = np.isfinite(V).all(axis=1)
    V[~finite] = flat[~finite]
    # A row that stopped keeps the voltages of its last mismatch; the rows still moving get theirs now.
    residual = previous
    if active.size:
        with np.errstate(invalid="ignore", over="ignore"):
            residual[active] = np.abs(_mismatch(V[active], P[active], Q[active])[0]).max(axis=1)
    X = np.concatenate([V.real, V.imag, U], axis=1)
    converged = finite & (residual <= H_TOLERANCE)
    return X, np.where(converged, np.abs(X[:, :24]).max(axis=1) - 1.0, NOT_CONVERGED)


class DistributedGenerationReformulation(RCMReformulation):
    """RCM36-RCM39 searched through the four generator sizes, the voltages decoded by a flat-start power flow."""

    def __init__(self, number: int) -> None:
        if not 36 <= number <= 39:
            raise ValueError(f"No distributed generation reformulation of RCM{number}; they are RCM36 to RCM39.")
        super().__init__(number, np.zeros(4), np.ones(4), 0)

    def _solve(self, U: Array) -> tuple[Array, Array]:
        return decode_dg(U)


__all__ = ["DistributedGenerationReformulation", "H_TOLERANCE", "decode_dg"]
