"""RCM47-RCM49 reformulated: seven controls, decoded by a damped Newton solve of the microgrid's power flow.

The 12 equalities of the islanded microgrid problems are the real and imaginary current mismatches of its
six buses. Given the frequency ``w``, the reactive droop gains ``Qc`` and the direction ``a`` of the active
droop gains ``(Pc4, Pc5, Pc6) = K a``, they are a square system in the 11 voltage components (``x1..x10``
and ``x12``) and the scale ``K``, which absorbs the active power balance. The genotype is ``t`` in
``[0, 1]^7``: ``w = t1``, ``a = (t2, t3, t4)`` and ``Qc = 500 t^3`` for ``t5..t7`` (the cube resolves the
small gains); ``Pc = 0`` and ``Qc = 0`` are reachable exactly.

The solve is a damped Newton method with a central-difference Jacobian, at most 40 iterations, each
halving its step up to 12 times until the norm of the mismatch decreases. It starts from the flat start:
voltages 1, ``x12 = 1 - 1e-5`` (so ``V1 = 1``) and ``K`` from the load plus 3 % losses; a row with
``w = 1`` or ``a = 0`` is not solved. A row stops when ``||h||_2 < 1e-14`` or when no step decreases it.
The decoding fails when ``max |h| > 1e-10`` (failure measure ``1e3 + min(log10(1 + max |h|), 100)``) or a
variable leaves its bounds (failure measure: the sum of the excesses, each divided by the variable's
range). The decoding is the same for the three problems; only the objectives differ.
"""

from __future__ import annotations

import numpy as np

from vamos.foundation.problem.rcm import RCM
from vamos.foundation.problem.rcm._common import Array
from vamos.foundation.problem.rcm.power_system import _MICROGRID_LINES, _MICROGRID_P, _MICROGRID_Q
from vamos.foundation.problem.rcm.reformulations._base import NOT_CONVERGED, RCMReformulation

H_TOLERANCE = 1e-10  # the largest |h| of a decoded design (the problems allow 1e-4)
NEWTON_TOLERANCE = 1e-14  # a row stops when ||h||_2 falls below this
MAX_ITERATIONS = 40
LINE_SEARCH_STEPS = 12
QC_MAX = 500.0


def _admittance(w: Array) -> Array:
    """The bus admittance matrices (rows x 6 x 6) at the frequencies ``w``: the line reactances scale with ``w``."""
    Y = np.zeros((w.shape[0], 6, 6), dtype=complex)
    for start, end, resistance, reactance in _MICROGRID_LINES:
        y = 1.0 / (resistance + 1j * (reactance * w))
        Y[:, start, end] = Y[:, start, end] - y
        Y[:, end, start] = Y[:, start, end]
    for bus in range(6):
        for start, end, resistance, reactance in _MICROGRID_LINES:
            if bus in (start, end):
                Y[:, bus, bus] = Y[:, bus, bus] + 1.0 / (resistance + 1j * (reactance * w))
    return Y


def _assemble(Z: Array, w: Array, a: Array, qc: Array) -> Array:
    """The design x1..x18 from the unknowns ``Z = (x1..x10, x12, K)`` and the controls."""
    X = np.empty((Z.shape[0], 18))
    X[:, 0:10] = Z[:, 0:10]
    X[:, 10] = w
    X[:, 11] = Z[:, 10]
    X[:, 12:15] = a * Z[:, 11:12]
    X[:, 15:18] = qc
    return X


def residual(X: Array) -> Array:
    """The 12 current mismatches h of the designs (rows x 18): real parts of buses 1-6, then imaginary parts."""
    V = np.zeros((X.shape[0], 6), dtype=complex)
    V[:, 1:6] = X[:, 0:5] + 1j * X[:, 5:10]
    w = X[:, 10]
    V[:, 0] = X[:, 11] + 1e-5
    Pc = np.zeros((X.shape[0], 6))
    Qc = np.zeros((X.shape[0], 6))
    Pc[:, 3:6] = X[:, 12:15]
    Qc[:, 3:6] = X[:, 15:18]
    Y = _admittance(w)
    current = np.zeros(V.shape, dtype=complex)
    for k in range(6):
        current = current + Y[:, :, k] * V[:, k : k + 1]
    # The droop exponents of the reference data make the voltage factors of the loads 1.
    p_specified = Pc * (1 - w)[:, None] - np.asarray(_MICROGRID_P)
    q_specified = Qc * (1 - np.sqrt(V.real**2 + V.imag**2)) - np.asarray(_MICROGRID_Q)
    specified = np.conj((p_specified + 1j * q_specified) / V)
    return np.concatenate([(current.real - specified.real), (current.imag - specified.imag)], axis=1)


def _residual(Z: Array, w: Array, a: Array, qc: Array) -> Array:
    with np.errstate(all="ignore"):
        return residual(_assemble(Z, w, a, qc))


def _jacobian(Z: Array, w: Array, a: Array, qc: Array) -> Array:
    """The central-difference Jacobian ``dh/dZ`` (rows x 12 x 12)."""
    J = np.empty((Z.shape[0], 12, 12))
    for j in range(12):
        step = 1e-7 * np.maximum(1.0, np.abs(Z[:, j]))
        plus = Z.copy()
        plus[:, j] += step
        minus = Z.copy()
        minus[:, j] -= step
        J[:, :, j] = (_residual(plus, w, a, qc) - _residual(minus, w, a, qc)) / (2 * step[:, None])
    return J


def _solve(J: Array, rhs: Array) -> Array:
    """The batched ``J d = rhs``; NaN for the rows that are singular or not finite."""
    d = np.full(rhs.shape, np.nan)
    ok = np.isfinite(J).all(axis=(1, 2)) & np.isfinite(rhs).all(axis=1)
    rows = np.flatnonzero(ok)
    if rows.size:
        try:
            d[rows] = np.linalg.solve(J[rows], rhs[rows][:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            for row in rows:
                try:
                    d[row] = np.linalg.solve(J[row], rhs[row])
                except np.linalg.LinAlgError:
                    d[row] = np.nan  # a singular row stops: its step is not finite
    return d


def _newton(Z: Array, w: Array, a: Array, qc: Array, active: Array) -> Array:
    """The damped Newton solve of the rows ``active`` of Z, in place."""
    norm = np.linalg.norm(_residual(Z, w, a, qc), axis=1)
    norm[~np.isfinite(norm)] = np.inf
    active = np.asarray(active & np.isfinite(norm), dtype=bool)
    for _ in range(MAX_ITERATIONS):
        active &= norm >= NEWTON_TOLERANCE
        rows = np.flatnonzero(active)
        if rows.size == 0:
            break
        Zr, wr, ar, qr = Z[rows], w[rows], a[rows], qc[rows]
        d = _solve(_jacobian(Zr, wr, ar, qr), -_residual(Zr, wr, ar, qr))
        dead = ~np.isfinite(d).all(axis=1)
        d[dead] = 0.0
        best = norm[rows].copy()
        updated = Zr.copy()
        moved = np.zeros(rows.size, dtype=bool)
        pending = np.asarray(~dead, dtype=bool)
        t = 1.0
        for _ in range(LINE_SEARCH_STEPS):
            if not pending.any():
                break
            p = np.flatnonzero(pending)
            trial = Zr[p] + t * d[p]
            trial_norm = np.linalg.norm(_residual(trial, wr[p], ar[p], qr[p]), axis=1)
            trial_norm[~np.isfinite(trial_norm)] = np.inf
            better = trial_norm < best[p]
            updated[p[better]] = trial[better]
            best[p[better]] = trial_norm[better]
            moved[p[better]] = True
            pending[p[better]] = False
            t *= 0.5
        Z[rows] = updated
        norm[rows] = best
        active[rows[~moved]] = False
    return Z


def decode_microgrid(T: Array) -> tuple[Array, Array]:
    """The controls (rows x 7) -> designs x1..x18 and the failure measure (positive where it failed)."""
    T = np.clip(T, 0.0, 1.0)
    w, a, qc = T[:, 0], T[:, 1:4], QC_MAX * T[:, 4:7] ** 3
    Z = np.zeros((T.shape[0], 12))
    Z[:, 0:5] = 1.0
    Z[:, 10] = 1.0 - 1e-5
    with np.errstate(all="ignore"):
        Z[:, 11] = np.sum(np.asarray(_MICROGRID_P)) * 1.03 / ((1.0 - w) * a.sum(axis=1))
    unsolvable = ~np.isfinite(Z[:, 11]) | (a.sum(axis=1) <= 0) | (w >= 1.0)
    Z[unsolvable, 11] = 1.0
    Z = _newton(Z, w, a, qc, ~unsolvable)
    X = _assemble(Z, w, a, qc)
    lower, upper = np.asarray(RCM["rcm47"][1], dtype=float), np.asarray(RCM["rcm47"][2], dtype=float)
    with np.errstate(all="ignore"):
        largest = np.abs(residual(X)).max(axis=1)
        excess = ((np.maximum(lower - X, 0.0) + np.maximum(X - upper, 0.0)) / (upper - lower)).sum(axis=1)
    largest[~np.isfinite(largest)] = np.inf
    converged = largest <= H_TOLERANCE
    excess[~np.isfinite(excess)] = NOT_CONVERGED
    measure = np.where(converged, excess, NOT_CONVERGED + np.minimum(np.log10(1.0 + largest), 100.0))
    return X, measure


class MicrogridReformulation(RCMReformulation):
    """RCM47-RCM49 searched through seven controls, the rest decoded by a damped Newton solve from a flat start."""

    def __init__(self, number: int) -> None:
        if not 47 <= number <= 49:
            raise ValueError(f"No microgrid reformulation of RCM{number}; they are RCM47 to RCM49.")
        super().__init__(number, np.zeros(7), np.ones(7), 0)

    def _solve(self, U: Array) -> tuple[Array, Array]:
        return decode_microgrid(U)


__all__ = ["H_TOLERANCE", "MicrogridReformulation", "decode_microgrid", "residual"]
