"""RCM36-RCM50: power system problems of the CEC 2021 suite, as jMetal's classes define them.

- **RCM36-RCM39**, optimal sizing of single-phase distributed generation (``AbstractRCMDistributedGenerationSizing``):
  the real and imaginary voltages of buses 4-15 of a 15-bus unbalanced network, then the active and reactive
  sizes of the generators at buses 7 and 15. 24 equalities: the power mismatch at buses 4-15.
- **RCM40-RCM46**, optimal power flow (``AbstractRCMOptimalPowerFlow``): the voltages of buses 2-14 of a 14-bus
  network, then the active and reactive generation at buses 2, 3, 6 and 8. 26 equalities: the power mismatch at
  buses 2-14. f2 of RCM40 and RCM41 adds ``imag(V1 conj(I2))``, as the reference code does.
- **RCM47-RCM49**, droop control of an islanded microgrid (``AbstractRCMIslandedMicrogridDroopControl``): the
  voltages of buses 2-6, the frequency, the voltage of bus 1 and the droop gains of buses 4-6. 12 equalities:
  the current mismatch at the six buses.
- **RCM50**, power distribution system planning: six generator outputs, with one equality, the power balance.

jMetal relaxes every equality ``h = 0`` to ``|h| <= 1e-4``. The network sums run in jMetal's order.
"""

from __future__ import annotations

import functools

import numpy as np

from vamos.foundation.problem.rcm import _networks
from vamos.foundation.problem.rcm._common import Array, Entry, relaxed_equality

# Generator buses (0-based) of the optimal power flow, and the fuel cost b p + c p^2 of the slack bus and of
# those generators (jMetal's constant terms are 0).
_OPF_GENERATORS = (1, 2, 5, 7)
_OPF_FUEL = ((0, 2.0, 0.02), (1, 1.75, 0.0175), (2, 1.0, 0.0625), (5, 3.25, 0.00834), (7, 3.0, 0.025))

# Microgrid lines: from bus, to bus, resistance, reactance (scaled by the frequency); and the nominal loads.
_MICROGRID_LINES = (
    (0, 1, 0.2666005332010664, 0.07432946865893732),
    (0, 3, 0.186000372000744, 0.08180916361832725),
    (1, 4, 0.124000248000496, 0.05843511687023374),
    (1, 2, 0.093000186000372, 0.43078368156736313),
    (2, 5, 0.031000062000124, 0.01168702337404675),
)
_MICROGRID_P = (0.16138989835002382, 0.0, 0.21451873678268588, 0.0, 0.0, 0.0)
_MICROGRID_Q = (0.10680528035555388, 0.0, 0.15161777012574437, 0.0, 0.0, 0.0)

# RCM50: loss coefficients (times 1e-6), and the cost a + b p + c p^2 and emission alpha + beta p + gamma p^2
# of each generator.
_RCM50_LOSS = (
    (140, 17, 15, 19, 26, 22),
    (17, 60, 13, 16, 15, 20),
    (15, 13, 65, 17, 24, 19),
    (19, 16, 17, 71, 30, 25),
    (26, 15, 24, 30, 69, 32),
    (22, 20, 19, 25, 32, 85),
)
_RCM50_COST = (
    (756.7988, 38.539, 0.15247),
    (451.3251, 46.1591, 0.10587),
    (1243.5311, 38.3055, 0.03546),
    (1049.9977, 40.3965, 0.02803),
    (1356.6592, 38.2704, 0.01799),
    (1658.5696, 36.3278, 0.02111),
)
_RCM50_EMISSION = (
    (13.8593, 0.32767, 0.00419),
    (13.8593, 0.32767, 0.00419),
    (40.2669, -0.54551, 0.00683),
    (40.2669, -0.54551, 0.00683),
    (42.8955, -0.51116, 0.00461),
    (42.8955, -0.51116, 0.00461),
)


@functools.cache
def _network(prefix: str) -> tuple[Array, Array, Array, Array]:
    """The read-only conductance, susceptance, active and reactive demand of the ``DG`` or ``OPF`` network."""

    def parse(name: str) -> Array:
        values = np.array(str(getattr(_networks, f"{prefix}_{name}")).split(), dtype=float)
        values.setflags(write=False)
        return values

    p = parse("P")
    n = p.size
    return parse("G").reshape(n, n), parse("B").reshape(n, n), p, parse("Q")


def _total(rows: Array) -> Array:
    """The sum of the rows, first to last, as jMetal's loops add them."""
    total = np.zeros(rows.shape[1:])
    for row in rows:
        total = total + row
    return total


def _currents(g: Array, b: Array, vr: Array, vi: Array) -> tuple[Array, Array]:
    """``I = Y V`` with ``Y = g + jb`` of shape (n, n, 1) or (n, n, N) and ``V = vr + j vi`` of shape (n, N)."""
    ir = np.zeros(vr.shape)
    ii = np.zeros(vr.shape)
    for k in range(vr.shape[0]):
        ir = ir + (g[:, k] * vr[k] - b[:, k] * vi[k])
        ii = ii + (g[:, k] * vi[k] + b[:, k] * vr[k])
    return ir, ii


def _dg_flow(x: Array) -> tuple[Array, Array, Array, list[Array]]:
    """Active and reactive power of each bus, phase balance and constraints of RCM36-RCM39."""
    g, b, p, q = _network("DG")
    vr = np.zeros((15, x.shape[1]))
    vi = np.zeros((15, x.shape[1]))
    vr[0] = 1.0
    vr[1], vi[1] = np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)
    vr[2], vi[2] = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    vr[3:], vi[3:] = x[0:12], x[12:24]
    pdg = np.zeros(vr.shape)
    qdg = np.zeros(vr.shape)
    pdg[6], pdg[14], qdg[6], qdg[14] = x[24], x[25], x[26], x[27]
    ir, ii = _currents(g[:, :, None], b[:, :, None], vr, vi)
    psp = vr * ir + vi * ii
    qsp = vi * ir - vr * ii
    del_p = psp[3:] - pdg[3:] - p[3:, None]
    del_q = qsp[3:] - qdg[3:] - q[3:, None]
    # |I1 + I2 + I3| + |I1 + exp(j 4 pi / 3) I2 + exp(j 2 pi / 3) I3|
    rot1r, rot1i = np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)
    rot2r, rot2i = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    t1r, t1i = ir[0] + ir[1] + ir[2], ii[0] + ii[1] + ii[2]
    t2r = ir[0] + (rot1r * ir[1] - rot1i * ii[1]) + (rot2r * ir[2] - rot2i * ii[2])
    t2i = ii[0] + (rot1r * ii[1] + rot1i * ir[1]) + (rot2r * ii[2] + rot2i * ir[2])
    phase_balance = np.hypot(t1r, t1i) + np.hypot(t2r, t2i)
    return psp, qsp, phase_balance, [relaxed_equality(h) for h in (*del_p, *del_q)]


def _opf_flow(x: Array) -> tuple[Array, Array, dict[str, Array], list[Array]]:
    """Active and reactive power of each bus, the other objective terms and constraints of RCM40-RCM46."""
    g, b, p, q = _network("OPF")
    vr = np.zeros((14, x.shape[1]))
    vi = np.zeros((14, x.shape[1]))
    vr[0] = 1.0
    vr[1:], vi[1:] = x[0:13], x[13:26]
    pg = np.zeros(vr.shape)
    qg = np.zeros(vr.shape)
    for k, bus in enumerate(_OPF_GENERATORS):
        pg[bus], qg[bus] = x[26 + k], x[30 + k]
    ir, ii = _currents(g[:, :, None], b[:, :, None], vr, vi)
    psp = vr * ir + vi * ii
    qsp = vi * ir - vr * ii
    del_p = psp[1:] - pg[1:] + p[1:, None]
    del_q = qsp[1:] - qg[1:] + q[1:, None]
    deviation = _total((1 - np.hypot(vr[1:], vi[1:])) ** 2)
    fuel = np.zeros(x.shape[1])
    for bus, linear, quadratic in _OPF_FUEL:
        fuel = fuel + (linear * pg[bus] + quadratic * pg[bus] * pg[bus])
    terms = {
        "reactive": vi[0] * ir[1] - vr[0] * ii[1] + (_total(qsp) - qsp[0]),
        "deviation": deviation,
        "fuel": fuel,
    }
    return psp, qsp, terms, [relaxed_equality(h) for h in (*del_p, *del_q)]


def _microgrid_flow(x: Array) -> tuple[Array, Array, Array, list[Array]]:
    """Active and reactive power of each bus, voltage deviation and constraints of RCM47-RCM49."""
    n_points = x.shape[1]
    vr = np.zeros((6, n_points))
    vi = np.zeros((6, n_points))
    vr[1:], vi[1:] = x[0:5], x[5:10]
    w = x[10]
    vr[0] = x[11] + 1e-5
    pc = np.zeros(vr.shape)
    qc = np.zeros(vr.shape)
    pc[3:], qc[3:] = x[12:15], x[15:18]
    yr = np.zeros((6, 6, n_points))
    yi = np.zeros((6, 6, n_points))
    for start, end, resistance, reactance in _MICROGRID_LINES:
        scaled = reactance * w
        denominator = resistance * resistance + scaled * scaled
        branch_r = resistance / denominator
        branch_i = -scaled / denominator
        yr[start, end] = yr[start, end] - branch_r
        yi[start, end] = yi[start, end] - branch_i
        yr[end, start] = yr[start, end]
        yi[end, start] = yi[start, end]
        yr[start, start] = yr[start, start] + branch_r
        yi[start, start] = yi[start, start] + branch_i
        yr[end, end] = yr[end, end] + branch_r
        yi[end, end] = yi[end, end] + branch_i
    ir, ii = _currents(yr, yi, vr, vi)
    magnitude = np.hypot(vr, vi)
    # The droop exponents of the reference data make the voltage factor of the active power 1.
    psp = pc * (1 - w) - np.asarray(_MICROGRID_P)[:, None]
    qsp = qc * (1 - magnitude) - np.asarray(_MICROGRID_Q)[:, None]
    squared = vr * vr + vi * vi
    del_ir = ir - (psp * vr + qsp * vi) / squared
    del_ii = ii - (psp * vi - qsp * vr) / squared
    deviation = _total((1 - magnitude) ** 2)
    return psp, qsp, deviation, [relaxed_equality(h) for h in (*del_ir, *del_ii)]


def _rcm36(x: Array) -> tuple[list[Array], list[Array]]:
    psp, _, balance, c = _dg_flow(x)
    return [balance, _total(psp)], c


def _rcm37(x: Array) -> tuple[list[Array], list[Array]]:
    _, qsp, balance, c = _dg_flow(x)
    return [balance, _total(qsp)], c


def _rcm38(x: Array) -> tuple[list[Array], list[Array]]:
    psp, qsp, _, c = _dg_flow(x)
    return [_total(qsp), _total(psp)], c


def _rcm39(x: Array) -> tuple[list[Array], list[Array]]:
    psp, qsp, balance, c = _dg_flow(x)
    return [balance, _total(psp), _total(qsp)], c


def _rcm40(x: Array) -> tuple[list[Array], list[Array]]:
    psp, _, terms, c = _opf_flow(x)
    return [_total(psp), terms["reactive"]], c


def _rcm41(x: Array) -> tuple[list[Array], list[Array]]:
    psp, _, terms, c = _opf_flow(x)
    return [_total(psp), terms["reactive"], terms["deviation"]], c


def _rcm42(x: Array) -> tuple[list[Array], list[Array]]:
    psp, _, terms, c = _opf_flow(x)
    return [_total(psp), terms["deviation"]], c


def _rcm43(x: Array) -> tuple[list[Array], list[Array]]:
    psp, _, terms, c = _opf_flow(x)
    return [terms["fuel"], _total(psp)], c


def _rcm44(x: Array) -> tuple[list[Array], list[Array]]:
    psp, qsp, terms, c = _opf_flow(x)
    return [terms["fuel"], _total(psp), _total(qsp)], c


def _rcm45(x: Array) -> tuple[list[Array], list[Array]]:
    psp, _, terms, c = _opf_flow(x)
    return [terms["fuel"], _total(psp), terms["deviation"]], c


def _rcm46(x: Array) -> tuple[list[Array], list[Array]]:
    psp, qsp, terms, c = _opf_flow(x)
    return [terms["fuel"], _total(psp), _total(qsp), terms["deviation"]], c


def _rcm47(x: Array) -> tuple[list[Array], list[Array]]:
    psp, qsp, _, c = _microgrid_flow(x)
    return [_total(psp), _total(qsp)], c


def _rcm48(x: Array) -> tuple[list[Array], list[Array]]:
    psp, _, deviation, c = _microgrid_flow(x)
    return [_total(psp), deviation], c


def _rcm49(x: Array) -> tuple[list[Array], list[Array]]:
    psp, qsp, deviation, c = _microgrid_flow(x)
    return [_total(psp), _total(qsp), deviation], c


def _rcm50(x: Array) -> tuple[list[Array], list[Array]]:
    """Power distribution system planning: cost and emission, with the power balance ``sum(p) = 1200 + losses``."""
    loss = np.zeros(x.shape[1])
    for i, row in enumerate(_RCM50_LOSS):
        for j, coefficient in enumerate(row):
            loss = loss + x[i] * (coefficient * 1e-6) * x[j]
    cost = np.zeros(x.shape[1])
    emission = np.zeros(x.shape[1])
    for i, ((a, b, c), (alpha, beta, gamma)) in enumerate(zip(_RCM50_COST, _RCM50_EMISSION)):
        cost = cost + (a + b * x[i] + c * x[i] * x[i])
        emission = emission + (alpha + beta * x[i] + gamma * x[i] * x[i])
    return [cost, emission], [relaxed_equality(_total(x) - 1200 - loss)]


_DG_BOUNDS = ((-1.0,) * 24 + (0.0,) * 4, (1.0,) * 28)
_OPF_BOUNDS = ((-1.0,) * 26 + (0.0,) * 8, (1.0,) * 34)
# Voltages of buses 2-6 in [-1, 1], frequency and voltage of bus 1 in [0, 2], droop gains in [0, 500].
_MICROGRID_BOUNDS = ((-1.0,) * 10 + (0.0,) * 8, (1.0,) * 10 + (2.0, 2.0) + (500.0,) * 6)

PROBLEMS: dict[str, Entry] = {
    "rcm36": (_rcm36, *_DG_BOUNDS, 2, 24),
    "rcm37": (_rcm37, *_DG_BOUNDS, 2, 24),
    "rcm38": (_rcm38, *_DG_BOUNDS, 2, 24),
    "rcm39": (_rcm39, *_DG_BOUNDS, 3, 24),
    "rcm40": (_rcm40, *_OPF_BOUNDS, 2, 26),
    "rcm41": (_rcm41, *_OPF_BOUNDS, 3, 26),
    "rcm42": (_rcm42, *_OPF_BOUNDS, 2, 26),
    "rcm43": (_rcm43, *_OPF_BOUNDS, 2, 26),
    "rcm44": (_rcm44, *_OPF_BOUNDS, 3, 26),
    "rcm45": (_rcm45, *_OPF_BOUNDS, 3, 26),
    "rcm46": (_rcm46, *_OPF_BOUNDS, 4, 26),
    "rcm47": (_rcm47, *_MICROGRID_BOUNDS, 2, 12),
    "rcm48": (_rcm48, *_MICROGRID_BOUNDS, 2, 12),
    "rcm49": (_rcm49, *_MICROGRID_BOUNDS, 3, 12),
    "rcm50": (_rcm50, (10.0, 10.0, 35.0, 35.0, 125.0, 130.0), (125.0, 150.0, 210.0, 225.0, 315.0, 325.0), 2, 1),
}

__all__ = ["PROBLEMS"]
