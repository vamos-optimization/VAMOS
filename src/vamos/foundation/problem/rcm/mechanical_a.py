"""RCM01-RCM11: mechanical design problems of the CEC 2021 suite, as jMetal's classes define them.

RCM03 is CRE21 (the two-bar truss), RCM05 CRE23 (the disc brake) and RCM11 CRE51 (water resource planning):
jMetal's classes give the same values, so they share the formulas.
"""

from __future__ import annotations

import numpy as np

from vamos.foundation.problem.classic_constrained import CLASSICS
from vamos.foundation.problem.rcm._common import Array, Entry, java_round
from vamos.foundation.problem.real_world.tanabe_ishibuchi_re3_a import disc_brake, two_bar_truss


def _rcm01(x: Array) -> tuple[list[Array], list[Array]]:
    """Pressure vessel design; the numbers of plates x1 and x2 are rounded."""
    x1, x2, x3, x4 = java_round(x[0]), java_round(x[1]), x[2], x[3]
    z1 = 0.0625 * x1
    z2 = 0.0625 * x2
    f1 = 1.7781 * z1 * x3**2 + 0.6224 * z1 * x2 * x4 + 3.1661 * z1**2 * x4 + 19.84 * z1**2 * x3
    f2 = -np.pi * x3**2 * x4 - (4.0 / 3.0) * np.pi * x3**3
    g1 = 0.00954 * x3 - z2
    g2 = 0.0193 * x3 - z1
    return [f1, f2], [-g1, -g2]


def _rcm02(x: Array) -> tuple[list[Array], list[Array]]:
    """Vibrating platform design."""
    d1, d2, d3, b, length = x[0], x[1], x[2], x[3], x[4]
    rho1, rho2, rho3 = 100.0, 2770.0, 7780.0
    e1, e2 = 1.6, 70.0
    c1, c2, c3 = 500.0, 1500.0, 800.0
    mu = 2 * b * (rho1 * d1 + rho2 * (d2 - d1) + rho3 * (d3 - d2))
    # The reference code multiplies the last layer by its density, not by a modulus.
    ei = (2 * b / 3) * (e1 * d1**3 + e2 * (d2**3 - d1**3) + rho3 * (d3 - d2))
    f1 = (-np.pi) / (2 * length) ** 2 * np.abs(ei / mu) ** 0.5
    f2 = 2 * b * length * (c1 * d1 + c2 * (d2 - d1) + c3 * (d3 - d2))
    g = [mu * length - 2800, d1 - d2, d2 - d1 - 0.15, d2 - d3, d3 - d2 - 0.01]
    return [f1, f2], [-gi for gi in g]


def _rcm04(x: Array) -> tuple[list[Array], list[Array]]:
    """Welded beam design; the buckling load uses x3^2 + x4^6, where CRE22 has x3^2 x4^6."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    p, l, e = 6000.0, 14.0, 30e6
    tmax, sigmax, g = 13600.0, 30000.0, 12e6
    pc = (4.013 * e * np.sqrt((x3 * x3 + x4**6) / 36)) / (l * l) * (1 - x3 / (2 * l) * np.sqrt(e / (4 * g)))
    sigma = (6 * p * l) / (x4 * x3 * x3)
    j = 2 * (np.sqrt(2) * x1 * x2 * (x2 * x2 / 12 + ((x1 + x3) / 2) ** 2))
    r = np.sqrt(x2 * x2 / 4 + ((x1 + x3) / 2) ** 2)
    m = p * (l + x2 / 2)
    tho1 = p / (np.sqrt(2) * x1 * x2)
    tho2 = m * r / j
    tho = np.sqrt(tho1 * tho1 + 2 * tho1 * tho2 * x2 / (2 * r) + tho2 * tho2)
    f1 = 1.10471 * x1 * x1 * x2 + 0.04811 * x3 * x4 * (14 + x2)
    f2 = (4 * p * l**3) / (e * x4 * x3**3)
    return [f1, f2], [-(tho - tmax), -(sigma - sigmax), -(x1 - x4), -(p - pc)]


def speed_reducer(x: Array) -> tuple[list[Array], list[Array]]:
    """RCM06's and RCM13's objectives f1, f2 and f3 and their first ten constraints; x3 is rounded.

    f3 is the stress of the second shaft, which RCM06 bounds by 850 and RCM13 minimizes and bounds by 1100.
    """
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], java_round(x[2]), x[3], x[4], x[5], x[6]
    f1 = (
        0.7854 * x1 * x2 * x2 * (10 * x3 * x3 / 3 + 14.933 * x3 - 43.0934)
        - 1.508 * x1 * (x6 * x6 + x7 * x7)
        + 7.477 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    )
    f2 = np.sqrt((745 * x4 / (x2 * x3)) ** 2 + 1.69e7) / (0.1 * x6**3)
    f3 = np.sqrt((745 * x5 / (x2 * x3)) ** 2 + 1.575e8) / (0.1 * x7**3)
    g = [
        1 / (x1 * x2 * x2 * x3) - 1.0 / 27,
        1 / (x1 * x2 * x2 * x3 * x3) - 1.0 / 397.5,
        x4**3 / (x2 * x3 * x6**4) - 1.0 / 1.93,
        x5**3 / (x2 * x3 * x7**4) - 1.0 / 1.93,
        x2 * x3 - 40,
        x1 / x2 - 12,
        -x1 / x2 + 5,
        1.9 - x4 + 1.5 * x6,
        1.9 - x5 + 1.1 * x7,
        f2 - 1300,
    ]
    return [f1, f2, f3], [-gi for gi in g]


def _rcm06(x: Array) -> tuple[list[Array], list[Array]]:
    """Speed reducer design."""
    (f1, f2, f3), c = speed_reducer(x)
    return [f1, f2], [*c, -(f3 - 850)]


def _rcm07(x: Array) -> tuple[list[Array], list[Array]]:
    """Gear train design; unlike CRE25, the numbers of teeth are not rounded."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    f1 = np.abs(6.931 - x3 * x4 / (x1 * x2))
    f2 = np.maximum(np.maximum(x1, x2), np.maximum(x3, x4))
    return [f1, f2], [-(f1 / 6.931 - 0.5)]


def _rcm08(x: Array) -> tuple[list[Array], list[Array]]:
    """Car side impact design, with the reference code's f1 (x1 times x2) and its g2 (x2 x5 for x3 x5)."""
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
    vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
    f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
    f1 = 1.98 + 4.9 * x1 * 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-5 * x6 + 2.73 * x7
    g = [
        -1 + 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3,
        -0.32 + 0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x2 * x5 + 0.0154464 * x6,
        -0.32 + 0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2,
        -0.32
        + 0.214
        + 0.00817 * x5
        - 0.045195 * x1
        - 0.0135168 * x1
        + 0.03099 * x2 * x6
        - 0.018 * x2 * x7
        + 0.007176 * x3
        + 0.023232 * x3
        - 0.00364 * x5 * x6
        - 0.018 * x2 * x2,
        -32 + 33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728,
        -32 + 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7,
        -32 + 46.36 - 9.9 * x2 - 4.4505 * x1,
        f2 - 4,
        vmbp - 9.9,
    ]
    return [f1, f2, 0.5 * (vmbp + vfd)], [-gi for gi in g]


def _rcm09(x: Array) -> tuple[list[Array], list[Array]]:
    """Four-bar plane truss; no constraints, as in the reference evaluator."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    f, e, l = 10.0, 2e5, 200.0
    f1 = l * (2 * x1 + np.sqrt(2) * x2 + np.sqrt(2) * x3 + x4)
    f2 = f * l / e * (2 / x1 + 2 * np.sqrt(2) / x2 - 2 * np.sqrt(2) / x3 + 2 / x4)
    return [f1, f2], []


def _rcm10(x: Array) -> tuple[list[Array], list[Array]]:
    """Two-bar plane truss, with the load P = 104 of jMetal pull request #514."""
    x1, x2 = x[0], x[1]
    rho, h, p, e, rho0 = 0.283, 100.0, 104.0, 3e7, 2e4
    f1 = 2 * rho * h * x2 * np.sqrt(1 + x1 * x1)
    f2 = rho * h * (1 + x1 * x1) ** 1.5 * np.sqrt(1 + x1**4) / (2 * np.sqrt(2) * e * x1 * x1 * x2)
    g1 = p * (1 + x1) * np.sqrt(1 + x1 * x1) / (2 * np.sqrt(2) * x1 * x2) - rho0
    g2 = p * (-x1 + 1) * np.sqrt(1 + x1 * x1) / (2 * np.sqrt(2) * x1 * x2) - rho0
    return [f1, f2], [-g1, -g2]


PROBLEMS: dict[str, Entry] = {
    "rcm01": (_rcm01, (0.51, 0.51, 10.0, 10.0), (99.49, 99.49, 200.0, 200.0), 2, 2),
    "rcm02": (_rcm02, (0.05, 0.2, 0.2, 0.35, 3.0), (0.5, 0.5, 0.6, 0.5, 6.0), 2, 5),
    "rcm03": (two_bar_truss, (1e-5, 1e-5, 1.0), (100.0, 100.0, 3.0), 2, 3),
    "rcm04": (_rcm04, (0.125, 0.1, 0.1, 0.125), (5.0, 10.0, 10.0, 5.0), 2, 4),
    "rcm05": (disc_brake, (55.0, 75.0, 1000.0, 11.0), (80.0, 110.0, 3000.0, 20.0), 2, 4),
    "rcm06": (_rcm06, (2.6, 0.7, 16.51, 7.3, 7.3, 2.9, 5.0), (3.6, 0.8, 28.49, 8.3, 8.3, 3.9, 5.5), 2, 11),
    "rcm07": (_rcm07, (11.51, 11.51, 11.51, 11.51), (60.49, 60.49, 60.49, 60.49), 2, 1),
    "rcm08": (_rcm08, (0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4), (1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2), 3, 9),
    "rcm09": (_rcm09, (1.0, 1.4142135623730951, 1.4142135623730951, 1.0), (3.0, 3.0, 3.0, 3.0), 2, 0),
    "rcm10": (_rcm10, (0.1, 0.5), (2.0, 2.5), 2, 2),
    "rcm11": (CLASSICS["water"][0], (0.01, 0.01, 0.01), (0.45, 0.1, 0.1), 5, 7),
}

__all__ = ["PROBLEMS", "speed_reducer"]
