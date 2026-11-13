"""RCM22-RCM29: chemical engineering and process design problems of the CEC 2021 suite, as jMetal defines them.

jMetal relaxes each equality constraint ``h = 0`` to ``|h| <= 1e-4`` (:func:`relaxed_equality`), and rounds
the binary variables of RCM25, RCM26, RCM28 and RCM29 with Java's ``Math.round`` (:func:`java_round`).
"""

from __future__ import annotations

import numpy as np

from vamos.foundation.problem.rcm._common import Array, Entry, java_round, relaxed_equality


def _rcm22(x: Array) -> tuple[list[Array], list[Array]]:
    """Haverly's pooling problem."""
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]
    f1 = -9 * x1 - 15 * x2 + 6 * x3 + 16 * x4
    f2 = 10 * (x5 + x6)
    g1 = x9 * x7 + 2 * x5 - 2.5 * x1
    g2 = x9 * x8 + 2 * x6 - 1.5 * x2
    h = [x7 + x8 - x4 - x3, x1 - x5 - x7, x2 - x6 - x8, x9 * x7 + x9 * x8 - 3 * x3 - x4]
    return [f1, f2], [-g1, -g2, *(relaxed_equality(hi) for hi in h)]


def _rcm23(x: Array) -> tuple[list[Array], list[Array]]:
    """Reactor network design."""
    x1, x2, x3, x4, x5, x6 = x[0], x[1], x[2], x[3], x[4], x[5]
    k1 = 0.09755988
    k2 = 0.99 * k1
    k3 = 0.0391908
    k4 = 0.9 * k3
    f2 = np.sqrt(x5) + np.sqrt(x6)
    h = [
        k1 * x5 * x2 + x1 - 1,
        k3 * x5 * x3 + x3 + x1 - 1,
        k2 * x6 * x2 - x1 + x2,
        k4 * x6 * x4 + x2 - x1 + x4 - x3,
    ]
    return [-x4, f2], [-(f2 - 4), *(relaxed_equality(hi) for hi in h)]


def _rcm24(x: Array) -> tuple[list[Array], list[Array]]:
    """Heat exchanger network design; every constraint is an equality."""
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]
    f1 = 35 * x1**0.6 + 35 * x2**0.6
    f2 = 200 * x1 * x4 - x3
    f3 = 200 * x2 * x6 - x5
    h = [
        x3 - 1e4 * (x7 - 100),
        x5 - 1e4 * (300 - x7),
        x3 - 1e4 * (600 - x8),
        x5 - 1e4 * (900 - x9),
        x4 * np.log(np.abs(x8 - 100) + 1e-6) - x4 * np.log(np.abs(600 - x7) + 1e-6) - x8 + x7 + 500,
        x6 * np.log(np.abs(x9 - x7) + 1e-6) - x6 * np.log(600) - x9 + x7 + 600,
    ]
    return [f1, f2, f3], [relaxed_equality(hi) for hi in h]


def _rcm25(x: Array) -> tuple[list[Array], list[Array]]:
    """Process synthesis; x2 is binary, rounded."""
    x1, x2 = x[0], java_round(x[1])
    f2 = -x1 * x1 - x2
    return [x2 + 2 * x1, f2], [-(f2 + 1.25), -(x1 + x2 - 1.6)]


def _rcm26(x: Array) -> tuple[list[Array], list[Array]]:
    """Process synthesis and design; x3 is binary, rounded."""
    x1, x2, x3 = x[0], x[1], java_round(x[2])
    f1 = -x3 + x2 + 2 * x1
    f2 = -x1 * x1 - x2 + x1 * x3
    h1 = -2 * np.exp(-x2) + x1
    g1 = x2 - x1 + x3
    return [f1, f2], [-g1, relaxed_equality(h1)]


def _rcm27(x: Array) -> tuple[list[Array], list[Array]]:
    """Process flow sheeting; jMetal, like the reference code, does not round x3."""
    x1, x2, x3 = x[0], x[1], x[2]
    f1 = -0.7 * x3 + 0.8 + 5 * (0.5 - x1) ** 2
    g = [-(np.exp(x1 - 0.2) + x2), x2 + 1.1 * x3 - 1, x1 - x3 - 0.2]
    return [f1, x1 - x3], [-gi for gi in g]


def _rcm28(x: Array) -> tuple[list[Array], list[Array]]:
    """Two-reactor problem; y1 and y2 are binary, rounded."""
    x1, x2, v1, v2 = x[0], x[1], x[2], x[3]
    y1, y2, xx = java_round(x[4]), java_round(x[5]), x[6]
    z1 = 0.9 * (1 - np.exp(-0.5 * v1)) * x1
    z2 = 0.8 * (1 - np.exp(-0.4 * v2)) * x2
    f1 = 7.5 * y1 + 5.5 * y2 + 7 * v1 + 6 * v2 + 5 * xx
    g = [v1 - 10 * y1 - 1e-6, v2 - 10 * y2, x1 - 20 * y1, x2 - 20 * y2]
    h = [y1 + y2 - 1, z1 + z2 - 10, x1 + x2 - xx, z1 * y1 + z2 * y2 - 10]
    return [f1, x1 + x2], [*(-gi for gi in g), *(relaxed_equality(hi) for hi in h)]


def _rcm29(x: Array) -> tuple[list[Array], list[Array]]:
    """Process synthesis (second problem); x4 to x7 are binary, rounded."""
    x1, x2, x3 = x[0], x[1], x[2]
    x4, x5, x6, x7 = java_round(x[3]), java_round(x[4]), java_round(x[5]), java_round(x[6])
    f1 = (1 - x4) ** 2 + (1 - x5) ** 2 + (1 - x6) ** 2 - np.log(np.abs(1 + x7) + 1e-6)
    f2 = (1 - x1) ** 2 + (2 - x2) ** 2 + (3 - x3) ** 2
    g = [
        x1 + x2 + x3 + x4 + x5 + x6 - 5,
        x6**3 + x1 * x1 + x2 * x2 + x3 * x3 - 5.5,
        x1 + x4 - 1.2,
        x2 + x5 - 1.8,
        x3 + x6 - 2.5,
        x1 + x7 - 1.2,
        x5 * x5 + x2 * x2 - 1.64,
        x6 * x6 + x3 * x3 - 4.25,
        x5 * x5 + x3 * x3 - 4.64,
    ]
    return [f1, f2], [-gi for gi in g]


PROBLEMS: dict[str, Entry] = {
    "rcm22": (_rcm22, (0.0,) * 9, (100.0, 200.0, 100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 200.0), 2, 6),
    "rcm23": (_rcm23, (0.0, 0.0, 0.0, 0.0, 0.00001, 0.00001), (1.0, 1.0, 1.0, 1.0, 16.0, 16.0), 2, 5),
    "rcm24": (
        _rcm24,
        (0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 100.0, 100.0, 100.0),
        (10.0, 200.0, 100.0, 200.0, 2000000.0, 600.0, 600.0, 600.0, 900.0),
        3,
        6,
    ),
    "rcm25": (_rcm25, (0.0, -0.49), (1.6, 1.49), 2, 2),
    "rcm26": (_rcm26, (0.5, 0.5, -0.49), (1.4, 1.4, 1.49), 2, 2),
    "rcm27": (_rcm27, (0.2, -2.22554, -0.49), (1.0, -1.0, 1.49), 2, 3),
    "rcm28": (_rcm28, (0.0, 0.0, 0.0, 0.0, -0.49, -0.49, 0.0), (20.0, 20.0, 10.0, 10.0, 1.49, 1.49, 40.0), 2, 8),
    "rcm29": (_rcm29, (0.0, 0.0, 0.0, -0.49, -0.49, -0.49, -0.49), (100.0, 100.0, 100.0, 1.49, 1.49, 1.49, 1.49), 2, 9),
}

__all__ = ["PROBLEMS"]
