"""RCM22, RCM24 and RCM28 reformulated: closed-form decoders of three process design problems.

Each genotype maps to a complete design that meets every equality by construction. A gene in
``[-0.1, 1.1]`` is clipped to ``[0, 1]``, so the bounds of the free variables, where many optimal designs
lie, have a positive probability.

- **RCM22** (Haverly's pooling), 5 genes: the pool quality ``x9 = 1 + 2 u1``, which fixes the mix of feeds A
  and B (h1, h4); the share ``s = u2`` of the pool sent to product 1 (``s = 1`` when the quality exceeds 1.5,
  where g2 forbids product 2 from the pool); the pool size, a share ``u3`` of the largest one that keeps
  ``x3``, ``x4`` and ``x8`` within 100 and leaves room for the C that product 1 needs; and the feed C to
  products 1 and 2 (``u4``, ``u5``), each between its sulfur limit (g1, g2) and its bounds. ``x1 = x5 + x7``
  and ``x2 = x6 + x8`` (h2, h3).
- **RCM24** (heat exchanger network), 1 gene: h1-h4 give ``x8 = 700 - x7``, ``x9 = 600 + x7``,
  ``x3 = 1e4 (x7 - 100)`` and ``x5 = 1e4 (300 - x7)``; then h5 is ``2 t`` with ``x7 = 100 + t``, so
  ``t = u (0.99e-4 / 2 - 1e-11)``. Any feasible design is weakly dominated by the same one with
  ``x1 = x2 = 0``, so ``x1 = x2 = x4 = x6 = 0``.
- **RCM28** (two reactors), 2 genes: exactly one reactor is used (h1), the first when ``u1 < 0.5``; its
  volume runs with ``u2`` from the one that needs the largest flow, 20, to 10, and its flow is the one that
  gives ``z = 10`` exactly (h2, h4). The other reactor's volume and flow are 0, and ``x7`` is the flow (h3).
"""

from __future__ import annotations

import math

import numpy as np

from vamos.foundation.problem.rcm import RCM
from vamos.foundation.problem.rcm._common import EQUALITY_TOLERANCE, Array
from vamos.foundation.problem.rcm.reformulations._base import RCMReformulation

# The genes of the process problems span [-SNAP, 1 + SNAP] and are clipped to [0, 1].
SNAP = 0.1
# RCM24's h5 = 2 t is kept 1 % inside the tolerance; 1e-11 absorbs the rounding of x7 + 500 - x8.
RCM24_T_MAX = 0.99 * EQUALITY_TOLERANCE / 2.0 - 1e-11
# Reactor i of RCM28: z = c (1 - exp(-r v)) x.
_REACTORS = ((0.9, 0.5), (0.8, 0.4))


def _bounds(key: str) -> tuple[Array, Array]:
    _, lower, upper, _, _ = RCM[key]
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _unit(U: Array) -> Array:
    return np.clip(U, 0.0, 1.0)


def decode_rcm22(U: Array) -> Array:
    """Haverly's pooling: (quality, split, pool size, C to product 1, C to product 2) -> x1..x9."""
    u = _unit(U)
    q = 1.0 + 2.0 * u[:, 0]
    a = u[:, 0]  # share of feed A in the pool: 3 a + (1 - a) = q
    s = np.where(q > 1.5, 1.0, u[:, 1])
    over = np.maximum(q - 2.5, 0.0)  # product 1 needs C when the pool exceeds 2.5 %
    with np.errstate(divide="ignore"):
        caps = np.column_stack(
            [
                np.where(a > 0, 100.0 / a, np.inf),  # x3 <= 100
                np.where(a < 1, 100.0 / (1.0 - a), np.inf),  # x4 <= 100
                # x7 plus the least x5 within 100, with room for the safety factor of that least x5
                np.where(s > 0, 100.0 * (1.0 - 1e-8) / (s * (1.0 + 2.0 * over)), np.inf),
                np.where(s < 1, 100.0 / (1.0 - s), np.inf),  # x8 <= 100
            ]
        )
    pool = u[:, 2] * caps.min(axis=1)
    x7 = s * pool
    x8 = (1.0 - s) * pool
    x3 = a * pool
    x4 = (1.0 - a) * pool
    x5_low = 2.0 * x7 * over * (1.0 + 1e-9)  # g1: x7 (q - 2.5) <= 0.5 x5
    x5 = x5_low + u[:, 3] * np.maximum(100.0 - x7 - x5_low, 0.0)
    x6_high = np.minimum.reduce([2.0 * x8 * np.maximum(1.5 - q, 0.0) * (1.0 - 1e-9), np.full_like(x8, 100.0), 200.0 - x8])
    x6 = u[:, 4] * x6_high
    X = np.column_stack([x5 + x7, x6 + x8, x3, x4, x5, x6, x7, x8, q])
    return np.asarray(np.clip(X, *_bounds("rcm22")))


def decode_rcm24(U: Array) -> Array:
    """Heat exchanger network: t -> x1..x9 on the segment of designs that meet every equality."""
    t = _unit(U[:, 0]) * RCM24_T_MAX
    x7 = 100.0 + t
    zero = np.zeros_like(t)
    X = np.column_stack([zero, zero, 1e4 * (x7 - 100.0), zero, 1e4 * (300.0 - x7), zero, x7, 700.0 - x7, 600.0 + x7])
    return np.asarray(np.clip(X, *_bounds("rcm24")))


def decode_rcm28(U: Array) -> Array:
    """Two reactors: (reactor, volume) -> x1..x7 with one reactor used and z = 10."""
    first = U[:, 0] < 0.5
    w = _unit(U[:, 1])
    X = np.zeros((U.shape[0], 7))
    for i, rows in ((0, first), (1, ~first)):
        c, r = _REACTORS[i]
        v_min = -math.log(1.0 - 10.0 / (20.0 * c)) / r  # the volume that needs the largest flow, 20
        v = v_min + w[rows] * (10.0 - v_min)
        flow = np.minimum(10.0 / (c * (1.0 - np.exp(-r * v))), 20.0)
        X[rows, i] = flow  # x1 or x2
        X[rows, 2 + i] = v  # v1 or v2
        X[rows, 4 + i] = 1.0  # y1 or y2
        X[rows, 6] = flow  # x7 = x1 + x2
    return X


# number: decoder, genotype lower and upper bounds, the original's number of inequalities (listed first)
_DECODERS = {
    22: (decode_rcm22, (-SNAP,) * 5, (1.0 + SNAP,) * 5, 2),
    24: (decode_rcm24, (-SNAP,), (1.0 + SNAP,), 0),
    28: (decode_rcm28, (0.0, -SNAP), (1.0, 1.0 + SNAP), 4),
}


class ProcessReformulation(RCMReformulation):
    """RCM22, RCM24 or RCM28 searched through its closed-form decoder; the decoding never fails."""

    def __init__(self, number: int) -> None:
        if number not in _DECODERS:
            raise ValueError(f"No process reformulation of RCM{number}; they are RCM22, RCM24 and RCM28.")
        decoder, lower, upper, n_inequalities = _DECODERS[number]
        super().__init__(number, np.asarray(lower), np.asarray(upper), n_inequalities)
        self._decoder = decoder

    def _solve(self, U: Array) -> tuple[Array, Array]:
        return self._decoder(U), np.zeros(U.shape[0])


__all__ = ["ProcessReformulation", "RCM24_T_MAX", "SNAP", "decode_rcm22", "decode_rcm24", "decode_rcm28"]
