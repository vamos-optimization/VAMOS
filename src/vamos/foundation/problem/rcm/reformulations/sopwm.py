"""RCM30-RCM35 reformulated: switching angles that increase by construction.

The constraints of the SOPWM problems keep consecutive angles at least ``ORDER_THRESHOLD`` (1e-6 degrees)
apart. Each variant decodes its genotype into angles in ``[0, 90]`` that increase by at least ``GAP``,
1 % more than that threshold, so those constraints always hold:

- ``"cumulative"`` (the default), the cumulative-variable reformulation of Wang, Zhu, Nan and Ishibuchi
  (GECCO '25 Companion, doi:10.1145/3712255.3726689, Section 2.2): ``D`` shares ``t_i`` in ``[0, 1]`` and
  a total ``t_{D+1}`` in ``[0, 90 - (D - 1) GAP]``; ``t^_i = alpha t_i`` with
  ``alpha = t_{D+1} / sum t_i`` (equal shares when every ``t_i`` is 0), ``GAP`` is added to ``t^_2`` to
  ``t^_D``, and the angles are the cumulative sums ``y_i = t^_1 + ... + t^_i`` (Eqs. 18-21), in degrees
  instead of the paper's radians, with ``GAP`` for PlatEMO's ``1e-5``.
- ``"sorted"``: ``D`` angles in ``[-4.5, 94.5]``, clipped to ``[0, 90]`` and sorted; each angle is then
  raised to its predecessor plus ``GAP``, the last one capped at 90, and each lowered to its successor
  minus ``GAP``, from the last to the first; a first angle that rounding leaves below 0 becomes 0.
- ``"level"``: the sorted angles and one more gene ``u`` in ``[-0.1, 1.1]``, clipped to ``[0, 1]``, which
  sets the fundamental ``S = sum s_l cos(x_l)`` to ``m u``, so that ``f2 = m^2 (1 - u)^2``: the sorted
  angles are moved toward 90 (``90 - a (90 - x)``) when their fundamental exceeds ``m u``, and toward 0
  (``a x``) when it falls short and the angles at 0 reach ``m u``, with ``a`` in ``[0, 1]`` found by
  bisection (60 halvings) and the result sorted again.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from vamos.foundation.problem.rcm._common import Array
from vamos.foundation.problem.rcm.power_electronics import _PATTERNS
from vamos.foundation.problem.rcm.reformulations._base import RCMReformulation

# The decoded angles increase by at least GAP degrees, 1 % more than power_electronics.ORDER_THRESHOLD.
GAP = 1.01e-6
# The angles of the sorted and level genotypes span [-ANGLE_SNAP, 90 + ANGLE_SNAP] and are clipped to [0, 90].
ANGLE_SNAP = 4.5
# The level gene spans [-LEVEL_SNAP, 1 + LEVEL_SNAP] and is clipped to [0, 1].
LEVEL_SNAP = 0.1
VARIANTS = ("cumulative", "sorted", "level")
_BISECTIONS = 60


def _signs(key: str) -> tuple[float, Array]:
    modulation_index, pattern = _PATTERNS[key]
    return modulation_index, np.array([1.0 if symbol == "+" else -1.0 for symbol in pattern])


def sorted_angles(U: Array) -> Array:
    """The angles of the ``"sorted"`` variant: clipped to [0, 90], sorted, then at least GAP apart."""
    x = np.sort(np.clip(U, 0.0, 90.0), axis=1)
    d = x.shape[1]
    for i in range(1, d):
        x[:, i] = np.maximum(x[:, i], x[:, i - 1] + GAP)
    x[:, -1] = np.minimum(x[:, -1], 90.0)
    for i in range(d - 2, -1, -1):
        x[:, i] = np.minimum(x[:, i], x[:, i + 1] - GAP)
    # When angles crowd at 0, rounding in the backward pass can leave the first one about 1e-21 below 0.
    return np.asarray(np.clip(x, 0.0, 90.0))


def fundamental(signs: Array, x: Array) -> Array:
    """``sum_l s_l cos(x_l)`` of each row of angles in degrees: ``f2 = (fundamental - m)^2``."""
    return np.asarray(np.sum(signs[None, :] * np.cos(x * math.pi / 180.0), axis=1))


def _bisect(signs: Array, x: Array, target: Array, move: Callable[[Array, Array | float], Array]) -> Array:
    """The sorted angles of ``move(x, a)``, with ``a`` in [0, 1] such that their fundamental is ``target``.

    ``move(x, 1)`` is ``x``. When the fundamentals at ``a = 0`` and ``a = 1`` lie on the same side of the
    target, ``a = 0``.
    """
    side0 = fundamental(signs, sorted_angles(move(x, 0.0))) > target
    side1 = fundamental(signs, sorted_angles(move(x, 1.0))) > target
    low, high = np.zeros(x.shape[0]), np.ones(x.shape[0])
    for _ in range(_BISECTIONS):
        mid = 0.5 * (low + high)
        same = (fundamental(signs, sorted_angles(move(x, mid[:, None]))) > target) == side0
        low = np.where(same, mid, low)
        high = np.where(same, high, mid)
    a = np.where(side0 == side1, 0.0, high)
    return sorted_angles(move(x, a[:, None]))


def _toward_90(x: Array, a: Array | float) -> Array:
    return np.asarray(90.0 - a * (90.0 - x))


def _toward_0(x: Array, a: Array | float) -> Array:
    return np.asarray(a * x)


def level_angles(key: str, U: Array) -> Array:
    """The angles of the ``"level"`` variant of RCM30-RCM35 (``key``): sorted, then moved to the fundamental ``m u``."""
    modulation_index, signs = _signs(key)
    x = sorted_angles(U[:, :-1])
    target = modulation_index * np.clip(U[:, -1], 0.0, 1.0)
    level = fundamental(signs, x)
    out = x.copy()
    down = level > target
    if down.any():
        out[down] = _bisect(signs, x[down], target[down], _toward_90)
    up = (level < target) & (float(np.sum(signs)) > target)
    if up.any():
        out[up] = _bisect(signs, x[up], target[up], _toward_0)
    return out


def cumulative_angles(U: Array) -> Array:
    """The angles of the ``"cumulative"`` variant (Wang et al.'s Eqs. 18-21): shares ``t_1..t_D``, then the total."""
    t, total = U[:, :-1], U[:, -1]
    d = t.shape[1]
    shares = np.zeros(t.shape[0])
    for i in range(d):  # sum t_i from the first share to the last
        shares = shares + t[:, i]
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where((shares > 0.0)[:, None], (total / shares)[:, None] * t, (total / d)[:, None])
    scaled[:, 1:] = scaled[:, 1:] + GAP
    # Rounding can lift the last angle above 90 by about 1e-13 at the largest total; the gap keeps 1e-8 to spare.
    return np.asarray(np.minimum(np.cumsum(scaled, axis=1), 90.0))


class SOPWMReformulation(RCMReformulation):
    """RCM30-RCM35 with increasing angles by construction; ``variant`` is ``"cumulative"``, ``"sorted"`` or ``"level"``.

    The decoding never fails. The constraints are the original's ordering constraints, which always hold,
    and the decoding-failure constraint, which is 0.
    """

    variant: str

    def __init__(self, number: int, *, variant: str = "cumulative") -> None:
        if not 30 <= number <= 35:
            raise ValueError(f"No SOPWM reformulation of RCM{number}; they are RCM30 to RCM35.")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r} of the SOPWM reformulation; use one of {', '.join(VARIANTS)}.")
        d = len(_PATTERNS[f"rcm{number:02d}"][1])
        if variant == "cumulative":
            lower, upper = np.zeros(d + 1), np.append(np.ones(d), 90.0 - (d - 1) * GAP)
        else:
            lower, upper = np.full(d, -ANGLE_SNAP), np.full(d, 90.0 + ANGLE_SNAP)
            if variant == "level":
                lower, upper = np.append(lower, -LEVEL_SNAP), np.append(upper, 1.0 + LEVEL_SNAP)
        super().__init__(number, lower, upper, d - 1)
        self.variant = variant

    def _solve(self, U: Array) -> tuple[Array, Array]:
        if self.variant == "cumulative":
            X = cumulative_angles(U)
        elif self.variant == "sorted":
            X = sorted_angles(U)
        else:
            X = level_angles(self.original.key, U)
        return X, np.zeros(U.shape[0])


__all__ = [
    "ANGLE_SNAP",
    "GAP",
    "LEVEL_SNAP",
    "SOPWMReformulation",
    "VARIANTS",
    "cumulative_angles",
    "fundamental",
    "level_angles",
    "sorted_angles",
]
