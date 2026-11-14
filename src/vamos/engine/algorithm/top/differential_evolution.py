"""The differential evolution operators of ToP.

jMetal's port of ToP (pull request jMetal/jMetal#525, class ``ToPDifferentialEvolution``) has the
three operators of PlatEMO's implementation: DE/current-to-rand/1 and DE/rand-to-best/1/bin for
phase 1, and DE/rand/1/bin for phase 2. Each one copies a base vector and replaces each of its
variables, with probability CR (a uniform draw lower than CR), by the value of the mutant vector;
unlike VAMOS's ``de`` crossover and jMetal's ``DifferentialEvolutionCrossover``, no variable is
forced to come from the mutant (there is no ``j_rand``), so with CR < 1 a trial vector can be a
copy of its base vector. ToP then clamps the trial vector to the bounds (VAMOS's repair, ``clip``
by default) and mutates it.

The functions take the vectors as rows of arrays, one row per trial vector, and compute each
variable with jMetal's operations in jMetal's order, so the floating-point values are jMetal's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def current_to_rand_1(current: NDArray[Any], r1: NDArray[Any], r2: NDArray[Any], r3: NDArray[Any], f: float) -> FloatArray:
    """Mutant vectors of DE/current-to-rand/1, ``x_i + F (x_r1 - x_i) + F (x_r2 - x_r3)``, whose base vector is ``x_i``."""
    x = np.asarray(current, dtype=float)
    return np.asarray(x + f * (np.asarray(r1, dtype=float) - x) + f * (np.asarray(r2, dtype=float) - np.asarray(r3, dtype=float)))


def rand_to_best_1(best: NDArray[Any], r1: NDArray[Any], r2: NDArray[Any], r3: NDArray[Any], f: float) -> FloatArray:
    """Mutant vectors of DE/rand-to-best/1, ``x_r1 + F (x_best - x_r1) + F (x_r2 - x_r3)``, whose base vector is ``x_r1``."""
    base = np.asarray(r1, dtype=float)
    return np.asarray(base + f * (np.asarray(best, dtype=float) - base) + f * (np.asarray(r2, dtype=float) - np.asarray(r3, dtype=float)))


def rand_1(base: NDArray[Any], second: NDArray[Any], third: NDArray[Any], f: float) -> FloatArray:
    """Mutant vectors of DE/rand/1, ``x_1 + F (x_2 - x_3)``, whose base vector is ``x_1``."""
    x = np.asarray(base, dtype=float)
    return np.asarray(x + f * (np.asarray(second, dtype=float) - np.asarray(third, dtype=float)))


def binomial_crossover(base: NDArray[Any], mutant: NDArray[Any], draws: NDArray[Any], cr: float) -> FloatArray:
    """Trial vectors: the mutant's variable where its uniform draw is lower than ``cr``, the base vector's elsewhere.

    There is no ``j_rand``: every variable follows its own draw.
    """
    return np.asarray(np.where(np.asarray(draws, dtype=float) < cr, np.asarray(mutant, dtype=float), np.asarray(base, dtype=float)))


__all__ = ["binomial_crossover", "current_to_rand_1", "rand_1", "rand_to_best_1"]
