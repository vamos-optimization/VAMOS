"""The exponential crossover of CMOEA-MS's DE variation, ``RAND_1_EXP``.

CMOEA-MS's DE variation creates, for the i-th solution ``x_i`` of the population, the mutant
``x_i + F (x_r1 - x_r2)`` and crosses it with ``x_i``. With ``RAND_1_BIN``, PlatEMO's variation,
the binomial crossover takes each variable from the mutant with probability ``CR`` and one
variable at random always, which is VAMOS's ``de`` operator. With ``RAND_1_EXP`` the exponential
crossover of jMetal's ``DifferentialEvolutionCrossover`` (jMetal 7.10) takes a block of
consecutive variables from the mutant: it starts at a random variable, takes it, and takes the
next one, cyclically, while a uniform draw is lower than ``CR``, up to all of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def exponential_crossover(base: NDArray[Any], mutant: NDArray[Any], start: NDArray[Any], draws: NDArray[Any], cr: float) -> FloatArray:
    """The trial vectors of the exponential crossover of each row of ``base`` with the same row of ``mutant``.

    Row ``i`` takes from the mutant its variable ``start[i]`` and the following ones, cyclically,
    while the draws ``draws[i, 0]``, ``draws[i, 1]``, ... are lower than ``cr``, up to all the
    variables; the other variables come from ``base``. jMetal tests the ``n``-th draw too before it
    stops, so a row consumes the draws up to the first one that is not lower than ``cr``, or ``n``.
    """
    base_values = np.asarray(base, dtype=float)
    mutant_values = np.asarray(mutant, dtype=float)
    n_rows, n_var = base_values.shape
    uniforms = np.asarray(draws, dtype=float).reshape(n_rows, -1)
    first = np.asarray(start, dtype=int).reshape(n_rows)
    continued = uniforms[:, : n_var - 1] < float(cr)
    length = 1 + np.cumprod(continued, axis=1).sum(axis=1)
    offset = (np.arange(n_var)[None, :] - first[:, None]) % n_var
    return np.asarray(np.where(offset < length[:, None], mutant_values, base_values), dtype=float)


__all__ = ["exponential_crossover"]
