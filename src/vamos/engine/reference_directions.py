"""Das-Dennis simplex-lattice reference directions.

This leaf module imports no other VAMOS module, so algorithms, result archives
and subset selectors share one lattice without creating import cycles.
"""

from __future__ import annotations

from math import comb

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def lattice_size(n_obj: int, divisions: int) -> int:
    """Return the number of directions of a complete Das-Dennis lattice.

    A lattice with ``divisions`` divisions on the ``n_obj``-objective simplex has
    ``comb(divisions + n_obj - 1, n_obj - 1)`` directions.
    """
    if n_obj < 1:
        raise ValueError("n_obj must be >= 1")
    if divisions < 1:
        raise ValueError("divisions must be >= 1")
    return comb(divisions + n_obj - 1, n_obj - 1)


def largest_lattice_divisions(max_directions: int, n_obj: int) -> int:
    """Return the most divisions whose complete lattice has at most ``max_directions`` directions."""
    if n_obj < 2:
        raise ValueError("A Das-Dennis lattice needs n_obj >= 2.")
    if max_directions < n_obj:
        raise ValueError(f"The smallest Das-Dennis lattice for n_obj={n_obj} has {n_obj} directions, more than {max_directions}.")
    divisions = 1
    while lattice_size(n_obj, divisions + 1) <= max_directions:
        divisions += 1
    return divisions


def das_dennis(n_obj: int, divisions: int) -> FloatArray:
    """Return the complete Das-Dennis lattice, one direction per row.

    Each row has coordinates in ``{0, 1/divisions, ..., 1}`` summing to one, and
    every such point appears once, so the set is symmetric between objectives and
    contains each axis. Rows are in ascending lexicographic order of their
    integer coordinates.
    """
    count = lattice_size(n_obj, divisions)
    coords = np.empty((count, n_obj), dtype=float)
    current = [0] * n_obj
    row = 0

    def fill(depth: int, remaining: int) -> None:
        nonlocal row
        if depth == n_obj - 1:
            current[depth] = remaining
            coords[row] = current
            row += 1
            return
        for value in range(remaining + 1):
            current[depth] = value
            fill(depth + 1, remaining - value)

    fill(0, divisions)
    weights = np.clip(coords / divisions, 0.0, 1.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


__all__ = ["das_dennis", "largest_lattice_divisions", "lattice_size"]
