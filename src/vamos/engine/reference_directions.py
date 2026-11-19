"""Das-Dennis simplex-lattice reference directions.

This leaf module imports no other VAMOS module, so algorithms, result archives
and subset selectors share one lattice without creating import cycles.
"""

from __future__ import annotations

from math import comb, lcm

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


def two_layer_size(n_obj: int, divisions: int, inner_divisions: int) -> int:
    """Return the number of directions of a two-layer lattice: both complete lattices.

    Development after 1.4.0: new.
    """
    return lattice_size(n_obj, divisions) + lattice_size(n_obj, inner_divisions)


def two_layer_lattice(n_obj: int, divisions: int, inner_divisions: int) -> FloatArray:
    """Return Deb and Jain's two-layer reference directions, one per row.

    With many objectives a lattice of few divisions has every direction on the boundary of the
    simplex, and one more division multiplies the directions. Deb and Jain (2014) add an inside
    layer instead: the boundary layer is the complete lattice of ``divisions`` and the
    inside layer the complete lattice of ``inner_divisions`` shrunk by half towards the centre of
    the simplex, ``w / 2 + 1 / (2 n_obj)``, as pymoo's multi-layer directions with scaling 0.5 and
    PlatEMO's ``UniformPoint`` do. The boundary rows come first, each layer in the order of
    :func:`das_dennis`. Eight objectives with ``divisions=3`` and ``inner_divisions=2`` give
    120 + 36 = 156 directions.

    The inside layer must add directions that the boundary layer does not have: two objectives with
    ``divisions=4`` and ``inner_divisions=2`` would repeat all three inside directions, and a repeated
    direction is a niche that never gets its own individual. Such settings raise ``ValueError`` (see
    :func:`two_layer_shared`).

    Development after 1.4.0: new.
    """
    shared = two_layer_shared(n_obj, divisions, inner_divisions)
    if shared:
        raise ValueError(
            f"The inside layer of inner_divisions={inner_divisions} repeats {shared} of the directions of the boundary layer "
            f"of divisions={divisions} for n_obj={n_obj}; choose layers that share no direction."
        )
    return np.asarray(np.vstack(_two_layers(n_obj, divisions, inner_divisions)), dtype=float)


def two_layer_shared(n_obj: int, divisions: int, inner_divisions: int) -> int:
    """Return how many directions of the inside layer the boundary layer already has.

    Every coordinate of either layer is a multiple of ``1 / lcm(divisions, 2 inner_divisions n_obj)``,
    so the layers are compared exactly on that grid. A two-layer lattice needs 0.

    Development after 1.4.0: new.
    """
    boundary, inside = _two_layers(n_obj, divisions, inner_divisions)
    grid = lcm(divisions, 2 * inner_divisions * n_obj)
    boundary_points = {tuple(row) for row in np.rint(boundary * grid).astype(np.int64).tolist()}
    return sum(tuple(row) in boundary_points for row in np.rint(inside * grid).astype(np.int64).tolist())


def _two_layers(n_obj: int, divisions: int, inner_divisions: int) -> tuple[FloatArray, FloatArray]:
    """The boundary layer and the inside layer, shrunk by half towards the centre of the simplex."""
    boundary = das_dennis(n_obj, divisions)
    inside = np.asarray(das_dennis(n_obj, inner_divisions) * 0.5 + 0.5 / n_obj, dtype=float)
    return boundary, inside


__all__ = ["das_dennis", "largest_lattice_divisions", "lattice_size", "two_layer_lattice", "two_layer_shared", "two_layer_size"]
