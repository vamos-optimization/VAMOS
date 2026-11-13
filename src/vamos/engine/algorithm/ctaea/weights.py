"""Weight vectors of C-TAEA, as jMetal's ``UniformWeightVectors`` generates them.

jMetal's port of C-TAEA (pull request jMetal/jMetal#524) generates its weight vectors with the
normal-boundary intersection of Das and Dennis in up to two layers, the default method of the
reference code's ``UniformPoint``:

- the outer layer has the largest number of divisions ``H1`` whose lattice has at most the
  requested number of vectors ``N`` (at least one division);
- if ``H1`` is smaller than the number of objectives ``M``, an inner layer is added with the
  largest number of divisions ``H2`` such that both layers have at most ``N`` vectors, and it is
  shrunk by half towards the centre of the simplex (``w / 2 + 1 / (2 M)``);
- every component is raised to at least ``1e-6``, so that the Tchebycheff function never
  divides by zero.

The rows of each layer are in ascending lexicographic order of their integer coordinates, the
outer layer first, and every coordinate is ``i / H`` computed as a single division, as in
jMetal. The number of vectors, which may be smaller than ``N`` (91 for 100 and three
objectives), is C-TAEA's population size. VAMOS's :func:`~vamos.engine.reference_directions.das_dennis`
lists the same lattice in the same order but renormalizes every row by its sum, which can
change a coordinate in its last bit, and it has neither the inner layer nor the floor, so C-TAEA
uses this function instead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vamos.engine.reference_directions import lattice_size

FloatArray = NDArray[np.float64]

MINIMUM_COMPONENT = 1e-6
"""Lower bound of the components of the weight vectors, as in jMetal's port."""


def _simplex_layer(n_obj: int, divisions: int) -> FloatArray:
    """The complete lattice with ``divisions`` divisions, rows in ascending lexicographic order."""
    rows: list[tuple[int, ...]] = []

    def fill(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == n_obj - 1:
            rows.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            fill((*prefix, value), remaining - value)

    fill((), divisions)
    return np.asarray(np.asarray(rows, dtype=float) / divisions, dtype=float)


def layer_divisions(requested: int, n_obj: int) -> tuple[int, int]:
    """The divisions ``(H1, H2)`` of the outer and inner layers; ``H2 = 0`` means no inner layer."""
    if requested < 1:
        raise ValueError(f"C-TAEA needs at least one requested weight vector, got {requested}.")
    if n_obj < 2:
        raise ValueError(f"C-TAEA needs at least two objectives, got {n_obj}.")
    outer = 1
    while lattice_size(n_obj, outer + 1) <= requested:
        outer += 1
    inner = 0
    if outer < n_obj:
        outer_count = lattice_size(n_obj, outer)
        while outer_count + lattice_size(n_obj, inner + 1) <= requested:
            inner += 1
    return outer, inner


def weight_vector_count(requested: int, n_obj: int) -> int:
    """The number of weight vectors, which is C-TAEA's population size, for ``requested`` vectors and ``n_obj`` objectives.

    It is at most ``requested`` unless ``requested`` is smaller than ``n_obj``: one division per
    objective already gives ``n_obj`` vectors.
    """
    outer, inner = layer_divisions(requested, n_obj)
    return lattice_size(n_obj, outer) + (lattice_size(n_obj, inner) if inner else 0)


def ctaea_weight_vectors(requested: int, n_obj: int) -> FloatArray:
    """C-TAEA's weight vectors for ``requested`` vectors and ``n_obj`` objectives, one per row (see the module)."""
    outer, inner = layer_divisions(requested, n_obj)
    layers = [_simplex_layer(n_obj, outer)]
    if inner:
        shrink = 0.5
        layers.append((1.0 - shrink) / n_obj + _simplex_layer(n_obj, inner) * shrink)
    return np.asarray(np.maximum(np.vstack(layers), MINIMUM_COMPONENT), dtype=float)


__all__ = ["MINIMUM_COMPONENT", "ctaea_weight_vectors", "layer_divisions", "weight_vector_count"]
