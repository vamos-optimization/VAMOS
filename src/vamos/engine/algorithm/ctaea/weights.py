"""Weight vectors of C-TAEA, as jMetal's ``UniformWeightVectors`` generates them, or those of a file or lattice.

jMetal's port of C-TAEA (pull request jMetal/jMetal#524) generates its weight vectors with the
normal-boundary intersection of Das and Dennis in up to two layers, the default method of
PlatEMO's ``UniformPoint``:

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

**Development after 1.5.2:** :func:`configured_weight_vectors` takes the vectors of C-TAEA's
``weight_vectors`` option instead, one per individual: a file (or a directory with jMetal's
``W{M}D_{N}.dat``), or the lattice of ``divisions`` (and ``inner_divisions``) computed as above.
The same floor applies to them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.weight_vectors import (
    check_two_layers,
    existing_weight_file,
    load_or_generate_weight_vectors,
    vector_source,
)
from vamos.engine.reference_directions import lattice_size
from vamos.foundation.exceptions import ConfigurationError

FloatArray = NDArray[np.float64]

MINIMUM_COMPONENT = 1e-6
"""Lower bound of the components of the weight vectors, as in jMetal's port; it applies to the vectors of a file too."""


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


def _layers(n_obj: int, outer: int, inner: int) -> FloatArray:
    """The outer lattice and, when ``inner`` is not 0, the inner one shrunk by half towards the centre, without the floor."""
    layers = [_simplex_layer(n_obj, outer)]
    if inner:
        shrink = 0.5
        layers.append((1.0 - shrink) / n_obj + _simplex_layer(n_obj, inner) * shrink)
    return np.asarray(np.vstack(layers), dtype=float)


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
    objective already gives ``n_obj`` vectors. With the ``weight_vectors`` option
    (:func:`configured_weight_vectors`), the population is ``pop_size`` instead.
    """
    outer, inner = layer_divisions(requested, n_obj)
    return lattice_size(n_obj, outer) + (lattice_size(n_obj, inner) if inner else 0)


def ctaea_weight_vectors(requested: int, n_obj: int) -> FloatArray:
    """C-TAEA's weight vectors for ``requested`` vectors and ``n_obj`` objectives, one per row (see the module)."""
    outer, inner = layer_divisions(requested, n_obj)
    return np.asarray(np.maximum(_layers(n_obj, outer, inner), MINIMUM_COMPONENT), dtype=float)


def has_weight_vector_source(source: object) -> bool:
    """Whether a ``weight_vectors`` option names a file or a lattice, so that C-TAEA runs one individual per vector.

    ``None``, and a mapping with neither ``path`` nor ``divisions``, leave C-TAEA's default vectors.
    Development after 1.5.2: new.
    """
    if isinstance(source, (str, os.PathLike)):
        return True
    return isinstance(source, Mapping) and (source.get("path") is not None or source.get("divisions") is not None)


def configured_weight_vectors(pop_size: int, n_obj: int, source: object = None) -> FloatArray:
    """The weight vectors of a C-TAEA run: the default ones, or those of its ``weight_vectors`` option.

    Without a source (see :func:`has_weight_vector_source`), these are
    ``ctaea_weight_vectors(pop_size, n_obj)``, whose number may be smaller than ``pop_size``. With
    one, C-TAEA runs one individual per vector, so there must be exactly ``pop_size`` of them:

    - ``path``: a file with one vector per line, or a directory with jMetal's
      ``W{n_obj}D_{pop_size}.dat`` (such as the files VAMOS ships in ``vamos/resources/weights``),
      read by :func:`~vamos.engine.algorithm.components.weight_vectors.load_or_generate_weight_vectors`,
      which checks ``sha256`` when it is given. The rows are used as they are, not renormalized;
    - ``divisions``: the complete Das-Dennis lattice, and with ``inner_divisions`` the inner layer
      shrunk by half, computed as C-TAEA's default vectors, so that ``divisions=12`` with
      ``pop_size=91`` and three objectives gives the default vectors of ``pop_size=91``, bit for bit.

    Every component is raised to at least :data:`MINIMUM_COMPONENT`, as the default vectors are:
    jMetal's files have components equal to 0 (39 in 36 of the rows of ``W3D_100.dat``), with which the
    Tchebycheff values of the archive updates would divide by zero. A missing path, a directory
    without the file, layers that repeat directions, and a number of vectors other than
    ``pop_size`` raise :class:`~vamos.exceptions.ConfigurationError` before any evaluation.

    Development after 1.5.2: new.
    """
    options = vector_source(source, field="weight_vectors")
    if not has_weight_vector_source(options):
        return ctaea_weight_vectors(pop_size, n_obj)
    assert options is not None
    path, divisions, inner = options.get("path"), options.get("divisions"), options.get("inner_divisions")
    if path is not None:
        file = existing_weight_file(str(path), n_obj, pop_size)
        sha256 = options.get("sha256")
        weights = load_or_generate_weight_vectors(pop_size, n_obj, path=file, sha256=None if sha256 is None else str(sha256))
        _check_count(f"Weight vector file '{file}' holds", int(weights.shape[0]), pop_size, n_obj)
    else:
        outer, inside = int(cast(int, divisions)), 0 if inner is None else int(cast(int, inner))
        lattice = f"divisions={outer}" if not inside else f"divisions={outer} and inner_divisions={inside}"
        # The size first, so that a lattice of another size is never generated.
        count = lattice_size(n_obj, outer) + (lattice_size(n_obj, inside) if inside else 0)
        _check_count(f"The lattice of {lattice} has", count, pop_size, n_obj)
        if inside:
            check_two_layers(n_obj, outer, inside)
        weights = _layers(n_obj, outer, inside)
    return np.asarray(np.maximum(weights, MINIMUM_COMPONENT), dtype=float)


def _check_count(origin: str, count: int, pop_size: int, n_obj: int) -> None:
    """Raise :class:`~vamos.exceptions.ConfigurationError` unless there is one weight vector per individual."""
    if count != pop_size:
        raise ConfigurationError(
            f"{origin} {count} weight vectors for {n_obj} objectives, but C-TAEA's weight_vectors need one per individual, "
            f"pop_size={pop_size}.",
            suggestion=f"Set pop_size to {count}, or give weight vectors with {pop_size} rows.",
        )


__all__ = [
    "MINIMUM_COMPONENT",
    "configured_weight_vectors",
    "ctaea_weight_vectors",
    "has_weight_vector_source",
    "layer_divisions",
    "weight_vector_count",
]
