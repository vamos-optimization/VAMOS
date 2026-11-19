import io
import os
import stat
import warnings

import numpy as np

from vamos.engine.reference_directions import FloatArray, das_dennis, largest_lattice_divisions, lattice_size
from vamos.foundation.exceptions import ConfigurationError

# How far a weight file may depart from the simplex through rounding: jMetal's files round each component, so
# their rows sum to 1 only within about 2e-6, and a few components that should be zero are about -5e-17.
_ROUNDING_TOLERANCE = 1e-5


def read_weight_vectors(path: str, n_obj: int) -> FloatArray:
    """Read the weight vectors of a file, one per line, with ``n_obj`` components each.

    The components are separated by whitespace or by commas: jMetal's weight files use spaces, except
    ``W4D_165.dat``, and NSGA-III's reference-direction files have been comma-separated. Every component
    must be non-negative and every row must sum to 1, both within the rounding of jMetal's files (1e-5);
    a negative component within that rounding reads as 0. The rows are returned as they are, not
    renormalized. A path that is not a file raises :class:`~vamos.exceptions.ConfigurationError`.

    **Development after 1.4.0:** commas and whitespace in every file, and the rounding tolerance of 1e-5
    (1e-6 before, without the negative rounding).
    """
    if not os.path.isfile(path):
        raise ConfigurationError(
            f"Weight vector file '{path}' does not exist or is not a file.",
            suggestion="Pass the path of an existing file, one weight vector per line.",
        )
    return _validated_weights(_load_weights(path), n_obj)


def load_or_generate_weight_vectors(
    pop_size: int,
    n_obj: int,
    *,
    path: str | None = None,
    divisions: int | None = None,
    mode: str = "auto",
) -> FloatArray:
    """
    Load weight vectors from a file if available, otherwise generate a simplex-lattice
    design and persist it when a path is provided. Files are read by
    :func:`read_weight_vectors`, with whitespace or commas between the components.

    mode="auto" generates the complete Das-Dennis lattice: with ``divisions``, all of
    its vectors; without, the largest lattice with at most ``pop_size`` vectors. The
    lattice is never truncated, so the result can have fewer rows than ``pop_size``.

    mode="jmetalpy" enforces jMetalPy-style behavior:
    - n_obj == 2: uniform weights of size pop_size
    - n_obj > 2: load W{n_obj}D_{pop_size}.dat when available, otherwise
      generate an exact Das-Dennis lattice and reject incompatible sizes.
    """
    if mode == "jmetalpy":
        return _load_or_generate_jmetalpy_weights(pop_size, n_obj, path, divisions)
    if path and os.path.exists(path):
        weights = read_weight_vectors(path, n_obj)
        if weights.shape[0] != pop_size:
            raise ValueError(
                f"Weight file '{path}' contains {weights.shape[0]} vectors but pop_size={pop_size} requires exactly that many."
            )
        return weights

    weights = _generate_weights(pop_size, n_obj, divisions)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, weights, delimiter=",")

    return weights


def _load_weights(path: str) -> FloatArray:
    """The rows of a weight file whose components are separated by whitespace or commas."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    arr = np.loadtxt(io.StringIO(text.replace(",", " ")), ndmin=2)
    return np.asarray(arr, dtype=float)


def _load_or_generate_jmetalpy_weights(pop_size: int, n_obj: int, path: str | None, divisions: int | None) -> FloatArray:
    if n_obj < 2:
        return np.ones((pop_size, 1), dtype=float)
    if n_obj == 2:
        if divisions is not None and lattice_size(n_obj, int(divisions)) != pop_size:
            raise ValueError(
                f"MOEA/D weight vectors for divisions={divisions} require pop_size={lattice_size(n_obj, int(divisions))}, got {pop_size}."
            )
        vals = np.linspace(0.0, 1.0, pop_size, dtype=float)
        return np.column_stack([vals, 1.0 - vals])

    weight_path = _resolve_jmetalpy_weight_path(path, n_obj, pop_size)
    if weight_path is None and divisions is None and _exact_divisions(pop_size, n_obj) is None:
        # No complete lattice has pop_size vectors: use the jMetal weights VAMOS ships for it, if any.
        packaged = _packaged_weights(n_obj, pop_size)
        if packaged is not None:
            return packaged
    if weight_path is not None and os.path.exists(weight_path):
        weights = read_weight_vectors(weight_path, n_obj)
        if weights.shape[0] != pop_size:
            raise ValueError(f"Expected {pop_size} weight vectors in '{weight_path}', got {weights.shape[0]}.")
        return weights

    weights = _generate_weights(pop_size, n_obj, divisions=divisions, strict=True)
    if weight_path is not None:
        saved = False

        # Try to write, but fall back gracefully if path is read-only
        try:
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            np.savetxt(weight_path, weights)
            saved = True
        except (PermissionError, OSError) as e:
            warnings.warn(
                f"Could not save generated weights to '{weight_path}': {e}. Continuing with in-memory weights.",
                RuntimeWarning,
                stacklevel=2,
            )

        if saved:
            warnings.warn(
                f"Weight vector file not found: {weight_path}. Generated and saved simplex-lattice weights; "
                "results may differ from jMetalPy reference weights.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Weight vector file not found: {weight_path}. Generated simplex-lattice weights in memory; "
                "results may differ from jMetalPy reference weights.",
                RuntimeWarning,
                stacklevel=2,
            )
        return weights
    return weights


def describe_weight_vectors(
    pop_size: int,
    n_obj: int,
    *,
    path: str | None = None,
    divisions: int | None = None,
) -> dict[str, object]:
    """Where MOEA/D's weight vectors for ``pop_size`` come from, as a run artifact records it.

    The same choice as ``load_or_generate_weight_vectors(..., mode="jmetalpy")``, without writing a
    file: ``uniform`` with one or two objectives, ``file`` (a file of ``path``, by its name),
    ``packaged_file`` (the jMetal file ``W{n_obj}D_{pop_size}.dat`` that VAMOS ships), or
    ``das_dennis`` with its ``divisions``. ``vectors`` is the number of weight vectors, which is the
    population. A size without weights raises ``ValueError``, as loading them would.

    A run whose ``path`` lacks the file generates the lattice and saves it there, so a file that holds
    exactly that lattice is still ``das_dennis``: the description does not depend on whether an earlier
    run has written it.
    """
    if n_obj <= 2:
        return {"source": "uniform", "vectors": pop_size}
    weight_path = _resolve_jmetalpy_weight_path(path, n_obj, pop_size)
    if weight_path is None and divisions is None and _exact_divisions(pop_size, n_obj) is None:
        if _packaged_weights(n_obj, pop_size) is not None:
            return {"source": "packaged_file", "file": f"W{n_obj}D_{pop_size}.dat", "vectors": pop_size}
    if weight_path is not None and os.path.exists(weight_path) and not _holds_saved_lattice(weight_path, pop_size, n_obj, divisions):
        return {"source": "file", "file": os.path.basename(weight_path), "vectors": pop_size}
    if divisions is None:
        divisions = _choose_exact_divisions(pop_size, n_obj)
    elif lattice_size(n_obj, int(divisions)) != pop_size:
        raise ValueError(f"Das-Dennis simplex lattice for n_obj={n_obj}, divisions={divisions} does not have pop_size={pop_size} vectors.")
    return {"source": "das_dennis", "divisions": int(divisions), "vectors": pop_size}


def _holds_saved_lattice(path: str, pop_size: int, n_obj: int, divisions: int | None) -> bool:
    """Whether the weight file holds exactly the Das-Dennis lattice that a run saves there when the file is missing.

    Verification rebuilds the description from a stored configuration, so the path is untrusted: only a
    regular file no larger than the lattice's ``np.savetxt`` text is read, and a FIFO is never opened.
    """
    exact = _exact_divisions(pop_size, n_obj)
    if exact is None or (divisions is not None and int(divisions) != exact):
        return False
    limit = pop_size * (n_obj * 26 + 2)  # "%.18e" values of at most 25 characters, a separator, either line ending
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        info = os.stat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            return False
        descriptor = os.open(path, flags)
        try:
            data = os.read(descriptor, limit + 1)
        finally:
            os.close(descriptor)
        weights = np.atleast_2d(np.loadtxt(io.BytesIO(data))) if len(data) <= limit else None
    except (OSError, ValueError):
        return False
    return bool(weights is not None and weights.shape == (pop_size, n_obj) and np.array_equal(weights, das_dennis(n_obj, exact)))


def largest_weighted_population(pop_size: int, n_obj: int) -> int:
    """Return the largest population, at most ``pop_size``, for which MOEA/D has weight vectors.

    MOEA/D needs one weight vector per individual. With one or two objectives every population
    has them. With more, its weights are the complete Das-Dennis lattice with exactly that many
    vectors or the packaged jMetal file ``W{n_obj}D_{size}.dat``, so a size with neither gives way
    to the next smaller one that has them. A lattice is never truncated.
    """
    if n_obj <= 2:
        return pop_size
    if pop_size < n_obj:
        raise ValueError(f"The smallest Das-Dennis lattice for n_obj={n_obj} has {n_obj} vectors, more than pop_size={pop_size}.")
    size = pop_size
    while _exact_divisions(size, n_obj) is None and _packaged_weights(n_obj, size) is None:
        size -= 1
    return size


def _exact_divisions(pop_size: int, n_obj: int) -> int | None:
    try:
        return _choose_exact_divisions(pop_size, n_obj)
    except ValueError:
        return None


def _packaged_weights(n_obj: int, pop_size: int) -> FloatArray | None:
    """The jMetal weights VAMOS ships for ``pop_size`` vectors, renormalized; ``None`` without a matching file.

    jMetal's files round each component, so their rows sum to 1 only within about 2e-6.
    """
    from vamos.resources import weight_path

    try:
        path = weight_path(f"W{n_obj}D_{pop_size}.dat")
    except ValueError:
        return None
    weights = _load_weights(str(path))
    if weights.shape != (pop_size, n_obj):
        return None
    weights = _validated_weights(weights, n_obj)
    normalized: FloatArray = weights / weights.sum(axis=1, keepdims=True)
    return normalized


def _resolve_jmetalpy_weight_path(path: str | None, n_obj: int, pop_size: int) -> str | None:
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    filename = f"W{n_obj}D_{pop_size}.dat"
    return os.path.join(path, filename)


def _validated_weights(weights: FloatArray, n_obj: int) -> FloatArray:
    """``weights`` with the negative rounding of a zero component set to 0, or ``ValueError`` off the simplex."""
    if weights.ndim != 2:
        raise ValueError("Weight matrix must be 2D.")
    if weights.shape[1] != n_obj:
        raise ValueError(f"Expected weight vectors with {n_obj} columns, got {weights.shape[1]}.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weight vectors must be finite.")
    if np.any(weights < -_ROUNDING_TOLERANCE):
        raise ValueError("Weight vectors must be non-negative.")
    if np.any(np.abs(weights.sum(axis=1) - 1.0) > _ROUNDING_TOLERANCE):
        raise ValueError("Each weight vector must sum to 1.")
    return np.asarray(np.maximum(weights, 0.0), dtype=float)


def _generate_weights(pop_size: int, n_obj: int, divisions: int | None, *, strict: bool = False) -> FloatArray:
    if n_obj < 2:
        # Degenerate single-objective: return uniform weights.
        return np.ones((pop_size, 1), dtype=float)
    if divisions is None:
        divisions = _choose_exact_divisions(pop_size, n_obj) if strict else largest_lattice_divisions(pop_size, n_obj)
    elif strict:
        max_vectors = lattice_size(n_obj, divisions)
        if max_vectors != pop_size:
            raise ValueError(
                f"Das-Dennis simplex lattice for n_obj={n_obj}, divisions={divisions} has {max_vectors} vectors, "
                f"but pop_size={pop_size}. Choose pop_size={max_vectors} or compatible divisions."
            )
    return das_dennis(n_obj, divisions)


def _choose_exact_divisions(pop_size: int, n_obj: int) -> int:
    divisions = 1
    while True:
        count = lattice_size(n_obj, divisions)
        if count == pop_size:
            return divisions
        if count > pop_size:
            raise ValueError(
                f"No Das-Dennis simplex lattice has exactly pop_size={pop_size} vectors for n_obj={n_obj}. "
                "Choose a compatible pop_size or provide an explicit weight vector file."
            )
        divisions += 1
