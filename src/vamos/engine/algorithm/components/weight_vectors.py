import io
import os
import re
import stat
import warnings
from numbers import Integral

import numpy as np

from vamos.engine.algorithm.components.vector_files import (
    VECTOR_SOURCE_FIELDS,
    _checked_digest,
    _load_weights,
    _validated_weights,
    bind_config_vector_files,
    bind_vector_file,
    describe_vector_file,
    existing_weight_file,
    read_weight_vectors,
    vector_digest,
    vector_source,
)
from vamos.engine.reference_directions import (
    FloatArray,
    das_dennis,
    largest_lattice_divisions,
    lattice_size,
    two_layer_lattice,
    two_layer_shared,
    two_layer_size,
)
from vamos.foundation.exceptions import ConfigurationError


def load_or_generate_weight_vectors(
    pop_size: int,
    n_obj: int,
    *,
    path: str | None = None,
    divisions: int | None = None,
    mode: str = "auto",
    inner_divisions: int | None = None,
    sha256: str | None = None,
) -> FloatArray:
    """
    The weight (reference) vectors of MOEA/D, NSGA-III and RVEA: those of a file, or a simplex-lattice design.

    With ``sha256``, the vectors read from the file must have that digest (:func:`vector_digest`), or
    :class:`~vamos.exceptions.ConfigurationError` is raised before any evaluation: a run records the digest of
    the vectors it read, so that replay never runs other vectors from a file that has changed.

    ``path`` names a file, read by :func:`read_weight_vectors` with whitespace or commas between the
    components, or a directory with jMetal's ``W{n_obj}D_{pop_size}.dat`` files. ``divisions`` gives the
    complete Das-Dennis lattice and, with ``inner_divisions``, Deb and Jain's two-layer lattice
    (:func:`~vamos.engine.reference_directions.two_layer_lattice`). A lattice is never truncated.

    mode="auto" (NSGA-III): the file, which must exist and may hold any number of vectors (NSGA-III checks
    that ``pop_size`` covers them); without one, the lattice of ``divisions``, or the largest lattice with at
    most ``pop_size`` vectors, so the result can have fewer rows than ``pop_size``.

    mode="exact" (RVEA): one vector per individual. The file must exist and hold ``pop_size`` vectors, and
    a lattice must have exactly ``pop_size`` vectors.

    mode="jmetalpy" (MOEA/D) enforces jMetalPy-style behavior:
    - n_obj == 2: uniform weights of size pop_size
    - n_obj > 2: load W{n_obj}D_{pop_size}.dat when available, otherwise
      generate an exact Das-Dennis lattice and reject incompatible sizes.

    Development after 1.4.0: mode "exact", directories and ``inner_divisions`` in every mode, and a path
    that does not exist raises :class:`~vamos.exceptions.ConfigurationError` in every mode. Mode "auto"
    no longer generates a lattice for a missing file and saves it there, and MOEA/D no longer creates a
    missing directory; it still saves the lattice in an existing directory that lacks the file.
    """
    if mode not in ("auto", "exact", "jmetalpy"):
        raise ValueError(f"Unknown weight vector mode {mode!r}; expected 'auto', 'exact' or 'jmetalpy'.")
    if inner_divisions is not None:
        return _two_layer_weights(pop_size, n_obj, path, divisions, inner_divisions, exact=mode != "auto")
    if mode == "jmetalpy":
        return _load_or_generate_jmetalpy_weights(pop_size, n_obj, path, divisions, sha256)
    if path:
        file = existing_weight_file(path, n_obj, pop_size)
        weights = _checked_digest(read_weight_vectors(file, n_obj), file, sha256)
        if mode == "exact" and weights.shape[0] != pop_size:
            raise ValueError(f"Weight file '{file}' holds {weights.shape[0]} vectors, but pop_size={pop_size} needs one per individual.")
        return weights
    return _generate_weights(pop_size, n_obj, divisions, strict=mode == "exact")


def _two_layer_weights(
    pop_size: int, n_obj: int, path: str | None, divisions: int | None, inner_divisions: int, *, exact: bool
) -> FloatArray:
    """Deb and Jain's two-layer lattice; with ``exact``, it must have ``pop_size`` vectors."""
    if path:
        raise ConfigurationError("Weight vectors come from a file or a lattice: inner_divisions excludes path.")
    if divisions is None:
        raise ConfigurationError("inner_divisions needs divisions, the boundary layer of the two-layer lattice.")
    count = two_layer_size(n_obj, int(divisions), int(inner_divisions))
    if exact and count != pop_size:
        raise ValueError(
            f"The two-layer lattice of divisions={divisions} and inner_divisions={inner_divisions} has {count} vectors "
            f"for n_obj={n_obj}, but pop_size={pop_size} needs one per individual."
        )
    check_two_layers(n_obj, int(divisions), int(inner_divisions))
    return two_layer_lattice(n_obj, int(divisions), int(inner_divisions))


def check_two_layers(n_obj: int, divisions: int, inner_divisions: int) -> None:
    """Raise :class:`~vamos.exceptions.ConfigurationError` when the inside layer repeats boundary directions.

    A repeated vector is a niche that never gets an individual of its own, and RVEA's angle to its nearest
    other vector would be zero. Development after 1.4.0: new.
    """
    shared = two_layer_shared(n_obj, divisions, inner_divisions)
    if shared:
        raise ConfigurationError(
            f"The two-layer lattice of divisions={divisions} and inner_divisions={inner_divisions} repeats {shared} "
            f"direction(s) for n_obj={n_obj}: the inside layer must add directions that the boundary layer does not have.",
            suggestion="Choose other divisions or inner_divisions: with inner_divisions < n_obj and divisions < 2 n_obj "
            "the layers share none (8 objectives: divisions=3, inner_divisions=2).",
        )


def _load_or_generate_jmetalpy_weights(
    pop_size: int, n_obj: int, path: str | None, divisions: int | None, sha256: str | None = None
) -> FloatArray:
    if n_obj < 2:
        return np.ones((pop_size, 1), dtype=float)
    if n_obj == 2:
        if divisions is not None and lattice_size(n_obj, int(divisions)) != pop_size:
            raise ValueError(
                f"MOEA/D weight vectors for divisions={divisions} require pop_size={lattice_size(n_obj, int(divisions))}, got {pop_size}."
            )
        vals = np.linspace(0.0, 1.0, pop_size, dtype=float)
        return np.column_stack([vals, 1.0 - vals])

    if path and not os.path.exists(path):
        # Development after 1.4.0: a path that does not exist is an error; MOEA/D used to create it as a directory.
        existing_weight_file(path, n_obj, pop_size)
    weight_path = _resolve_jmetalpy_weight_path(path, n_obj, pop_size)
    if weight_path is None and divisions is None and _exact_divisions(pop_size, n_obj) is None:
        # No complete lattice has pop_size vectors: use the jMetal weights VAMOS ships for it, if any.
        packaged = _packaged_weights(n_obj, pop_size)
        if packaged is not None:
            return packaged
    if weight_path is not None and os.path.exists(weight_path):
        weights = _checked_digest(read_weight_vectors(weight_path, n_obj), weight_path, sha256)
        if weights.shape[0] != pop_size:
            raise ValueError(f"Expected {pop_size} weight vectors in '{weight_path}', got {weights.shape[0]}.")
        return weights

    if sha256 is not None:
        # The digest records the vectors of a file that a run read: generating the lattice instead would run others.
        raise ConfigurationError(
            f"Weight vector file '{weight_path}' that this configuration records ({sha256}) does not exist.",
            suggestion="Restore the file that the run read, or remove sha256 from the configuration.",
        )
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
    inner_divisions: int | None = None,
    sha256: str | None = None,
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

    Development after 1.4.0: ``two_layer`` with its ``divisions`` and ``inner_divisions``. With ``sha256``,
    the digest that :func:`bind_vector_file` records with the file a run read, the description comes from
    the configuration alone, without the file: ``das_dennis`` when the digest is that of the lattice of
    ``pop_size`` vectors (a lattice that an earlier run saved), else ``file``.
    """
    if inner_divisions is not None:
        if divisions is None:
            raise ValueError("inner_divisions needs divisions, the boundary layer of the two-layer lattice.")
        count = two_layer_size(n_obj, int(divisions), int(inner_divisions))
        if count != pop_size:
            raise ValueError(f"The two-layer lattice for n_obj={n_obj} has {count} vectors, not pop_size={pop_size}.")
        check_two_layers(n_obj, int(divisions), int(inner_divisions))
        return {"source": "two_layer", "divisions": int(divisions), "inner_divisions": int(inner_divisions), "vectors": pop_size}
    if n_obj <= 2:
        return {"source": "uniform", "vectors": pop_size}
    if path and sha256 is not None:
        exact = _exact_divisions(pop_size, n_obj)
        if exact is not None and (divisions is None or int(divisions) == exact) and vector_digest(das_dennis(n_obj, exact)) == sha256:
            return {"source": "das_dennis", "divisions": exact, "vectors": pop_size}
        return {"source": "file", "file": os.path.basename(path), "vectors": pop_size}
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


def weighted_populations(n_obj: int, max_pop_size: int) -> tuple[int, ...]:
    """The populations, up to ``max_pop_size``, for which MOEA/D has weight vectors without options.

    Every population from 2 with one or two objectives; with more, the sizes of the complete Das-Dennis
    lattices and of the packaged jMetal files ``W{n_obj}D_{size}.dat``. This is
    :func:`vamos.algorithms.moead_populations`. Development after 1.4.0: new.
    """
    if isinstance(n_obj, bool) or not isinstance(n_obj, Integral) or n_obj < 1:
        raise ConfigurationError(f"n_obj must be a positive integer; got {n_obj!r}.")
    if isinstance(max_pop_size, bool) or not isinstance(max_pop_size, Integral) or max_pop_size < 1:
        raise ConfigurationError(f"max_pop_size must be a positive integer; got {max_pop_size!r}.")
    if n_obj <= 2:
        return tuple(range(2, int(max_pop_size) + 1))
    sizes: set[int] = set()
    divisions = 1
    while (size := lattice_size(int(n_obj), divisions)) <= max_pop_size:
        sizes.add(size)
        divisions += 1
    sizes.update(size for size in _packaged_sizes(int(n_obj)) if size <= max_pop_size and _packaged_weights(int(n_obj), size) is not None)
    return tuple(sorted(sizes))


def _packaged_sizes(n_obj: int) -> list[int]:
    """The populations of the jMetal weight files that VAMOS ships for ``n_obj`` objectives."""
    from importlib import resources

    pattern = re.compile(rf"W{n_obj}D_(\d+)\.dat")
    names = (entry.name for entry in resources.files("vamos.resources").joinpath("weights").iterdir())
    return sorted(int(match.group(1)) for name in names if (match := pattern.fullmatch(name)))


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


__all__ = [
    "VECTOR_SOURCE_FIELDS",
    "bind_config_vector_files",
    "bind_vector_file",
    "check_two_layers",
    "describe_vector_file",
    "describe_weight_vectors",
    "existing_weight_file",
    "largest_weighted_population",
    "load_or_generate_weight_vectors",
    "read_weight_vectors",
    "vector_digest",
    "vector_source",
    "weighted_populations",
]
