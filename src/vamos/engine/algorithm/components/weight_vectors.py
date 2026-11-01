import os
import warnings

import numpy as np

from vamos.engine.reference_directions import FloatArray, das_dennis, largest_lattice_divisions, lattice_size


def load_or_generate_weight_vectors(
    pop_size: int,
    n_obj: int,
    *,
    path: str | None = None,
    divisions: int | None = None,
    mode: str = "auto",
) -> FloatArray:
    """
    Load weight vectors from CSV if available, otherwise generate a simplex-lattice
    design and persist it when a path is provided.

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
        weights = _load_weights(path, delimiter=",")
        _assert_valid_weights(weights, n_obj)
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


def _load_weights(path: str, delimiter: str | None = ",") -> FloatArray:
    if delimiter is None:
        arr = np.loadtxt(path)
    else:
        arr = np.loadtxt(path, delimiter=delimiter)
    arr = np.atleast_2d(arr).astype(float, copy=False)
    return arr


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
        weight_path = _packaged_weight_file(n_obj, pop_size)
    if weight_path is not None and os.path.exists(weight_path):
        weights = _load_weights(weight_path, delimiter=None)
        _assert_valid_weights(weights, n_obj)
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


def _exact_divisions(pop_size: int, n_obj: int) -> int | None:
    try:
        return _choose_exact_divisions(pop_size, n_obj)
    except ValueError:
        return None


def _packaged_weight_file(n_obj: int, pop_size: int) -> str | None:
    from vamos.resources import weight_path

    try:
        return str(weight_path(f"W{n_obj}D_{pop_size}.dat"))
    except ValueError:
        return None


def _resolve_jmetalpy_weight_path(path: str | None, n_obj: int, pop_size: int) -> str | None:
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    filename = f"W{n_obj}D_{pop_size}.dat"
    return os.path.join(path, filename)


def _assert_valid_weights(weights: FloatArray, n_obj: int) -> None:
    if weights.ndim != 2:
        raise ValueError("Weight matrix must be 2D.")
    if weights.shape[1] != n_obj:
        raise ValueError(f"Expected weight vectors with {n_obj} columns, got {weights.shape[1]}.")
    if np.any(weights < 0.0):
        raise ValueError("Weight vectors must be non-negative.")
    rows_sum = weights.sum(axis=1)
    # Allow very small numerical drift
    if np.any(np.abs(rows_sum - 1.0) > 1e-6):
        raise ValueError("Each weight vector must sum to 1.")


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
