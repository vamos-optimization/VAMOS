"""Files of weight and reference vectors: reading, options, content binding and run records.

MOEA/D's ``weight_vectors``, NSGA-III's ``reference_directions`` and RVEA's ``reference_vectors`` share
these helpers through :mod:`vamos.engine.algorithm.components.weight_vectors`.

Development after 1.4.0: new module.
"""

import dataclasses
import hashlib
import io
import os
import re
from collections.abc import Mapping
from difflib import get_close_matches
from numbers import Integral
from typing import Any

import numpy as np

from vamos.engine.reference_directions import FloatArray
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


_VECTOR_SOURCE_KEYS = ("path", "divisions", "inner_divisions", "sha256")
_DIGEST_PATTERN = r"sha256:[0-9a-f]{64}"


def vector_source(value: object, *, field: str = "reference vectors") -> dict[str, int | str | None] | None:
    """The options of a weight- or reference-vector source, checked.

    MOEA/D's ``weight_vectors``, NSGA-III's ``reference_directions`` and RVEA's ``reference_vectors``
    take ``None``, a path, or a mapping with these keys:

    - ``path``: a file with one vector per line, or a directory with jMetal's ``W{n_obj}D_{pop_size}.dat``
      files;
    - ``divisions``: the complete Das-Dennis lattice of that many divisions;
    - ``inner_divisions``: with ``divisions``, Deb and Jain's two-layer lattice (see
      :func:`~vamos.engine.reference_directions.two_layer_lattice`). It needs ``divisions`` and excludes ``path``;
    - ``sha256``: with ``path``, the digest of the vectors the file must hold (:func:`vector_digest`). A run
      records it, with the file that ``path`` names, so that replay runs the same vectors
      (:func:`bind_vector_file`).

    A path becomes ``{"path": path, "divisions": None}``. A mapping keeps the keys it has, so a configuration
    serializes as before, except ``inner_divisions=None``, which is left out. An unknown key, a path that is
    not a string, a count that is not a positive integer, or ``inner_divisions`` without ``divisions`` or with
    ``path`` raises :class:`~vamos.exceptions.ConfigurationError`.

    Development after 1.4.0: new; VAMOS 1.4.0 ignored unknown keys.
    """
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return {"path": os.fspath(value), "divisions": None}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} must be a mapping with path, divisions and inner_divisions, or a path; got {value!r}.")
    unknown = [key for key in value if key not in _VECTOR_SOURCE_KEYS]
    if unknown:
        hints = "".join(
            f" Did you mean '{match[0]}'?" for key in unknown if (match := get_close_matches(str(key), _VECTOR_SOURCE_KEYS, n=1))
        )
        raise ConfigurationError(f"Unknown {field} key(s): {', '.join(map(repr, unknown))}.{hints}")
    source: dict[str, int | str | None] = dict(value)
    path = source.get("path")
    if path is not None:
        if not isinstance(path, (str, os.PathLike)):
            raise ConfigurationError(f"{field} path must be a string path or None; got {path!r}.")
        source["path"] = str(os.fspath(path))
    for key in ("divisions", "inner_divisions"):
        count = source.get(key)
        if count is not None and (isinstance(count, bool) or not isinstance(count, Integral) or int(count) < 1):
            raise ConfigurationError(f"{field} {key} must be a positive integer or None; got {count!r}.")
    digest = source.get("sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(_DIGEST_PATTERN, digest)):
        raise ConfigurationError(f"{field} sha256 must be 'sha256:' and 64 lower-case hexadecimal digits; got {digest!r}.")
    if digest is not None and source.get("path") is None:
        raise ConfigurationError(f"{field} sha256 is the digest of the vectors of a file: it needs path.")
    if source.get("inner_divisions") is None:
        source.pop("inner_divisions", None)
    elif source.get("divisions") is None:
        raise ConfigurationError(f"{field} inner_divisions needs divisions, the boundary layer of the two-layer lattice.")
    elif source.get("path") is not None:
        raise ConfigurationError(f"{field} takes a file or a lattice: inner_divisions excludes path.")
    return source


def vector_digest(vectors: FloatArray) -> str:
    """The digest that binds a run to the vectors it read: ``sha256:`` and the SHA-256 of their shape and values.

    The values are the float64 vectors that :func:`read_weight_vectors` returns, so a file rewritten with
    other separators or number formats keeps its digest. Development after 1.4.0: new.
    """
    array = np.ascontiguousarray(vectors, dtype="<f8")
    rows, columns = array.shape
    return "sha256:" + hashlib.sha256(f"{rows}x{columns}:".encode("ascii") + array.tobytes()).hexdigest()


def _checked_digest(weights: FloatArray, file: str, sha256: str | None) -> FloatArray:
    if sha256 is not None and vector_digest(weights) != sha256:
        raise ConfigurationError(
            f"Weight vector file '{file}' no longer holds the vectors this configuration records ({sha256}).",
            suggestion="Restore the file that the run read, or remove sha256 from the configuration to run the file's current vectors.",
        )
    return weights


def bind_vector_file(source: object, n_obj: int, pop_size: int) -> object:
    """``source`` with its ``path`` resolved to the file that a run reads and the ``sha256`` of its vectors.

    A directory becomes its ``W{n_obj}D_{pop_size}.dat``. A source without a path, one that already has a
    digest, or a path whose file does not exist (MOEA/D then generates the lattice; NSGA-III and RVEA raise
    when the run starts) is returned unchanged. The resolved specification then records which file the run
    read and the digest of its vectors: verification rebuilds the record from them without the file, and
    replay refuses a file whose vectors have changed. Development after 1.4.0: new.
    """
    if not isinstance(source, Mapping) or not source.get("path") or source.get("sha256") is not None:
        return source
    path = str(source["path"])
    file = path if os.path.isfile(path) else os.path.join(path, f"W{n_obj}D_{pop_size}.dat")
    if not os.path.isfile(file):
        return source
    return {**source, "path": file, "sha256": vector_digest(read_weight_vectors(file, n_obj))}


VECTOR_SOURCE_FIELDS = ("weight_vectors", "reference_directions", "reference_vectors")
"""The configuration fields that name the vectors of MOEA/D, NSGA-III and RVEA."""


def bind_config_vector_files(config: Any, n_obj: int) -> Any:
    """A configuration dataclass with each vector file bound by :func:`bind_vector_file`.

    MOEA/D's ``weight_vectors`` with one or two objectives are uniform and read no file, so they stay as
    they are. Development after 1.4.0: new.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        return config
    pop_size = getattr(config, "pop_size", None)
    if not isinstance(pop_size, int) or isinstance(pop_size, bool):
        return config
    changes: dict[str, object] = {}
    for name in VECTOR_SOURCE_FIELDS:
        if not hasattr(config, name) or (name == "weight_vectors" and n_obj <= 2):
            continue
        value = getattr(config, name)
        bound = bind_vector_file(value, n_obj, pop_size)
        if bound is not value:
            changes[name] = bound
    return dataclasses.replace(config, **changes) if changes else config


def existing_weight_file(path: str, n_obj: int, pop_size: int) -> str:
    """The weight file that ``path`` names: itself, or ``W{n_obj}D_{pop_size}.dat`` in the directory ``path``.

    A path that does not exist, or a directory without that file, raises
    :class:`~vamos.exceptions.ConfigurationError`. Development after 1.4.0: new.
    """
    if os.path.isfile(path):
        return path
    name = f"W{n_obj}D_{pop_size}.dat"
    if os.path.isdir(path):
        file = os.path.join(path, name)
        if os.path.isfile(file):
            return file
        raise ConfigurationError(
            f"Weight vector directory '{path}' has no {name} for {n_obj} objectives and pop_size={pop_size}.",
            suggestion="Add the file, name the file itself, or choose a population that the directory has a file for.",
        )
    raise ConfigurationError(
        f"Weight vector path '{path}' does not exist.",
        suggestion="Pass an existing file, one vector per line, or a directory with jMetal's W<objectives>D_<size>.dat files.",
    )


def _load_weights(path: str) -> FloatArray:
    """The rows of a weight file whose components are separated by whitespace or commas."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    arr = np.loadtxt(io.StringIO(text.replace(",", " ")), ndmin=2)
    return np.asarray(arr, dtype=float)


def describe_vector_file(path: str, *, vectors: int | None = None) -> dict[str, object]:
    """The record of NSGA-III's or RVEA's vectors from ``path``: ``{"source": "file", "file": name}``.

    ``name`` is the name of the file in ``path``, which :func:`bind_vector_file` resolves when the run starts,
    and ``vectors``, when given, the number of vectors. The record comes from the configuration alone:
    verification rebuilds it from a stored configuration without the file, which may be absent or on
    another machine. Development after 1.4.0: new.
    """
    record: dict[str, object] = {"source": "file", "file": os.path.basename(os.path.normpath(path))}
    if vectors is not None:
        record["vectors"] = vectors
    return record


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
