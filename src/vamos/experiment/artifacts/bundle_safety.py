"""Structural and resource-limit validation for numerical ResultBundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn
from zipfile import ZipFile, ZipInfo

import numpy as np

from .errors import ArtifactResourceLimitError, MalformedResultBundleError, UnsupportedArrayDTypeError
from .models import ArtifactDescriptor, LoadLimits

ALLOWED_ARRAY_NAMES = (
    "F",
    "X",
    "G",
    "CV",
    "population/F",
    "population/X",
    "population/G",
    "population/CV",
    "archive/F",
    "archive/X",
    "archive/G",
    "archive/CV",
    "reference_directions",
)
_ALLOWED_DTYPE_KINDS = ("b", "i", "u", "f")
_COLUMN_DIMENSIONS = (
    ("n_obj", ("F", "population/F", "archive/F", "reference_directions")),
    ("n_var", ("X", "population/X", "archive/X")),
)


def validate_array_collection(
    arrays: Mapping[str, np.ndarray[Any, Any]],
    *,
    required_f: bool,
    limits: LoadLimits,
    operation: str,
) -> None:
    descriptor = result_descriptor()
    path = Path("result.npz")
    if required_f and "F" not in arrays:
        raise_malformed(descriptor, path, operation, "does not contain required F", "F", sorted(arrays))
    check_limit(len(arrays), limits.max_arrays, "max_arrays", descriptor=descriptor, path=path, operation=operation)
    total_elements = 0
    total_uncompressed_bytes = 0
    for name, value in arrays.items():
        if name not in ALLOWED_ARRAY_NAMES:
            raise_malformed(descriptor, path, operation, "contains an unknown array key", ALLOWED_ARRAY_NAMES, name)
        validate_dtype(value.dtype, name=name, descriptor=descriptor, path=path, operation=operation)
        validate_shape(name, value.shape, descriptor=descriptor, path=path, operation=operation)
        check_limit(
            value.ndim,
            limits.max_array_dimensions,
            "max_array_dimensions",
            descriptor=descriptor,
            path=path,
            operation=operation,
        )
        total_elements += value.size
        total_uncompressed_bytes += value.nbytes
    check_limit(
        total_elements,
        limits.max_total_elements,
        "max_total_elements",
        descriptor=descriptor,
        path=path,
        operation=operation,
    )
    check_limit(
        total_uncompressed_bytes,
        limits.max_total_uncompressed_bytes,
        "max_total_uncompressed_bytes",
        descriptor=descriptor,
        path=path,
        operation=operation,
    )
    contracts = {name: (tuple(value.shape), value.dtype) for name, value in arrays.items()}
    validate_contract_map(contracts, descriptor=descriptor, path=path, operation=operation)


def validate_contract_map(
    contracts: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]],
    *,
    descriptor: ArtifactDescriptor,
    path: Path,
    operation: str,
) -> None:
    for prefix in ("", "population/", "archive/"):
        f_name = f"{prefix}F"
        if f_name not in contracts:
            continue
        rows = contracts[f_name][0][0]
        for suffix in ("X", "G", "CV"):
            name = f"{prefix}{suffix}"
            if name in contracts and contracts[name][0][0] != rows:
                raise_malformed(
                    descriptor,
                    path,
                    operation,
                    f"contains row-misaligned arrays {f_name!r} and {name!r}",
                    rows,
                    contracts[name][0][0],
                )
    top_rows = contracts.get("F", ((0,), np.dtype("float64")))[0][0]
    for name in ("X", "G", "CV"):
        if name in contracts and contracts[name][0][0] != top_rows:
            raise_malformed(
                descriptor,
                path,
                operation,
                f"contains row-misaligned F and {name}",
                top_rows,
                contracts[name][0][0],
            )
    declared = descriptor.array_contract
    if declared is None:
        return
    if set(declared) != set(contracts):
        raise_malformed(descriptor, path, operation, "does not match manifest array keys", sorted(declared), sorted(contracts))
    for name, (shape, dtype) in contracts.items():
        spec = declared[name]
        if not isinstance(spec, Mapping):
            raise_malformed(descriptor, path, operation, "has invalid array_contract entry", "object", type(spec).__name__)
        if list(shape) != list(spec.get("shape", [])) or dtype.str != spec.get("dtype"):
            raise_malformed(
                descriptor,
                path,
                operation,
                f"array {name!r} differs from manifest array_contract",
                {"shape": spec.get("shape"), "dtype": spec.get("dtype")},
                {"shape": list(shape), "dtype": dtype.str},
            )


def validate_problem_columns(
    contracts: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]],
    resolved_spec: Mapping[str, Any],
    *,
    descriptor: ArtifactDescriptor,
    path: Path,
    operation: str,
) -> None:
    """Check objective columns against the resolved problem's ``n_obj`` and decision columns against its ``n_var``.

    A dimension that the problem configuration does not record as an integer is not checked.
    """
    problem = resolved_spec.get("problem")
    config = problem.get("config") if isinstance(problem, Mapping) else None
    if not isinstance(config, Mapping):
        return
    for dimension, names in _COLUMN_DIMENSIONS:
        expected = config.get(dimension)
        if isinstance(expected, bool) or not isinstance(expected, int):
            continue
        for name in names:
            if name in contracts and contracts[name][0][1] != expected:
                raise_malformed(
                    descriptor,
                    path,
                    operation,
                    f"array {name!r} has a column count different from the problem's {dimension}",
                    expected,
                    contracts[name][0][1],
                )


def validate_shape(
    name: str,
    shape: tuple[int, ...],
    *,
    descriptor: ArtifactDescriptor,
    path: Path,
    operation: str,
) -> None:
    expected_dimensions = 1 if name.endswith("/CV") or name == "CV" else 2
    if len(shape) != expected_dimensions or any(dim < 0 for dim in shape):
        raise_malformed(
            descriptor,
            path,
            operation,
            f"array {name!r} has an invalid shape",
            f"{expected_dimensions} dimensions with non-negative sizes",
            shape,
        )


def validate_dtype(
    dtype: np.dtype[Any],
    *,
    name: str,
    descriptor: ArtifactDescriptor,
    path: Path,
    operation: str,
) -> None:
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise UnsupportedArrayDTypeError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            field=f"result.npz#{name}",
            reason="uses a dtype outside the safe v1 allowlist",
            expected="bool, signed integer, unsigned integer, or floating dtype without object/structured fields",
            actual=dtype.str,
            action="Encode decisions numerically with a fixed-width dtype; never enable pickle or coerce away semantics.",
        )


def array_name(member: ZipInfo, *, descriptor: ArtifactDescriptor, path: Path, operation: str) -> str:
    filename = member.filename
    posix = PurePosixPath(filename)
    if not filename.endswith(".npy") or posix.is_absolute() or ".." in posix.parts or "\\" in filename:
        raise_malformed(descriptor, path, operation, "contains an unsafe or non-NPY ZIP member", "relative <key>.npy", filename)
    name = filename[:-4]
    if name not in ALLOWED_ARRAY_NAMES:
        raise_malformed(descriptor, path, operation, "contains an unknown array member", ALLOWED_ARRAY_NAMES, name)
    return name


def validate_member_layout(
    path: Path,
    archive: ZipFile,
    members: list[ZipInfo],
    *,
    descriptor: ArtifactDescriptor,
    operation: str,
) -> None:
    spans: list[tuple[int, int, str]] = []
    with path.open("rb") as raw:
        for member in members:
            raw.seek(member.header_offset)
            header = raw.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise_malformed(descriptor, path, operation, "contains an invalid local ZIP header", "PK local header", member.filename)
            name_bytes = int.from_bytes(header[26:28], "little")
            extra_bytes = int.from_bytes(header[28:30], "little")
            data_start = member.header_offset + 30 + name_bytes + extra_bytes
            data_end = data_start + member.compress_size
            if data_end > archive.start_dir:
                raise_malformed(
                    descriptor, path, operation, "contains a ZIP member overlapping the central directory", archive.start_dir, data_end
                )
            spans.append((member.header_offset, data_end, member.filename))
    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if previous[1] > current[0]:
            raise_malformed(
                descriptor,
                path,
                operation,
                "contains overlapping ZIP members",
                f"end of {previous[2]} <= start of {current[2]}",
                {"previous_end": previous[1], "current_start": current[0]},
            )


def check_limit(
    observed: int | float,
    configured: int | float,
    limit: str,
    *,
    descriptor: ArtifactDescriptor,
    path: Path,
    operation: str,
) -> None:
    if observed > configured:
        raise ArtifactResourceLimitError(
            operation=operation,
            limit=limit,
            configured=configured,
            observed=observed,
            artifact_role=descriptor.role,
            path=descriptor.path or path,
            action="Inspect the source and pass explicit trusted LoadLimits only if this artifact is expected.",
        )


def raise_malformed(
    descriptor: ArtifactDescriptor,
    path: Path,
    operation: str,
    reason: str,
    expected: Any,
    actual: Any,
) -> NoReturn:
    raise MalformedResultBundleError(
        operation=operation,
        artifact_role=descriptor.role,
        path=descriptor.path or path,
        reason=reason,
        expected=expected,
        actual=actual,
        action="Restore result.npz from the original run; never retry with pickle enabled.",
    )


def result_descriptor() -> ArtifactDescriptor:
    return ArtifactDescriptor(
        role="result_bundle",
        path="result.npz",
        media_type="application/vnd.vamos.result-bundle+npz",
        sha256="0" * 64,
        bytes=0,
        required_for=("load",),
        canonical=True,
    )


__all__ = [
    "array_name",
    "check_limit",
    "raise_malformed",
    "result_descriptor",
    "validate_array_collection",
    "validate_contract_map",
    "validate_dtype",
    "validate_member_layout",
    "validate_problem_columns",
    "validate_shape",
]
