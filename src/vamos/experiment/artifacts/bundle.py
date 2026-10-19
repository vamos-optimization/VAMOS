"""Safe NPZ ResultBundle writing and bounded loading."""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

import numpy as np
from numpy.lib import format as npformat
from numpy.typing import NDArray

from .bundle_safety import (
    array_name,
    check_limit,
    raise_malformed,
    result_descriptor,
    validate_array_collection,
    validate_contract_map,
    validate_dtype,
    validate_member_layout,
    validate_shape,
)
from .errors import ArtifactResourceLimitError, MalformedResultBundleError
from .models import ArtifactDescriptor, LoadLimits

_ALLOWED_COMPRESSIONS = (ZIP_STORED, ZIP_DEFLATED)


class ResultLike(Protocol):
    F: NDArray[Any] | None
    X: NDArray[Any] | None


class _ValidatedArrayWriter(Protocol):
    def __call__(self, file: BinaryIO, **arrays: NDArray[np.generic]) -> None: ...


def snapshot_result_arrays(result: ResultLike, *, limits: LoadLimits) -> dict[str, np.ndarray[Any, Any]]:
    """Copy every canonical numerical result array before persistence starts."""
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    raw_data = getattr(result, "data", {})
    data = raw_data if isinstance(raw_data, Mapping) else {}
    _capture_array(arrays, "F", result.F, limits=limits)
    _capture_array(arrays, "X", result.X, limits=limits)
    for key in ("G", "CV", "reference_directions"):
        _capture_array(arrays, key, data.get(key), limits=limits)
    if "reference_directions" not in arrays:
        _capture_array(arrays, "reference_directions", data.get("weights"), limits=limits)
    for namespace in ("population", "archive"):
        nested = data.get(namespace)
        if not isinstance(nested, Mapping):
            continue
        for key in ("F", "X", "G", "CV"):
            _capture_array(arrays, f"{namespace}/{key}", nested.get(key), limits=limits)
    for key in ("F", "X", "G", "CV"):
        _capture_array(arrays, f"archive/{key}", data.get(f"archive_{key}"), limits=limits)
    validate_array_collection(arrays, required_f=True, limits=limits, operation="save result")
    return arrays


def write_result_bundle(path: Path, arrays: Mapping[str, np.ndarray[Any, Any]], *, limits: LoadLimits) -> None:
    """Write a copied array mapping to NPZ and flush it before return."""
    copied = {name: np.array(value, copy=True, order="K", subok=False) for name, value in arrays.items()}
    validate_array_collection(copied, required_f=True, limits=limits, operation="save result")
    with path.open("xb") as handle:
        # Validation excludes control keywords such as NumPy 2.2's allow_pickle.
        # Pass only arrays, retaining compatibility with older NumPy releases.
        cast(_ValidatedArrayWriter, np.savez)(handle, **copied)
        handle.flush()
        os.fsync(handle.fileno())


def inspect_result_bundle(
    path: Path,
    *,
    descriptor: ArtifactDescriptor,
    limits: LoadLimits,
    required_f: bool,
    operation: str,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Validate ZIP/NPY metadata without materializing any array."""
    try:
        observed_artifact_bytes = path.stat().st_size
    except OSError as exc:
        raise MalformedResultBundleError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            reason="cannot be inspected",
            expected="readable regular NPZ ResultBundle",
            actual=type(exc).__name__,
            action="Restore result.npz from the original run.",
        ) from exc
    check_limit(
        observed_artifact_bytes,
        limits.max_artifact_bytes,
        "max_artifact_bytes",
        descriptor=descriptor,
        path=path,
        operation=operation,
    )
    contracts: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    total_uncompressed = 0
    total_elements = 0
    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            check_limit(
                len(members),
                limits.max_zip_members,
                "max_zip_members",
                descriptor=descriptor,
                path=path,
                operation=operation,
            )
            check_limit(
                len(members),
                limits.max_arrays,
                "max_arrays",
                descriptor=descriptor,
                path=path,
                operation=operation,
            )
            validate_member_layout(path, archive, members, descriptor=descriptor, operation=operation)
            names: set[str] = set()
            for member in members:
                name = array_name(member, descriptor=descriptor, path=path, operation=operation)
                if name in names:
                    raise_malformed(
                        descriptor,
                        path,
                        operation,
                        "contains duplicate array member names",
                        "unique .npy members",
                        name,
                    )
                names.add(name)
                if member.compress_type not in _ALLOWED_COMPRESSIONS or member.flag_bits & 0x1:
                    raise_malformed(
                        descriptor,
                        path,
                        operation,
                        "uses an unsupported or encrypted ZIP member",
                        "unencrypted ZIP_STORED or ZIP_DEFLATED NPY members",
                        {"name": member.filename, "compression": member.compress_type, "flags": member.flag_bits},
                    )
                total_uncompressed += member.file_size
                check_limit(
                    total_uncompressed,
                    limits.max_total_uncompressed_bytes,
                    "max_total_uncompressed_bytes",
                    descriptor=descriptor,
                    path=path,
                    operation=operation,
                )
                ratio = math.inf if member.compress_size == 0 and member.file_size else member.file_size / max(1, member.compress_size)
                check_limit(
                    ratio,
                    limits.max_compression_ratio,
                    "max_compression_ratio",
                    descriptor=descriptor,
                    path=path,
                    operation=operation,
                )
                with archive.open(member, "r") as stream:
                    version = npformat.read_magic(stream)  # type: ignore[no-untyped-call]
                    if version == (1, 0):
                        shape, _fortran, dtype = npformat.read_array_header_1_0(stream, max_header_size=limits.max_npy_header_bytes)  # type: ignore[no-untyped-call,call-arg]
                    elif version == (2, 0):
                        shape, _fortran, dtype = npformat.read_array_header_2_0(stream, max_header_size=limits.max_npy_header_bytes)  # type: ignore[no-untyped-call,call-arg]
                    else:
                        raise_malformed(
                            descriptor,
                            path,
                            operation,
                            "uses an unsupported NPY header version",
                            "NPY 1.0 or 2.0",
                            version,
                        )
                    validate_dtype(dtype, name=name, descriptor=descriptor, path=path, operation=operation)
                    validate_shape(name, shape, descriptor=descriptor, path=path, operation=operation)
                    elements = math.prod(shape)
                    array_bytes = elements * dtype.itemsize
                    check_limit(
                        len(shape),
                        limits.max_array_dimensions,
                        "max_array_dimensions",
                        descriptor=descriptor,
                        path=path,
                        operation=operation,
                    )
                    check_limit(
                        array_bytes,
                        limits.max_array_bytes,
                        "max_array_bytes",
                        descriptor=descriptor,
                        path=path,
                        operation=operation,
                    )
                    total_elements += elements
                    check_limit(
                        total_elements,
                        limits.max_total_elements,
                        "max_total_elements",
                        descriptor=descriptor,
                        path=path,
                        operation=operation,
                    )
                    if stream.tell() + array_bytes != member.file_size:
                        raise_malformed(
                            descriptor,
                            path,
                            operation,
                            "has an NPY payload length inconsistent with its header",
                            stream.tell() + array_bytes,
                            member.file_size,
                        )
                    contracts[name] = (tuple(int(dim) for dim in shape), dtype)
    except ArtifactResourceLimitError:
        raise
    except (BadZipFile, OSError, ValueError, EOFError, struct.error) as exc:
        raise MalformedResultBundleError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            reason="is not a safe, well-formed NPZ ResultBundle",
            expected="bounded NPZ containing valid NPY 1.0/2.0 numerical arrays",
            actual=f"{type(exc).__name__}: {exc}",
            action="Restore result.npz from the original run; never retry with pickle enabled.",
        ) from exc
    if required_f and "F" not in contracts:
        raise_malformed(descriptor, path, operation, "does not contain required F", "F.npy", sorted(contracts))
    validate_contract_map(contracts, descriptor=descriptor, path=path, operation=operation)
    return contracts


def load_result_bundle(
    path: Path,
    *,
    descriptor: ArtifactDescriptor,
    limits: LoadLimits,
    required_f: bool,
    operation: str,
) -> dict[str, np.ndarray[Any, Any]]:
    """Inspect, then materialize a ResultBundle with pickle disabled."""
    contracts = inspect_result_bundle(
        path,
        descriptor=descriptor,
        limits=limits,
        required_f=required_f,
        operation=operation,
    )
    try:
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {name: np.array(bundle[name], copy=True, order="K", subok=False) for name in contracts}
    except (OSError, ValueError, EOFError, BadZipFile) as exc:
        raise MalformedResultBundleError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            reason="could not be materialized after header validation",
            expected="safe numerical arrays loadable with allow_pickle=False",
            actual=f"{type(exc).__name__}: {exc}",
            action="Restore result.npz from the original run; never retry with pickle enabled.",
        ) from exc
    validate_array_collection(arrays, required_f=required_f, limits=limits, operation=operation)
    return arrays


def array_contract(arrays: Mapping[str, np.ndarray[Any, Any]]) -> dict[str, dict[str, Any]]:
    return {name: {"dtype": value.dtype.str, "shape": [int(dim) for dim in value.shape]} for name, value in sorted(arrays.items())}


def _capture_array(
    arrays: dict[str, np.ndarray[Any, Any]],
    name: str,
    value: Any,
    *,
    limits: LoadLimits,
) -> None:
    if value is None:
        return
    try:
        array = np.array(value, copy=True, order="K", subok=False)
    except (TypeError, ValueError) as exc:
        raise MalformedResultBundleError(
            operation="save result",
            artifact_role="result_bundle",
            path="result.npz",
            reason=f"array {name!r} cannot be converted safely",
            expected="rectangular NumPy numerical array",
            actual=type(value).__name__,
            action="Provide a numeric ndarray with the contract-defined shape.",
        ) from exc
    validate_dtype(array.dtype, name=name, descriptor=result_descriptor(), path=Path("result.npz"), operation="save result")
    validate_shape(name, array.shape, descriptor=result_descriptor(), path=Path("result.npz"), operation="save result")
    check_limit(
        array.nbytes,
        limits.max_array_bytes,
        "max_array_bytes",
        descriptor=result_descriptor(),
        path=Path("result.npz"),
        operation="save result",
    )
    arrays[name] = array


__all__ = [
    "ResultLike",
    "array_contract",
    "inspect_result_bundle",
    "load_result_bundle",
    "snapshot_result_arrays",
    "write_result_bundle",
]
