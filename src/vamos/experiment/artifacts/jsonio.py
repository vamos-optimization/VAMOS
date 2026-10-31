"""Canonical and defensive JSON handling for v1 run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import ArtifactResourceLimitError, DuplicateJSONKeyError, ManifestValidationError


def normalize_json(value: Any, *, field: str = "$") -> Any:
    """Create a detached, canonical-hashable JSON value.

    Tuple-based configuration is normalized to JSON arrays. Unsupported values
    fail explicitly rather than being converted with ``str()``.
    """
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            return unicodedata.normalize("NFC", value)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestValidationError(
                operation="serialize run artifact",
                field=field,
                reason="contains a non-finite JSON number",
                expected="finite JSON number",
                actual=value,
                action="Move non-finite numerical values to result.npz or correct the metadata.",
            )
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ManifestValidationError(
                    operation="serialize run artifact",
                    field=field,
                    reason="contains a non-string or empty object key",
                    expected="non-empty string keys",
                    actual=raw_key,
                    action="Use explicit non-empty JSON object keys.",
                )
            key = unicodedata.normalize("NFC", raw_key)
            if key in normalized:
                raise DuplicateJSONKeyError(
                    operation="serialize run artifact",
                    field=field,
                    reason="contains keys that collide after Unicode NFC normalization",
                    expected="unique normalized keys",
                    actual=key,
                    action="Rename one of the colliding keys.",
                )
            normalized[key] = normalize_json(item, field=f"{field}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [normalize_json(item, field=f"{field}[{index}]") for index, item in enumerate(value)]
    raise ManifestValidationError(
        operation="serialize run artifact",
        field=field,
        reason="contains a value that is not JSON data",
        expected="null, boolean, integer, finite float, string, array, or object",
        actual=type(value).__name__,
        action="Replace executable or custom objects with explicit configuration-only JSON.",
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using the normative v1 canonical JSON form."""
    normalized = normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stored_json_bytes(value: Any) -> bytes:
    """Serialize normalized human-readable JSON with UTF-8, LF, and trailing LF."""
    normalized = normalize_json(value)
    text = json.dumps(normalized, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    return (text + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_self_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest after omitting only ``integrity.manifest_sha256``."""
    payload = normalize_json(manifest)
    if not isinstance(payload, dict):
        raise AssertionError("normalize_json returned a non-object for a manifest")
    integrity = payload.get("integrity")
    if isinstance(integrity, Mapping):
        mutable_integrity = dict(integrity)
        mutable_integrity.pop("manifest_sha256", None)
        payload["integrity"] = mutable_integrity
    return sha256_bytes(canonical_json_bytes(payload))


def load_json_file(
    path: Path,
    *,
    operation: str,
    artifact_role: str,
    max_bytes: int,
    max_depth: int,
) -> dict[str, Any]:
    """Read JSON with byte/depth limits, duplicate rejection, and no constants."""
    value, _ = read_json_file(path, operation=operation, artifact_role=artifact_role, max_bytes=max_bytes, max_depth=max_depth)
    return value


def read_json_file(
    path: Path,
    *,
    operation: str,
    artifact_role: str,
    max_bytes: int,
    max_depth: int,
    read_bytes: Callable[[Path], bytes] | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Like :func:`load_json_file`, also returning the exact bytes parsed, from a single read.

    ``read_bytes`` replaces ``Path.read_bytes`` for that one read.
    """
    try:
        observed_bytes = path.stat().st_size
    except OSError as exc:
        raise ManifestValidationError(
            operation=operation,
            artifact_role=artifact_role,
            path=path,
            reason="cannot be inspected",
            expected="readable regular JSON file",
            actual=type(exc).__name__,
            action="Restore the artifact and ensure it is readable.",
        ) from exc
    if observed_bytes > max_bytes:
        raise ArtifactResourceLimitError(
            operation=operation,
            limit="max_json_bytes",
            configured=max_bytes,
            observed=observed_bytes,
            artifact_role=artifact_role,
            path=path,
            action="Inspect the source and pass explicit trusted LoadLimits only if this size is expected.",
        )
    try:
        payload = path.read_bytes() if read_bytes is None else read_bytes(path)
    except OSError as exc:
        raise ManifestValidationError(
            operation=operation,
            artifact_role=artifact_role,
            path=path,
            reason="cannot be read",
            expected="readable UTF-8 JSON",
            actual=type(exc).__name__,
            action="Restore the artifact and ensure it is readable.",
        ) from exc
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ManifestValidationError(
            operation=operation,
            artifact_role=artifact_role,
            path=path,
            reason="contains a forbidden UTF-8 byte-order mark",
            expected="UTF-8 without BOM",
            actual="UTF-8 BOM",
            action="Restore the canonical JSON bytes from the original run.",
        )

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DuplicateJSONKeyError(
                    operation=operation,
                    artifact_role=artifact_role,
                    path=path,
                    field="$",
                    reason="contains a duplicate JSON object key",
                    expected="unique object keys",
                    actual=key,
                    action="Restore a canonical manifest; do not guess which duplicate value is authoritative.",
                )
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise ManifestValidationError(
            operation=operation,
            artifact_role=artifact_role,
            path=path,
            field="$",
            reason="contains a non-finite JSON literal",
            expected="finite JSON numbers",
            actual=value,
            action="Restore canonical JSON and keep numerical NaN/Infinity values only in result.npz.",
        )

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=reject_duplicate, parse_constant=reject_constant)
    except (ManifestValidationError, ArtifactResourceLimitError):
        raise
    except UnicodeDecodeError as exc:
        raise ManifestValidationError(
            operation=operation,
            artifact_role=artifact_role,
            path=path,
            reason="is not valid UTF-8",
            expected="UTF-8 JSON",
            actual=f"byte {exc.start}",
            action="Restore the canonical JSON bytes from the original run.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(
            operation=operation,
            artifact_role=artifact_role,
            path=path,
            field=f"line {exc.lineno}, column {exc.colno}",
            reason="is malformed JSON",
            expected="valid canonical JSON object",
            actual=exc.msg,
            action="Restore the canonical JSON bytes from the original run.",
        ) from exc
    if not isinstance(value, dict):
        raise ManifestValidationError(
            operation=operation,
            artifact_role=artifact_role,
            path=path,
            field="$",
            reason="has the wrong top-level type",
            expected="JSON object",
            actual=type(value).__name__,
            action="Restore a v1 JSON document with an object at its root.",
        )
    observed_depth = _json_depth(value)
    if observed_depth > max_depth:
        raise ArtifactResourceLimitError(
            operation=operation,
            limit="max_json_depth",
            configured=max_depth,
            observed=observed_depth,
            artifact_role=artifact_role,
            path=path,
            action="Inspect the source and pass explicit trusted LoadLimits only if this nesting is expected.",
        )
    return value, payload


def _json_depth(value: Any) -> int:
    maximum = 1
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        maximum = max(maximum, depth)
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return maximum


__all__ = [
    "canonical_json_bytes",
    "load_json_file",
    "manifest_self_hash",
    "normalize_json",
    "read_json_file",
    "sha256_bytes",
    "sha256_file",
    "stored_json_bytes",
]
