"""Canonical JSON and closed-schema primitives for StudyManifest v1."""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from vamos.experiment.artifacts.errors import ArtifactResourceLimitError, DuplicateJSONKeyError, RunArtifactError
from vamos.experiment.artifacts.jsonio import (
    canonical_json_bytes,
    normalize_json,
    read_json_file,
    sha256_bytes,
    stored_json_bytes,
)

from .errors import MalformedStudyError, StudyResourceLimitError, UnsupportedStudySchemaError
from .models import SCHEMA_VERSION

_DIGEST = r"[0-9a-f]{64}"
_TASK_ID = r"sha256:[0-9a-f]{64}"
_TIMESTAMP = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z"


def canonical_json(value: Any) -> bytes:
    """Return the shared RunManifest canonical JSON form with study errors."""
    try:
        return canonical_json_bytes(value)
    except RunArtifactError as exc:
        reason = "NON_FINITE_NUMBER" if "non-finite" in exc.reason else "MALFORMED_JSON_VALUE"
        raise MalformedStudyError(
            operation="serialize study document",
            reason=reason,
            field=exc.field,
            expected=exc.expected,
            actual=exc.actual,
            action="Use finite, JSON-only values with unique NFC-normalized keys.",
        ) from exc


def document_self_hash(value: Mapping[str, Any]) -> str:
    """Hash a canonical document while omitting only its own semantic hash."""
    payload = normalize_json(value)
    if not isinstance(payload, dict):
        raise AssertionError("study document must be an object")
    integrity = payload.get("integrity")
    if isinstance(integrity, Mapping):
        mutable = dict(integrity)
        mutable.pop("document_sha256", None)
        payload["integrity"] = mutable
    return sha256_bytes(canonical_json(payload))


def seal_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached document with its semantic self-hash populated."""
    payload = copy.deepcopy(dict(value))
    payload["integrity"] = {"document_sha256": document_self_hash(payload)}
    return payload


def stored_document_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the normative human-readable bytes used for stored documents."""
    try:
        return stored_json_bytes(value)
    except RunArtifactError as exc:
        raise MalformedStudyError(
            operation="serialize study document",
            reason="MALFORMED_JSON_VALUE",
            field=exc.field,
            expected=exc.expected,
            actual=exc.actual,
            action="Use finite, JSON-only values with unique NFC-normalized keys.",
        ) from exc


def load_json(path: Path, *, operation: str, role: str, max_bytes: int, max_depth: int) -> dict[str, Any]:
    """Load bounded JSON and translate shared parser failures to study errors."""
    value, _ = read_json(path, operation=operation, role=role, max_bytes=max_bytes, max_depth=max_depth)
    return value


def read_json(path: Path, *, operation: str, role: str, max_bytes: int, max_depth: int) -> tuple[dict[str, Any], bytes]:
    """Like :func:`load_json`, also returning the exact bytes parsed, from a single read.

    Callers that check canonical bytes must not read the file again: a concurrent writer
    may have replaced it in between.
    """
    from .writing import read_bytes_shared

    try:
        return read_json_file(
            path,
            read_bytes=read_bytes_shared,
            operation=operation,
            artifact_role=role,
            max_bytes=max_bytes,
            max_depth=max_depth,
        )
    except DuplicateJSONKeyError as exc:
        _translated(exc, reason="DUPLICATE_JSON_KEY", role=role)
    except ArtifactResourceLimitError as exc:
        raise StudyResourceLimitError(
            operation=operation,
            reason="RESOURCE_LIMIT",
            document_role=role,
            path=path,
            expected=exc.expected,
            actual=exc.actual,
            action="Inspect the study source; raise limits only for a trusted study.",
        ) from exc
    except RunArtifactError as exc:
        reason = "NON_FINITE_NUMBER" if "non-finite" in exc.reason else "MALFORMED_JSON"
        _translated(exc, reason=reason, role=role)


def _translated(exc: RunArtifactError, *, reason: str, role: str) -> NoReturn:
    raise MalformedStudyError(
        operation=exc.operation,
        reason=reason,
        document_role=role,
        field=exc.field,
        path=exc.path,
        expected=exc.expected,
        actual=exc.actual,
        action="Restore the canonical StudyManifest v1 document; do not guess its intended contents.",
    ) from exc


def require_fields(value: Mapping[str, Any], fields: set[str], *, role: str) -> None:
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        reason = "UNKNOWN_FIELD" if unknown else "MISSING_FIELD"
        raise MalformedStudyError(
            operation="load study",
            reason=reason,
            document_role=role,
            field="$",
            expected=sorted(fields),
            actual={"missing": missing, "unknown": unknown},
            action="Restore a document with exactly the closed V1 field set.",
        )


def validate_header(value: Mapping[str, Any], *, role: str, document_type: str) -> None:
    if value.get("document_type") != document_type:
        raise UnsupportedStudySchemaError(
            operation="load study",
            reason="UNSUPPORTED_SCHEMA",
            document_role=role,
            field="$.document_type",
            expected=document_type,
            actual=value.get("document_type"),
            action="Use a canonical V1 study or regenerate this pre-release artifact.",
        )
    if value.get("schema_version") != SCHEMA_VERSION:
        raise UnsupportedStudySchemaError(
            operation="load study",
            reason="UNSUPPORTED_SCHEMA",
            document_role=role,
            field="$.schema_version",
            expected=SCHEMA_VERSION,
            actual=value.get("schema_version"),
            action="Regenerate the study with this VAMOS version; future schemas are not guessed.",
        )


def require_uuid4(value: object, *, field: str, role: str) -> str:
    if not isinstance(value, str):
        _malformed(field, role, "lowercase UUIDv4", value, "INVALID_IDENTITY")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        _malformed(field, role, "lowercase UUIDv4", value, "INVALID_IDENTITY", cause=exc)
    if parsed.version != 4 or str(parsed) != value:
        _malformed(field, role, "lowercase UUIDv4", value, "INVALID_IDENTITY")
    return value


def require_digest(value: object, *, field: str, role: str, prefixed: bool = False) -> str:
    pattern = _TASK_ID if prefixed else _DIGEST
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        _malformed(field, role, "sha256:<64 lowercase hex>" if prefixed else "64 lowercase hex", value, "INVALID_IDENTITY")
    return value


def require_timestamp(value: object, *, field: str, role: str) -> str:
    if not isinstance(value, str) or re.fullmatch(_TIMESTAMP, value) is None:
        _malformed(field, role, "RFC 3339 UTC timestamp with Z suffix", value, "INVALID_TIMESTAMP")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        _malformed(field, role, "valid RFC 3339 UTC timestamp", value, "INVALID_TIMESTAMP", cause=exc)
    return value


def require_int(value: object, *, field: str, role: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _malformed(field, role, f"integer >= {minimum}", value, "INVALID_FIELD")
    return value


def _malformed(
    field: str,
    role: str,
    expected: object,
    actual: object,
    reason: str,
    *,
    cause: Exception | None = None,
) -> NoReturn:
    error = MalformedStudyError(
        operation="load study",
        reason=reason,
        document_role=role,
        field=field,
        expected=expected,
        actual=actual,
        action="Restore the canonical document from a trusted copy.",
    )
    if cause is None:
        raise error
    raise error from cause


__all__ = [
    "canonical_json",
    "document_self_hash",
    "load_json",
    "require_digest",
    "require_fields",
    "require_int",
    "require_timestamp",
    "require_uuid4",
    "seal_document",
    "stored_document_bytes",
    "validate_header",
]
