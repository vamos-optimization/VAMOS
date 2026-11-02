"""Typed errors for the v1 run-artifact boundary."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MESSAGE_LIMIT = 500
# Windows drive (not inside a word, so not the ``s:/`` of ``https://``) or UNC paths, and POSIX paths
# not preceded by a word character, dot, colon or slash (so ``1/n``, ``a/b`` and URLs are kept).
_ABSOLUTE_PATH_PATTERN = r"""(?<!\w)[A-Za-z]:[\\/][^\s'"<>|]*|\\\\[^\s'"<>|]+|(?<![\w.~:/\\])/[^\s'"<>|]+"""


def sanitized_message(exc: BaseException, *, empty: str = "The error has no message.") -> str:
    """One bounded line describing ``exc``, safe to persist.

    Line breaks are collapsed, absolute paths are replaced by ``<path>`` and the text is cut to
    ``MESSAGE_LIMIT`` characters. A typed error's ``field`` is added when its message omits it.
    """
    message = " ".join(str(exc).split())
    field = getattr(exc, "field", None)
    if isinstance(field, str) and field and field not in message:
        message = f"{message} (field {field})" if message else f"field {field}"
    message = re.sub(_ABSOLUTE_PATH_PATTERN, "<path>", message)
    return message[:MESSAGE_LIMIT] if message else empty


class RunArtifactError(Exception):
    """Base error carrying actionable, machine-readable failure details."""

    category = "run_artifact"

    def __init__(
        self,
        *,
        operation: str,
        reason: str,
        expected: Any,
        actual: Any,
        action: str,
        field: str | None = None,
        artifact_role: str | None = None,
        path: str | Path | None = None,
        optimization_executed: bool = False,
    ) -> None:
        self.operation = operation
        self.reason = reason
        self.expected = expected
        self.actual = actual
        self.action = action
        self.field = field
        self.artifact_role = artifact_role
        self.path = str(path) if path is not None else None
        self.optimization_executed = optimization_executed
        location = field or artifact_role or "run artifact"
        if self.path is not None:
            location = f"{location} at {self.path}"
        message = f"Cannot {operation}: {location} {reason}. Expected {expected!r}; received {actual!r}. {action}"
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        """Return stable structured fields for future CLI rendering."""
        return {
            "operation": self.operation,
            "category": self.category,
            "reason": self.reason,
            "expected": self.expected,
            "actual": self.actual,
            "action": self.action,
            "field": self.field,
            "artifact_role": self.artifact_role,
            "path": self.path,
            "optimization_executed": self.optimization_executed,
        }


class ManifestValidationError(RunArtifactError):
    """A manifest is malformed or violates v1 semantic constraints."""

    category = "manifest_schema"


class MissingManifestFieldError(ManifestValidationError):
    """A required manifest or source-result field is unavailable."""

    category = "manifest_field_missing"


class DuplicateJSONKeyError(ManifestValidationError):
    """A JSON object contains a duplicate key."""

    category = "manifest_parse"


class UnsupportedSchemaError(ManifestValidationError):
    """The document identity or schema version is unsupported."""

    category = "unsupported_schema"


class UnsupportedArtifactLayoutError(UnsupportedSchemaError):
    """The directory is not a v1 run artifact."""

    category = "unsupported_artifact_layout"


class ArtifactIntegrityError(RunArtifactError):
    """A referenced artifact or the manifest self-hash is invalid."""

    category = "artifact_integrity"

    def __init__(
        self,
        *,
        operation: str,
        reason: str,
        expected: Any,
        actual: Any,
        action: str,
        field: str | None = None,
        artifact_role: str | None = None,
        path: str | Path | None = None,
        expected_sha256: str | None = None,
        actual_sha256: str | None = None,
        expected_bytes: int | None = None,
        actual_bytes: int | None = None,
        state: str | None = None,
    ) -> None:
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        self.expected_bytes = expected_bytes
        self.actual_bytes = actual_bytes
        self.state = state
        super().__init__(
            operation=operation,
            reason=reason,
            expected=expected,
            actual=actual,
            action=action,
            field=field,
            artifact_role=artifact_role,
            path=path,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload.update(
            {
                "expected_sha256": self.expected_sha256,
                "actual_sha256": self.actual_sha256,
                "expected_bytes": self.expected_bytes,
                "actual_bytes": self.actual_bytes,
                "state": self.state,
            }
        )
        return payload


class ArtifactMissingError(ArtifactIntegrityError):
    """A required artifact is missing."""

    category = "artifact_missing"


class UnsafeArtifactPathError(RunArtifactError):
    """An artifact path is not safely confined to its run directory."""

    category = "unsafe_artifact_path"


class UnsupportedArrayDTypeError(RunArtifactError):
    """A result array uses a dtype outside the safe v1 allowlist."""

    category = "unsupported_array_dtype"


class MalformedResultBundleError(RunArtifactError):
    """A result NPZ/NPY structure is malformed or internally inconsistent."""

    category = "malformed_result_bundle"


class ArtifactResourceLimitError(RunArtifactError):
    """Parsing was refused before exceeding a configured resource limit."""

    category = "artifact_resource_limit"

    def __init__(
        self,
        *,
        operation: str,
        limit: str,
        configured: int | float,
        observed: int | float,
        artifact_role: str,
        path: str | Path,
        action: str,
    ) -> None:
        self.limit = limit
        self.configured = configured
        self.observed = observed
        super().__init__(
            operation=operation,
            reason=f"exceeds resource limit {limit}",
            expected=f"{limit} <= {configured}",
            actual=observed,
            action=action,
            artifact_role=artifact_role,
            path=path,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload.update({"limit": self.limit, "configured": self.configured, "observed": self.observed})
        return payload


class OutputCollisionError(RunArtifactError):
    """A save destination already exists and must not be overwritten."""

    category = "output_collision"


class IncompleteRunError(RunArtifactError):
    """A run has no usable canonical numerical result."""

    category = "incomplete_run"


class IncompleteRunMetadataError(RunArtifactError):
    """A result cannot be saved without complete execution metadata."""

    category = "incomplete_run_metadata"


class VerificationRequirementError(RunArtifactError):
    """A requested verification/replayability level was not met."""

    category = "verification_requirement"


class EnvironmentIncompatibilityError(VerificationRequirementError):
    """The current environment cannot satisfy the requested replay level."""

    category = "environment_incompatibility"


class ReplayUnavailableError(RunArtifactError):
    """The source run is not eligible for executable replay."""

    category = "replay_unavailable"


class ComponentNotReconstructableError(ReplayUnavailableError):
    """A persisted component cannot be reconstructed from built-ins."""

    category = "component_not_reconstructable"


class ResolvedSpecMismatchError(ReplayUnavailableError):
    """Reconstruction does not reproduce the persisted resolved spec."""

    category = "resolved_spec_mismatch"


class UnsupportedReplayProviderError(ReplayUnavailableError):
    """A persisted provider is outside the built-in replay trust boundary."""

    category = "unsupported_replay_provider"


class ReplayExecutionError(RunArtifactError):
    """Optimization began but did not produce a replay result."""

    category = "replay_execution"


class ReplayResultMismatchError(ReplayExecutionError):
    """Replay completed but exact array comparison failed."""

    category = "replay_result_mismatch"


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactMissingError",
    "ArtifactResourceLimitError",
    "ComponentNotReconstructableError",
    "DuplicateJSONKeyError",
    "EnvironmentIncompatibilityError",
    "IncompleteRunMetadataError",
    "IncompleteRunError",
    "MalformedResultBundleError",
    "ManifestValidationError",
    "MissingManifestFieldError",
    "OutputCollisionError",
    "ReplayExecutionError",
    "ReplayResultMismatchError",
    "ReplayUnavailableError",
    "ResolvedSpecMismatchError",
    "RunArtifactError",
    "UnsafeArtifactPathError",
    "UnsupportedArrayDTypeError",
    "UnsupportedArtifactLayoutError",
    "UnsupportedReplayProviderError",
    "UnsupportedSchemaError",
    "VerificationRequirementError",
    "sanitized_message",
]
