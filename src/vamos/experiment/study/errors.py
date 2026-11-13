"""Typed, actionable errors for StudyManifest v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vamos.foundation.exceptions import VAMOSError


class StudyError(VAMOSError):
    """Base error with stable fields suitable for future command rendering.

    It is also a ``VAMOSError``; its ``message`` is the whole error text and it has no ``suggestion``.
    """

    category = "study"

    def __init__(
        self,
        *,
        operation: str,
        reason: str,
        expected: Any,
        actual: Any,
        action: str,
        document_role: str | None = None,
        field: str | None = None,
        path: str | Path | None = None,
        entity_id: str | None = None,
        study_id: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        current_state: str | None = None,
        expected_state: str | None = None,
        objective_evaluation_began: bool = False,
        canonical_run_published: bool = False,
        published: bool = False,
    ) -> None:
        self.operation = operation
        self.reason = reason
        self.expected = expected
        self.actual = actual
        self.action = action
        self.document_role = document_role
        self.field = field
        self.path = str(path) if path is not None else None
        self.entity_id = entity_id
        self.study_id = study_id
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.current_state = current_state
        self.expected_state = expected_state
        self.objective_evaluation_began = objective_evaluation_began
        self.canonical_run_published = canonical_run_published
        self.published = published
        self.execution_occurred = objective_evaluation_began
        location = field or document_role or entity_id or "study"
        if self.path is not None:
            location = f"{location} at {self.path}"
        super().__init__(f"Cannot {operation}: {location} ({reason}). Expected {expected!r}; received {actual!r}. {action}")

    def as_dict(self) -> dict[str, Any]:
        """Return stable structured diagnostics."""
        return {
            "operation": self.operation,
            "category": self.category,
            "reason": self.reason,
            "expected": self.expected,
            "actual": self.actual,
            "action": self.action,
            "document_role": self.document_role,
            "field": self.field,
            "path": self.path,
            "entity_id": self.entity_id,
            "study_id": self.study_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "current_state": self.current_state,
            "expected_state": self.expected_state,
            "objective_evaluation_began": self.objective_evaluation_began,
            "canonical_run_published": self.canonical_run_published,
            "published": self.published,
            "execution_occurred": self.execution_occurred,
        }


class InvalidStudySpecError(StudyError):
    category = "study_spec"


class UnresolvedStudyTaskError(StudyError):
    category = "study_resolution"


class DuplicateStudyTaskError(UnresolvedStudyTaskError):
    category = "duplicate_study_task"


class MalformedStudyError(StudyError):
    category = "study_schema"


class UnsupportedStudySchemaError(MalformedStudyError):
    category = "unsupported_study_schema"


class MissingStudyDocumentError(MalformedStudyError):
    category = "missing_study_document"


class StudyIntegrityError(MalformedStudyError):
    category = "study_integrity"


class PlanMismatchError(StudyIntegrityError):
    category = "plan_mismatch"


class UnsafeStudyPathError(MalformedStudyError):
    category = "unsafe_study_path"


class StudyResourceLimitError(MalformedStudyError):
    category = "study_resource_limit"


class StudyOutputCollisionError(StudyError):
    category = "study_output_collision"


class StudyInfrastructureError(StudyError):
    category = "study_infrastructure"


class UnsupportedStudyExecutionStateError(StudyError):
    category = "study_execution_state"


class InvalidStudyTransitionError(StudyError):
    category = "study_transition"


class StudyExecutionError(StudyError):
    category = "study_execution"


class StudyRunPublicationError(StudyInfrastructureError):
    category = "study_run_publication"


class StudyRunVerificationError(StudyInfrastructureError):
    category = "study_run_verification"


class StudyFinalizationError(StudyInfrastructureError):
    category = "study_finalization"


class StudyEventAppendError(StudyInfrastructureError):
    category = "study_event_append"


class StudyCheckpointError(StudyIntegrityError):
    category = "study_checkpoint"


class ReferencedRunMissingError(StudyIntegrityError):
    category = "referenced_run_missing"


class ReferencedRunCorruptError(StudyIntegrityError):
    category = "referenced_run_corrupt"


class RetryNotAllowedError(StudyError):
    category = "study_retry_not_allowed"


class RetryLimitError(RetryNotAllowedError):
    category = "study_retry_limit"


class ResumeEnvironmentIncompatibilityError(StudyInfrastructureError):
    category = "resume_environment_incompatibility"


__all__ = [
    "DuplicateStudyTaskError",
    "InvalidStudySpecError",
    "MalformedStudyError",
    "MissingStudyDocumentError",
    "PlanMismatchError",
    "ReferencedRunCorruptError",
    "ReferencedRunMissingError",
    "ResumeEnvironmentIncompatibilityError",
    "RetryLimitError",
    "RetryNotAllowedError",
    "StudyError",
    "StudyEventAppendError",
    "StudyExecutionError",
    "StudyFinalizationError",
    "StudyInfrastructureError",
    "StudyIntegrityError",
    "StudyCheckpointError",
    "InvalidStudyTransitionError",
    "StudyOutputCollisionError",
    "StudyRunPublicationError",
    "StudyRunVerificationError",
    "StudyResourceLimitError",
    "UnresolvedStudyTaskError",
    "UnsafeStudyPathError",
    "UnsupportedStudySchemaError",
    "UnsupportedStudyExecutionStateError",
]
