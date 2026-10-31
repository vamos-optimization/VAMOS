"""Complete, data-only loading and verification of StudyManifest v1 roots."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vamos.experiment.artifacts.jsonio import sha256_bytes

from .decoding import decode_manifest, decode_plan, decode_spec
from .errors import (
    MalformedStudyError,
    MissingStudyDocumentError,
    PlanMismatchError,
    StudyIntegrityError,
    StudyResourceLimitError,
)
from .journal import derive_effective_study
from .journal_loading import load_event_journal
from .limits import StudyLoadLimits
from .models import DocumentReference, Study, StudyManifest, StudySpec
from .paths import confined_study_path
from .record_loading import ObserveRunReference, RunVerification, load_records
from .serialization import read_json, stored_document_bytes


@dataclass(slots=True)
class _ReadBudget:
    limits: StudyLoadLimits
    documents: int = 0
    total_bytes: int = 0

    def account(self, size: int, *, role: str, path: str) -> None:
        self.documents += 1
        self.total_bytes += size
        if self.documents > self.limits.max_documents:
            self._raise("max_documents", self.limits.max_documents, self.documents, role, path)
        if self.total_bytes > self.limits.max_total_bytes:
            self._raise("max_total_bytes", self.limits.max_total_bytes, self.total_bytes, role, path)

    @staticmethod
    def _raise(limit: str, configured: int, observed: int, role: str, path: str) -> None:
        raise StudyResourceLimitError(
            operation="load study",
            reason="RESOURCE_LIMIT",
            document_role=role,
            path=path,
            expected={limit: configured},
            actual=observed,
            action="Inspect the source and pass larger explicit limits only for a trusted study.",
        )


def load_study(path: str | Path, *, limits: StudyLoadLimits | None = None) -> Study:
    """Load and fully verify one canonical study without executing code."""
    return _load_study(path, limits=limits, run_verification="all", tolerate_run_errors=False)


def load_study_projection(
    path: str | Path,
    *,
    limits: StudyLoadLimits | None = None,
    observe_run_reference: ObserveRunReference | None = None,
) -> Study:
    """Load a report projection with metadata-only, issue-tolerant run checks."""
    return _load_study(
        path,
        limits=limits,
        run_verification="metadata",
        tolerate_run_errors=True,
        observe_run_reference=observe_run_reference,
    )


def _load_study(
    path: str | Path,
    *,
    limits: StudyLoadLimits | None,
    run_verification: RunVerification,
    tolerate_run_errors: bool,
    observe_run_reference: ObserveRunReference | None = None,
) -> Study:
    configured = limits or StudyLoadLimits()
    root = _root(path)
    budget = _ReadBudget(configured)
    manifest_raw, _ = _read(root, "study-manifest.json", "study_manifest", configured.max_manifest_bytes, budget)
    manifest = decode_manifest(manifest_raw)
    _validate_strings(manifest_raw, configured, "study_manifest")
    _validate_reference_contract(manifest)

    spec_raw, spec_bytes = _read_reference(root, manifest.spec, configured.max_spec_bytes, budget)
    _verify_reference(manifest.spec, spec_raw, spec_bytes)
    spec = decode_spec(spec_raw)

    plan_raw, plan_bytes = _read_reference(root, manifest.plan, configured.max_plan_bytes, budget)
    _verify_reference(manifest.plan, plan_raw, plan_bytes, plan=True)
    plan = decode_plan(plan_raw)
    if plan.task_count > configured.max_tasks:
        raise StudyResourceLimitError(
            operation="load study",
            reason="RESOURCE_LIMIT",
            document_role="resolved_plan",
            field="$.task_count",
            expected=f"at most {configured.max_tasks}",
            actual=plan.task_count,
            action="Split the trusted campaign into smaller studies.",
        )

    def read(relative: str, role: str, max_bytes: int) -> tuple[dict[str, Any], bytes]:
        return _read(root, relative, role, max_bytes, budget)

    tasks, attempts = load_records(
        root,
        manifest,
        plan.tasks,
        configured,
        read,
        run_verification=run_verification,
        tolerate_run_errors=tolerate_run_errors,
        observe_run_reference=observe_run_reference,
    )
    events = load_event_journal(root, manifest, configured, read)
    _cross_validate(manifest, spec, plan.plan_id)
    effective = derive_effective_study(
        root,
        manifest,
        tasks,
        attempts,
        events,
        run_verification=run_verification,
        tolerate_run_errors=tolerate_run_errors,
        observe_run_reference=observe_run_reference,
    )
    reconciliation_required = (
        manifest.checkpoint_sequence != events[-1].sequence or tasks != effective.tasks or attempts != effective.attempts
    )
    return Study(
        root=root,
        manifest=effective.manifest,
        spec=spec,
        plan=plan,
        tasks=effective.tasks,
        attempts=effective.attempts,
        events=events,
        stored_checkpoint_sequence=manifest.checkpoint_sequence,
        stored_checkpoint_event_sha256=manifest.checkpoint_event_sha256,
        reconciliation_required=reconciliation_required,
    )


def _root(path: str | Path) -> Path:
    candidate = Path(path)
    try:
        if not candidate.exists() or not candidate.is_dir():
            raise MissingStudyDocumentError(
                operation="load study",
                reason="MISSING_STUDY_ROOT",
                document_role="study_root",
                path=candidate,
                expected="existing canonical StudyManifest v1 directory",
                actual="missing or not a directory",
                action="Select the complete study directory containing study-manifest.json.",
            )
        return candidate.resolve(strict=True)
    except MissingStudyDocumentError:
        raise
    except OSError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="UNREADABLE_STUDY_ROOT",
            document_role="study_root",
            path=candidate,
            expected="readable study directory",
            actual=type(exc).__name__,
            action="Restore permissions or select a readable canonical study.",
        ) from exc


def _read(
    root: Path,
    relative: str,
    role: str,
    max_bytes: int,
    budget: _ReadBudget,
) -> tuple[dict[str, Any], bytes]:
    target = confined_study_path(root, relative, role=role, must_exist=True)
    if not target.exists() or not target.is_file():
        raise MissingStudyDocumentError(
            operation="load study",
            reason="MISSING_DOCUMENT",
            document_role=role,
            path=relative,
            expected="regular canonical JSON file",
            actual="missing or not a file",
            action="Restore the complete study directory from a trusted copy.",
        )
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="UNREADABLE_DOCUMENT",
            document_role=role,
            path=relative,
            expected="readable regular file",
            actual=type(exc).__name__,
            action="Restore the canonical document and its permissions.",
        ) from exc
    budget.account(size, role=role, path=relative)
    # One read: parsing and the canonical-bytes check must see the same replacement, and a
    # concurrent writer on Windows cannot replace the file while it is open.
    value, payload = read_json(target, operation="load study", role=role, max_bytes=max_bytes, max_depth=budget.limits.max_json_depth)
    expected_bytes = stored_document_bytes(value)
    if payload != expected_bytes:
        raise StudyIntegrityError(
            operation="load study",
            reason="NON_CANONICAL_BYTES",
            document_role=role,
            path=relative,
            expected=sha256_bytes(expected_bytes),
            actual=sha256_bytes(payload),
            action="Restore the exact canonical JSON bytes; semantic reformatting is not a persisted write.",
        )
    _validate_strings(value, budget.limits, role)
    return value, payload


def _read_reference(
    root: Path,
    reference: DocumentReference,
    max_bytes: int,
    budget: _ReadBudget,
) -> tuple[dict[str, Any], bytes]:
    return _read(root, reference.path, reference.role, max_bytes, budget)


def _verify_reference(
    reference: DocumentReference,
    value: Mapping[str, Any],
    payload: bytes,
    *,
    plan: bool = False,
) -> None:
    file_hash = sha256_bytes(payload)
    integrity = value.get("integrity")
    semantic = integrity.get("document_sha256") if isinstance(integrity, Mapping) else None
    matches = reference.bytes == len(payload) and hmac.compare_digest(reference.sha256, file_hash)
    matches = matches and isinstance(semantic, str) and hmac.compare_digest(reference.semantic_sha256, semantic)
    if not matches:
        error = PlanMismatchError if plan else StudyIntegrityError
        raise error(
            operation="load study",
            reason="PLAN_MISMATCH" if plan else "DOCUMENT_REFERENCE_MISMATCH",
            document_role=reference.role,
            path=reference.path,
            expected={
                "bytes": reference.bytes,
                "sha256": reference.sha256,
                "semantic_sha256": reference.semantic_sha256,
            },
            actual={"bytes": len(payload), "sha256": file_hash, "semantic_sha256": semantic},
            action="Restore the immutable published document and matching root manifest.",
        )


def _cross_validate(
    manifest: StudyManifest,
    spec: StudySpec,
    plan_id: str,
) -> None:
    _validate_manifest_identity(manifest, spec, plan_id)
    _validate_manifest_policy(manifest, spec)


def _validate_manifest_identity(manifest: StudyManifest, spec: StudySpec, plan_id: str) -> None:
    if spec.study_id != manifest.study_id or plan_id != manifest.plan_id:
        raise PlanMismatchError(
            operation="load study",
            reason="PLAN_MISMATCH",
            expected={"study_id": manifest.study_id, "plan_id": manifest.plan_id},
            actual={"study_id": spec.study_id, "plan_id": plan_id},
            action="Restore the matching spec, plan, and manifest set.",
        )


def _validate_manifest_policy(manifest: StudyManifest, spec: StudySpec) -> None:
    if (manifest.on_error, manifest.max_attempts_per_task) != (
        spec.on_error,
        spec.max_attempts_per_task,
    ):
        _unexpected(
            "study_manifest.policy",
            (spec.on_error, spec.max_attempts_per_task),
            (manifest.on_error, manifest.max_attempts_per_task),
            reason="POLICY_MISMATCH",
        )


def _validate_reference_contract(manifest: StudyManifest) -> None:
    expected = {
        "spec": ("study-spec.json", "study_spec"),
        "plan": ("plan.json", "resolved_plan"),
    }
    for name, reference in (("spec", manifest.spec), ("plan", manifest.plan)):
        path, role = expected[name]
        if reference.path != path or reference.role != role or reference.required_for != ("inspect", "run", "resume"):
            _unexpected(
                f"study_manifest.{name}",
                {"path": path, "role": role, "required_for": ("inspect", "run", "resume")},
                reference,
            )


def _validate_strings(value: object, limits: StudyLoadLimits, role: str) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str) and len(item.encode("utf-8")) > limits.max_string_bytes:
            raise StudyResourceLimitError(
                operation="load study",
                reason="RESOURCE_LIMIT",
                document_role=role,
                expected=f"strings at most {limits.max_string_bytes} UTF-8 bytes",
                actual=len(item.encode("utf-8")),
                action="Remove oversized untrusted metadata.",
            )
        if isinstance(item, Mapping):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _unexpected(role: str, expected: object, actual: object, *, reason: str = "UNEXPECTED_LAYOUT") -> None:
    raise StudyIntegrityError(
        operation="load study",
        reason=reason,
        document_role=role,
        expected=expected,
        actual=actual,
        action="Restore the complete canonical study tree; do not infer state from extra or missing files.",
    )


__all__ = ["load_study", "load_study_projection"]
