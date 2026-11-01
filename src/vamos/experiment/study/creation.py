"""Non-destructive atomic publication of complete planned studies."""

from __future__ import annotations

import os
import secrets
import shutil
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vamos.experiment.artifacts.jsonio import sha256_bytes
from vamos.experiment.artifacts.paths import windows_path_length_error

from .documents import (
    initial_event_document,
    manifest_document,
    plan_document,
    semantic_hash,
    study_spec_document,
    task_document,
)
from .errors import StudyError, StudyInfrastructureError, StudyOutputCollisionError
from .identity import new_uuid4
from .limits import StudyLoadLimits
from .loading import load_study
from .models import DocumentReference, ResolvedStudyPlan, Study, StudySpec
from .planning import resolve_spec
from .serialization import stored_document_bytes
from .writing import fsync_directory, write_bytes_atomic

_PhaseHook = Callable[[str], None]


def create_study(
    spec: StudySpec,
    *,
    output: str | Path,
    limits: StudyLoadLimits | None = None,
) -> Study:
    """Resolve and atomically publish a study without executing any task."""
    if not isinstance(spec, StudySpec):
        from .errors import InvalidStudySpecError

        raise InvalidStudySpecError(
            operation="create study",
            reason="INVALID_STUDY_SPEC",
            expected="validated StudySpec",
            actual=type(spec).__name__,
            action="Construct vamos.StudySpec(...) before calling create_study.",
        )
    destination = Path(output).absolute()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StudyInfrastructureError(
            operation="create study",
            reason="ATOMIC_PUBLICATION_FAILED",
            path=destination.parent,
            expected="writable parent directory for sibling staging",
            actual=_failure(exc),
            action="Choose a writable output parent; no study directory was published.",
        ) from exc
    _reject_existing(destination)
    plan = resolve_spec(spec)
    return publish_study(spec, plan=plan, destination=destination, limits=limits)


def publish_study(
    spec: StudySpec,
    *,
    plan: ResolvedStudyPlan,
    destination: Path,
    limits: StudyLoadLimits | None = None,
    phase_hook: _PhaseHook | None = None,
) -> Study:
    """Publish already resolved state; ``phase_hook`` is an internal test seam."""
    configured = limits or StudyLoadLimits()
    destination = destination.absolute()
    _reject_existing(destination)
    # Keep the full 128 bits of entropy while limiting transient path growth on
    # Windows, where otherwise an ordinary pytest/user temp root can hit MAX_PATH.
    token = secrets.token_urlsafe(16)
    staging = destination.parent / f".{destination.name}.vamos-study-staging-{token}"
    owns_staging = False
    published = False
    hook = phase_hook or _publication_phase
    try:
        staging.mkdir(exist_ok=False)
        owns_staging = True
        hook("staging_created")
        _populate_staging(staging, spec=spec, plan=plan, hook=hook)
        fsync_directory(staging)
        hook("documents_written")
        load_study(staging, limits=configured)
        hook("staging_verified")
        _reject_existing(destination)
        hook("before_publish")
        os.rename(staging, destination)
        published = True
        fsync_directory(destination.parent)
        hook("published")
        return load_study(destination, limits=configured)
    except StudyError:
        raise
    except Exception as exc:
        raise StudyInfrastructureError(
            operation="create study",
            reason="ATOMIC_PUBLICATION_FAILED",
            path=destination,
            expected="complete study directory published once or no destination",
            actual=_failure(exc),
            published=published,
            action=(
                "Load the destination to verify it; publication completed before the infrastructure error."
                if published
                else "Correct the storage failure and retry at an absent destination."
            ),
        ) from exc
    finally:
        if owns_staging and not published:
            _remove_owned_staging(staging, parent=destination.parent, token=token)


def _failure(exc: Exception) -> str:
    """The failure's type and message, with the path-length explanation of a Windows long-path failure."""
    shown: Exception = exc
    if isinstance(exc, OSError):
        shown = windows_path_length_error(exc) or exc
    message = str(shown)
    return f"{type(shown).__name__}: {message}" if message else type(shown).__name__


def _populate_staging(staging: Path, *, spec: StudySpec, plan: ResolvedStudyPlan, hook: _PhaseHook) -> None:
    study_id = new_uuid4()
    event_id = new_uuid4()
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    spec_doc = study_spec_document(spec, study_id)
    spec_bytes = stored_document_bytes(spec_doc)
    write_bytes_atomic(staging / "study-spec.json", spec_bytes)
    hook("spec_written")

    plan_doc = plan_document(plan)
    if semantic_hash(plan_doc) != plan.document_sha256:
        raise AssertionError("resolved plan semantic hash changed during document construction")
    plan_bytes = stored_document_bytes(plan_doc)
    write_bytes_atomic(staging / "plan.json", plan_bytes)
    hook("plan_written")

    for task in plan.tasks:
        digest = task.task_id.removeprefix("sha256:")
        task_doc = task_document(
            study_id=study_id,
            task=task,
            max_attempts_per_task=spec.max_attempts_per_task,
        )
        write_bytes_atomic(staging / "tasks" / digest / "task.json", stored_document_bytes(task_doc))
    hook("tasks_written")

    event_doc = initial_event_document(study_id=study_id, event_id=event_id, timestamp=timestamp)
    event_bytes = stored_document_bytes(event_doc)
    write_bytes_atomic(staging / "events" / "00000000000000000001.json", event_bytes)
    hook("event_written")

    manifest_doc = manifest_document(
        study_id=study_id,
        plan_id=plan.plan_id,
        timestamp=timestamp,
        on_error=spec.on_error,
        max_attempts_per_task=spec.max_attempts_per_task,
        task_count=plan.task_count,
        spec_reference=_reference("study-spec.json", "study_spec", spec_doc, spec_bytes),
        plan_reference=_reference("plan.json", "resolved_plan", plan_doc, plan_bytes),
        event_sha256=sha256_bytes(event_bytes),
    )
    write_bytes_atomic(staging / "study-manifest.json", stored_document_bytes(manifest_doc))
    hook("manifest_written")


def _reference(
    path: str,
    role: str,
    document: Mapping[str, Any],
    payload: bytes,
) -> DocumentReference:
    return DocumentReference(
        path=path,
        role=role,
        required_for=("inspect", "run", "resume"),
        semantic_sha256=semantic_hash(document),
        sha256=sha256_bytes(payload),
        bytes=len(payload),
    )


def _reject_existing(path: Path) -> None:
    if os.path.lexists(path):
        raise StudyOutputCollisionError(
            operation="create study",
            reason="OUTPUT_COLLISION",
            path=path,
            expected="destination path that does not exist",
            actual="occupied output path",
            action="Choose another output directory; VAMOS never overwrites, reuses, or merges studies.",
        )


def _remove_owned_staging(staging: Path, *, parent: Path, token: str) -> None:
    if not os.path.lexists(staging):
        return
    expected_suffix = f".vamos-study-staging-{token}"
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved_staging_parent = staging.parent.resolve(strict=True)
    except OSError:
        return
    if resolved_parent != resolved_staging_parent or not staging.name.endswith(expected_suffix):
        return
    shutil.rmtree(staging)


def _publication_phase(_phase: str) -> None:
    """Internal monkeypatch seam for deterministic crash-boundary tests."""


__all__ = ["create_study", "publish_study"]
