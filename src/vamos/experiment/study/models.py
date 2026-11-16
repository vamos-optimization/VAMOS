"""Immutable models for planned and sequentially executed StudyManifest v1 roots."""

from __future__ import annotations

import numbers
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from vamos.experiment.artifacts.errors import RunArtifactError
from vamos.experiment.artifacts.jsonio import canonical_json_bytes, normalize_json
from vamos.experiment.artifacts.models import LoadLimits, deep_freeze, deep_thaw

from .errors import InvalidStudySpecError
from .limits import StudyLoadLimits

if TYPE_CHECKING:
    from .report_models import StudyReport, StudySummary

SCHEMA_VERSION = "1.0.0"
OnErrorPolicy = Literal["fail_fast", "continue"]
StudyState = Literal["created", "running", "paused", "completed", "completed_with_failures", "failed", "cancelled"]
TaskState = Literal["pending", "running", "succeeded", "failed", "interrupted", "cancelled", "skipped"]
AttemptState = Literal["created", "running", "succeeded", "failed", "interrupted", "cancelled"]


def _invalid(field_name: str, expected: object, actual: object, *, reason: str = "INVALID_STUDY_SPEC") -> NoReturn:
    raise InvalidStudySpecError(
        operation="validate study spec",
        reason=reason,
        field=field_name,
        expected=expected,
        actual=actual,
        action="Correct the StudySpec before creating a study; no directory was published.",
    )


def _positive_optional(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
        _invalid(name, "positive integer or None", value)
    return operator.index(value)


def _names(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _invalid(name, "sequence of non-empty strings", type(values).__name__)
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            _invalid(f"{name}[{index}]", "non-empty string", value)
        result.append(value.strip())
    return tuple(result)


def _seeds(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _invalid("seeds", "sequence of integers", type(values).__name__)
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            _invalid(f"seeds[{index}]", "integer seed", value)
        result.append(int(value))
    return tuple(result)


def _frozen_json_mapping(name: str, value: Mapping[str, object] | None, *, max_bytes: int) -> Mapping[str, Any]:
    raw: Mapping[str, object] = {} if value is None else value
    if not isinstance(raw, Mapping):
        _invalid(name, "JSON object", type(raw).__name__)
    try:
        normalized = normalize_json(raw, field=f"$.{name}")
        encoded = canonical_json_bytes(normalized)
    except RunArtifactError as exc:
        reason = "NON_FINITE_NUMBER" if "non-finite" in exc.reason else "INVALID_STUDY_SPEC"
        _invalid(name, "bounded JSON-only object", exc.actual, reason=reason)
    if not isinstance(normalized, dict):
        raise AssertionError("normalized mapping is not an object")
    if len(encoded) > max_bytes:
        _invalid(name, f"at most {max_bytes} canonical JSON bytes", len(encoded), reason="RESOURCE_LIMIT")
    return cast(Mapping[str, Any], deep_freeze(normalized))


@dataclass(frozen=True, slots=True)
class StudySpec:
    """Validated, immutable user intent for a deterministic study plan.

    Seeds are explicit. All defaults selected during creation are frozen into
    each task's resolved run specification before publication.
    """

    problems: Sequence[str]
    algorithms: Sequence[str]
    seeds: Sequence[int]
    max_evaluations: int | None = None
    pop_size: int | None = None
    engine: str | None = None
    eval_strategy: str = "serial"
    n_var: int | None = None
    n_obj: int | None = None
    problem_kwargs: Mapping[str, object] | None = None
    algorithm_configs: Mapping[str, object] | None = None
    on_error: OnErrorPolicy = "fail_fast"
    max_attempts_per_task: int = 3
    labels: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None
    study_id: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problems", _names("problems", self.problems))
        object.__setattr__(self, "algorithms", _names("algorithms", self.algorithms))
        object.__setattr__(self, "seeds", _seeds(self.seeds))
        object.__setattr__(self, "max_evaluations", _positive_optional("max_evaluations", self.max_evaluations))
        object.__setattr__(self, "pop_size", _positive_optional("pop_size", self.pop_size))
        object.__setattr__(self, "n_var", _positive_optional("n_var", self.n_var))
        object.__setattr__(self, "n_obj", _positive_optional("n_obj", self.n_obj))
        if self.engine is not None and (not isinstance(self.engine, str) or not self.engine.strip()):
            _invalid("engine", "non-empty string or None", self.engine)
        if self.eval_strategy not in {"serial", "multiprocessing", "dask"}:
            _invalid("eval_strategy", "'serial', 'multiprocessing', or 'dask'", self.eval_strategy)
        if self.on_error not in ("fail_fast", "continue"):
            _invalid("on_error", "'fail_fast' or 'continue'", self.on_error)
        attempts = _positive_optional("max_attempts_per_task", self.max_attempts_per_task)
        if attempts is None:
            raise AssertionError("max_attempts_per_task cannot be None")
        object.__setattr__(self, "max_attempts_per_task", attempts)
        object.__setattr__(self, "problem_kwargs", _frozen_json_mapping("problem_kwargs", self.problem_kwargs, max_bytes=256 * 1024))
        object.__setattr__(
            self, "algorithm_configs", _frozen_json_mapping("algorithm_configs", self.algorithm_configs, max_bytes=1024 * 1024)
        )
        object.__setattr__(self, "labels", _frozen_json_mapping("labels", self.labels, max_bytes=256 * 1024))
        object.__setattr__(self, "metadata", _frozen_json_mapping("metadata", self.metadata, max_bytes=1024 * 1024))

    def as_intent_dict(self) -> dict[str, Any]:
        """Return a detached JSON representation without document identity."""
        return {
            "matrix": {
                "problems": list(self.problems),
                "algorithms": list(self.algorithms),
                "seeds": list(self.seeds),
            },
            "run_defaults": {
                "max_evaluations": self.max_evaluations,
                "pop_size": self.pop_size,
                "engine": self.engine,
                "eval_strategy": self.eval_strategy,
                "n_var": self.n_var,
                "n_obj": self.n_obj,
                "problem_kwargs": deep_thaw(self.problem_kwargs),
                "algorithm_configs": deep_thaw(self.algorithm_configs),
            },
            "policy": {
                "on_error": self.on_error,
                "max_attempts_per_task": self.max_attempts_per_task,
            },
            "labels": deep_thaw(self.labels),
            "metadata": deep_thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanTask:
    plan_index: int
    requested_run: Mapping[str, Any]
    resolved_run_spec: Mapping[str, Any]
    task_id: str
    task_spec_sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedStudyPlan:
    plan_id: str
    tasks: tuple[PlanTask, ...]
    document_sha256: str

    @property
    def task_count(self) -> int:
        return len(self.tasks)


@dataclass(frozen=True, slots=True)
class DocumentReference:
    path: str
    role: str
    required_for: tuple[str, ...]
    semantic_sha256: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class StudyCounts:
    tasks: int
    pending: int
    running: int
    succeeded: int
    failed: int
    interrupted: int
    cancelled: int
    skipped: int


@dataclass(frozen=True, slots=True)
class StudyManifest:
    study_id: str
    plan_id: str
    state: StudyState
    created_at: str
    updated_at: str
    execution_id: str | None
    on_error: OnErrorPolicy
    max_attempts_per_task: int
    spec: DocumentReference
    plan: DocumentReference
    counts: StudyCounts
    checkpoint_sequence: int
    checkpoint_event_sha256: str
    document_sha256: str


@dataclass(frozen=True, slots=True)
class AttemptReference:
    attempt_id: str
    attempt_number: int
    path: str
    role: str
    required_for: tuple[str, ...]
    semantic_sha256: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class Retryability:
    retryable: bool
    category: str | None
    attempts_remaining: int


@dataclass(frozen=True, slots=True)
class TaskRecord:
    study_id: str
    task_id: str
    plan_index: int
    state: TaskState
    attempts: tuple[AttemptReference, ...]
    current_attempt_id: str | None
    selected_success_attempt_id: str | None
    retryability: Retryability
    reason: Mapping[str, Any] | None
    claim_epoch: int
    document_sha256: str


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    study_id: str
    task_id: str
    attempt_id: str
    attempt_number: int
    execution_id: str
    status: AttemptState
    timestamps: Mapping[str, Any]
    lease_evidence: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None
    run_reference: Mapping[str, Any] | None
    document_sha256: str


@dataclass(frozen=True, slots=True)
class StudyEvent:
    sequence: int
    event_id: str
    event_type: str
    entity_kind: str
    entity_id: str
    transition_from: str | None
    transition_to: str
    execution_id: str | None
    timestamp: str
    reason: Mapping[str, Any] | None
    payload: Mapping[str, Any]
    previous_event_sha256: str | None
    document_sha256: str
    file_sha256: str = ""


@dataclass(frozen=True, slots=True)
class Study:
    """Immutable, data-only handle for a verified persisted study."""

    root: Path
    manifest: StudyManifest
    spec: StudySpec
    plan: ResolvedStudyPlan
    tasks: tuple[TaskRecord, ...]
    attempts: tuple[AttemptRecord, ...] = field(repr=False)
    events: tuple[StudyEvent, ...] = field(repr=False)
    stored_checkpoint_sequence: int = field(repr=False)
    stored_checkpoint_event_sha256: str = field(repr=False)
    reconciliation_required: bool = field(repr=False)
    # The limits this handle was loaded with; every reload by run(), resume(), retry(), cancel(),
    # inspect() and summarize() uses them, and run_limits bound the study's own runs.
    limits: StudyLoadLimits = field(default=StudyLoadLimits(), kw_only=True, repr=False, compare=False)
    run_limits: LoadLimits = field(default=LoadLimits(), kw_only=True, repr=False, compare=False)

    @property
    def study_id(self) -> str:
        return self.manifest.study_id

    @property
    def plan_id(self) -> str:
        return self.manifest.plan_id

    @property
    def status(self) -> StudyState:
        return self.manifest.state

    def run(self) -> Study:
        """Execute this newly created durable study sequentially."""
        from .execution import execute_study

        return execute_study(self)

    def cancel(self) -> Study:
        """Cancel this study or request cancellation from its local runner."""
        from .execution import cancel_study

        return cancel_study(self)

    def resume(self, *, retry_failed: bool = False) -> Study:
        """Reconcile this study and execute its eligible unfinished tasks."""
        from .recovery import resume_study

        return resume_study(self, retry_failed=retry_failed)

    def retry(self, *, failed_only: bool = True) -> Study:
        """Explicitly retry eligible terminal attempts without changing the plan."""
        from .recovery import retry_study

        return retry_study(self, failed_only=failed_only)

    def inspect(self) -> StudyReport:
        """Reload and project current durable state without modifying it."""
        from .projection import project_study

        return project_study(self).report

    def summarize(self) -> StudySummary:
        """Return a deterministic in-memory summary without writing files."""
        from .projection import project_study

        return project_study(self).summary


__all__ = [
    "AttemptRecord",
    "AttemptReference",
    "DocumentReference",
    "OnErrorPolicy",
    "PlanTask",
    "ResolvedStudyPlan",
    "Retryability",
    "SCHEMA_VERSION",
    "Study",
    "StudyCounts",
    "StudyEvent",
    "StudyManifest",
    "StudySpec",
    "TaskRecord",
]
