"""Panel-free progress, failures and hypervolume of a durable study for VAMOS Studio.

Everything here reads the study through the data-only ``load_study`` view, its
``inspect()`` report and its ``summarize()`` projection; nothing writes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vamos.study_artifacts import load_study
from vamos.ux.studio.algorithm_catalog import algorithm_label
from vamos.ux.studio.batch import BatchFailure, BatchRun, normalized_hypervolumes, run_rows, summary_rows
from vamos.ux.studio.data import RunRecord, load_runs_from_study

TERMINAL_TASK_STATES = ("succeeded", "failed", "cancelled", "skipped")
TERMINAL_STUDY_STATES = ("completed", "completed_with_failures", "cancelled", "failed")


@dataclass(frozen=True)
class TaskRow:
    """One task of a study, named by its problem, algorithm and seed."""

    problem: str
    algorithm: str
    seed: int | None
    state: str
    failure: str = ""
    retryable: bool = False

    def label(self) -> str:
        seed = f", seed {self.seed}" if self.seed is not None else ""
        return f"{self.problem} / {self.algorithm}{seed}"


@dataclass(frozen=True)
class StudyProgress:
    """A study's state, counts and timing at one moment."""

    root: Path
    study_id: str
    state: str
    counts: Mapping[str, int]
    next_actions: tuple[str, ...] = ()
    running: tuple[TaskRow, ...] = ()
    failures: tuple[TaskRow, ...] = ()
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    updated_at: str = ""
    issues: tuple[str, ...] = field(default=())

    @property
    def tasks(self) -> int:
        return int(self.counts.get("tasks", 0))

    @property
    def finished(self) -> int:
        return sum(int(self.counts.get(state, 0)) for state in TERMINAL_TASK_STATES)

    @property
    def fraction(self) -> float:
        return self.finished / self.tasks if self.tasks else 1.0

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STUDY_STATES


def component_name(component_id: object) -> str:
    """``vamos.problem:zdt1@1`` -> ``zdt1``."""
    text = str(component_id or "unknown")
    return text.split(":", 1)[-1].split("@", 1)[0]


def study_progress(root: Path, *, now: datetime | None = None) -> StudyProgress:
    """Project the current state of the study at ``root``."""
    study = load_study(root)
    report = study.inspect().as_dict()
    summary = study.summarize().as_dict()
    rows = [_task_row(row) for row in _mappings(summary.get("rows"))]
    attempts = _mappings(report.get("attempts"))
    counts = {str(key): int(value) for key, value in _mapping(report.get("counts")).items() if isinstance(value, int)}
    elapsed, eta = _timing(attempts, counts, terminal=str(report.get("state")) in TERMINAL_STUDY_STATES, now=now)
    timestamps = _mapping(report.get("timestamps"))
    return StudyProgress(
        root=Path(root),
        study_id=str(report.get("study_id", "")),
        state=str(report.get("state", "unknown")),
        counts=counts,
        next_actions=tuple(str(action) for action in _sequence(report.get("next_actions"))),
        running=tuple(row for row in rows if row.state == "running"),
        failures=tuple(row for row in rows if row.state == "failed"),
        elapsed_seconds=elapsed,
        eta_seconds=eta,
        updated_at=str(timestamps.get("updated_at", "")),
        issues=tuple(str(_mapping(issue).get("message", issue)) for issue in _sequence(report.get("issues"))),
    )


def journal_marker(root: Path) -> int:
    """How many journal events the study has, a cheap signal that it changed.

    The journal is append-only, so an unchanged count means an unchanged study;
    the count is only a trigger to reload through ``load_study``.
    """
    try:
        with os.scandir(Path(root) / "events") as entries:
            return sum(1 for entry in entries if entry.name.endswith(".json"))
    except OSError:
        return -1


def study_hypervolume(
    root: Path, failures: Sequence[TaskRow] = (), *, cache: dict[str, RunRecord] | None = None
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summary and per-run rows of the normalized hypervolume of the completed runs."""
    runs = [
        BatchRun(
            problem=run.problem_name,
            algorithm=algorithm_label(run.algorithm_name),
            seed=run.seed,
            F=np.atleast_2d(np.asarray(run.fun, dtype=float)),
            G=None if run.constraints is None else np.atleast_2d(np.asarray(run.constraints, dtype=float)),
        )
        for run in load_runs_from_study(root, cache=cache)
        if np.asarray(run.fun).size
    ]
    hv = normalized_hypervolumes(runs)
    failed = [BatchFailure(row.problem, row.algorithm, row.seed, row.failure) for row in failures]
    return summary_rows(runs, hv, failed), run_rows(runs, hv)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def _task_row(row: Mapping[str, object]) -> TaskRow:
    failure = _mapping(row.get("failure"))
    seed = row.get("seed")
    return TaskRow(
        problem=component_name(row.get("problem_id")),
        algorithm=algorithm_label(component_name(row.get("algorithm_id"))),
        seed=seed if isinstance(seed, int) else None,
        state=str(row.get("state", "unknown")),
        failure=str(failure.get("message") or row.get("failure_code") or ""),
        retryable=bool(row.get("retryable", False)),
    )


def _timing(
    attempts: Sequence[Mapping[str, object]],
    counts: Mapping[str, int],
    *,
    terminal: bool,
    now: datetime | None,
) -> tuple[float | None, float | None]:
    started = [moment for moment in (_timestamp(attempt.get("started_at")) for attempt in attempts) if moment is not None]
    if not started:
        return None, None
    durations: list[float] = []
    completed: list[datetime] = []
    for attempt in attempts:
        begin, end = _timestamp(attempt.get("started_at")), _timestamp(attempt.get("completed_at"))
        if begin is not None and end is not None:
            durations.append((end - begin).total_seconds())
            completed.append(end)
    current = now or datetime.now(timezone.utc)
    end_time = max(completed) if terminal and completed else current
    elapsed = (end_time - min(started)).total_seconds()
    remaining = int(counts.get("pending", 0)) + int(counts.get("running", 0)) + int(counts.get("interrupted", 0))
    eta = float(np.mean(durations)) * remaining if durations and remaining and not terminal else None
    return elapsed, eta


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _mappings(value: object) -> list[Mapping[str, object]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, (list, tuple)) else []


__all__ = [
    "StudyProgress",
    "TaskRow",
    "component_name",
    "format_duration",
    "journal_marker",
    "study_hypervolume",
    "study_progress",
]
