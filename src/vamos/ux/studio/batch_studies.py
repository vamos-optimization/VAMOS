"""Panel-free planning, creation and listing of the batch studies of VAMOS Studio.

A batch is an ordinary durable study: Studio plans it with ``plan_study``,
publishes it with ``create_study`` and lets a background worker run it, so the
same study can be inspected, resumed or summarized with ``vamos study``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vamos.study_artifacts import StudySpec, create_study, load_study, plan_study
from vamos.ux.studio.study_jobs import JobStatus, job_dir_for, job_status, start_job
from vamos.ux.studio.study_progress import StudyProgress

BATCH_PREFIX = "batch-"
OnErrorPolicy = Literal["fail_fast", "continue"]


@dataclass(frozen=True)
class PlanCheck:
    """The outcome of planning a batch, without writing anything."""

    valid: bool
    text: str
    task_count: int = 0


@dataclass(frozen=True)
class StudyListing:
    """One study found under a folder, with its Studio runner."""

    root: Path
    study_id: str
    state: str
    tasks: int
    finished: int
    failed: int
    updated_at: str
    job: JobStatus

    def label(self) -> str:
        runner = " · running in Studio" if self.job.active else ""
        return f"{self.root.name} — {self.state}, {self.finished}/{self.tasks} tasks{runner}"


def batch_spec(
    *,
    problems: Sequence[str],
    algorithms: Sequence[str],
    seeds: Sequence[int],
    max_evaluations: int,
    pop_size: int | None,
    on_error: str,
) -> StudySpec:
    """A study of every algorithm on every problem for every seed."""
    policy: OnErrorPolicy = "continue" if on_error == "continue" else "fail_fast"
    return StudySpec(
        problems=list(problems),
        algorithms=list(algorithms),
        seeds=[int(seed) for seed in seeds],
        max_evaluations=int(max_evaluations),
        pop_size=None if pop_size is None else int(pop_size),
        on_error=policy,
    )


def default_batch_folder(base: Path, *, now: float | None = None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() if now is None else now))
    return base / f"{BATCH_PREFIX}{stamp}"


def check_plan(spec: StudySpec, output: Path) -> PlanCheck:
    """Resolve the exact plan and describe it, or explain why it cannot run."""
    try:
        report = plan_study(spec, output=output).as_dict()
    except Exception as exc:
        return PlanCheck(False, f"The batch cannot be planned: {exc}")
    errors = [str(error) for error in _sequence(report.get("errors"))]
    output_info = _mapping(report.get("output"))
    available = bool(output_info.get("available", True))
    populations = sorted({int(size) for size in _sequence(report.get("population_sizes")) if isinstance(size, int)})
    task_count = _int(report.get("task_count"))
    budget = _int(report.get("total_evaluation_budget"))
    lines = [
        f"**{task_count} runs**, {budget:,} evaluations in total.",
        f"Population sizes: {', '.join(str(size) for size in populations) or 'none'}.",
        f"Failure policy: {report.get('failure_policy', spec.on_error)}.",
        f"Output: `{output}` ({'available' if available else 'already in use'}).",
    ]
    lines += [f"Error: {error}" for error in errors]
    valid = bool(report.get("valid", False)) and available and not errors
    return PlanCheck(valid, "\n\n".join(lines), task_count)


def create_and_start(spec: StudySpec, output: Path) -> JobStatus:
    """Publish the study and start a background worker that runs it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    study = create_study(spec, output=output)
    return start_job(study.root, study.study_id, "run")


def list_studies(base: Path, *, limit: int = 50) -> list[StudyListing]:
    """Studies at or below ``base``, most recently updated first."""
    base = Path(base).expanduser()
    if not base.is_dir():
        return []
    candidates = [base] if (base / "study-manifest.json").is_file() else []
    candidates += [
        manifest.parent
        for manifest in base.rglob("study-manifest.json")
        if manifest.parent != base and ".vamos-studio" not in manifest.parts and "vamos-study-staging" not in str(manifest.parent.name)
    ]
    listings: list[StudyListing] = []
    for root in candidates:
        try:
            report = load_study(root).inspect().as_dict()
        except Exception:
            continue
        counts = _mapping(report.get("counts"))
        study_id = str(report.get("study_id", ""))
        listings.append(
            StudyListing(
                root=root.resolve(),
                study_id=study_id,
                state=str(report.get("state", "unknown")),
                tasks=_int(counts.get("tasks")),
                finished=sum(_int(counts.get(state)) for state in ("succeeded", "failed", "cancelled", "skipped")),
                failed=_int(counts.get("failed")),
                updated_at=str(_mapping(report.get("timestamps")).get("updated_at", "")),
                job=job_status(job_dir_for(root.resolve(), study_id)),
            )
        )
    listings.sort(key=lambda listing: listing.updated_at, reverse=True)
    return listings[:limit]


@dataclass(frozen=True)
class RunnerActions:
    """What Studio may do next with a study, and why."""

    stop: bool = False
    start: bool = False
    resume: bool = False
    retry_failed: bool = False
    open_results: bool = False
    needs_confirmation: bool = False
    guidance: str = ""


def runner_actions(progress: StudyProgress | None, job: JobStatus) -> RunnerActions:
    """Offer only the operations the study's state and its Studio runner allow."""
    if progress is None:
        return RunnerActions()
    open_results = int(progress.counts.get("succeeded", 0)) > 0
    if job.active:
        return RunnerActions(stop=job.phase == "running", open_results=open_results, guidance=job.describe())
    actions = set(progress.next_actions)
    owner_may_exist = "wait_for_current_owner_or_resume_after_interruption" in actions
    resume = bool(actions & {"resume", "resume_to_reconcile"}) or owner_may_exist
    guidance = ""
    if owner_may_exist and job.phase == "none":
        guidance = (
            "The study is marked as running, but no Studio runner owns it. If no other process is running it "
            "(for example `vamos study run` in a terminal), confirm and resume it."
        )
    elif owner_may_exist:
        guidance = "The runner of this study ended before the study finished; resume it to rerun the interrupted task."
    elif "create_new_study" in actions:
        guidance = "The study failed for an infrastructure reason; create a new batch."
    elif "restore_referenced_run_evidence" in actions:
        guidance = "Some run files that the study references are missing or unreadable; see `vamos study inspect`."
    elif job.phase == "failed" and job.message:
        guidance = f"The last operation failed: {job.message}"
    return RunnerActions(
        start="run" in actions,
        resume=resume,
        retry_failed="retry_failed" in actions,
        open_results=open_results,
        needs_confirmation=owner_may_exist and job.phase == "none",
        guidance=guidance,
    )


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


__all__ = [
    "PlanCheck",
    "RunnerActions",
    "StudyListing",
    "batch_spec",
    "check_plan",
    "create_and_start",
    "default_batch_folder",
    "list_studies",
    "runner_actions",
]
