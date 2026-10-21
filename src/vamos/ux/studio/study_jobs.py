"""Background runners for the durable studies that VAMOS Studio starts.

Studio starts one worker process per study operation (run, resume or retry).
The worker owns the study for its whole lifetime, exactly like
``vamos study run`` in a terminal, so Studio itself never writes to a study.
The job files live outside the study, in
``<parent>/.vamos-studio/jobs/<study-id>/``:

- ``job.json``: operation, study path and start time, written by Studio;
- ``heartbeat.json``: the worker's process id, unique token and a timestamp, refreshed every
  few seconds while it runs;
- ``stop.json``: a stop request addressed to one worker's unique token;
- ``outcome.json``: how the worker ended, written by the worker;
- ``worker.log``: the worker's output.

A fresh heartbeat indicates recent activity, not process identity. Studio
requests a stop using the worker's unique token; the matching worker exits
itself. Studio never sends a signal to a PID read from a file.
Stopping a worker unwinds its owning thread before acknowledging the stop,
so atomic writes clean up their temporary files. The task in progress is lost,
the study stays ``running``, and resuming it reconciles and reruns that task.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

JobOperation = Literal["run", "resume", "retry"]
JobPhase = Literal["none", "starting", "running", "stopping", "finished", "failed", "stopped", "lost"]

HEARTBEAT_SECONDS = 2.0
STOP_POLL_SECONDS = 0.25
STOPPED_BY_RUNNER = "The runner stopped at the request of Studio."
STALE_AFTER_SECONDS = 15.0
STARTUP_GRACE_SECONDS = 60.0
WORKER_MODULE = "vamos.ux.studio.study_worker"

JOB_FILE = "job.json"
HEARTBEAT_FILE = "heartbeat.json"
OUTCOME_FILE = "outcome.json"
STOP_FILE = "stop.json"
LOG_FILE = "worker.log"


class JobConflictError(RuntimeError):
    """A Studio worker is already active for this study."""


@dataclass(frozen=True)
class JobStatus:
    """What Studio knows about the latest worker of one study."""

    phase: JobPhase
    operation: str | None = None
    pid: int | None = None
    started_at: float | None = None
    heartbeat_age: float | None = None
    message: str = ""

    @property
    def active(self) -> bool:
        return self.phase in ("starting", "running", "stopping")

    def describe(self) -> str:
        operation = self.operation or "operation"
        if self.phase == "none":
            return "No Studio runner has worked on this study."
        if self.phase == "starting":
            return f"The Studio runner is starting the {operation}."
        if self.phase == "running":
            age = f", heartbeat {self.heartbeat_age:.0f} s ago" if self.heartbeat_age is not None else ""
            return f"A Studio runner is executing the {operation} in the background (process {self.pid}{age})."
        if self.phase == "stopped":
            return f"The {operation} was stopped from Studio."
        if self.phase == "stopping":
            return f"A stop was requested; waiting for the {operation} runner to exit."
        if self.phase == "lost":
            return f"The Studio runner of the {operation} ended without reporting an outcome."
        return f"The {operation} ended: {self.message}" if self.message else f"The {operation} ended."


def jobs_root(study_root: Path) -> Path:
    return study_root.parent / ".vamos-studio" / "jobs"


def job_dir_for(study_root: Path, study_id: str) -> Path:
    return jobs_root(study_root) / study_id


def job_status(job_dir: Path, *, now: float | None = None) -> JobStatus:
    """Classify the latest worker of a study from its job files."""
    job = _read_json(job_dir / JOB_FILE)
    if job is None:
        return JobStatus("none")
    operation = str(job.get("operation", "")) or None
    started_at = _number(job.get("started_at"))
    outcome = _read_json(job_dir / OUTCOME_FILE)
    if outcome is not None:
        status = str(outcome.get("status", "failed"))
        phase: JobPhase = "finished" if status == "finished" else "stopped" if status == "stopped" else "failed"
        return JobStatus(phase, operation, started_at=started_at, message=str(outcome.get("message", "")))
    current = time.time() if now is None else now
    heartbeat = _read_json(job_dir / HEARTBEAT_FILE)
    beat = _number(heartbeat.get("time")) if heartbeat else None
    if beat is not None:
        age = max(0.0, current - beat)
        pid = heartbeat.get("pid") if heartbeat else None
        phase = "running" if age <= STALE_AFTER_SECONDS else "lost"
        request = _read_json(job_dir / STOP_FILE)
        token = heartbeat.get("worker_id") if heartbeat else None
        if phase == "running" and isinstance(token, str) and token and request and request.get("worker_id") == token:
            phase = "stopping"
        return JobStatus(phase, operation, pid=pid if isinstance(pid, int) else None, started_at=started_at, heartbeat_age=age)
    if started_at is not None and current - started_at <= STARTUP_GRACE_SECONDS:
        return JobStatus("starting", operation, started_at=started_at)
    return JobStatus("lost", operation, started_at=started_at)


def start_job(study_root: Path, study_id: str, operation: JobOperation, *, retry_failed: bool = False) -> JobStatus:
    """Start a detached worker for ``operation`` on the study at ``study_root``."""
    job_dir = job_dir_for(study_root, study_id)
    if job_status(job_dir).active:
        raise JobConflictError("A Studio runner is already working on this study.")
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in (HEARTBEAT_FILE, OUTCOME_FILE, STOP_FILE):
        (job_dir / name).unlink(missing_ok=True)
    write_json(
        job_dir / JOB_FILE, {"operation": operation, "study": str(study_root), "started_at": time.time(), "retry_failed": retry_failed}
    )
    command = [sys.executable, "-m", WORKER_MODULE, "--study", str(study_root), "--job-dir", str(job_dir), "--operation", operation]
    if retry_failed:
        command.append("--retry-failed")
    with (job_dir / LOG_FILE).open("ab") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {operation} ===\n".encode())
        log.flush()
        process = _spawn_detached(command, log)
    threading.Thread(target=process.wait, name="vamos-studio-reaper", daemon=True).start()
    return job_status(job_dir)


def stop_job(job_dir: Path) -> JobStatus:
    """Ask the identified worker to exit; never signal a potentially reused PID."""
    status = job_status(job_dir)
    if status.phase == "starting":
        raise JobConflictError("The runner is still starting; stop it once it reports a heartbeat.")
    if status.phase != "running" or status.pid is None:
        return status
    heartbeat = _read_json(job_dir / HEARTBEAT_FILE)
    token = heartbeat.get("worker_id") if heartbeat else None
    if not isinstance(token, str) or not token:
        raise JobConflictError("This runner has no worker identity; stop it from the terminal that started it.")
    write_json(job_dir / STOP_FILE, {"worker_id": token, "time": time.time()})
    return job_status(job_dir)


def log_tail(job_dir: Path, *, max_bytes: int = 16_384) -> str:
    """The end of the worker log, read without loading the whole file."""
    path = job_dir / LOG_FILE
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    return text if size <= max_bytes else text.split("\n", 1)[-1]


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` atomically, so readers never see a partial file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _spawn_detached(command: list[str], log: BinaryIO) -> subprocess.Popen[bytes]:
    """Start ``command`` so that stopping the Studio server does not stop it."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


__all__ = [
    "HEARTBEAT_SECONDS",
    "STOP_POLL_SECONDS",
    "STOPPED_BY_RUNNER",
    "JobConflictError",
    "JobOperation",
    "JobStatus",
    "job_dir_for",
    "job_status",
    "jobs_root",
    "log_tail",
    "start_job",
    "stop_job",
    "write_json",
]
