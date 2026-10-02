"""Launch GUI runs in isolated worker processes, monitor them, and cancel them.

Each run lives in its own job directory::

    <jobs_root>/<timestamp>-<algorithm>-<problem>/
        request.json   validated run request
        events.jsonl   progress events (see ``events``)
        worker.log     worker stdout/stderr
        run/           canonical stored run, written on success

Running optimizations out of process keeps the interface responsive (no GIL
contention or JIT compilation in the server), isolates crashes, and makes
cancellation a process termination instead of a cooperative protocol.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .events import EVENTS_FILENAME, append_event, has_terminal_event

REQUEST_FILENAME = "request.json"
LOG_FILENAME = "worker.log"
RUN_DIRNAME = "run"
ENGINES = ("numpy", "numba", "moocore")
WORKER_MODULE = "vamos.experiment.gui.worker"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class RunRequest:
    """Built-in problem/algorithm run requested from the GUI."""

    problem: str
    algorithm: str = "auto"
    max_evaluations: int = 10000
    pop_size: int | None = None
    seed: int = 1
    engine: str = "numpy"

    def validate(self) -> None:
        """Raise ``ValueError`` with an actionable message if the request is invalid."""
        from vamos.engine.algorithm.registry import get_algorithms_registry
        from vamos.foundation.problem.registry import available_problem_names

        if self.problem not in set(available_problem_names()):
            raise ValueError(f"Unknown problem {self.problem!r}. Choose a registered built-in problem.")
        algorithms = set(get_algorithms_registry()) | {"auto"}
        if self.algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm {self.algorithm!r}. Available: {', '.join(sorted(algorithms))}.")
        if self.engine not in ENGINES:
            raise ValueError(f"Unknown engine {self.engine!r}. Available: {', '.join(ENGINES)}.")
        if not _is_int(self.max_evaluations) or self.max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive integer.")
        if self.pop_size is not None and (not _is_int(self.pop_size) or self.pop_size < 2):
            raise ValueError("pop_size must be an integer >= 2 when given.")
        if self.pop_size is not None and self.max_evaluations < self.pop_size:
            raise ValueError("max_evaluations must be >= pop_size because the initial population consumes pop_size evaluations.")
        if not _is_int(self.seed) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RunRequest:
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown run request fields: {', '.join(unknown)}.")
        if not isinstance(data.get("problem"), str):
            raise ValueError("Run request field 'problem' must be a string.")
        return cls(**{key: data[key] for key in known if key in data})


@dataclass
class JobHandle:
    """A launched (or re-attached) GUI job."""

    job_dir: Path
    request: RunRequest
    process: subprocess.Popen[bytes] | None = None

    @property
    def events_path(self) -> Path:
        return self.job_dir / EVENTS_FILENAME

    @property
    def run_dir(self) -> Path:
        return self.job_dir / RUN_DIRNAME

    @property
    def log_path(self) -> Path:
        return self.job_dir / LOG_FILENAME

    def returncode(self) -> int | None:
        """Return the worker exit code, or ``None`` while it is running (or unknown)."""
        return None if self.process is None else self.process.poll()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait(self, timeout: float | None = None) -> int | None:
        if self.process is None:
            return None
        return self.process.wait(timeout=timeout)

    def cancel(self, timeout: float = 5.0) -> bool:
        """Terminate a running worker and record a ``cancelled`` end event.

        Returns ``False`` when the worker had already finished.
        """
        if self.process is None or self.process.poll() is not None:
            return False
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)
        if not has_terminal_event(self.events_path):
            append_event(
                self.events_path,
                {"type": "end", "status": "cancelled", "run_dir": None, "evaluations": None, "error": "Cancelled from the GUI."},
            )
        return True


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:40] or "run"


def _new_job_dir(jobs_root: Path, request: RunRequest) -> Path:
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(request.algorithm)}-{_slug(request.problem)}"
    candidate = jobs_root / stem
    suffix = 2
    while candidate.exists():
        candidate = jobs_root / f"{stem}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def launch_job(request: RunRequest, jobs_root: str | Path, *, python: str | None = None) -> JobHandle:
    """Validate ``request``, create its job directory, and start the worker process."""
    request.validate()
    job_dir = _new_job_dir(Path(jobs_root), request)
    (job_dir / REQUEST_FILENAME).write_text(json.dumps(request.to_json(), indent=2) + "\n", encoding="utf-8")
    command = [python or sys.executable, "-m", WORKER_MODULE, str(job_dir)]
    with (job_dir / LOG_FILENAME).open("wb") as log:
        process = subprocess.Popen(command, cwd=job_dir, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return JobHandle(job_dir=job_dir, request=request, process=process)


__all__ = [
    "ENGINES",
    "LOG_FILENAME",
    "REQUEST_FILENAME",
    "RUN_DIRNAME",
    "JobHandle",
    "RunRequest",
    "launch_job",
]
