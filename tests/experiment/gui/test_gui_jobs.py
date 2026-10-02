from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vamos.experiment.artifacts import load_run
from vamos.experiment.gui.catalog import discover_runs, load_front
from vamos.experiment.gui.events import read_events
from vamos.experiment.gui.jobs import JobHandle, RunRequest, launch_job
from vamos.experiment.gui.monitor import ProgressMonitor
from vamos.experiment.gui.worker import run_job

pytestmark = pytest.mark.gui


def _wait(job: JobHandle, timeout: float = 180.0) -> ProgressMonitor:
    monitor = ProgressMonitor(job.events_path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        monitor.poll()
        if job.returncode() is not None:
            monitor.poll()
            return monitor
        time.sleep(0.05)
    job.cancel()
    raise AssertionError(f"GUI worker did not finish within {timeout} s")


def test_job_streams_progress_and_stores_a_canonical_run(tmp_path: Path) -> None:
    request = RunRequest(problem="zdt1", algorithm="nsgaii", max_evaluations=600, pop_size=20, seed=7)
    job = launch_job(request, tmp_path / "jobs")
    state = _wait(job).state
    assert job.returncode() == 0, job.log_path.read_text(encoding="utf-8")
    assert (state.status, state.run_dir, state.evaluations) == ("succeeded", "run", 600)
    assert state.front is not None and state.front.shape[1] == 2
    assert json.loads((job.job_dir / "request.json").read_text(encoding="utf-8")) == request.to_json()
    assert load_run(job.run_dir).status == "succeeded"
    F, X = load_front(job.run_dir)
    assert F.shape[1] == 2
    assert X is not None and X.shape == (F.shape[0], 30)
    (summary,) = discover_runs(tmp_path)
    assert (summary.algorithm, summary.problem, summary.seed, summary.evaluations) == ("nsgaii", "zdt1", 7, 600)
    assert summary.relative_path == f"jobs/{job.job_dir.name}/run"


def test_optimization_failure_is_reported_as_a_failed_job(tmp_path: Path) -> None:
    request = RunRequest(problem="dtlz2", algorithm="moead", max_evaluations=500, pop_size=50, seed=1)
    job = launch_job(request, tmp_path)
    state = _wait(job).state
    assert job.returncode() == 1
    assert state.status == "failed"
    assert state.error is not None and "pop_size" in state.error
    assert "Traceback" in job.log_path.read_text(encoding="utf-8")
    assert not job.run_dir.exists()


def test_worker_rejects_an_invalid_stored_request(tmp_path: Path) -> None:
    (tmp_path / "request.json").write_text('{"problem": "zdt1", "bogus": 1}', encoding="utf-8")
    assert run_job(tmp_path) == 2
    events = read_events(tmp_path / "events.jsonl")[0]
    assert events[-1]["status"] == "failed"
    assert "Unknown run request fields" in str(events[-1]["error"])


def test_cancel_terminates_a_running_worker(tmp_path: Path) -> None:
    job = launch_job(RunRequest(problem="zdt1", algorithm="nsgaii", max_evaluations=5_000_000, pop_size=100, seed=1), tmp_path)
    monitor = ProgressMonitor(job.events_path)
    deadline = time.monotonic() + 120.0
    while monitor.state.front is None and time.monotonic() < deadline and job.returncode() is None:
        monitor.poll()
        time.sleep(0.05)
    assert monitor.state.front is not None, job.log_path.read_text(encoding="utf-8")
    assert job.cancel()
    assert job.returncode() is not None
    monitor.poll()
    assert monitor.state.status == "cancelled"
    assert not job.cancel()
    assert not job.run_dir.exists()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"problem": "nope"}, "Unknown problem"),
        ({"algorithm": "nsga2"}, "Unknown algorithm"),
        ({"engine": "cuda"}, "Unknown engine"),
        ({"max_evaluations": 0}, "max_evaluations"),
        ({"max_evaluations": True}, "max_evaluations"),
        ({"pop_size": 1}, "pop_size"),
        ({"pop_size": 50, "max_evaluations": 10}, ">= pop_size"),
        ({"seed": -1}, "seed"),
    ],
)
def test_invalid_requests_fail_before_anything_is_created(tmp_path: Path, changes: dict[str, object], message: str) -> None:
    request = RunRequest(**{"problem": "zdt1", **changes})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        launch_job(request, tmp_path / "jobs")
    assert not (tmp_path / "jobs").exists()


def test_request_json_round_trip_and_rejection() -> None:
    request = RunRequest(problem="zdt1", pop_size=10)
    assert RunRequest.from_json(request.to_json()) == request
    with pytest.raises(ValueError, match="Unknown run request fields"):
        RunRequest.from_json({"problem": "zdt1", "x": 1})
    with pytest.raises(ValueError, match="'problem'"):
        RunRequest.from_json({"algorithm": "nsgaii"})
