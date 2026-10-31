"""Tests for the background study runners and study progress of VAMOS Studio."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vamos.study_artifacts import StudySpec, create_study, load_study
from vamos.ux.studio import study_jobs, study_worker
from vamos.ux.studio.batch_studies import batch_spec, check_plan, default_batch_folder, list_studies, runner_actions
from vamos.ux.studio.data import build_fronts, load_runs_from_study
from vamos.ux.studio.study_jobs import JobConflictError, JobStatus, job_dir_for, job_status, log_tail, start_job, stop_job, write_json
from vamos.ux.studio.study_progress import StudyProgress, component_name, format_duration, study_hypervolume, study_progress


def _tiny_spec(**overrides: object) -> StudySpec:
    settings: dict[str, object] = {
        "problems": ["zdt1"],
        "algorithms": ["nsgaii"],
        "seeds": [0, 1],
        "pop_size": 8,
        "max_evaluations": 40,
        "engine": "numpy",
        "on_error": "continue",
    }
    settings.update(overrides)
    return StudySpec(**settings)  # type: ignore[arg-type]


def _job(tmp_path: Path, **files: dict[str, object]) -> Path:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True)
    for name, payload in files.items():
        write_json(job_dir / f"{name}.json", payload)
    return job_dir


def test_job_phases_follow_the_job_files(tmp_path: Path) -> None:
    now = 1_000_000.0

    assert job_status(tmp_path / "missing").phase == "none"
    assert job_status(_job(tmp_path / "a", job={"operation": "run", "started_at": now - 5}), now=now).phase == "starting"
    running = job_status(
        _job(tmp_path / "b", job={"operation": "run", "started_at": now - 50}, heartbeat={"pid": 4321, "time": now - 3}), now=now
    )
    assert (running.phase, running.pid, running.active) == ("running", 4321, True)
    assert (
        job_status(
            _job(tmp_path / "c", job={"operation": "run", "started_at": now - 90}, heartbeat={"pid": 1, "time": now - 60}), now=now
        ).phase
        == "lost"
    )
    assert job_status(_job(tmp_path / "d", job={"operation": "run", "started_at": now - 600}), now=now).phase == "lost"
    finished = job_status(
        _job(tmp_path / "e", job={"operation": "resume"}, outcome={"status": "finished", "message": "the study is completed"})
    )
    assert (finished.phase, finished.active) == ("finished", False)
    assert "completed" in finished.describe()


def test_stop_requests_target_a_worker_token_without_signalling_a_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signalled: list[int] = []
    monkeypatch.setattr(study_jobs.os, "kill", lambda pid, _sig: signalled.append(pid))
    stale = _job(
        tmp_path / "stale", job={"operation": "run", "started_at": time.time() - 600}, heartbeat={"pid": 111, "time": time.time() - 600}
    )
    fresh = _job(
        tmp_path / "fresh",
        job={"operation": "run", "started_at": time.time()},
        heartbeat={"pid": 222, "worker_id": "worker-a", "time": time.time()},
    )

    assert stop_job(stale).phase == "lost"
    assert stop_job(fresh).phase == "stopping"
    assert job_status(fresh).active
    assert signalled == []
    assert json.loads((fresh / "stop.json").read_text())["worker_id"] == "worker-a"
    assert not (fresh / "outcome.json").exists()


def test_stop_never_signals_the_process_id_in_a_heartbeat(tmp_path: Path) -> None:
    # A worker killed without an outcome leaves a fresh heartbeat for a while; its process id may then belong to
    # an unrelated process, which stopping the job must not touch.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        job_dir = _job(
            tmp_path,
            job={"operation": "run", "started_at": time.time()},
            heartbeat={"pid": unrelated.pid, "worker_id": "former-worker", "time": time.time()},
        )

        status = stop_job(job_dir)

        assert status.phase == "stopping"
        assert unrelated.poll() is None, "stopping the job killed an unrelated process"
    finally:
        unrelated.kill()
        unrelated.wait()


def test_stop_refuses_a_heartbeat_without_worker_identity(tmp_path: Path) -> None:
    job_dir = _job(tmp_path, job={"operation": "run"}, heartbeat={"pid": 222, "time": time.time()})
    with pytest.raises(JobConflictError, match="no worker identity"):
        stop_job(job_dir)
    assert not (job_dir / "stop.json").exists()


def test_a_replacement_worker_ignores_an_old_stop_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = _job(tmp_path, job={"operation": "run"}, stop={"worker_id": "old-worker"})
    done = threading.Event()
    monkeypatch.setattr(done, "wait", lambda _timeout: done.set())
    study_worker._beat(job_dir, done, "replacement-worker")
    assert json.loads((job_dir / "heartbeat.json").read_text())["worker_id"] == "replacement-worker"
    assert not (job_dir / "outcome.json").exists()


def test_only_the_addressed_worker_interrupts_its_main_thread_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_dir = _job(tmp_path, job={"operation": "run"}, stop={"worker_id": "worker-a"})
    done = threading.Event()
    interrupts: list[int] = []
    polls: list[float] = []

    def wait(timeout: float) -> None:
        polls.append(timeout)
        if len(polls) == 3:
            done.set()

    monkeypatch.setattr(study_worker._thread, "interrupt_main", interrupts.append)
    monkeypatch.setattr(done, "wait", wait)

    study_worker._beat(job_dir, done, "worker-a")

    assert interrupts == [signal.SIGTERM]
    assert job_status(job_dir).phase == "stopping"
    assert not (job_dir / "outcome.json").exists(), "only the owning thread can acknowledge its completed cleanup"


def test_stop_ends_the_detached_worker_without_terminal_study_cancellation(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(max_evaluations=100_000_000), output=tmp_path / "study")
    start_job(study.root, study.study_id, "run")
    job_dir = job_dir_for(study.root, study.study_id)
    deadline = time.monotonic() + 30
    while job_status(job_dir).phase == "starting" and time.monotonic() < deadline:
        time.sleep(0.1)
    assert stop_job(job_dir).phase in {"stopping", "stopped"}
    while job_status(job_dir).active and time.monotonic() < deadline:
        time.sleep(0.1)
    assert job_status(job_dir).phase == "stopped", log_tail(job_dir)
    assert load_study(study.root).inspect().as_dict()["state"] in {"created", "running"}
    assert job_status(job_dir).message == study_jobs.STOPPED_BY_RUNNER
    beat = (job_dir / "heartbeat.json").stat().st_mtime
    time.sleep(1.5 * study_jobs.HEARTBEAT_SECONDS)
    assert (job_dir / "heartbeat.json").stat().st_mtime == beat


def test_stop_during_an_atomic_attempt_write_cleans_up_before_acknowledgment(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(seeds=[0]), output=tmp_path / "study")
    job_dir = job_dir_for(study.root, study.study_id)
    job_dir.mkdir(parents=True)
    write_json(job_dir / "job.json", {"operation": "run", "started_at": time.time()})
    # Stop precisely while an attempt's temporary file is open. A hard process
    # exit strands that file and makes the canonical study unreadable.
    script = """
import sys
import time
from pathlib import Path
from vamos.experiment.study import writing
from vamos.ux.studio import study_worker

original_write = writing.write_bytes_atomic
original_fsync = writing.os.fsync
inside_attempt = False

def write(path, payload):
    global inside_attempt
    inside_attempt = path.parent.name == "attempts"
    try:
        original_write(path, payload)
    finally:
        inside_attempt = False

def fsync(fd):
    if inside_attempt:
        Path(sys.argv[2], "inside-attempt-write").touch()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            time.sleep(0.05)
        raise RuntimeError("the test did not stop the worker")
    original_fsync(fd)

writing.write_bytes_atomic = write
writing.os.fsync = fsync
raise SystemExit(study_worker.main([
    "--study", sys.argv[1], "--job-dir", sys.argv[2], "--operation", "run"
]))
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(study.root), str(job_dir)])
    try:
        deadline = time.monotonic() + 30
        while not (job_dir / "inside-attempt-write").exists() and time.monotonic() < deadline:
            assert process.poll() is None, "worker exited before the interrupted write"
            time.sleep(0.05)
        assert (job_dir / "inside-attempt-write").exists()
        assert list(study.root.rglob(".tmp-*")), "the test must interrupt an actual open temporary file"
        assert stop_job(job_dir).phase in {"stopping", "stopped"}
        assert process.wait(timeout=30) == study_worker.STOPPED_EXIT_CODE
        assert job_status(job_dir).phase == "stopped"
        assert not list(study.root.rglob(".tmp-*"))
        assert load_study(study.root).resume().status == "completed"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_a_starting_runner_cannot_be_stopped_or_doubled(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(), output=tmp_path / "study")
    job_dir = job_dir_for(study.root, study.study_id)
    job_dir.mkdir(parents=True)
    write_json(job_dir / "job.json", {"operation": "run", "started_at": time.time()})

    with pytest.raises(JobConflictError):
        stop_job(job_dir)
    with pytest.raises(JobConflictError):
        start_job(study.root, study.study_id, "run")


def test_worker_runs_the_study_and_records_its_outcome(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(), output=tmp_path / "study")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    write_json(job_dir / "job.json", {"operation": "run", "started_at": time.time()})

    previous_handler = signal.getsignal(signal.SIGTERM)
    code = study_worker.main(["--study", str(study.root), "--job-dir", str(job_dir), "--operation", "run"])

    status = job_status(job_dir)
    assert code == 0
    assert status.phase == "finished"
    assert "completed" in status.message
    assert (job_dir / "heartbeat.json").is_file()
    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_worker_reports_an_operation_the_study_refuses(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(), output=tmp_path / "study").run()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    write_json(job_dir / "job.json", {"operation": "run", "started_at": time.time()})

    code = study_worker.main(["--study", str(study.root), "--job-dir", str(job_dir), "--operation", "run"])

    assert code == 1
    assert job_status(job_dir).phase == "failed"


def test_a_background_runner_completes_the_study(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(), output=tmp_path / "study")

    start_job(study.root, study.study_id, "run")
    job_dir = job_dir_for(study.root, study.study_id)
    deadline = time.time() + 120
    while job_status(job_dir).active and time.time() < deadline:
        time.sleep(0.5)

    assert job_status(job_dir).phase == "finished", log_tail(job_dir)
    assert study_progress(study.root).state == "completed"
    assert "=== " in log_tail(job_dir)


def test_progress_and_hypervolume_of_a_completed_study(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(algorithms=["nsgaii", "spea2"]), output=tmp_path / "study").run()

    progress = study_progress(study.root)
    summary, runs = study_hypervolume(study.root, progress.failures)

    assert (progress.state, progress.tasks, progress.finished, progress.fraction) == ("completed", 4, 4, 1.0)
    assert progress.terminal and progress.eta_seconds is None
    assert progress.elapsed_seconds is not None and progress.elapsed_seconds >= 0
    assert {row["Algorithm"] for row in summary} == {"NSGA-II", "SPEA2"}
    assert len(runs) == 4


def test_failed_tasks_are_listed_with_their_failure(tmp_path: Path) -> None:
    # MOEA/D needs 91 weight vectors for three objectives, so its task fails.
    study = create_study(_tiny_spec(problems=["dtlz2"], algorithms=["moead"], seeds=[0]), output=tmp_path / "study").run()

    progress = study_progress(study.root)

    assert progress.state == "completed_with_failures"
    assert [row.label() for row in progress.failures] == ["dtlz2 / MOEA/D, seed 0"]
    assert progress.failures[0].failure


def test_constraints_reach_loaded_runs_and_fronts(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(problems=["c1dtlz1"], seeds=[0]), output=tmp_path / "study").run()

    runs = load_runs_from_study(study.root)
    fronts = build_fronts(runs)

    assert runs[0].constraints is not None
    assert fronts[0].constraints is not None
    assert fronts[0].constraints.shape[0] == fronts[0].points_F.shape[0]


def _progress(state: str, actions: tuple[str, ...], succeeded: int = 0) -> StudyProgress:
    return StudyProgress(root=Path("s"), study_id="id", state=state, counts={"tasks": 2, "succeeded": succeeded}, next_actions=actions)


def test_runner_actions_follow_the_study_and_its_runner() -> None:
    idle = JobStatus("none")

    assert runner_actions(_progress("created", ("run",)), idle).start
    running = runner_actions(
        _progress("running", ("wait_for_current_owner_or_resume_after_interruption",), 1), JobStatus("running", "run", pid=5)
    )
    assert (running.stop, running.resume, running.open_results) == (True, False, True)
    orphan = runner_actions(_progress("running", ("wait_for_current_owner_or_resume_after_interruption",)), idle)
    assert orphan.resume and orphan.needs_confirmation
    stopped = runner_actions(_progress("running", ("wait_for_current_owner_or_resume_after_interruption",)), JobStatus("stopped", "run"))
    assert stopped.resume and not stopped.needs_confirmation
    assert runner_actions(_progress("completed_with_failures", ("retry_failed",), 1), idle).retry_failed
    assert runner_actions(None, idle) == runner_actions(None, JobStatus("running"))


def test_plan_check_explains_an_impossible_population(tmp_path: Path) -> None:
    # RVEA needs one individual per reference vector: no three-objective lattice has 50 (45 or 55 do).
    spec = batch_spec(problems=["dtlz2"], algorithms=["rvea"], seeds=[0], max_evaluations=100, pop_size=50, on_error="continue")

    plan = check_plan(spec, tmp_path / "batch")

    assert not plan.valid
    assert "REFERENCE_DIRECTION_POPULATION_MISMATCH" in plan.text


def test_plan_check_describes_a_valid_batch(tmp_path: Path) -> None:
    spec = batch_spec(
        problems=["zdt1", "zdt2"], algorithms=["nsgaii", "nsgaiii"], seeds=[0, 1], max_evaluations=1000, pop_size=None, on_error="fail_fast"
    )

    plan = check_plan(spec, tmp_path / "batch")

    assert plan.valid
    assert plan.task_count == 8
    assert "available" in plan.text
    assert default_batch_folder(tmp_path, now=0).name.startswith("batch-")


def test_list_studies_finds_batches_and_ignores_runner_files(tmp_path: Path) -> None:
    study = create_study(_tiny_spec(), output=tmp_path / "batches" / "one").run()
    (tmp_path / "batches" / ".vamos-studio" / "jobs").mkdir(parents=True)

    listings = list_studies(tmp_path / "batches")

    assert [listing.root for listing in listings] == [study.root.resolve()]
    assert listings[0].finished == listings[0].tasks == 2
    assert "completed" in listings[0].label()
    assert list_studies(tmp_path / "missing") == []


def test_small_helpers() -> None:
    assert component_name("vamos.problem:zdt1@1") == "zdt1"
    assert format_duration(None) == "unknown"
    assert format_duration(42) == "42 s"
    assert format_duration(3725) == "1 h 02 min"
    assert load_study  # the data-only reader stays the only way the monitor reads a study
    assert datetime.now(timezone.utc)
