from __future__ import annotations

from pathlib import Path

import pytest

from vamos.experiment.gui.events import append_event
from vamos.experiment.gui.monitor import ProgressMonitor

pytestmark = pytest.mark.gui


def test_monitor_folds_start_progress_and_end_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    monitor = ProgressMonitor(path)
    assert not monitor.poll()
    assert monitor.state.status == "pending"

    append_event(path, {"type": "start", "time": 0.0, "request": {"max_evaluations": 200}})
    append_event(
        path,
        {
            "type": "progress",
            "time": 0.5,
            "generation": 1,
            "update": 1,
            "evaluations": 100,
            "front": [[0.0, 1.0], [1.0, 0.0]],
            "front_size": 2,
            "hypervolume": 0.5,
        },
    )
    assert monitor.poll()
    state = monitor.state
    assert (state.status, state.evaluations, state.fraction, state.n_obj) == ("running", 100, 0.5, 2)
    assert state.front is not None and state.front.shape == (2, 2)
    assert (state.convergence_x, state.convergence_y) == ([100.0], [0.5])

    append_event(
        path, {"type": "progress", "time": 0.8, "update": 2, "evaluations": None, "front": [], "front_size": 0, "hypervolume": None}
    )
    append_event(path, {"type": "end", "time": 1.2, "status": "succeeded", "run_dir": "run", "evaluations": 200, "error": None})
    assert monitor.poll()
    assert state.finished
    assert (state.status, state.run_dir, state.evaluations, state.fraction, state.elapsed) == ("succeeded", "run", 200, 1.0, 1.2)
    assert state.front is not None and state.front.shape == (2, 2)
    assert len(state.convergence_x) == 1
    assert not monitor.poll()


def test_convergence_uses_the_update_counter_without_evaluations(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_event(path, {"type": "progress", "update": 3, "front": [[1.0, 2.0]], "front_size": 1, "hypervolume": 0.25})
    monitor = ProgressMonitor(path)
    assert monitor.poll()
    assert monitor.state.convergence_x == [3.0]
    assert monitor.state.fraction is None


def test_unknown_end_status_and_silent_worker_exit_are_failures(tmp_path: Path) -> None:
    monitor = ProgressMonitor(tmp_path / "events.jsonl")
    monitor.apply({"type": "end", "status": "weird"})
    assert monitor.state.status == "failed"

    silent = ProgressMonitor(tmp_path / "missing.jsonl")
    silent.mark_exited(3, log_path=tmp_path / "worker.log")
    assert silent.state.status == "failed"
    assert silent.state.error is not None and "code 3" in silent.state.error and "worker.log" in silent.state.error
    silent.mark_exited(4)
    assert "code 3" in silent.state.error
