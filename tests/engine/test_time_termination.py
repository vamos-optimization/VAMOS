"""The ``("max_time", seconds)`` termination: a wall-clock limit, recorded as not reproducible."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import pytest

import vamos
from vamos.engine.hooks.live_viz import NoOpLiveVisualization
from vamos.foundation.exceptions import ConfigurationError

ALGORITHMS = (
    "agemoea",
    "ccmo",
    "cmoeams",
    "ctaea",
    "ibea",
    "moead",
    "nsgaii",
    "nsgaiii",
    "pps",
    "rvea",
    "smpso",
    "smsemoa",
    "spea2",
    "top",
)


class _Clock:
    """A clock that advances one second each time it is read."""

    def __init__(self) -> None:
        self._ticks = itertools.count()

    def __call__(self) -> float:
        return float(next(self._ticks))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    from vamos.experiment import time_limit

    fake = _Clock()
    monkeypatch.setattr(time_limit, "_clock", fake)
    return fake


def _run(algorithm: str, termination: Any, **kwargs: Any) -> vamos.OptimizationResult:
    return vamos.optimize("zdt1", algorithm=algorithm, termination=termination, pop_size=20, seed=1, engine="numpy", **kwargs)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_stops_at_the_time_limit(algorithm: str, clock: _Clock) -> None:
    # The clock starts at 0 and gains a second per check: the run stops at the first check after 3 s.
    result = _run(algorithm, ("max_time", 3))

    evaluations = int(result.data["evaluations"])
    assert 20 <= evaluations <= 20 * 10
    assert result.meta["time_limit_reached"] is True


def test_the_limit_is_checked_once_per_generation(clock: _Clock) -> None:
    result = _run("nsgaii", ("max_time", 3))

    # Start at 0; generation 0 is checked at 1 s, generation 1 at 2 s and generation 2 at 3 s.
    assert result.data["evaluations"] == 60


def test_an_evaluation_cap_stops_first(clock: _Clock) -> None:
    result = _run("nsgaii", ("max_time", {"seconds": 1000, "max_evaluations": 100}))

    assert result.data["evaluations"] == 100
    assert result.meta["time_limit_reached"] is False


def test_a_real_time_limit_stops_a_long_budget() -> None:
    started = time.perf_counter()
    result = _run("nsgaii", ("max_time", 0.2))

    assert time.perf_counter() - started < 30
    assert result.data["evaluations"] >= 20


def test_the_user_callback_still_runs_and_can_stop_first(clock: _Clock) -> None:
    class Stopper:
        def __init__(self) -> None:
            self.generations: list[int] = []

        def on_start(self, ctx: object = None) -> None:
            return None

        def on_generation(self, generation: int, F: Any = None, X: Any = None, stats: Any = None) -> None:
            self.generations.append(generation)

        def on_end(self, final_F: Any = None, final_stats: Any = None) -> None:
            return None

        def should_stop(self) -> bool:
            return len(self.generations) >= 2

    callback = Stopper()
    result = _run("nsgaii", ("max_time", 1000), live_viz=callback)

    assert callback.generations == [0, 1]
    assert result.data["evaluations"] == 40
    assert result.meta["time_limit_reached"] is False


def test_a_callback_derived_from_the_no_op_one_is_kept(clock: _Clock) -> None:
    class Stopper(NoOpLiveVisualization):
        def __init__(self) -> None:
            self.generations: list[int] = []
            self.ended = False

        def on_generation(self, generation: int, F: Any = None, X: Any = None, stats: Any = None) -> None:
            self.generations.append(generation)

        def on_end(self, final_F: Any = None, final_stats: Any = None) -> None:
            self.ended = True

        def should_stop(self) -> bool:
            return len(self.generations) >= 2

    callback = Stopper()
    result = _run("nsgaii", ("max_time", 1000), live_viz=callback)

    assert callback.generations == [0, 1] and callback.ended
    assert result.data["evaluations"] == 40
    assert result.meta["time_limit_reached"] is False


def test_the_manifest_records_the_limit_and_refuses_an_exact_replay(clock: _Clock, tmp_path: Path) -> None:
    result = _run("nsgaii", ("max_time", 3))
    resolved = result.meta["run_artifact_resolved_spec"]

    assert resolved["termination"]["component_id"] == "vamos.termination:max_time@1"
    assert resolved["termination"]["config"] == {"seconds": 3.0, "max_evaluations": None}
    assert resolved["determinism"]["declared"] is False

    stored = vamos.save_result(result, tmp_path / "run")
    manifest = stored.manifest
    assert manifest["outcome"]["termination_reason"] == "max_time"
    assert manifest["replayability"]["declared_level"] != "exact"
    assert "wall_clock_termination" in {reason["code"] for reason in manifest["replayability"]["reasons"]}
    report = vamos.verify_run(stored.root)
    assert report.effective_replayability != "exact"
    assert "wall_clock_termination" in {reason.code for reason in report.reasons}


def test_a_capped_run_records_which_limit_stopped_it(clock: _Clock, tmp_path: Path) -> None:
    result = _run("nsgaii", ("max_time", {"seconds": 1000, "max_evaluations": 100}))

    assert result.meta["run_artifact_resolved_spec"]["termination"]["config"] == {"seconds": 1000.0, "max_evaluations": 100}
    assert vamos.save_result(result, tmp_path / "run").manifest["outcome"]["termination_reason"] == "max_evaluations"


@pytest.mark.parametrize(
    ("termination", "reason"),
    [(("max_time", 3), "max_time"), (("max_time", {"seconds": 1000, "max_evaluations": 100}), "max_evaluations")],
    ids=["time", "cap"],
)
def test_saving_a_loaded_run_keeps_its_termination_reason(clock: _Clock, tmp_path: Path, termination: Any, reason: str) -> None:
    stored = vamos.save_result(_run("nsgaii", termination), tmp_path / "run")
    assert stored.manifest["outcome"]["termination_reason"] == reason

    loaded = vamos.load_result(stored.root)
    assert loaded.meta["time_limit_reached"] is (reason == "max_time")
    copy = vamos.save_result(loaded, tmp_path / "copy")
    assert copy.manifest["outcome"]["termination_reason"] == reason


def test_a_loaded_run_without_a_time_limit_has_no_time_limit_flag(tmp_path: Path) -> None:
    stored = vamos.save_result(_run("nsgaii", ("max_evaluations", 40)), tmp_path / "run")
    assert "time_limit_reached" not in vamos.load_result(stored.root).meta


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1.0,
        float("inf"),
        float("nan"),
        "5",
        True,
        {"max_evaluations": 100},
        {"seconds": 5, "max_evaluations": 0},
        {"seconds": 5, "other": 1},
    ],
)
def test_invalid_limits_are_rejected(value: Any) -> None:
    with pytest.raises(ConfigurationError, match="max_time"):
        _run("nsgaii", ("max_time", value))


def test_max_evaluations_cannot_be_given_beside_a_time_limit() -> None:
    with pytest.raises(ConfigurationError, match="max_evaluations"):
        _run("nsgaii", ("max_time", 5), max_evaluations=100)
