"""Tests for following and stopping Studio runs."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from vamos.foundation.problem.registry import make_problem_selection
from vamos.ux.studio.algorithm_catalog import default_algorithm_config, studio_algorithms
from vamos.ux.studio.live_run import MAX_SNAPSHOTS, LiveRun
from vamos.ux.studio.problem_builder_backend import compile_objective_function, run_preview_optimization
from vamos.ux.studio.services import _run_algorithm


class _StopAfter(LiveRun):
    """Request a stop once ``generations`` generations have been reported."""

    def __init__(self, budget: int, generations: int) -> None:
        super().__init__(budget)
        self.reported = 0
        self.limit = generations

    def on_generation(self, generation: int, F: Any = None, X: Any = None, stats: dict[str, Any] | None = None) -> None:
        super().on_generation(generation, F=F, X=X, stats=stats)
        self.reported += 1
        if self.reported >= self.limit:
            self.request_stop()


def test_live_run_records_fronts_and_evaluations() -> None:
    live = LiveRun(budget=1000)

    live.on_generation(1, F=np.array([[0.0, 1.0]]), stats={"evals": 200})
    live.on_generation(2, F=np.array([[0.5, 0.5]]), stats={"evals": 400})

    assert (live.generation, live.evaluations, live.fraction) == (2, 400, 0.4)
    assert [number for number, _ in live.snapshots] == [1, 2]
    assert live.latest() is not None and live.latest().tolist() == [[0.5, 0.5]]
    assert not live.should_stop()
    live.request_stop()
    assert live.should_stop() and live.stop_requested


def test_live_run_keeps_a_bounded_number_of_fronts() -> None:
    live = LiveRun(budget=10**9)

    for generation in range(5 * MAX_SNAPSHOTS):
        live.on_generation(generation, F=np.zeros((2, 2)))

    assert len(live.snapshots) <= MAX_SNAPSHOTS + 1
    assert live.snapshots[-1][0] >= 4 * MAX_SNAPSHOTS


def test_the_final_front_replaces_a_generation_already_recorded() -> None:
    live = LiveRun(budget=1000)

    live.on_generation(3, F=np.array([[0.0, 1.0]]), stats={"evals": 300})
    live.on_end(final_F=np.array([[0.2, 0.8]]))

    assert [number for number, _ in live.snapshots] == [3]
    assert live.latest().tolist() == [[0.2, 0.8]]


@pytest.mark.parametrize("algorithm", studio_algorithms())
def test_every_algorithm_records_each_generation_once(algorithm: str) -> None:
    selection = make_problem_selection("dtlz2" if algorithm in {"nsgaiii", "rvea"} else "zdt1")
    config = default_algorithm_config(algorithm, encoding="real", n_var=selection.n_var, n_obj=selection.n_obj, pop_size=20)
    live = LiveRun(budget=200)

    _run_algorithm(
        selection.instantiate(),
        algorithm=algorithm,
        algorithm_config=config,
        termination=("max_evaluations", 4 * int(config.to_dict()["pop_size"])),
        seed=0,
        engine="numpy",
        live_viz=live,
    )

    numbers = [number for number, _ in live.snapshots]
    assert numbers and len(numbers) == len(set(numbers)), f"duplicated generations: {numbers}"


@pytest.mark.parametrize("algorithm", studio_algorithms())
def test_every_algorithm_stops_when_asked(algorithm: str) -> None:
    problem = make_problem_selection("zdt1").instantiate()
    config = default_algorithm_config(algorithm, encoding="real", n_var=problem.n_var, n_obj=problem.n_obj, pop_size=20)
    budget = 200_000
    live = _StopAfter(budget, generations=3)

    _run_algorithm(
        problem,
        algorithm=algorithm,
        algorithm_config=config,
        termination=("max_evaluations", budget),
        seed=0,
        engine="numpy",
        live_viz=live,
    )

    assert live.stop_requested
    assert 0 < live.evaluations < budget // 10


def test_preview_child_process_stops_when_cancelled() -> None:
    code = "s = 0.0\nfor i in range(2000):\n    s += i * 1e-12\nreturn [x[0], 1.0 - x[0] + s]"
    fn = compile_objective_function(code, trusted_local_code=True)
    cancel = threading.Event()
    threading.Timer(3.0, cancel.set).start()
    started = time.monotonic()

    with pytest.raises(InterruptedError, match="Stopped"):
        run_preview_optimization(
            fn,
            n_var=5,
            n_obj=2,
            bounds=[(0.0, 1.0)] * 5,
            algorithm="nsgaii",
            budget=5_000_000,
            pop_size=100,
            seed=0,
            objective_code=code,
            timeout_seconds=120.0,
            trusted_local_code=True,
            cancel=cancel,
        )

    assert time.monotonic() - started < 60.0
