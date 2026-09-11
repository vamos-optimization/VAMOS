from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vamos.experiment.cli._tune_runtime import (
    BUILDERS,
    _population_front_for_scoring,
    build_task,
)


class _FakeResult:
    def __init__(self) -> None:
        # Deliberately better than the population. The tuning scorer must ignore
        # this top-level value because it may come from an external archive.
        self.F = np.array([[0.0, 0.0]])
        self.data = {
            "population": {
                "F": np.array(
                    [
                        [1.0, 1.0],
                        [0.5, 2.0],
                        [2.0, 0.5],
                        [0.2, 0.2],
                    ]
                ),
                # The apparently best row is infeasible and must not influence
                # the scored front.
                "G": np.array([[0.0], [0.0], [0.0], [1.0]]),
            }
        }


def test_population_front_for_scoring_ignores_top_level_result_and_infeasible_rows() -> None:
    front = _population_front_for_scoring(_FakeResult())

    expected = np.array(
        [
            [1.0, 1.0],
            [0.5, 2.0],
            [2.0, 0.5],
        ]
    )
    assert front.shape == expected.shape
    assert {tuple(row) for row in front.tolist()} == {tuple(row) for row in expected.tolist()}


def test_cli_tuning_task_excludes_external_archive_controls() -> None:
    algo_space = BUILDERS["nsgaii"]()
    param_space = algo_space.to_param_space()
    assert "use_external_archive" in param_space.params

    args = SimpleNamespace(
        instances="",
        problem="zdt1",
        algorithm="nsgaii",
        backend="random",
        seed=1,
        n_seeds=2,
        aggregate_mode="mean",
    )
    task = build_task(args, param_space, budget_per_run=100)

    assert "use_external_archive" not in task.param_space.params
    assert "archive_unbounded" not in task.param_space.params
    assert "archive_prune_policy" not in task.param_space.params
    sample = task.param_space.sample(np.random.default_rng(7))
    assert not {"use_external_archive", "archive_unbounded", "archive_prune_policy"} & sample.keys()
