from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vamos.engine.tuning import TrialResult, build_nsgaii_config_space
from vamos.experiment.cli._tune_runtime import (
    CLI_EXCLUDED_TUNING_PARAMS,
    _score_hypervolume_result,
    build_cli_param_space,
)
from vamos.experiment.cli.tune import _reject_legacy_cli_history
from vamos.foundation.quality_indicators.hypervolume import hypervolume


def test_cli_space_excludes_archive_controls_but_programmatic_space_keeps_them() -> None:
    cli_space = build_cli_param_space("nsgaii")
    excluded = set(CLI_EXCLUDED_TUNING_PARAMS)

    assert excluded.isdisjoint(cli_space.params)
    assert all(condition.param_name not in excluded for condition in cli_space.conditions)
    assert all(not any(name in condition.expr for name in excluded) for condition in cli_space.conditions)

    programmatic_space = build_nsgaii_config_space().to_param_space()
    assert "use_external_archive" in programmatic_space.params
    assert "archive_unbounded" in programmatic_space.params
    assert "archive_prune_policy" in programmatic_space.params


def test_hv_scorer_filters_infeasible_rows_before_scoring() -> None:
    ref = np.array([5.0, 5.0])
    feasible_F = np.array([[1.0, 4.0], [4.0, 1.0]])
    result = SimpleNamespace(
        F=np.vstack([feasible_F, np.array([[0.5, 0.5]])]),
        G=np.array([[-1.0], [0.0], [1.0]]),
    )

    score = _score_hypervolume_result(result, ref, failure_score=-123.0)

    assert score == pytest.approx(hypervolume(feasible_F, ref))
    assert score < hypervolume(result.F, ref)


def test_hv_scorer_uses_failure_score_when_no_feasible_rows_remain() -> None:
    result = SimpleNamespace(
        F=np.array([[1.0, 1.0], [2.0, 0.5]]),
        G=np.array([[0.1], [3.0]]),
    )

    assert _score_hypervolume_result(result, np.array([5.0, 5.0]), failure_score=-7.5) == pytest.approx(-7.5)


def test_hv_scorer_accepts_one_dimensional_constraint_vector() -> None:
    result = SimpleNamespace(
        F=np.array([[1.0, 4.0], [4.0, 1.0]]),
        G=np.array([-1.0, 0.5]),
    )

    score = _score_hypervolume_result(result, np.array([5.0, 5.0]), failure_score=-1.0)

    assert score == pytest.approx(hypervolume(np.array([[1.0, 4.0]]), np.array([5.0, 5.0])))


def test_hv_scorer_rejects_misaligned_constraint_rows() -> None:
    result = SimpleNamespace(
        F=np.array([[1.0, 4.0], [4.0, 1.0]]),
        G=np.array([[-1.0]]),
    )

    with pytest.raises(ValueError, match="aligned F/G rows"):
        _score_hypervolume_result(result, np.array([5.0, 5.0]), failure_score=0.0)


def test_cli_rejects_legacy_persisted_history_with_archive_controls() -> None:
    history = [
        TrialResult(
            trial_id=1,
            config={"pop_size": 40, "use_external_archive": True},
            score=1.0,
        )
    ]

    with pytest.raises(RuntimeError, match="fresh persisted tuning study"):
        _reject_legacy_cli_history({"pop_size": 40}, history)
