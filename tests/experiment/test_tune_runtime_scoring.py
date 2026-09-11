from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vamos.engine.tuning import TrialResult, build_nsgaii_config_space
from vamos.experiment.cli._tune_runtime import (
    CLI_EXCLUDED_TUNING_PARAMS,
    _score_hypervolume_result,
    build_cli_param_space,
    reject_legacy_cli_history,
    reject_legacy_optuna_study_before_run,
)
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


def test_hv_scorer_reads_constraints_from_optimization_result_payload() -> None:
    ref = np.array([5.0, 5.0])
    feasible_F = np.array([[1.0, 4.0], [4.0, 1.0]])
    all_F = np.vstack([feasible_F, np.array([[0.5, 0.5]])])
    result = SimpleNamespace(
        F=all_F,
        data={"G": np.array([[-1.0], [0.0], [1.0]])},
    )

    score = _score_hypervolume_result(result, ref, failure_score=-123.0)

    assert score == pytest.approx(hypervolume(feasible_F, ref))
    assert score < hypervolume(all_F, ref)


def test_hv_scorer_uses_failure_score_when_no_feasible_rows_remain() -> None:
    result = SimpleNamespace(
        F=np.array([[1.0, 1.0], [2.0, 0.5]]),
        data={"G": np.array([[0.1], [3.0]])},
    )

    assert _score_hypervolume_result(result, np.array([5.0, 5.0]), failure_score=-7.5) == pytest.approx(-7.5)


def test_hv_scorer_accepts_one_dimensional_constraint_vector() -> None:
    result = SimpleNamespace(
        F=np.array([[1.0, 4.0], [4.0, 1.0]]),
        data={"G": np.array([-1.0, 0.5])},
    )

    score = _score_hypervolume_result(result, np.array([5.0, 5.0]), failure_score=-1.0)

    assert score == pytest.approx(hypervolume(np.array([[1.0, 4.0]]), np.array([5.0, 5.0])))


def test_hv_scorer_rejects_misaligned_constraint_rows() -> None:
    result = SimpleNamespace(
        F=np.array([[1.0, 4.0], [4.0, 1.0]]),
        data={"G": np.array([[-1.0]])},
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
        reject_legacy_cli_history({"pop_size": 40}, history)


def _install_fake_optuna(
    monkeypatch: pytest.MonkeyPatch,
    *,
    study_name: str,
    trial_config: dict[str, object],
) -> None:
    fake = ModuleType("optuna")
    fake.get_all_study_summaries = lambda *, storage: [SimpleNamespace(study_name=study_name)]  # type: ignore[attr-defined]
    fake.load_study = lambda *, study_name, storage: SimpleNamespace(  # type: ignore[attr-defined]
        trials=[SimpleNamespace(user_attrs={"config": dict(trial_config)}, params={})]
    )
    monkeypatch.setitem(sys.modules, "optuna", fake)


def test_cli_rejects_legacy_named_optuna_study_before_new_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_optuna(
        monkeypatch,
        study_name="legacy-study",
        trial_config={"pop_size": 40, "use_external_archive": False},
    )
    args = SimpleNamespace(
        backend="optuna",
        optuna_storage="sqlite:///legacy.db",
        optuna_study_name="legacy-study",
        optuna_load_if_exists=True,
    )

    with pytest.raises(RuntimeError, match="fresh --optuna-study-name"):
        reject_legacy_optuna_study_before_run(args)


def test_cli_allows_current_named_optuna_study(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_optuna(
        monkeypatch,
        study_name="current-study",
        trial_config={"pop_size": 40, "mutation_eta": 20.0},
    )
    args = SimpleNamespace(
        backend="optuna",
        optuna_storage="sqlite:///current.db",
        optuna_study_name="current-study",
        optuna_load_if_exists=True,
    )

    reject_legacy_optuna_study_before_run(args)
