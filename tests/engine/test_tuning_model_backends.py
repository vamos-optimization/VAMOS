from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vamos.engine.tuning import (
    Instance,
    ModelBasedTuner,
    ParamSpace,
    Real,
    TuningTask,
    available_model_based_backends,
)
from vamos.engine.tuning._model_backend_utils import build_configspace, sample_from_optuna_trial
from vamos.engine.tuning.racing.bridge import build_agemoea_config_space, build_moead_config_space, build_nsgaii_config_space
from vamos.engine.tuning.racing.param_space import Boolean, Categorical, Condition, Int


def _conditional_space() -> ParamSpace:
    # Children deliberately precede their parents, including a nested condition.
    return ParamSpace(
        params={
            "depth": Int("depth", 1, 5),
            "rate": Real("rate", 0.01, 1.0, log=True),
            "enabled": Boolean("enabled"),
            "method": Categorical("method", ["tree", "linear"]),
        },
        conditions=[
            Condition("depth", "cfg['enabled'] == True"),
            Condition("enabled", "cfg['method'] == 'tree'"),
            Condition("rate", "cfg['method'] == 'linear'"),
        ],
    )


@pytest.mark.parametrize(
    "values, expected",
    [
        ({"method": "linear", "rate": 0.1}, {"method", "rate"}),
        ({"method": "tree", "enabled": False}, {"method", "enabled"}),
        ({"method": "tree", "enabled": True, "depth": 3}, {"method", "enabled", "depth"}),
    ],
)
def test_optuna_suggests_only_active_parameters(values, expected):
    class Trial:
        def __init__(self):
            self.requested = []

        def suggest(self, name, *args, **kwargs):
            self.requested.append(name)
            return values[name]

        suggest_float = suggest
        suggest_int = suggest
        suggest_categorical = suggest

    trial = Trial()
    space = _conditional_space()
    config = sample_from_optuna_trial(trial, space)
    assert config == values
    assert set(trial.requested) == expected
    space.validate(config)


def test_configspace_samples_only_active_parameters():
    pytest.importorskip("ConfigSpace")
    space = _conditional_space()
    cs = build_configspace(space, seed=5)
    assert len(cs.conditions) == 3
    branches = set()
    for sampled in cs.sample_configuration(50):
        config = dict(sampled)
        space.validate(config)
        assert set(config) == {name for name in space.params if space.is_active(name, config)}
        branches.add((config["method"], config.get("enabled")))
    assert branches == {("linear", None), ("tree", False), ("tree", True)}


@pytest.mark.parametrize("backend", ["optuna", "bohb_optuna", "smac3", "bohb"])
def test_conditional_trials_and_history(backend, tmp_path, monkeypatch):
    pytest.importorskip({"optuna": "optuna", "bohb_optuna": "optuna", "smac3": "smac", "bohb": "hpbandster"}[backend])
    if backend in {"smac3", "bohb"}:
        pytest.importorskip("ConfigSpace")
    monkeypatch.chdir(tmp_path)
    task = _make_task()
    task.param_space = _conditional_space()
    task.budget_per_run = 1
    evaluated = []

    def evaluate(config, ctx):
        assert set(config) == {name for name in task.param_space.params if task.param_space.is_active(name, config)}
        evaluated.append(dict(config))
        return float(ctx.budget) + float(config.get("depth", config.get("rate", 0.0)))

    # Continue beyond the startup samples so the backend can fit its model.
    tuner = ModelBasedTuner(task=task, backend=backend, max_trials=16, seed=5)
    best, history = tuner.run(evaluate, verbose=False)
    assert evaluated
    assert best in evaluated
    assert all(trial.config in evaluated for trial in history)


def test_optuna_persists_only_active_parameters(tmp_path):
    optuna = pytest.importorskip("optuna")
    storage = f"sqlite:///{(tmp_path / 'conditional.sqlite3').as_posix()}"
    task = _make_task()
    task.param_space = _conditional_space()
    for _ in range(2):
        tuner = ModelBasedTuner(
            task=task,
            max_trials=3,
            seed=5,
            optuna_storage_url=storage,
            optuna_study_name="conditional",
        )
        best, history = tuner.run(lambda config, ctx: float(ctx.budget), verbose=False)
    study = optuna.load_study(study_name="conditional", storage=storage)
    assert len(study.trials) == len(history) == 6
    for trial, result in zip(study.trials, history):
        assert trial.params == trial.user_attrs["config"] == result.config
        assert set(trial.params) == {name for name in task.param_space.params if task.param_space.is_active(name, trial.params)}
    assert best == study.best_trial.params


@pytest.mark.parametrize(
    "expr",
    [
        "cfg['method'] == 'tree' or cfg['method'] == 'linear'",
        "cfg['method'] != 'tree' and cfg['enabled'] == True",
        "not (cfg['method'] == 'tree' or cfg['enabled'] == False)",
        "cfg['enabled']",
        "not cfg['enabled']",
        "cfg['count'] < 3",
        "cfg['count'] <= 3",
        "cfg['count'] > 3",
        "cfg['count'] >= 3",
        "not (cfg['count'] <= 3)",
    ],
)
def test_configspace_condition_expression_parity(expr):
    pytest.importorskip("ConfigSpace")
    space = ParamSpace(
        params={
            "value": Real("value", 0.0, 1.0),
            "method": Categorical("method", ["tree", "linear", "other"]),
            "enabled": Boolean("enabled"),
            "count": Int("count", 1, 5),
        },
        conditions=[Condition("value", expr)],
    )
    for sampled in build_configspace(space, seed=3).sample_configuration(40):
        config = dict(sampled)
        assert ("value" in config) == space.is_active("value", config)
        space.validate(config)


@pytest.mark.parametrize("builder", [build_nsgaii_config_space, build_moead_config_space, build_agemoea_config_space])
def test_builtin_configspace_preserves_combined_archive_conditions(builder):
    pytest.importorskip("ConfigSpace")
    space = builder().to_param_space()
    seen = set()
    for sampled in build_configspace(space, seed=5).sample_configuration(100):
        config = dict(sampled)
        space.validate(config)
        assert set(config) == {name for name in space.params if space.is_active(name, config)}
        enabled = config["use_external_archive"]
        unbounded = config.get("archive_unbounded")
        assert ("archive_prune_policy" in config) == (enabled and not unbounded)
        assert ("mutation_eta" in config) == (config["mutation"] in {"pm", "polynomial", "linked_polynomial"})
        seen.add((enabled, unbounded))
    assert seen == {(False, None), (True, False), (True, True)}


@pytest.mark.parametrize(
    "expr",
    [
        "cfg['enabled'] == True or cfg['method'] == 'linear'",
        "not (cfg['enabled'] == False)",
        "cfg['enabled'] != False",
    ],
)
def test_configspace_keeps_descendants_inactive_when_parent_is_missing(expr):
    pytest.importorskip("ConfigSpace")
    space = _conditional_space()
    space.conditions[0] = Condition("depth", expr)
    for sampled in build_configspace(space, seed=5).sample_configuration(30):
        config = dict(sampled)
        assert ("depth" in config) == space.is_active("depth", config)
        space.validate(config)


@pytest.mark.parametrize("backend", ["optuna", "configspace"])
@pytest.mark.parametrize(
    "conditions, message",
    [
        ([Condition("missing", "cfg['depth'] == 2")], "unknown parameter"),
        ([Condition("depth", "cfg['missing'] == 2")], "unknown parent"),
        ([Condition("depth", "cfg['depth'] == 2")], "cycle"),
        ([Condition("depth", "cfg['rate'] == 1"), Condition("rate", "cfg['depth'] == 2")], "cycle"),
        ([Condition("depth", "cfg['method'")], "Invalid condition syntax"),
    ],
)
def test_invalid_condition_dependencies_fail_before_sampling(backend, conditions, message):
    space = _conditional_space()
    space.conditions = conditions
    with pytest.raises(ValueError, match=message):
        if backend == "optuna":
            sample_from_optuna_trial(None, space)
        else:
            pytest.importorskip("ConfigSpace")
            build_configspace(space, seed=0)


@pytest.mark.parametrize("expr", ["cfg.get('method') == 'tree'", "cfg['depth'] < cfg['rate']", "__import__('os')"])
def test_configspace_rejects_untranslatable_conditions(expr):
    pytest.importorskip("ConfigSpace")
    space = _conditional_space()
    space.conditions = [Condition("enabled", expr)]
    with pytest.raises(ValueError, match="Cannot translate condition for 'enabled'"):
        build_configspace(space, seed=0)


def _make_task() -> TuningTask:
    return TuningTask(
        name="model_backends_smoke",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[1, 2],
        budget_per_run=4,
        maximize=True,
        aggregator=np.mean,
    )


def _eval_fn(config: dict, ctx) -> float:
    # Peak near x=0.8 and use budget to emulate multi-fidelity signal.
    x = float(config["x"])
    base = 1.0 - abs(x - 0.8)
    return float(base * (ctx.budget / 4.0))


def test_available_model_based_backends_keys():
    backends = available_model_based_backends()
    assert set(backends) == {"optuna", "bohb_optuna", "smac3", "bohb"}


def test_missing_smac3_dependency_raises():
    if available_model_based_backends()["smac3"]:
        pytest.skip("smac3 deps are installed in this environment.")
    tuner = ModelBasedTuner(task=_make_task(), max_trials=1, backend="smac3", seed=0, n_jobs=1)
    with pytest.raises(RuntimeError, match="smac"):
        tuner.run(_eval_fn, verbose=False)


def test_missing_bohb_dependency_raises():
    if available_model_based_backends()["bohb"]:
        pytest.skip("bohb deps are installed in this environment.")
    tuner = ModelBasedTuner(task=_make_task(), max_trials=1, backend="bohb", seed=0, n_jobs=1)
    with pytest.raises(RuntimeError, match="hpbandster"):
        tuner.run(_eval_fn, verbose=False)


def test_optuna_backend_smoke():
    if not available_model_based_backends()["optuna"]:
        pytest.skip("optuna not installed.")
    tuner = ModelBasedTuner(task=_make_task(), max_trials=3, backend="optuna", seed=0, n_jobs=1)
    best, history = tuner.run(_eval_fn, verbose=False)
    assert "x" in best
    assert len(history) >= 1
    assert all("x" in t.config for t in history)


def test_fidelity_subsampling_scales_instances_and_seeds():
    task = TuningTask(
        name="fidelity_subsample_smoke",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[
            Instance("zdt1", 2),
            Instance("zdt2", 2),
            Instance("dtlz1", 2),
            Instance("wfg1", 2),
        ],
        seeds=[1, 2, 3, 4],
        budget_per_run=100,
        maximize=True,
        aggregator=np.mean,
    )
    tuner = ModelBasedTuner(
        task=task,
        max_trials=1,
        backend="optuna",
        seed=7,
        n_jobs=1,
        budget_levels=[25, 100],
        fidelity_min_instance_frac=0.5,
        fidelity_min_seed_count=1,
        fidelity_max_seed_count=4,
    )
    ctxs: list[tuple[int, int, int, int | None]] = []

    def eval_fn(_config: dict, ctx) -> float:
        ctxs.append((int(ctx.budget), int(ctx.seed), int(ctx.fidelity_level), ctx.previous_budget))
        return 1.0

    _ = tuner._eval_config_at_budget({"x": 0.5}, eval_fn, budget=25)
    low = list(ctxs)
    assert len(low) == 2  # 2 instances * 1 seed
    assert all(level == 0 for _, _, level, _ in low)
    assert all(prev is None for _, _, _, prev in low)

    ctxs.clear()
    _ = tuner._eval_config_at_budget({"x": 0.5}, eval_fn, budget=100)
    high = list(ctxs)
    assert len(high) == 16  # 4 instances * 4 seeds
    assert all(level == 1 for _, _, level, _ in high)
    assert all(int(prev) == 25 for _, _, _, prev in high)


def test_optuna_storage_resume_and_trace(tmp_path: Path):
    if not available_model_based_backends()["optuna"]:
        pytest.skip("optuna not installed.")

    db_path = tmp_path / "resume_optuna.sqlite3"
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study_name = "resume_optuna_smoke"

    tuner_a = ModelBasedTuner(
        task=_make_task(),
        max_trials=1,
        backend="optuna",
        seed=3,
        n_jobs=1,
        optuna_storage_url=storage_url,
        optuna_study_name=study_name,
        optuna_load_if_exists=True,
    )
    _, history_a = tuner_a.run(_eval_fn, verbose=False)
    assert len(history_a) >= 1
    assert db_path.exists()
    assert any("fidelity_trace" in h.details for h in history_a)

    tuner_b = ModelBasedTuner(
        task=_make_task(),
        max_trials=1,
        backend="optuna",
        seed=3,
        n_jobs=1,
        optuna_storage_url=storage_url,
        optuna_study_name=study_name,
        optuna_load_if_exists=True,
    )
    _, history_b = tuner_b.run(_eval_fn, verbose=False)
    assert len(history_b) >= len(history_a) + 1


def test_bohb_optuna_backend_smoke():
    if not available_model_based_backends()["bohb_optuna"]:
        pytest.skip("optuna not installed.")
    tuner = ModelBasedTuner(
        task=_make_task(),
        max_trials=3,
        backend="bohb_optuna",
        seed=0,
        n_jobs=1,
        bohb_reduction_factor=2,
    )
    best, history = tuner.run(_eval_fn, verbose=False)
    assert "x" in best
    assert len(history) >= 1
