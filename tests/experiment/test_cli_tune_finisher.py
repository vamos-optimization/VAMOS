from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vamos.engine.tuning import EvalContext, Instance
from vamos.experiment.cli import tune as tune_cli


def _simple_eval(config: dict[str, float], ctx: EvalContext) -> float:
    _ = ctx
    return float(config["x"])


def test_statistical_finisher_selects_best_candidate():
    candidates = [
        {"score": 0.2, "config": {"x": 0.2}},
        {"score": 0.9, "config": {"x": 0.9}},
    ]
    instances = [Instance(name="zdt1", n_var=10, kwargs={}), Instance(name="zdt2", n_var=10, kwargs={})]
    seeds = [1, 2, 3]

    result = tune_cli._run_statistical_finisher(
        candidates=candidates,
        eval_fn=_simple_eval,
        instances=instances,
        seeds=seeds,
        budget=100,
        aggregator=lambda vals: float(np.mean(vals)),
        alpha=0.05,
        min_blocks=3,
        failure_score=0.0,
        use_friedman=True,
    )

    assert result is not None
    assert float(result["winner_config"]["x"]) == 0.9
    assert int(result["num_candidates"]) == 2
    assert int(result["num_blocks"]) == len(instances) * len(seeds)
    assert any(bool(row["selected"]) for row in result["candidate_rows"])


def test_statistical_finisher_replaces_failed_scores_with_failure_score():
    instances = [Instance(name="zdt1", n_var=10, kwargs={})]
    seeds = [1, 2, 3]

    def flaky_eval(config: dict[str, float], ctx: EvalContext) -> float:
        if int(ctx.seed) == 2:
            raise RuntimeError("boom")
        if int(ctx.seed) == 3:
            return float("nan")
        return float(config["x"])

    result = tune_cli._run_statistical_finisher(
        candidates=[{"score": 0.5, "config": {"x": 0.5}}],
        eval_fn=flaky_eval,
        instances=instances,
        seeds=seeds,
        budget=50,
        aggregator=lambda vals: float(np.mean(vals)),
        alpha=0.05,
        min_blocks=2,
        failure_score=-1.0,
        use_friedman=True,
    )

    assert result is not None
    block_scores = [float(row["score"]) for row in result["block_rows"]]
    assert -1.0 in block_scores
    assert all(np.isfinite(block_scores))
    assert float(result["winner_config"]["x"]) == 0.5


def test_split_instances_suite_stratified_manifest_has_suite():
    instances = [
        Instance(name="zdt1", n_var=10, kwargs={}),
        Instance(name="zdt2", n_var=10, kwargs={}),
        Instance(name="dtlz1", n_var=10, kwargs={}),
        Instance(name="dtlz2", n_var=10, kwargs={}),
        Instance(name="wfg1", n_var=10, kwargs={}),
        Instance(name="wfg2", n_var=10, kwargs={}),
    ]
    train, validation, test, manifest = tune_cli._split_instances(
        instances,
        train_frac=0.6,
        validation_frac=0.2,
        split_seed=42,
        strategy="suite_stratified",
    )

    all_names = {inst.name for inst in instances}
    split_names = {inst.name for inst in train + validation + test}
    assert all_names == split_names
    assert len(validation) >= 1
    assert len(test) >= 1
    assert all("suite" in row for row in manifest)


def test_backend_fallback_switches_to_random(monkeypatch, tmp_path: Path):
    called: dict[str, str] = {}

    def fake_available() -> dict[str, bool]:
        return {"optuna": False, "bohb_optuna": False, "smac3": False, "bohb": False}

    def fake_run_backend(args, task, eval_fn, resolved_jobs):  # noqa: ANN001
        _ = task, eval_fn, resolved_jobs
        called["backend"] = str(args.backend)
        return {"pop_size": 10}, []

    monkeypatch.setattr(tune_cli, "available_model_based_backends", fake_available)
    monkeypatch.setattr(tune_cli, "_run_backend", fake_run_backend)

    tune_cli.main(
        [
            "--instances",
            "zdt1,zdt2,zdt3",
            "--algorithm",
            "nsgaii",
            "--backend",
            "optuna",
            "--backend-fallback",
            "random",
            "--no-run-validation",
            "--no-run-test",
            "--no-run-statistical-finisher",
            "--n-seeds",
            "2",
            "--n-jobs",
            "1",
            "--output-dir",
            str(tmp_path / "tune_out"),
            "--name",
            "fallback_case",
        ]
    )

    assert called["backend"] == "random"


def test_backend_fallback_error_mode_raises(monkeypatch):
    def fake_available() -> dict[str, bool]:
        return {"optuna": False, "bohb_optuna": False, "smac3": False, "bohb": False}

    monkeypatch.setattr(tune_cli, "available_model_based_backends", fake_available)

    with pytest.raises(RuntimeError, match="not available"):
        tune_cli.main(
            [
                "--instances",
                "zdt1,zdt2,zdt3",
                "--algorithm",
                "nsgaii",
                "--backend",
                "optuna",
                "--backend-fallback",
                "error",
                "--no-run-validation",
                "--no-run-test",
                "--no-run-statistical-finisher",
            ]
        )


def test_cli_validation_reports_failed_runs():
    from vamos.experiment.cli._tune_post import evaluate_config_split
    from vamos.experiment.cli._tune_runtime import make_evaluator

    # pop_size=150 cannot run within 100 evaluations: every validation run fails.
    eval_fn = make_evaluator(
        problem_key="zdt1",
        n_var=6,
        n_obj=2,
        algorithm_name="nsgaii",
        fixed_pop_size=150,
        ref_point_str=None,
        warm_start=False,
        runtime_penalty=0.0,
        failure_score=0.0,
        logger=lambda: __import__("logging").getLogger("test"),
    )

    metrics = evaluate_config_split(
        config={},
        eval_fn=eval_fn,
        instances=[Instance(name="zdt1", n_var=6, kwargs={})],
        seeds=[1, 2, 3],
        budget=100,
        aggregator=lambda vals: float(np.mean(vals)),
        failure_score=0.0,
    )

    assert (metrics["rows_total"], metrics["rows_ok"], metrics["fail_rate"]) == (3, 0, 1.0)
    assert np.isnan(metrics["score_agg"])


@pytest.mark.parametrize("backend", ["random", "racing"])
def test_cli_tune_exits_nonzero_when_every_evaluation_fails(tmp_path: Path, backend: str):
    # A budget of 10 evaluations cannot run any population the NSGA-II space samples.
    argv = [
        "--instances",
        "zdt1",
        "--algorithm",
        "nsgaii",
        "--backend",
        backend,
        "--budget",
        "10",
        "--tune-budget",
        "4",
        "--n-seeds",
        "1",
        "--n-jobs",
        "1",
        "--no-run-validation",
        "--no-run-test",
        "--no-run-statistical-finisher",
        "--output-dir",
        str(tmp_path / "out"),
    ]

    with pytest.raises(SystemExit) as caught:
        tune_cli.main(argv)

    assert caught.value.code not in (0, None)
    assert "every one of the" in str(caught.value.code)
    assert "failed" in str(caught.value.code)


def test_tuning_summary_counts_failed_evaluations(tmp_path: Path, monkeypatch):
    import json

    from vamos.experiment.cli import _tune_runtime

    calls = {"n": 0}
    real = _tune_runtime.make_evaluator

    def flaky_evaluator(**kwargs):
        inner = real(**kwargs)

        def eval_fn(config, ctx):
            calls["n"] += 1
            return float("nan") if calls["n"] % 2 else 1.0

        inner.inner = eval_fn
        return inner

    monkeypatch.setattr(tune_cli, "make_evaluator", flaky_evaluator)
    out = tmp_path / "out"
    tune_cli.main(
        [
            "--instances",
            "zdt1",
            "--algorithm",
            "nsgaii",
            "--backend",
            "random",
            "--tune-budget",
            "3",
            "--n-seeds",
            "2",
            "--n-jobs",
            "1",
            "--no-run-validation",
            "--no-run-test",
            "--no-run-statistical-finisher",
            "--output-dir",
            str(out),
        ]
    )

    summaries = list(out.rglob("tuning_summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["evaluations"] == 6
    assert summary["failed_evaluations"] == 3


def _fail_for_half_the_configurations(config, ctx):
    # Fails for about half of the configurations, the same ones in every process.
    import json
    import zlib

    return float("nan") if zlib.crc32(json.dumps(config, sort_keys=True, default=str).encode()) % 2 else 1.0


def test_tuning_summary_counts_evaluations_run_in_worker_processes(tmp_path: Path, monkeypatch):
    # With --n-jobs 2, racing evaluates in joblib worker processes, which count in their own copy of
    # the evaluator; the summary must still count every evaluation.
    import json

    from vamos.experiment.cli import _tune_runtime

    real = _tune_runtime.make_evaluator

    def odd_seed_evaluator(**kwargs):
        inner = real(**kwargs)
        inner.inner = _fail_for_half_the_configurations
        return inner

    monkeypatch.setattr(tune_cli, "make_evaluator", odd_seed_evaluator)
    out = tmp_path / "out"
    tune_cli.main(
        [
            "--instances",
            "zdt1",
            "--algorithm",
            "nsgaii",
            "--backend",
            "racing",
            "--tune-budget",
            "8",
            "--n-seeds",
            "2",
            "--n-jobs",
            "2",
            "--no-run-validation",
            "--no-run-test",
            "--no-run-statistical-finisher",
            "--output-dir",
            str(out),
        ]
    )

    (summary_path,) = out.rglob("tuning_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["evaluations"] > 0
    assert 0 < summary["failed_evaluations"] < summary["evaluations"]


def _validation_inputs(tmp_path: Path):
    import argparse
    import logging
    from types import SimpleNamespace

    from vamos.engine.tuning import Instance, TrialResult
    from vamos.engine.tuning.racing.param_space import ParamSpace, Real

    args = argparse.Namespace(run_validation=True, validation_budget=0, budget=100, validation_topk=2, failure_score=0.0)
    task = SimpleNamespace(param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}), aggregator=lambda vals: float(np.mean(vals)))
    history = [TrialResult(trial_id=0, config={"x": 0.9}, score=2.0), TrialResult(trial_id=1, config={"x": 0.1}, score=1.0)]
    return dict(
        args=args,
        out_dir=tmp_path,
        history=history,
        task=task,
        validation_instances=[Instance(name="zdt1", n_var=6, kwargs={})],
        validation_seeds=[1, 2],
        logger=logging.getLogger("test"),
    )


def test_validation_never_chooses_a_candidate_that_failed_every_run(tmp_path: Path):
    from vamos.experiment.cli._tune_flow import run_validation_stage

    # The best-trained candidate (x = 0.9) fails every validation run; the other one validates.
    def eval_fn(config, ctx):
        return float("nan") if config["x"] == 0.9 else 1.0

    champions = run_validation_stage(eval_fn=eval_fn, **_validation_inputs(tmp_path))

    assert champions == {"champion_global": {"x": 0.1}, "champion_robust": {"x": 0.1}, "champion_fast": {"x": 0.1}}


def test_validation_chooses_no_champion_when_every_candidate_failed(tmp_path: Path):
    from vamos.experiment.cli._tune_flow import run_validation_stage

    champions = run_validation_stage(eval_fn=lambda config, ctx: float("nan"), **_validation_inputs(tmp_path))

    assert champions == {}
    assert not (tmp_path / "selected_configs_validation.json").exists()
    assert (tmp_path / "validation_metrics.csv").exists()


def test_validation_counts_failed_runs_with_the_failure_score(tmp_path: Path):
    from vamos.experiment.cli._tune_flow import run_validation_stage

    inputs = _validation_inputs(tmp_path)
    inputs["validation_seeds"] = list(range(10))

    # x = 0.9 scores 10 once and fails nine times; x = 0.1 scores 2 on every run. With the default failure
    # score of 0, the first averages 1 and the second 2.
    def eval_fn(config, ctx):
        if config["x"] == 0.9:
            return 10.0 if ctx.seed == 0 else float("nan")
        return 2.0

    champions = run_validation_stage(eval_fn=eval_fn, **inputs)

    assert champions["champion_global"] == {"x": 0.1}


def test_tuning_summary_best_score_matches_best_config(tmp_path: Path):
    import argparse
    import json

    from vamos.engine.tuning import ParamSpace, Real, TrialResult, TuningTask
    from vamos.experiment.cli._tune_post import persist_artifacts

    task = TuningTask(
        name="summary",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance(name="zdt1", n_var=6, kwargs={})],
        seeds=[1],
        budget_per_run=100,
    )
    history = [TrialResult(0, {"x": 0.9}, 0.95), TrialResult(1, {"x": 0.4}, 0.80), TrialResult(2, {"x": 0.4}, 0.85)]
    args = argparse.Namespace(
        backend="racing",
        problem="zdt1",
        algorithm="nsgaii",
        n_var=6,
        n_obj=2,
        seed=1,
        aggregate_mode="mean",
        runtime_penalty=0.0,
        failure_score=0.0,
    )

    # A statistical finisher or racing's final choice can pick a configuration other than the top history score.
    persist_artifacts(tmp_path, args, task, {"x": 0.4}, history, 1.0, 1, available_backends_fn=dict)

    summary = json.loads((tmp_path / "tuning_summary.json").read_text(encoding="utf-8"))
    assert summary["best_score"] == 0.85
    assert summary["best_history_score"] == 0.95
