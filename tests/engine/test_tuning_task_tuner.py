import numpy as np
import pytest

from vamos.engine.tuning.racing.param_space import ParamSpace, Real
from vamos.engine.tuning.racing.random_search_tuner import RandomSearchTuner
from vamos.engine.tuning.racing.tuning_task import EvalContext, Instance, TuningTask


def test_tuning_task_eval_config_aggregates_scores():
    space = ParamSpace(params={"x": Real("x", 0.0, 1.0)})
    task = TuningTask(
        name="demo",
        param_space=space,
        instances=[Instance("p", 2)],
        seeds=[1, 2],
        budget_per_run=1,
        maximize=True,
        aggregator=np.mean,
    )

    def eval_fn(config, ctx: EvalContext) -> float:
        return ctx.seed  # deterministic per seed

    cfg = {"x": 0.5}
    score = task.eval_config(cfg, eval_fn)
    assert score == 1.5  # mean of seeds 1 and 2


def test_random_search_tuner_tracks_best():
    space = ParamSpace(params={"x": Real("x", 0.0, 1.0)})
    task = TuningTask(
        name="demo",
        param_space=space,
        instances=[Instance("p", 2)],
        seeds=[0],
        budget_per_run=1,
        maximize=True,
        aggregator=np.mean,
    )

    def eval_fn(config, ctx: EvalContext) -> float:
        return config["x"]

    tuner = RandomSearchTuner(task=task, max_trials=5, seed=0)
    best, history = tuner.run(eval_fn, verbose=False)
    assert len(history) <= 5
    assert "x" in best
    # best x should be near the maximum sampled
    scores = [t.score for t in history]
    assert np.isclose(max(scores), best["x"])


def _task(failure_score: float | None = None) -> TuningTask:
    return TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0, 1],
        budget_per_run=1,
        maximize=True,
        aggregator=np.mean,
        failure_score=failure_score,
    )


def test_random_search_survives_failures_and_ignores_nan():
    calls = {"n": 0}

    def eval_fn(config, ctx: EvalContext) -> float:
        calls["n"] += 1
        if calls["n"] <= 2:
            return float("nan")  # the whole first trial fails
        if calls["n"] == 5:
            raise RuntimeError("one bad run")
        return float(config["x"])

    best, history = RandomSearchTuner(task=_task(), max_trials=6, seed=0).run(eval_fn, verbose=False)

    assert len(history) == 6
    assert history[0].details == {"evaluations": 2, "failed_evaluations": 2}
    assert history[2].details["failed_evaluations"] == 1
    clean = [trial for trial in history if trial.details["failed_evaluations"] == 0]
    assert best == max(clean, key=lambda trial: trial.score).config
    assert all(np.isfinite(trial.score) for trial in history)


def test_task_failure_score_replaces_failed_evaluations():
    task = _task(failure_score=-5.0)

    evaluation = task.evaluate({"x": 0.5}, lambda config, ctx: float("nan") if ctx.seed == 0 else 1.0)

    assert (evaluation.score, evaluation.evaluations, evaluation.failed_evaluations) == (-2.0, 2, 1)
    assert task.eval_config({"x": 0.5}, lambda config, ctx: float("inf")) == -5.0


def test_random_search_raises_when_every_evaluation_fails():
    from vamos.engine.tuning.racing.tuning_task import TuningFailedError

    with pytest.raises(TuningFailedError, match="every one of the 6 evaluations failed"):
        RandomSearchTuner(task=_task(), max_trials=3, seed=0).run(lambda config, ctx: float("nan"), verbose=False)
