from threading import Barrier, Lock
from typing import Any

import pytest
from joblib import parallel_config

from src.vamos.engine.tuning.racing.core import RacingTuner
from src.vamos.engine.tuning.racing.param_space import ParamSpace, Real
from src.vamos.engine.tuning.racing.scenario import Scenario
from src.vamos.engine.tuning.racing.tuning_task import EvalContext, Instance, TuningTask


def test_parallel_execution_preserves_results():
    """Two workers overlap and retain sequential results, independent of host speed."""
    task = TuningTask(
        name="test_parallel",
        param_space=ParamSpace({"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("inst1", 1)],
        seeds=[0],
        budget_per_run=1,
        maximize=False,
    )

    def evaluate(config: dict[str, Any], ctx: EvalContext) -> float:
        return abs(config["x"] - 0.5)

    sequential = RacingTuner(task, Scenario(max_experiments=8, n_jobs=1, verbose=False), max_initial_configs=8)
    expected = sequential.run(evaluate)

    rendezvous = Barrier(2, timeout=10)
    lock = Lock()
    active = peak = calls = 0

    def parallel_evaluate(config: dict[str, Any], ctx: EvalContext) -> float:
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
            first_pair = calls <= 2
        try:
            # The first evaluation cannot finish until another worker enters.
            # A serial implementation times out and produces failure scores.
            if first_pair:
                rendezvous.wait()
            return evaluate(config, ctx)
        finally:
            with lock:
                active -= 1

    parallel = RacingTuner(task, Scenario(max_experiments=8, n_jobs=2, verbose=False), max_initial_configs=8)
    with parallel_config(backend="threading"):
        actual = parallel.run(parallel_evaluate)

    assert peak == 2
    assert calls == 8
    assert active == 0
    assert actual == expected


def test_joblib_pickling_lambda():
    """
    Ensure that we catch pickling errors if user passes unpicklable stuff?
    Actually RacingTuner takes a function. If we pass a lambda it might fail with joblib
    unless using cloudpickle (joblib usually handles some lambdas but not all).
    Let's just ensure standard functions work.
    """
    param_space = ParamSpace({"x": Real("x", 0.0, 1.0)})
    instances = [Instance("inst1", 1)]
    seeds = [0]

    task = TuningTask(name="test_pickle", param_space=param_space, instances=instances, seeds=seeds, budget_per_run=1, maximize=False)

    scenario = Scenario(max_experiments=10, n_jobs=2, verbose=False)
    tuner = RacingTuner(task, scenario, max_initial_configs=2)

    # Use a lambda - likely to fail with standard pickle, but joblib (loky) might handle it.
    # If it fails, users should define top-level functions.
    # "AttributeError: Can't pickle local object..."
    # We won't test for failure, just documenting behavior.

    # Let's test that a local function works (joblib sometimes handles this)
    def local_eval(config, ctx):
        return 0.0

    try:
        tuner.run(local_eval)
    except Exception as e:
        pytest.fail(f"Parallel execution failed with local function: {e}")
