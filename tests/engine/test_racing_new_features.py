"""Tests for new racing features: convergence detection, checkpoint I/O and elitist restarts."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import vamos.engine.tuning.racing.core as core_module
from vamos.engine.tuning import build_nsgaii_config_space
from vamos.engine.tuning.racing.core import RacingTuner
from vamos.engine.tuning.racing.io import load_checkpoint, save_checkpoint
from vamos.engine.tuning.racing.param_space import Categorical, Condition, ParamSpace, Real, _active_assignment
from vamos.engine.tuning.racing.refill import make_neighbor_config
from vamos.engine.tuning.racing.scenario import Scenario
from vamos.engine.tuning.racing.tuning_task import Instance, TuningTask


def test_convergence_detection_stops_early():
    """Test that racing stops when score stagnates for convergence_window stages."""
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0, 1, 2, 3, 4, 5],
        budget_per_run=1,
        maximize=False,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=1000,
        min_survivors=1,
        elimination_fraction=0.01,
        start_instances=1,
        use_statistical_tests=False,
        instance_order_random=False,
        seed_order_random=False,
        convergence_window=3,
        convergence_threshold=0.05,
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=3)

    call_count = [0]

    def eval_fn(config, ctx):
        call_count[0] += 1
        # Return constant score to trigger convergence
        return 0.5

    tuner.run(eval_fn, verbose=False)

    # Should stop early due to convergence
    assert tuner._stage_index < len(tuner._schedule), "Should have stopped early"
    assert len(tuner._best_score_history) >= 3, "Should have tracked at least convergence_window scores"


def test_convergence_disabled_by_default():
    """Test that convergence detection is disabled when window is 0."""
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0, 1],
        budget_per_run=1,
        maximize=False,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=100,
        min_survivors=1,
        elimination_fraction=0.01,
        start_instances=1,
        use_statistical_tests=False,
        instance_order_random=False,
        seed_order_random=False,
        # Default: convergence_window=0
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=3)

    def eval_fn(config, ctx):
        return 0.5

    tuner.run(eval_fn, verbose=False)

    # Should run all stages since convergence is disabled
    assert tuner._stage_index == len(tuner._schedule)


def test_convergence_not_triggered_by_high_variance_plateau():
    """Do not declare convergence when recent best scores are too volatile."""
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0],
        budget_per_run=1,
        maximize=True,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=100,
        convergence_window=4,
        convergence_threshold=0.05,
        use_statistical_tests=False,
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=2)

    # Improvement from oldest to newest is small/negative, but variance is large.
    tuner._best_score_history = [1.0, 0.75, 1.25, 0.98]

    assert tuner._check_convergence() is False


def test_checkpoint_save_load_roundtrip():
    """Test checkpoint save and load preserves data."""
    best_configs = [{"x": 0.5, "pop_size": 100}]
    elite_archive = [{"config": {"x": 0.5}, "score": 0.123}]
    metadata = {"stage_index": 10, "num_experiments": 500}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.json"

        save_checkpoint(best_configs, elite_archive, path, metadata=metadata)

        loaded = load_checkpoint(path)

        assert loaded["best_configs"] == best_configs
        assert loaded["elite_archive"] == elite_archive
        assert loaded["metadata"] == metadata


def test_warm_start_with_initial_configs():
    """Test that initial_configs parameter works as warm start."""
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0],
        budget_per_run=1,
        maximize=False,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=10,
        min_survivors=1,
        elimination_fraction=0.5,
        use_statistical_tests=False,
    )

    warm_start = [{"x": 0.42}, {"x": 0.84}]

    tuner = RacingTuner(
        task=task,
        scenario=scenario,
        seed=0,
        max_initial_configs=5,
        initial_configs=warm_start,
    )

    def eval_fn(config, ctx):
        return config["x"]  # Lower x is better

    best_cfg, history = tuner.run(eval_fn, verbose=False)

    # Check that warm start configs were included
    trial_x_values = [t.config["x"] for t in history]
    assert 0.42 in trial_x_values, "Warm start config should be in history"
    assert 0.84 in trial_x_values, "Warm start config should be in history"


def _restart_space() -> ParamSpace:
    # use_x -> x_kind -> x_eta, with conditions on the direct parent only.
    return ParamSpace(
        params={
            "use_x": Categorical("use_x", [True, False]),
            "x_kind": Categorical("x_kind", ["a", "b"]),
            "x_eta": Real("x_eta", 1.0, 10.0),
            "q": Real("q", 0.0, 1.0),
        },
        conditions=[Condition("x_kind", "cfg['use_x'] == True"), Condition("x_eta", "cfg['x_kind'] == 'b'")],
    )


def test_neighbor_configs_follow_the_space_conditions():
    space = _restart_space()
    rng = np.random.default_rng(0)
    elites = [{"use_x": False, "q": 0.3}, {"use_x": True, "x_kind": "a", "q": 0.3}, {"use_x": True, "x_kind": "b", "x_eta": 4.0, "q": 0.3}]
    neighbors = [make_neighbor_config(elite, space, rng) for elite in elites for _ in range(300)]

    for neighbor in neighbors:
        assert neighbor == _active_assignment(space, neighbor)
        space.validate(neighbor)
    # Children that a perturbation activates are drawn from their own range.
    assert any("x_eta" in neighbor for neighbor in neighbors[:600])
    # Keys outside the space travel with the elite unchanged.
    assert make_neighbor_config({"use_x": False, "q": 0.3, "population_size": 40}, space, rng)["population_size"] == 40


@pytest.mark.parametrize("use_statistical_tests", [False, True])
def test_elitist_restarts_with_builtin_space_do_not_crash(use_statistical_tests: bool):
    space = build_nsgaii_config_space().to_param_space()
    task = TuningTask(name="t", param_space=space, instances=[Instance("a", 1)], seeds=list(range(10)), budget_per_run=1)
    scenario = Scenario(max_experiments=300, use_statistical_tests=use_statistical_tests, use_elitist_restarts=True, verbose=False)

    def eval_fn(config, ctx):
        noise = np.random.default_rng([ctx.seed, int(config["crossover_prob"] * 1e9)]).normal(0.0, 0.05)
        return float(config["crossover_prob"]) + float(noise)

    best, history = RacingTuner(task, scenario, seed=0).run(eval_fn)

    assert len(history) > 20, "the race should have spawned restart configurations"
    for config in [best, *(trial.config for trial in history)]:
        assert config == _active_assignment(space, config)
        space.validate(config)


def test_elitist_restarts_keep_eliminating_after_refill(monkeypatch):
    task = TuningTask(
        name="t",
        param_space=ParamSpace(params={"q": Real("q", 0.0, 1.0)}),
        instances=[Instance("a", 1)],
        seeds=list(range(40)),
        budget_per_run=1,
        maximize=True,
    )
    scenario = Scenario(max_experiments=600, use_statistical_tests=True, use_elitist_restarts=True, verbose=False)
    evaluations = []

    def eval_fn(config, ctx):
        evaluations.append(ctx.seed)
        noise = np.random.default_rng([ctx.seed, int(round(config["q"] * 1e9))]).normal(0.0, 0.2)
        return config["q"] + float(noise)

    stages = []
    original = core_module.eliminate_configs

    def spy(configs, *, task, scenario):
        alive = [state for state in configs if state.alive]
        eliminated = original(configs, task=task, scenario=scenario)
        stages.append((len(alive), sum(state.alive for state in configs), {tuple(state.blocks) for state in alive}))
        return eliminated

    monkeypatch.setattr(core_module, "eliminate_configs", spy)
    best, history = RacingTuner(task, scenario, seed=3, max_initial_configs=12).run(eval_fn)

    eliminating = [index for index, (before, after, _) in enumerate(stages) if after < before]
    assert len(history) > 12, "the race should have spawned restart configurations"
    assert len(eliminating) > 3, f"eliminations stopped after the first refill: stages {eliminating}"
    # Configurations spawned by a restart are tested on the same blocks as the elites.
    assert all(len(block_sets) == 1 for _, _, block_sets in stages)
    assert len(evaluations) <= scenario.max_experiments


def test_warm_start_never_receives_another_blocks_checkpoint():
    task = TuningTask(
        name="w",
        param_space=ParamSpace(params={"q": Real("q", 0.0, 1.0)}),
        instances=[Instance("zdt1", 1), Instance("dtlz2", 1)],
        seeds=[0, 1],
        budget_per_run=300,
    )
    scenario = Scenario(
        max_experiments=100,
        use_multi_fidelity=True,
        fidelity_levels=(100, 300),
        fidelity_min_configs=1,
        fidelity_promotion_ratio=1.0,
        instance_order_random=False,
        seed_order_random=False,
        verbose=False,
    )
    received = {}

    def eval_fn(config, ctx):
        block = f"{ctx.instance.name}/seed{ctx.seed}"
        if ctx.fidelity_level == 0 and block == "zdt1/seed0":
            raise RuntimeError("run failed")
        if ctx.fidelity_level == 1:
            received[block] = (ctx.checkpoint, ctx.previous_budget)
        return config["q"], f"checkpoint-of-{block}"

    RacingTuner(task, scenario, seed=0, max_initial_configs=1).run(eval_fn)

    # The failed block starts again from scratch; every other block resumes its own run.
    assert received["zdt1/seed0"] == (None, None)
    for block in ("zdt1/seed1", "dtlz2/seed0", "dtlz2/seed1"):
        assert received[block] == (f"checkpoint-of-{block}", 100)
