from types import SimpleNamespace

import numpy as np
import pytest

from vamos.engine.tuning.racing.core import RacingTuner
from vamos.engine.tuning.racing.elimination import eliminate_configs
from vamos.engine.tuning.racing.param_space import Categorical, Condition, ParamSpace, Real
from vamos.engine.tuning.racing.sampler import ModelBasedSampler
from vamos.engine.tuning.racing.scenario import Scenario
from vamos.engine.tuning.racing.state import ConfigState
from vamos.engine.tuning.racing.tuning_task import Instance, TuningTask


def test_elimination_keeps_history_when_lengths_differ():
    task = SimpleNamespace(aggregator=np.mean, maximize=False)
    scenario = SimpleNamespace(
        min_survivors=1,
        elimination_fraction=0.5,
        use_statistical_tests=False,
        min_blocks_before_elimination=1,
        alpha=0.05,
    )
    configs = [
        ConfigState(config_id=0, config={}, alive=True, scores=[0.1, 0.1, 1.0]),
        ConfigState(config_id=1, config={}, alive=True, scores=[0.5]),
    ]

    eliminated = eliminate_configs(configs, task=task, scenario=scenario)

    assert eliminated is True
    assert configs[0].alive is True
    assert configs[1].alive is False


def test_rank_elimination_waits_for_min_blocks():
    task = SimpleNamespace(aggregator=np.mean, maximize=False)
    scenario = SimpleNamespace(
        min_survivors=1,
        elimination_fraction=0.5,
        use_statistical_tests=False,
        min_blocks_before_elimination=3,
        alpha=0.05,
    )
    configs = [
        ConfigState(config_id=0, config={}, alive=True, scores=[0.1, 0.1]),
        ConfigState(config_id=1, config={}, alive=True, scores=[0.5, 0.5]),
    ]

    eliminated = eliminate_configs(configs, task=task, scenario=scenario)

    assert eliminated is False
    assert all(c.alive for c in configs)


def test_stage_index_advances_without_elimination():
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0, 1],
        budget_per_run=1,
        maximize=True,
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
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=3)

    def eval_fn(config, ctx):
        return 0.0

    tuner.run(eval_fn, verbose=False)

    assert tuner._stage_index == len(tuner._schedule)


def test_model_based_sampler_respects_inactive_params():
    space = ParamSpace(
        params={
            "method": Categorical("method", ["a", "b"]),
            "sigma": Real("sigma", 0.0, 1.0),
        },
        conditions=[Condition("sigma", "cfg['method'] == 'a'")],
    )
    sampler = ModelBasedSampler(space, exploration_prob=0.0, min_samples_to_model=1)
    sampler.update([{"method": "a", "sigma": 0.5}, {"method": "b"}])

    rng = np.random.default_rng(1)
    cfg = sampler.sample(rng)

    assert cfg["method"] == "b"
    assert "sigma" not in cfg


# Two instances on very different score levels: a mean taken over a different
# subset of blocks is dominated by the instance mix, not by the quality ``q``.
_INSTANCE_OFFSET = {"easy": 100.0, "hard": 0.0}
_QUALITIES = [float(q) for q in np.linspace(0.05, 0.95, 12)]


def _offset_task(n_seeds: int) -> TuningTask:
    return TuningTask(
        name="offset",
        param_space=ParamSpace(params={"q": Real("q", 0.0, 1.0)}),
        instances=[Instance("easy", 1), Instance("hard", 1)],
        seeds=list(range(n_seeds)),
        budget_per_run=1,
        maximize=True,
    )


def _shared_block_means(observed: dict[tuple[float, str, int], float], qualities: list[float]) -> dict[float, float]:
    blocks = {q: {(inst, seed) for (seen_q, inst, seed) in observed if seen_q == q} for q in qualities}
    shared = set.intersection(*blocks.values())
    assert shared, "survivors must share at least one block"
    return {q: float(np.mean([observed[(q, inst, seed)] for inst, seed in shared])) for q in qualities}


@pytest.mark.parametrize("use_statistical_tests", [False, True])
@pytest.mark.parametrize("tuner_seed", range(6))
def test_racing_returns_a_survivor_over_shared_blocks(use_statistical_tests: bool, tuner_seed: int) -> None:
    noise = 0.5 if use_statistical_tests else 0.01
    observed: dict[tuple[float, str, int], float] = {}

    def eval_fn(config, ctx):
        q = float(config["q"])
        inst = ctx.instance.name
        rng = np.random.default_rng([ctx.seed, 0 if inst == "easy" else 1, round(q * 1e6)])
        score = _INSTANCE_OFFSET[inst] + q + float(rng.normal(0.0, noise))
        observed[(q, inst, ctx.seed)] = score
        return score

    scenario = Scenario(max_experiments=10_000, use_statistical_tests=use_statistical_tests, min_survivors=1)
    tuner = RacingTuner(
        _offset_task(n_seeds=10),
        scenario,
        seed=tuner_seed,
        max_initial_configs=len(_QUALITIES),
        initial_configs=[{"q": q} for q in _QUALITIES],
    )

    best, history = tuner.run(eval_fn)

    survivors = [float(row.config["q"]) for row in history if row.details["alive"]]
    assert float(best["q"]) in survivors, f"returned q={best['q']:.3f} was eliminated; survivors={survivors}"
    means = _shared_block_means(observed, survivors)
    assert float(best["q"]) == max(means, key=lambda q: means[q])


@pytest.mark.parametrize(("max_experiments", "final_budget"), [(10_000, 4), (100, 2)])
def test_multi_fidelity_racing_returns_a_final_level_survivor(max_experiments: int, final_budget: int) -> None:
    # Scores are not comparable across fidelities: a cheap run scores higher.
    # 100 experiments fit levels 0 and 1 (12 x 6 + 3 x 6) but not level 2.
    observed: dict[tuple[float, str, int], float] = {}

    def eval_fn(config, ctx):
        q = float(config["q"])
        inst = ctx.instance.name
        score = _INSTANCE_OFFSET[inst] + q + 10.0 / ctx.budget
        if ctx.budget == final_budget:
            observed[(q, inst, ctx.seed)] = score
        return score

    scenario = Scenario(
        max_experiments=max_experiments,
        use_multi_fidelity=True,
        fidelity_levels=(1, 2, 4),
        fidelity_promotion_ratio=0.3,
        fidelity_min_configs=2,
        fidelity_warm_start=False,
        min_survivors=1,
    )
    tuner = RacingTuner(
        _offset_task(n_seeds=3),
        scenario,
        seed=0,
        max_initial_configs=len(_QUALITIES),
        initial_configs=[{"q": q} for q in _QUALITIES],
    )

    best, history = tuner.run(eval_fn)

    survivors = [float(row.config["q"]) for row in history if row.details["alive"]]
    assert float(best["q"]) in survivors, f"returned q={best['q']:.3f} was eliminated; survivors={survivors}"
    means = _shared_block_means(observed, survivors)
    assert float(best["q"]) == max(means, key=lambda q: means[q])


def test_final_selection_compares_late_survivors_on_shared_blocks_only() -> None:
    # A restarted configuration scored only on the easy instance must not beat a
    # veteran that is better on the block they share. When the survivors share no
    # block, those with the fewest blocks are left out instead of compared on
    # their own blocks.
    tuner = RacingTuner(_offset_task(n_seeds=2), Scenario(max_experiments=10), seed=0, max_initial_configs=1)
    veteran = ConfigState(config_id=0, config={"q": 0.9}, scores=[100.9, 0.9, 100.9], blocks=[(0, 0), (1, 0), (0, 1)])
    newcomer = ConfigState(config_id=1, config={"q": 0.5}, scores=[100.5], blocks=[(0, 1)])
    eliminated = ConfigState(config_id=2, config={"q": 1.0}, alive=False, scores=[101.0], blocks=[(0, 0)])

    best, history = tuner._finalize_results([veteran, newcomer, eliminated])
    assert best is not None and best.config == {"q": 0.9}
    assert best.score == 100.9
    assert [row.details["alive"] for row in history] == [True, True, False]

    stranger = ConfigState(config_id=3, config={"q": 0.1}, scores=[100.1], blocks=[(0, 0)])
    best, _ = tuner._finalize_results([newcomer, stranger, veteran])
    assert best is not None and best.config == {"q": 0.9}
