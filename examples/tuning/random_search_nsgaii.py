"""Small executable example of algorithm-configuration tuning in VAMOS.

This example tunes two NSGA-II operator parameters with the experimental
``vamos.engine.tuning`` API. Candidate configurations are evaluated through
the stable ``optimize`` and ``NSGAIIConfig`` interfaces.

The setup is intentionally small enough for documentation smoke tests. For a
scientific study, pre-register a larger training design and evaluate the chosen
configuration on held-out problems/seeds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.engine.tuning import (
    EvalContext,
    Instance,
    ParamSpace,
    RandomSearchTuner,
    Real,
    TuningTask,
)


def _reference_front(problem_name: str, n_points: int = 101) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, n_points)
    if problem_name == "zdt1":
        f2 = 1.0 - np.sqrt(f1)
    elif problem_name == "zdt2":
        f2 = 1.0 - np.square(f1)
    else:
        raise ValueError(
            f"This teaching example only defines reference fronts for zdt1/zdt2, got {problem_name!r}."
        )
    return np.column_stack((f1, f2))


def _igd(front: np.ndarray, problem_name: str) -> float:
    """Compute a small self-contained IGD estimate against an analytic ZDT front."""
    reference = _reference_front(problem_name)
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def evaluate_config(config: dict[str, Any], ctx: EvalContext) -> float:
    """Run one NSGA-II configuration on the problem/seed supplied by the tuner."""
    algorithm_config = (
        NSGAIIConfig.builder()
        .pop_size(20)
        .selection("tournament")
        .crossover("sbx", prob=float(config["crossover_prob"]), eta=20.0)
        .mutation(
            "pm",
            prob=1.0 / ctx.instance.n_var,
            eta=float(config["mutation_eta"]),
        )
        .build()
    )
    result = optimize(
        ctx.instance.name,
        algorithm="nsgaii",
        algorithm_config=algorithm_config,
        max_evaluations=ctx.budget,
        n_var=ctx.instance.n_var,
        seed=ctx.seed,
        engine="numpy",
    )
    if result.F is None or result.F.size == 0:
        raise RuntimeError("NSGA-II returned no objective vectors.")
    return _igd(result.F, ctx.instance.name)


def main() -> None:
    space = ParamSpace(
        params={
            "crossover_prob": Real("crossover_prob", 0.7, 1.0),
            "mutation_eta": Real("mutation_eta", 10.0, 40.0),
        }
    )
    task = TuningTask(
        name="nsgaii_zdt_training_demo",
        param_space=space,
        instances=[
            Instance(name="zdt1", n_var=30),
            Instance(name="zdt2", n_var=30),
        ],
        seeds=[0, 1],
        budget_per_run=80,
        maximize=False,
        aggregator=lambda values: float(np.median(values)),
    )

    tuner = RandomSearchTuner(task=task, max_trials=2, seed=7)
    best_config, history = tuner.run(evaluate_config, verbose=False)

    print("Best configuration:", best_config)
    print("Trial scores (median IGD; lower is better):")
    for trial in history:
        print(f"  trial={trial.trial_id} score={trial.score:.6f} config={trial.config}")


if __name__ == "__main__":
    main()
