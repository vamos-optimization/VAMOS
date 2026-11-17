"""Smoke profile of ``vamos tune``: tiny budgets, and a population that fits them."""

from __future__ import annotations

import argparse
from typing import Any

from vamos.engine.algorithm.catalog import ALGORITHM_MINIMUMS
from vamos.engine.algorithm.variants import canonical_algorithm_name
from vamos.engine.tuning import Int, ParamSpace

from ._tune_irace import smoke_tune_budget
from ._tune_runtime import MODEL_BACKENDS

SMOKE_BUDGET = 30
SMOKE_TUNE_BUDGET = 4
SMOKE_N_SEEDS = 2
SMOKE_N_JOBS = 1


def _clamp_positive(value: int, ceiling: int) -> int:
    return max(1, min(int(value), int(ceiling)))


def apply_smoke_mode(args: argparse.Namespace) -> dict[str, Any]:
    args.budget = _clamp_positive(int(args.budget), SMOKE_BUDGET)
    args.tune_budget = smoke_tune_budget(args, _clamp_positive(int(args.tune_budget), SMOKE_TUNE_BUDGET))
    args.n_seeds = _clamp_positive(int(args.n_seeds), SMOKE_N_SEEDS)
    args.n_jobs = SMOKE_N_JOBS
    # The largest population of a smoke run: half its budget, because the opposition-based
    # initializer evaluates a population and its opposite.
    args.pop_size = _clamp_positive(int(args.pop_size), int(args.budget) // 2)
    args.initial_configs = _clamp_positive(int(args.initial_configs), int(args.tune_budget))
    args.validation_topk = _clamp_positive(int(args.validation_topk), int(args.tune_budget))
    args.finisher_topk = _clamp_positive(int(args.finisher_topk), int(args.tune_budget))
    args.run_validation = False
    args.run_test = False
    args.run_statistical_finisher = False
    args.multi_fidelity = False
    args.show_progress_bar = False
    if str(args.backend) in MODEL_BACKENDS and str(args.backend_fallback) == "error":
        args.backend_fallback = "random"
    return {
        "budget": int(args.budget),
        "tune_budget": int(args.tune_budget),
        "n_seeds": int(args.n_seeds),
        "n_jobs": int(args.n_jobs),
        "pop_size": int(args.pop_size),
        "initial_configs": int(args.initial_configs),
        "run_validation": bool(args.run_validation),
        "run_test": bool(args.run_test),
        "run_statistical_finisher": bool(args.run_statistical_finisher),
        "multi_fidelity": bool(args.multi_fidelity),
        "backend_fallback": str(args.backend_fallback),
    }


# The algorithms whose population can be a reference lattice, with their names in messages.
_LATTICE_LABELS = {"moead": "MOEA/D", "nsgaiii": "NSGA-III", "rvea": "RVEA"}


def smallest_population(algorithm: str, n_obj: int) -> int:
    """The smallest population ``algorithm`` runs with on ``n_obj`` objectives.

    NSGA-III's and RVEA's populations are complete Das-Dennis lattices, and so are MOEA/D's weight
    vectors with three or more objectives; the smallest lattice has one direction per objective.
    Otherwise it is the minimum of :data:`vamos.engine.algorithm.catalog.ALGORITHM_MINIMUMS`.
    """
    base = canonical_algorithm_name(algorithm).split("_")[0]
    minimum = ALGORITHM_MINIMUMS.get(base, {}).get("pop_size", 1)
    if base in ("nsgaiii", "rvea") or (base == "moead" and n_obj > 2):
        return max(minimum, int(n_obj))
    return minimum


def smoke_param_space(
    param_space: ParamSpace, *, max_population: int, n_obj: int, algorithm: str, budget: int
) -> tuple[ParamSpace, dict[str, int]]:
    """``param_space`` for a smoke run, and the settings that the run fixes, its population among them.

    The spaces do not tune ``pop_size``: every run takes the one of the command, which the smoke
    profile caps at half its budget of a few dozen evaluations. The population never goes below the
    algorithm's smallest one (:func:`smallest_population`): with many objectives, a
    reference-direction algorithm runs its smallest lattice. When even that does not fit the smoke
    budget, the command stops with a message instead of failing every evaluation.

    Returns the space and the fixed settings, which the evaluator sets in every configuration:
    ``pop_size``, and any integer parameter of a single value, which leaves the space because SMAC3,
    BOHB and irace reject an integer range without width.
    """
    floor = smallest_population(algorithm, n_obj)
    if floor > budget:
        label = _LATTICE_LABELS.get(canonical_algorithm_name(algorithm).split("_")[0])
        why = " (its smallest reference lattice)" if label else ""
        raise SystemExit(
            f"vamos tune --smoke: {label or algorithm} with {n_obj} objectives needs a population of at least {floor}{why}, "
            f"which does not fit the smoke budget of {budget} evaluations per run. "
            "Run it without --smoke, with a --budget above that population."
        )
    params = dict(param_space.params)
    fixed = {name: int(spec.low) for name, spec in params.items() if isinstance(spec, Int) and spec.low == spec.high}
    if fixed:
        params = {name: spec for name, spec in params.items() if name not in fixed}
        conditions = [condition for condition in param_space.conditions if condition.param_name in params]
        param_space = ParamSpace(params=params, conditions=conditions)
    # A space that loses no parameter, such as the view of a built-in grammar, stays as it is.
    return param_space, {**fixed, "pop_size": max(int(max_population), floor)}


__all__ = ["apply_smoke_mode", "smallest_population", "smoke_param_space"]
