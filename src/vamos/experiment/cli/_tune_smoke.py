"""Smoke profile of ``vamos tune``: tiny budgets, and a search space whose populations fit them."""

from __future__ import annotations

import argparse
import math
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


def _capped(param: Int, ceiling: int, *, floor: int = 1) -> Int:
    high = min(int(param.high), int(ceiling))
    # Keep two values at least: SMAC3, BOHB and irace reject an integer range without width.
    low = max(int(floor), min(int(param.low), high - 1))
    return Int(param.name, low, high, log=param.log, role=param.role)


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


def smoke_param_space(param_space: ParamSpace, *, max_population: int, n_obj: int, algorithm: str, budget: int) -> ParamSpace:
    """``param_space`` with every sampled population at most ``max_population`` and feasible.

    The built-in spaces sample ``pop_size`` in [20, 200], and RVEA's population is its number of
    reference directions, ``comb(n_partitions + n_obj - 1, n_obj - 1)``, while a smoke run has a few
    dozen evaluations: without the cap, ``optimize`` rejects most sampled configurations. The cap
    never goes below the algorithm's smallest population (:func:`smallest_population`): with many
    objectives, a reference-direction algorithm samples its smallest lattice, and a size one above
    it, which MOEA/D and NSGA-III round down to that lattice. When even that does not fit the smoke
    budget, the command stops with a message instead of failing every evaluation.
    """
    floor = smallest_population(algorithm, n_obj)
    params = dict(param_space.params)
    pop_size = params.get("pop_size")
    partitions = params.get("n_partitions")
    needed = floor + 1 if isinstance(pop_size, Int) else floor
    if needed > budget:
        label = _LATTICE_LABELS.get(canonical_algorithm_name(algorithm).split("_")[0])
        why = " (its smallest reference lattice)" if label else ""
        raise SystemExit(
            f"vamos tune --smoke: {label or algorithm} with {n_obj} objectives needs a population of at least {floor}{why}, "
            f"which does not fit the smoke budget of {budget} evaluations per run. "
            "Run it without --smoke, with a --budget above that population."
        )
    if isinstance(pop_size, Int):
        params["pop_size"] = _capped(pop_size, max(max_population, floor + 1), floor=floor)
    if isinstance(partitions, Int):
        n_dirs = max(2, int(n_obj)) - 1
        limit = max(max_population, floor)
        fitting = [h for h in range(1, int(partitions.high) + 1) if math.comb(h + n_dirs, n_dirs) <= limit]
        # One partition gives the n_obj axis directions, so it always fits; when no second count
        # fits, the range has a single value, which SMAC3, BOHB and irace cannot sample.
        params["n_partitions"] = _capped(partitions, max(fitting))
    return ParamSpace(params=params, conditions=list(param_space.conditions))


__all__ = ["apply_smoke_mode", "smallest_population", "smoke_param_space"]
