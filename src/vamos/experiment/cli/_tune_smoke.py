"""Smoke profile of ``vamos tune``: tiny budgets, and a search space whose populations fit them."""

from __future__ import annotations

import argparse
import math
from typing import Any

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


def _capped(param: Int, ceiling: int) -> Int:
    high = min(int(param.high), int(ceiling))
    # Keep two values at least: SMAC3, BOHB and irace reject an integer range without width.
    low = max(1, min(int(param.low), high - 1))
    return Int(param.name, low, high, log=param.log, role=param.role)


def smoke_param_space(param_space: ParamSpace, *, max_population: int, n_obj: int) -> ParamSpace:
    """``param_space`` with every sampled population at most ``max_population``.

    The built-in spaces sample ``pop_size`` in [20, 200], and RVEA's population is its number of
    reference directions, ``comb(n_partitions + n_obj - 1, n_obj - 1)``, while a smoke run has a few
    dozen evaluations: without the cap, ``optimize`` rejects most sampled configurations.
    """
    params = dict(param_space.params)
    pop_size = params.get("pop_size")
    if isinstance(pop_size, Int):
        params["pop_size"] = _capped(pop_size, max_population)
    partitions = params.get("n_partitions")
    if isinstance(partitions, Int):
        n_dirs = max(2, int(n_obj)) - 1
        fitting = [h for h in range(1, int(partitions.high) + 1) if math.comb(h + n_dirs, n_dirs) <= max_population]
        if fitting:
            params["n_partitions"] = _capped(partitions, max(fitting))
    return ParamSpace(params=params, conditions=list(param_space.conditions))


__all__ = ["apply_smoke_mode", "smoke_param_space"]
