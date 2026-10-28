from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from .racing.tuning_task import EvalContext, Instance, TuningTask

ScoreEvalFn = Callable[[dict[str, Any], EvalContext], float]


class FidelityTunerLike(Protocol):
    task: TuningTask
    backend: str
    seed: int
    budget_levels: list[int] | None
    fidelity_min_instance_frac: float
    fidelity_min_seed_count: int | None
    fidelity_max_seed_count: int | None
    fidelity_selection_seed: int | None
    _fidelity_cache: dict[int, tuple[Sequence[Instance], Sequence[int], dict[str, Any]]]


def suite_key(instance: Instance) -> str:
    kwargs = dict(getattr(instance, "kwargs", {}) or {})
    for key in ("suite", "family", "group"):
        value = kwargs.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    name = str(getattr(instance, "name", "")).strip().lower()
    if "_" in name:
        return name.split("_", 1)[0]
    return name if name else "default"


def default_optuna_study_name(tuner: FidelityTunerLike) -> str:
    raw = f"{tuner.task.name}_{tuner.backend}_{int(tuner.seed)}"
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw)
    return safe or f"vamos_tuning_{int(tuner.seed)}"


def resolve_seed_bounds(tuner: FidelityTunerLike) -> tuple[int, int]:
    total = max(1, len(tuner.task.seeds))
    max_count = int(tuner.fidelity_max_seed_count) if tuner.fidelity_max_seed_count is not None else total
    max_count = min(total, max(1, max_count))
    min_count = int(tuner.fidelity_min_seed_count) if tuner.fidelity_min_seed_count is not None else max_count
    min_count = min(max_count, max(1, min_count))
    return min_count, max_count


def resolve_budget_levels(tuner: FidelityTunerLike) -> list[int]:
    """Return the increasing per-run budgets of the fidelity schedule.

    Without ``budget_levels`` there is one fidelity, the task's full budget. Given levels are
    capped at that budget, which closes the schedule when it is missing.
    """
    full_budget = int(tuner.task.budget_per_run)
    levels = sorted({min(full_budget, int(v)) for v in tuner.budget_levels or [] if int(v) > 0})
    if not levels or levels[-1] != full_budget:
        levels.append(full_budget)
    return levels


def budget_fraction(tuner: FidelityTunerLike, budget: int) -> float:
    b = int(min(int(tuner.task.budget_per_run), max(1, int(budget))))
    levels = resolve_budget_levels(tuner)
    b_min = int(levels[0])
    b_max = int(levels[-1])
    if b_max <= b_min:
        return 1.0
    frac = (float(b) - float(b_min)) / float(b_max - b_min)
    return float(np.clip(frac, 0.0, 1.0))


def fidelity_level_info(tuner: FidelityTunerLike, budget: int) -> tuple[int, int | None]:
    levels = resolve_budget_levels(tuner)
    b = int(min(int(tuner.task.budget_per_run), max(1, int(budget))))
    idx = 0
    for i, level in enumerate(levels):
        if b >= int(level):
            idx = int(i)
        else:
            break
    prev = int(levels[idx - 1]) if idx > 0 else None
    return int(idx), prev


def resolve_fidelity_slice(tuner: FidelityTunerLike, budget: int) -> tuple[Sequence[Instance], Sequence[int], dict[str, Any]]:
    b = int(min(int(tuner.task.budget_per_run), max(1, int(budget))))
    cached = tuner._fidelity_cache.get(int(b))
    if cached is not None:
        return cached

    all_instances = list(tuner.task.instances)
    all_seeds = [int(s) for s in tuner.task.seeds]
    if not all_instances:
        raise RuntimeError("Tuning task has no instances.")
    if not all_seeds:
        raise RuntimeError("Tuning task has no seeds.")

    frac = budget_fraction(tuner, b)
    min_inst_frac = float(tuner.fidelity_min_instance_frac)
    inst_frac = min_inst_frac + frac * (1.0 - min_inst_frac)
    target_instances = int(max(1, min(len(all_instances), int(round(float(len(all_instances)) * inst_frac)))))
    if target_instances >= len(all_instances):
        selected_instances = list(all_instances)
    else:
        rng_seed = int(tuner.fidelity_selection_seed if tuner.fidelity_selection_seed is not None else tuner.seed)
        rng = np.random.default_rng(rng_seed + int(7919 * b))
        groups: dict[str, list[int]] = {}
        for idx, inst in enumerate(all_instances):
            groups.setdefault(suite_key(inst), []).append(int(idx))
        group_names = sorted(groups)
        for name in group_names:
            rng.shuffle(groups[name])

        selected_idx: list[int] = []
        if target_instances <= len(group_names):
            chosen = rng.choice(np.asarray(group_names, dtype=object), size=target_instances, replace=False)
            for name in chosen.tolist():
                selected_idx.append(int(groups[str(name)].pop()))
        else:
            for name in group_names:
                selected_idx.append(int(groups[name].pop()))
            remainder: list[int] = []
            for name in group_names:
                remainder.extend(int(v) for v in groups[name])
            rng.shuffle(remainder)
            missing = int(target_instances - len(selected_idx))
            selected_idx.extend(remainder[:missing])

        selected_instances = [all_instances[i] for i in sorted(set(selected_idx))]
        if len(selected_instances) < target_instances:
            used = {id(inst) for inst in selected_instances}
            for inst in all_instances:
                if id(inst) in used:
                    continue
                selected_instances.append(inst)
                if len(selected_instances) >= target_instances:
                    break

    min_seed_count, max_seed_count = resolve_seed_bounds(tuner)
    target_seed_count = int(round(float(min_seed_count) + frac * float(max_seed_count - min_seed_count)))
    target_seed_count = max(1, min(len(all_seeds), target_seed_count))
    selected_seeds = list(all_seeds[:target_seed_count]) if target_seed_count < len(all_seeds) else list(all_seeds)

    fidelity_level, previous_budget = fidelity_level_info(tuner, b)
    meta = {
        "budget": int(b),
        "budget_fraction": float(frac),
        "instances_used": int(len(selected_instances)),
        "instances_total": int(len(all_instances)),
        "seeds_used": int(len(selected_seeds)),
        "seeds_total": int(len(all_seeds)),
        "fidelity_level": int(fidelity_level),
        "previous_budget": (None if previous_budget is None else int(previous_budget)),
    }
    resolved = (selected_instances, selected_seeds, meta)
    tuner._fidelity_cache[int(b)] = resolved
    return resolved


def eval_config_at_budget(
    tuner: FidelityTunerLike,
    config: dict[str, Any],
    eval_fn: ScoreEvalFn,
    budget: int,
) -> float:
    scores: list[float] = []
    b = int(min(int(tuner.task.budget_per_run), max(1, int(budget))))
    instances, seeds, fidelity_meta = resolve_fidelity_slice(tuner, b)
    fidelity_level = int(fidelity_meta.get("fidelity_level", 0))
    previous_budget = fidelity_meta.get("previous_budget", None)
    for inst in instances:
        for seed in seeds:
            ctx = EvalContext(
                instance=inst,
                seed=int(seed),
                budget=b,
                fidelity_level=int(fidelity_level),
                previous_budget=(None if previous_budget is None else int(previous_budget)),
            )
            result = eval_fn(config, ctx)
            if isinstance(result, tuple):
                scores.append(float(result[0]))
            else:
                scores.append(float(result))
    if not scores:
        raise RuntimeError("No scores computed for configuration.")
    return float(tuner.task.aggregator(scores))


__all__ = [
    "ScoreEvalFn",
    "budget_fraction",
    "default_optuna_study_name",
    "eval_config_at_budget",
    "fidelity_level_info",
    "resolve_budget_levels",
    "resolve_fidelity_slice",
    "resolve_seed_bounds",
    "suite_key",
]
