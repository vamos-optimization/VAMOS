from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed  # type: ignore[import-untyped]

from .eval_types import EvalFn, EvalResult
from .random_search_tuner import TrialResult
from .state import ConfigState, raise_if_every_evaluation_failed
from .tuning_task import EvalContext

if TYPE_CHECKING:
    from .core import RacingTuner


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _eval_worker_warmstart(
    eval_fn: EvalFn,
    config: dict[str, Any],
    ctx: EvalContext,
) -> EvalResult | None:
    """The result of one evaluation, or ``None`` when it raised."""
    try:
        return eval_fn(config, ctx)
    except Exception:
        _logger().warning("[multi-fidelity] eval_fn failed; assigning the failure score.", exc_info=True)
        return None


def _record_result(
    tuner: RacingTuner,
    state: ConfigState,
    result: EvalResult | None,
    *,
    block_key: tuple[int, int],
    fidelity_level: int,
    budget: int,
    warm_start: bool,
) -> None:
    """Store one evaluation; an exception or a non-finite score is a failure and keeps no checkpoint."""
    if result is None:
        score, failed = tuner.task.failure_value(), True
    else:
        score, failed = tuner.task.checked_score(result[0] if isinstance(result, tuple) else result)
        if warm_start and not failed and isinstance(result, tuple) and len(result) == 2:
            # A checkpoint and the budget that produced it are kept together.
            state.checkpoint_map[block_key] = result[1]
            state.last_budget_map[block_key] = budget
    state.scores.append(score)
    state.blocks.append(block_key)
    state.fidelity_scores.setdefault(fidelity_level, []).append(score)
    state.failures += int(failed)


def _run_multi_fidelity(
    tuner: RacingTuner,
    eval_fn: EvalFn,
    verbose: bool | None = None,
) -> tuple[dict[str, Any], list[TrialResult]]:
    verbose_flag = tuner.scenario.verbose if verbose is None else verbose
    fidelity_levels = list(tuner.scenario.fidelity_levels)
    promotion_ratio = tuner.scenario.fidelity_promotion_ratio
    min_configs = tuner.scenario.fidelity_min_configs

    configs: list[ConfigState] = tuner._sample_initial_configs()
    num_experiments = 0

    inst_indices = list(range(len(tuner.instances)))
    seed_indices = list(range(len(tuner.seeds)))
    if tuner.scenario.instance_order_random:
        tuner.rng.shuffle(inst_indices)
    if tuner.scenario.seed_order_random:
        tuner.rng.shuffle(seed_indices)
    blocks = [(inst_idx, seed_idx) for seed_idx in seed_indices for inst_idx in inst_indices]
    n_blocks = len(blocks)

    if n_blocks == 0:
        raise ValueError("Multi-fidelity tuning requires at least one instance and one seed.")

    if verbose_flag:
        _logger().info(
            "[multi-fidelity] Starting with %d configs, %d fidelity levels: %s",
            len(configs),
            len(fidelity_levels),
            fidelity_levels,
        )

    for fidelity_idx, budget in enumerate(fidelity_levels):
        alive_count = tuner._num_alive(configs)
        if alive_count == 0:
            if verbose_flag:
                _logger().info("[multi-fidelity] No configs alive, stopping.")
            break

        stage_evals = alive_count * n_blocks
        if num_experiments + stage_evals > tuner.scenario.max_experiments:
            if verbose_flag:
                _logger().info("[multi-fidelity] Experiment budget exhausted.")
            break

        if verbose_flag:
            _logger().info(
                "[multi-fidelity] Fidelity %d/%d (budget=%d): evaluating %d configs across %d blocks",
                fidelity_idx + 1,
                len(fidelity_levels),
                budget,
                alive_count,
                n_blocks,
            )

        for inst_idx, seed_idx in blocks:
            _run_stage_with_budget(tuner, configs, inst_idx, seed_idx, eval_fn, budget, fidelity_idx)

        num_experiments += stage_evals

        scored_configs: list[tuple[int, float]] = []
        for idx, state in enumerate(configs):
            if not state.alive:
                continue
            scores = state.fidelity_scores.get(fidelity_idx, [])
            if scores:
                agg = float(tuner.task.aggregator(scores))
            else:
                agg = float("-inf") if tuner.task.maximize else float("inf")
            scored_configs.append((idx, agg))

        if tuner.task.maximize:
            scored_configs.sort(key=lambda x: x[1], reverse=True)
        else:
            scored_configs.sort(key=lambda x: x[1])

        is_final_level = fidelity_idx == len(fidelity_levels) - 1
        if is_final_level:
            n_keep = max(tuner.scenario.min_survivors, min_configs)
        else:
            n_keep = max(int(len(scored_configs) * promotion_ratio), min_configs)

        survivors = {idx for idx, _ in scored_configs[:n_keep]}
        eliminated = 0
        for idx, state in enumerate(configs):
            if state.alive and idx not in survivors:
                state.alive = False
                eliminated += 1

        if verbose_flag and eliminated > 0:
            _logger().info(
                "[multi-fidelity] Eliminated %d configs, %d survivors promoted",
                eliminated,
                len(survivors),
            )

    raise_if_every_evaluation_failed(configs, tuner="RacingTuner")
    best_state, history = tuner._finalize_results(configs)

    if best_state is None:
        raise RuntimeError("Multi-fidelity tuning finished without a valid configuration.")

    if verbose_flag and best_state.score is not None:
        _logger().info(
            "[multi-fidelity] Best score=%.6f after %d fidelity levels.",
            best_state.score,
            len(fidelity_levels),
        )

    return best_state.config, history


def _run_stage_with_budget(
    tuner: RacingTuner,
    configs: list[ConfigState],
    inst_idx: int,
    seed_idx: int,
    eval_fn: EvalFn,
    budget: int,
    fidelity_level: int = 0,
) -> None:
    instance = tuner.instances[inst_idx]
    seed = tuner.seeds[seed_idx]
    warm_start = tuner.scenario.fidelity_warm_start
    block_key = (inst_idx, seed_idx)

    tasks = []
    indices = []
    for idx, state in enumerate(configs):
        if not state.alive:
            continue
        if warm_start:
            # Only this block's own earlier run; a block whose run failed starts again from scratch.
            checkpoint = state.checkpoint_map.get(block_key)
            prev_budget = state.last_budget_map.get(block_key)
        else:
            checkpoint = None
            prev_budget = None

        ctx = EvalContext(
            instance=instance,
            seed=seed,
            budget=budget,
            fidelity_level=fidelity_level,
            previous_budget=prev_budget if warm_start else None,
            checkpoint=checkpoint if warm_start else None,
        )
        tasks.append((state.config, ctx, idx))
        indices.append(idx)

    if not tasks:
        return

    if tuner.scenario.n_jobs == 1:
        results = [_eval_worker_warmstart(eval_fn, cfg, ctx) for cfg, ctx, _ in tasks]
    else:
        results = Parallel(n_jobs=tuner.scenario.n_jobs)(delayed(_eval_worker_warmstart)(eval_fn, cfg, ctx) for cfg, ctx, _ in tasks)
    for (_cfg, _ctx, idx), result in zip(tasks, results):
        _record_result(
            tuner,
            configs[idx],
            result,
            block_key=block_key,
            fidelity_level=fidelity_level,
            budget=budget,
            warm_start=warm_start,
        )


__all__ = ["_run_multi_fidelity", "_run_stage_with_budget"]
