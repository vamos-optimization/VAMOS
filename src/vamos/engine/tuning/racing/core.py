from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import-untyped]

from .elimination import eliminate_configs, update_elite_archive
from .eval_types import EvalFn
from .multi_fidelity import _run_multi_fidelity
from .param_space import ParamSpace
from .random_search_tuner import TrialResult
from .refill import refill_population
from .sampler import ModelBasedSampler, Sampler
from .scenario import Scenario
from .schedule import build_schedule
from .state import ConfigState, EliteEntry
from .tuning_task import EvalContext, Instance, TuningTask


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _failure_score(maximize: bool) -> float:
    # Use a finite sentinel so aggregators and statistical tests keep working.
    return -1e30 if maximize else 1e30


def _eval_worker(
    eval_fn: EvalFn,
    config: dict[str, Any],
    ctx: EvalContext,
    maximize: bool,
) -> float:
    try:
        result = eval_fn(config, ctx)
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)
    except Exception:
        _logger().warning("[racing] eval_fn failed in worker; assigning failure score.", exc_info=True)
        return _failure_score(maximize)


_BlockScores = dict[tuple[int, int], float]


def _final_evidence(survivors: list[ConfigState], *, multi_fidelity: bool) -> list[tuple[ConfigState, _BlockScores]]:
    """Map each survivor's final scores to the blocks that produced them."""
    if not multi_fidelity:
        return [(state, dict(zip(state.blocks, state.scores))) for state in survivors if state.scores]

    # Survivors are compared on the last fidelity level they reached, which is
    # the final level unless the experiment budget stopped the run earlier.
    reached = [max(state.fidelity_scores) for state in survivors if state.fidelity_scores]
    if not reached:
        return []
    level = max(reached)
    evidence: list[tuple[ConfigState, _BlockScores]] = []
    for state in survivors:
        level_scores = state.fidelity_scores.get(level, [])
        if level_scores:
            # That level is the most recent one, so its blocks close the record.
            level_blocks = state.blocks[-len(level_scores) :]
            evidence.append((state, dict(zip(level_blocks, level_scores))))
    return evidence


def _select_final(configs: list[ConfigState], *, task: TuningTask, multi_fidelity: bool) -> EliteEntry | None:
    """Choose the returned configuration among the survivors only.

    Survivors are aggregated over the blocks they all share, so a configuration
    cannot win because it was scored on easier instances or on fewer blocks.
    When survivors share no block, those with the fewest blocks are left out
    until some block is shared.
    """
    pool = _final_evidence([state for state in configs if state.alive], multi_fidelity=multi_fidelity)
    shared: set[tuple[int, int]] = set()
    while pool:
        shared = set.intersection(*(set(by_block) for _, by_block in pool))
        if shared:
            break
        fewest = min(len(by_block) for _, by_block in pool)
        richer = [entry for entry in pool if len(entry[1]) > fewest]
        if not richer:
            break
        pool = richer

    best: EliteEntry | None = None
    for state, by_block in pool:
        blocks = [block for block in by_block if block in shared] if shared else list(by_block)
        agg_score = float(task.aggregator([by_block[block] for block in blocks]))
        if np.isnan(agg_score):
            continue
        if best is None or (agg_score > best.score if task.maximize else agg_score < best.score):
            best = EliteEntry(config=state.config, score=agg_score)
    return best


class RacingTuner:
    """Racing tuner for algorithm configuration."""

    def __init__(
        self,
        task: TuningTask,
        scenario: Scenario,
        seed: int = 0,
        max_initial_configs: int = 20,
        sampler: Sampler | None = None,
        initial_configs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.task = task
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.max_initial_configs = max_initial_configs
        self._stage_index: int = 0
        self._elite_archive: list[EliteEntry] = []
        self._next_config_id: int = 0  # Re-indexed below
        self._best_score_history: list[float] = []  # For convergence detection

        # Injection of default/user configurations
        self.initial_configs_payload = initial_configs or []

        self.param_space: ParamSpace = task.param_space
        self.instances: Sequence[Instance] = list(task.instances)
        self.seeds: Sequence[int] = list(task.seeds)

        if sampler is None:
            # Prefer model-based sampling by default in large/conditional spaces.
            self.sampler: Sampler = ModelBasedSampler(self.param_space)
        else:
            self.sampler = sampler

        self._schedule: list[tuple[int, int]] = build_schedule(
            self.instances,
            self.seeds,
            start_instances=self.scenario.start_instances,
            instance_order_random=self.scenario.instance_order_random,
            seed_order_random=self.scenario.seed_order_random,
            rng=self.rng,
        )

    def _sample_initial_configs(self) -> list[ConfigState]:
        configs: list[ConfigState] = []
        config_id_counter = 0

        # 1. Add injected configs
        for user_cfg in self.initial_configs_payload:
            state = ConfigState(config_id=config_id_counter, config=user_cfg, alive=True)
            configs.append(state)
            config_id_counter += 1

        # 2. Sample remainder
        needed = max(0, self.max_initial_configs - len(configs))
        for _ in range(needed):
            cfg = self.sampler.sample(self.rng)
            state = ConfigState(config_id=config_id_counter, config=cfg, alive=True)
            configs.append(state)
            config_id_counter += 1

        self._next_config_id = config_id_counter
        return configs

    def _current_budget(self, stage_index: int | None = None) -> int:
        base_budget = self.task.budget_per_run
        if self.scenario.use_adaptive_budget:
            if self.scenario.initial_budget_per_run is not None:
                base_budget = self.scenario.initial_budget_per_run

            stage = self._stage_index if stage_index is None else stage_index
            budget = int(round(base_budget * (self.scenario.budget_growth_factor**stage)))

            if self.scenario.max_budget_per_run is not None:
                budget = min(budget, self.scenario.max_budget_per_run)

            if self.task.budget_per_run is not None:
                budget = min(budget, self.task.budget_per_run)

            if budget <= 0:
                budget = max(1, self.task.budget_per_run or 1)
            return budget

        if base_budget is None or base_budget <= 0:
            raise ValueError("task.budget_per_run must be positive when adaptive budget is disabled")
        return base_budget

    def run(
        self,
        eval_fn: EvalFn,
        verbose: bool | None = None,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        # Dispatch to multi-fidelity if enabled
        if self.scenario.use_multi_fidelity:
            return _run_multi_fidelity(self, eval_fn, verbose)

        verbose_flag = self.scenario.verbose if verbose is None else verbose

        schedule = list(self._schedule)
        configs: list[ConfigState] = self._sample_initial_configs()
        num_experiments = 0

        for inst_idx, seed_idx in schedule:
            if self.scenario.max_stages is not None and self._stage_index >= self.scenario.max_stages:
                if verbose_flag:
                    _logger().info("[racing] Reached maximum number of stages.")
                break

            if self._num_alive(configs) == 0:
                break

            stage_alive = self._num_alive(configs)
            remaining_budget = self.scenario.max_experiments - num_experiments
            if remaining_budget <= 0:
                if verbose_flag:
                    _logger().info("[racing] Experiment budget exhausted.")
                break
            partial_eval_limit: int | None = None
            if remaining_budget < stage_alive:
                partial_eval_limit = int(remaining_budget)
                if verbose_flag:
                    _logger().info(
                        "[racing] Partial stage due to budget: evaluating %s/%s alive configs.",
                        partial_eval_limit,
                        stage_alive,
                    )

            if verbose_flag:
                _logger().info(
                    "[racing] Stage %s: instance %s, seed idx %s, alive=%s",
                    self._stage_index,
                    inst_idx,
                    seed_idx,
                    stage_alive,
                )

            stage_eval_count = self._run_stage(
                configs,
                inst_idx,
                seed_idx,
                eval_fn,
                max_evals=partial_eval_limit,
            )
            num_experiments += stage_eval_count

            eliminated_any = False
            if partial_eval_limit is None:
                eliminated_any = eliminate_configs(configs, task=self.task, scenario=self.scenario)

            if eliminated_any:
                self._elite_archive = update_elite_archive(
                    configs,
                    task=self.task,
                    scenario=self.scenario,
                    elite_archive=self._elite_archive,
                )
                if self.scenario.use_elitist_restarts:
                    played = self._played_blocks(configs)
                    n_before = len(configs)
                    self._next_config_id = refill_population(
                        configs,
                        scenario=self.scenario,
                        param_space=self.param_space,
                        sampler=self.sampler,
                        elite_archive=self._elite_archive,
                        target_population_size=self.scenario.target_population_size or self.max_initial_configs,
                        rng=self.rng,
                        next_config_id=self._next_config_id,
                        # Only as many as can be evaluated on the blocks already played.
                        max_new=(self.scenario.max_experiments - num_experiments) // max(1, len(played)),
                    )
                    num_experiments += self._catch_up(configs, set(range(n_before, len(configs))), played, eval_fn)
            if isinstance(self.sampler, ModelBasedSampler):
                survivor_configs = [c.config for c in configs if c.alive]
                self.sampler.update(survivor_configs)

            # Track best score for convergence detection
            current_best = self._get_current_best_score(configs)
            if current_best is not None:
                self._best_score_history.append(current_best)

            reached_budget = num_experiments >= self.scenario.max_experiments
            reached_min_survivors = self._num_alive(configs) <= self.scenario.min_survivors
            reached_convergence = self._check_convergence()

            self._stage_index += 1

            if reached_budget:
                if verbose_flag:
                    _logger().info("[racing] Reached maximum experiment budget.")
                break

            if reached_min_survivors:
                if verbose_flag:
                    _logger().info("[racing] Reached minimum survivors, stopping early.")
                break

            if reached_convergence:
                if verbose_flag:
                    _logger().info("[racing] Converged after %s stages (no improvement).", self._stage_index)
                break

            if partial_eval_limit is not None:
                if verbose_flag:
                    _logger().info("[racing] Budget exhausted after partial stage.")
                break

        best_state, history = self._finalize_results(configs)
        if best_state is None:
            raise RuntimeError("RacingTuner finished without a valid configuration.")

        if verbose_flag and best_state.score is not None:
            _logger().info(
                "[racing] Best score=%.6f after stage %s.",
                best_state.score,
                self._stage_index,
            )

        return best_state.config, history

    def _run_stage(
        self,
        configs: list[ConfigState],
        inst_idx: int,
        seed_idx: int,
        eval_fn: EvalFn,
        max_evals: int | None = None,
        *,
        only: set[int] | None = None,
        budget: int | None = None,
    ) -> int:
        instance = self.instances[inst_idx]
        seed = self.seeds[seed_idx]
        block = (inst_idx, seed_idx)
        budget = self._current_budget() if budget is None else budget

        # Identify jobs to run
        tasks = []
        indices = []
        for idx, state in enumerate(configs):
            if not state.alive or (only is not None and idx not in only):
                continue
            ctx = EvalContext(instance=instance, seed=seed, budget=budget)
            tasks.append((state.config, ctx))
            indices.append(idx)

        if not tasks:
            return 0

        if max_evals is not None and max_evals < len(tasks):
            take = int(max(max_evals, 0))
            if take <= 0:
                return 0
            chosen = self.rng.choice(len(tasks), size=take, replace=False)
            chosen = np.sort(chosen)
            tasks = [tasks[int(i)] for i in chosen]
            indices = [indices[int(i)] for i in chosen]

        if self.scenario.n_jobs == 1:
            # Sequential execution (avoid overhead)
            for i, (cfg, ctx) in enumerate(tasks):
                try:
                    result = eval_fn(cfg, ctx)
                    score = float(result[0]) if isinstance(result, tuple) else float(result)
                except Exception:
                    _logger().warning(
                        "[racing] eval_fn failed for config_id=%s; assigning failure score.",
                        configs[indices[i]].config_id,
                        exc_info=True,
                    )
                    score = _failure_score(self.task.maximize)
                configs[indices[i]].scores.append(score)
                configs[indices[i]].blocks.append(block)
        else:
            # Parallel execution with joblib
            results = Parallel(n_jobs=self.scenario.n_jobs)(
                delayed(_eval_worker)(eval_fn, cfg, ctx, self.task.maximize) for cfg, ctx in tasks
            )

            for i, score in enumerate(results):
                configs[indices[i]].scores.append(score)
                configs[indices[i]].blocks.append(block)
        return len(tasks)

    def _played_blocks(self, configs: list[ConfigState]) -> list[tuple[int, int]]:
        """The blocks the alive configurations have played, in the order they played them."""
        played: list[tuple[int, int]] = []
        for state in configs:
            if state.alive and len(state.blocks) > len(played):
                played = list(state.blocks)
        return played

    def _catch_up(self, configs: list[ConfigState], new: set[int], played: list[tuple[int, int]], eval_fn: EvalFn) -> int:
        """Evaluate configurations spawned by a restart on the blocks the race has played.

        The new configurations then enter the elimination tests on the same blocks as
        the elites, as in elitist irace (López-Ibáñez et al., 2016, §3.4). Each block is
        replayed at the budget of the stage that first played it.
        """
        if not new:
            return 0
        return sum(
            self._run_stage(configs, inst_idx, seed_idx, eval_fn, only=new, budget=self._current_budget(stage))
            for stage, (inst_idx, seed_idx) in enumerate(played)
        )

    def _num_alive(self, configs: list[ConfigState]) -> int:
        return sum(1 for c in configs if c.alive)

    def _get_current_best_score(self, configs: list[ConfigState]) -> float | None:
        """Get the current best aggregated score among alive configs."""
        best: float | None = None
        for state in configs:
            if not state.alive or not state.scores:
                continue
            agg = float(self.task.aggregator(state.scores))
            if best is None:
                best = agg
            elif self.task.maximize and agg > best:
                best = agg
            elif not self.task.maximize and agg < best:
                best = agg
        return best

    def _check_convergence(self) -> bool:
        """Check if best score has stagnated for convergence_window stages."""
        window = self.scenario.convergence_window
        if window <= 0:
            return False  # Disabled

        history = self._best_score_history
        if len(history) < window:
            return False

        # Get scores over the last 'window' stages
        recent = history[-window:]
        oldest = recent[0]
        newest = recent[-1]
        recent_arr = np.asarray(recent, dtype=float)
        if recent_arr.size == 0 or not np.all(np.isfinite(recent_arr)):
            return False

        # Compute relative improvement
        if self.task.maximize:
            delta = newest - oldest
        else:
            delta = oldest - newest
        if abs(oldest) < 1e-12:
            # Avoid division by zero; when near zero use signed absolute-diff.
            improvement = delta
        else:
            improvement = delta / abs(oldest)

        # Also require low volatility in the recent window to avoid declaring
        # convergence on noisy oscillations around a plateau.
        mean_abs = float(np.mean(np.abs(recent_arr)))
        scale = mean_abs if mean_abs > 1e-12 else 1.0
        volatility = float(np.std(recent_arr) / scale)

        return improvement < self.scenario.convergence_threshold and volatility < self.scenario.convergence_threshold

    def _finalize_results(self, configs: list[ConfigState]) -> tuple[EliteEntry | None, list[TrialResult]]:
        history: list[TrialResult] = []

        for state in configs:
            scores = state.scores
            details = {"num_evals": len(state.scores), "alive": state.alive}
            if self.scenario.use_multi_fidelity:
                final_level = len(self.scenario.fidelity_levels) - 1
                final_scores = state.fidelity_scores.get(final_level) or []
                details["final_fidelity_level"] = int(final_level)
                details["num_final_evals"] = int(len(final_scores))
                if final_scores:
                    scores = final_scores

            if not scores:
                agg_score = float("nan")
            else:
                agg_score = float(self.task.aggregator(scores))

            history.append(
                TrialResult(
                    trial_id=state.config_id,
                    config=state.config,
                    score=agg_score,
                    details=details,
                )
            )

        return _select_final(configs, task=self.task, multi_fidelity=self.scenario.use_multi_fidelity), history


__all__ = ["RacingTuner"]
