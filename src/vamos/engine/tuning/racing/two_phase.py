"""
Two-phase tuner: Successive Halving screening + an F-race refinement.

Phase 1 (Screening):
    Samples many configurations and evaluates them at progressively increasing
    budgets, using rank-based promotion (Successive Halving).  This is cheap
    and quickly filters out clearly bad regions of the hyperparameter space.

Phase 2 (Refinement):
    Takes the survivors from Phase 1 and races them block by block at full
    budget with F-race tests (Friedman on the ranks of each block and Conover's
    post-hoc comparison, or a paired Wilcoxon test with two alive). It is a
    single race over those survivors, not irace's iterated racing, which samples
    new configurations around the elites; the ``irace`` backend runs that.

Usage:

    >>> from vamos.engine.tuning.racing import (
    ...     TwoPhaseTuner, TwoPhaseScenario, TuningTask, Instance, WarmStartEvaluator,
    ... )
    >>> scenario = TwoPhaseScenario(
    ...     phase1_configs=30,
    ...     phase1_budgets=(1000, 3000),
    ...     phase1_promotion_ratio=0.3,
    ...     phase1_min_survivors=5,
    ...     phase2_max_experiments=600,
    ... )
    >>> tuner = TwoPhaseTuner(task=task, scenario=scenario, seed=42)
    >>> best_config, history = tuner.run(evaluator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import-untyped]

from .core import RacingTuner
from .eval_types import EvalFn
from .random_search_tuner import TrialResult
from .sampler import Sampler, UniformSampler
from .scenario import Scenario
from .state import ConfigState
from .tuning_task import EvalContext, TuningTask


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _failure_score(maximize: bool) -> float:
    return -1e30 if maximize else 1e30


def _screening_worker(
    eval_fn: EvalFn,
    config: dict[str, Any],
    ctx: EvalContext,
    maximize: bool,
) -> float:
    """Worker for parallel phase-1 evaluation; returns a scalar score."""
    try:
        result = eval_fn(config, ctx)
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)
    except Exception:
        _logger().warning("[two-phase] eval_fn failed in screening worker; assigning failure score.", exc_info=True)
        return _failure_score(maximize)


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class TwoPhaseScenario:
    """Configuration for two-phase tuning (screening + racing).

    Phase 1 parameters control Successive Halving:
      * ``phase1_configs`` – number of random configurations to sample.
      * ``phase1_budgets`` – strictly-increasing tuple of cheap evaluation
        budgets.  After each level the worst configs are eliminated.
      * ``phase1_promotion_ratio`` – fraction promoted to the next level.
      * ``phase1_min_survivors`` – hard floor on survivors (also the number
        that enters Phase 2).

    Phase 2 parameters control irace-style racing:
      * ``phase2_max_experiments`` – experiment budget for the racing phase
        (config × instance × seed evaluations).
      * The per-run evaluation budget comes from ``TuningTask.budget_per_run``.
    """

    # -- Phase 1: Successive Halving (screening) ----------------------------
    phase1_configs: int = 30
    """Number of random configurations to sample for screening."""

    phase1_budgets: tuple[int, ...] = (1000, 3000)
    """Strictly-increasing evaluation budgets for each screening level."""

    phase1_promotion_ratio: float = 0.3
    """Fraction of configs promoted at each screening level (top-K by rank)."""

    phase1_min_survivors: int = 5
    """Minimum survivors at each level; also the number entering Phase 2."""

    # -- Phase 2: irace-style Racing (refinement) ---------------------------
    phase2_max_experiments: int = 600
    """Maximum config × instance × seed evaluations in Phase 2."""

    phase2_min_survivors: int = 1
    """Minimum configurations that survive the final race."""

    phase2_use_statistical_tests: bool = True
    """Use Friedman + paired t-tests for elimination (irace-style)."""

    phase2_alpha: float = 0.05
    """Significance level for the statistical tests."""

    phase2_min_blocks_before_elimination: int = 3
    """Minimum (instance × seed) blocks before first elimination attempt."""

    phase2_use_elitist_restarts: bool = False
    """Refill population with local neighbours of elites after elimination."""

    # -- General -------------------------------------------------------------
    n_jobs: int = 1
    """Number of parallel workers (-1 = all cores)."""

    verbose: bool = True
    """Print progress information."""

    def __post_init__(self) -> None:
        if self.phase1_configs < 2:
            raise ValueError("phase1_configs must be >= 2")
        if len(self.phase1_budgets) < 1:
            raise ValueError("phase1_budgets must have at least 1 level")
        for i in range(len(self.phase1_budgets) - 1):
            if self.phase1_budgets[i] >= self.phase1_budgets[i + 1]:
                raise ValueError("phase1_budgets must be strictly increasing")
        if not (0.0 < self.phase1_promotion_ratio <= 1.0):
            raise ValueError("phase1_promotion_ratio must be in (0, 1]")
        if self.phase1_min_survivors < 1:
            raise ValueError("phase1_min_survivors must be >= 1")
        if self.phase2_max_experiments <= 0:
            raise ValueError("phase2_max_experiments must be > 0")
        if self.phase2_min_survivors < 1:
            raise ValueError("phase2_min_survivors must be >= 1")
        if not (0.0 < self.phase2_alpha < 1.0):
            raise ValueError("phase2_alpha must be in (0, 1)")


# ---------------------------------------------------------------------------
# Two-Phase Tuner
# ---------------------------------------------------------------------------


class TwoPhaseTuner:
    """Two-phase tuner combining Successive Halving with irace-style racing.

    Parameters
    ----------
    task : TuningTask
        Defines the parameter space, instances, seeds, and full evaluation
        budget (``budget_per_run`` is the Phase 2 per-run budget).
    scenario : TwoPhaseScenario
        Controls both phases (screening budgets, promotion ratio, racing
        experiment budget, statistical test settings, etc.).
    seed : int
        Random seed for reproducibility.
    sampler : Sampler | None
        Configuration sampler for Phase 1.  Defaults to ``UniformSampler``.
    """

    def __init__(
        self,
        task: TuningTask,
        scenario: TwoPhaseScenario,
        seed: int = 0,
        sampler: Sampler | None = None,
    ) -> None:
        self.task = task
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.param_space = task.param_space
        self.instances = list(task.instances)
        self.seeds = list(task.seeds)

        if sampler is None:
            self.sampler: Sampler = UniformSampler(self.param_space)
        else:
            self.sampler = sampler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        eval_fn: EvalFn,
        verbose: bool | None = None,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        """Run both phases and return ``(best_config, combined_history)``."""
        verbose_flag = self.scenario.verbose if verbose is None else verbose
        sc = self.scenario

        # -- Phase 1: Successive Halving -----------------------------------
        if verbose_flag:
            _logger().info(
                "[two-phase] Phase 1: screening %d configs with budgets %s",
                sc.phase1_configs,
                sc.phase1_budgets,
            )

        survivors, phase1_history = self._run_screening(eval_fn, verbose_flag)

        if verbose_flag:
            _logger().info(
                "[two-phase] Phase 1 complete: %d survivors promoted to Phase 2",
                len(survivors),
            )

        # Reset evaluator normalization bounds so Phase 2 is not distorted
        # by cheap Phase 1 evaluations.
        if hasattr(eval_fn, "reset_bounds") and callable(eval_fn.reset_bounds):
            eval_fn.reset_bounds()
            if verbose_flag:
                _logger().info(
                    "[two-phase] Reset evaluator bounds between phases",
                )

        # -- Phase 2: irace-style Racing -----------------------------------
        if verbose_flag:
            _logger().info(
                "[two-phase] Phase 2: racing %d configs at full budget (budget_per_run=%s)",
                len(survivors),
                self.task.budget_per_run,
            )

        best_config, phase2_history = self._run_racing(
            eval_fn,
            survivors,
            verbose_flag,
            trial_id_offset=sc.phase1_configs,
        )

        all_history = phase1_history + phase2_history

        if verbose_flag:
            _logger().info("[two-phase] Tuning complete.")

        return best_config, all_history

    # ------------------------------------------------------------------
    # Phase 1 – Successive Halving
    # ------------------------------------------------------------------

    def _run_screening(
        self,
        eval_fn: EvalFn,
        verbose: bool,
    ) -> tuple[list[dict[str, Any]], list[TrialResult]]:
        """Cheap rank-based screening across multiple budget levels."""
        sc = self.scenario

        # 1. Sample initial configs
        configs: list[ConfigState] = []
        for i in range(sc.phase1_configs):
            cfg = self.sampler.sample(self.rng)
            configs.append(ConfigState(config_id=i, config=cfg, alive=True))

        # 2. Build evaluation blocks (instance × seed), shuffled
        inst_indices = list(range(len(self.instances)))
        seed_indices = list(range(len(self.seeds)))
        self.rng.shuffle(inst_indices)
        self.rng.shuffle(seed_indices)
        blocks = [(ii, si) for si in seed_indices for ii in inst_indices]

        # 3. Iterate over budget levels
        for level_idx, budget in enumerate(sc.phase1_budgets):
            alive_configs = [c for c in configs if c.alive]
            n_alive = len(alive_configs)
            if n_alive <= sc.phase1_min_survivors:
                break

            if verbose:
                _logger().info(
                    "[screening] Level %d/%d (budget=%d): evaluating %d configs on %d blocks",
                    level_idx + 1,
                    len(sc.phase1_budgets),
                    budget,
                    n_alive,
                    len(blocks),
                )

            # Evaluate all alive configs on every block at this budget
            for inst_idx, seed_idx in blocks:
                self._eval_block(
                    configs,
                    inst_idx,
                    seed_idx,
                    budget,
                    level_idx,
                    eval_fn,
                )

            # Rank-based promotion
            self._promote(configs, level_idx, verbose)

        # 4. Collect results
        survivor_cfgs = [s.config for s in configs if s.alive]
        history = self._build_phase1_history(configs)
        return survivor_cfgs, history

    def _eval_block(
        self,
        configs: list[ConfigState],
        inst_idx: int,
        seed_idx: int,
        budget: int,
        level_idx: int,
        eval_fn: EvalFn,
    ) -> None:
        """Evaluate all alive configs on a single (instance, seed) block."""
        instance = self.instances[inst_idx]
        seed = self.seeds[seed_idx]

        tasks: list[tuple[dict[str, Any], EvalContext]] = []
        task_indices: list[int] = []

        for idx, state in enumerate(configs):
            if not state.alive:
                continue
            ctx = EvalContext(
                instance=instance,
                seed=seed,
                budget=budget,
                fidelity_level=level_idx,
            )
            tasks.append((state.config, ctx))
            task_indices.append(idx)

        if not tasks:
            return

        if self.scenario.n_jobs == 1:
            for i, (cfg, ctx) in enumerate(tasks):
                try:
                    result = eval_fn(cfg, ctx)
                    score = float(result[0]) if isinstance(result, tuple) else float(result)
                except Exception:
                    _logger().warning(
                        "[two-phase] eval_fn failed in screening for config_id=%s; assigning failure score.",
                        configs[task_indices[i]].config_id,
                        exc_info=True,
                    )
                    score = _failure_score(self.task.maximize)
                configs[task_indices[i]].fidelity_scores.setdefault(
                    level_idx,
                    [],
                ).append(score)
        else:
            results = Parallel(n_jobs=self.scenario.n_jobs)(
                delayed(_screening_worker)(eval_fn, cfg, ctx, self.task.maximize) for cfg, ctx in tasks
            )
            for i, score in enumerate(results):
                configs[task_indices[i]].fidelity_scores.setdefault(
                    level_idx,
                    [],
                ).append(float(score))

    def _promote(
        self,
        configs: list[ConfigState],
        level_idx: int,
        verbose: bool,
    ) -> None:
        """Rank alive configs at *level_idx* and eliminate the bottom tier."""
        sc = self.scenario
        scored: list[tuple[int, float]] = []
        for idx, state in enumerate(configs):
            if not state.alive:
                continue
            scores = state.fidelity_scores.get(level_idx, [])
            if scores:
                agg = float(self.task.aggregator(scores))
            else:
                agg = float("-inf") if self.task.maximize else float("inf")
            scored.append((idx, agg))

        if self.task.maximize:
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored.sort(key=lambda x: x[1])

        n_keep = max(
            int(len(scored) * sc.phase1_promotion_ratio),
            sc.phase1_min_survivors,
        )
        survivors_set = {idx for idx, _ in scored[:n_keep]}

        eliminated = 0
        for idx, state in enumerate(configs):
            if state.alive and idx not in survivors_set:
                state.alive = False
                eliminated += 1

        if verbose and eliminated > 0:
            _logger().info(
                "[screening] Eliminated %d, %d survivors promoted",
                eliminated,
                len(survivors_set),
            )

    @staticmethod
    def _build_phase1_history(
        configs: list[ConfigState],
    ) -> list[TrialResult]:
        history: list[TrialResult] = []
        for state in configs:
            if state.fidelity_scores:
                last_level = max(state.fidelity_scores.keys())
                scores = state.fidelity_scores[last_level]
            else:
                scores = []
            agg = float(np.mean(scores)) if scores else float("nan")
            history.append(
                TrialResult(
                    trial_id=state.config_id,
                    config=state.config,
                    score=agg,
                    details={"phase": "screening", "alive": state.alive},
                )
            )
        return history

    # ------------------------------------------------------------------
    # Phase 2 – irace-style Racing
    # ------------------------------------------------------------------

    def _run_racing(
        self,
        eval_fn: EvalFn,
        survivor_configs: list[dict[str, Any]],
        verbose: bool,
        trial_id_offset: int = 0,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        """Run block-by-block racing with statistical tests at full budget."""
        sc = self.scenario

        # Use a near-zero elimination_fraction so that rank-based elimination
        # is effectively disabled.  This prevents noisy early blocks from
        # killing configs before the Friedman + paired t-tests have enough
        # data (min_blocks_before_elimination).  Once statistical tests
        # activate, THEY drive elimination — which is the whole point of
        # irace-style racing.
        racing_scenario = Scenario(
            max_experiments=sc.phase2_max_experiments,
            use_multi_fidelity=False,
            use_statistical_tests=sc.phase2_use_statistical_tests,
            alpha=sc.phase2_alpha,
            min_survivors=sc.phase2_min_survivors,
            min_blocks_before_elimination=sc.phase2_min_blocks_before_elimination,
            elimination_fraction=0.01,  # defer to statistical tests
            use_elitist_restarts=sc.phase2_use_elitist_restarts,
            n_jobs=sc.n_jobs,
            verbose=verbose,
        )

        racer = RacingTuner(
            task=self.task,
            scenario=racing_scenario,
            seed=int(self.rng.integers(0, 2**31)),
            max_initial_configs=len(survivor_configs),
            initial_configs=survivor_configs,
        )

        best_config, raw_history = racer.run(eval_fn, verbose=verbose)

        # Annotate Phase 2 history: add "phase" tag and offset trial_ids
        # so they don't collide with Phase 1 ids.
        history: list[TrialResult] = []
        for h in raw_history:
            details = dict(h.details)
            details["phase"] = "racing"
            history.append(
                TrialResult(
                    trial_id=h.trial_id + trial_id_offset,
                    config=h.config,
                    score=h.score,
                    details=details,
                )
            )

        return best_config, history


__all__ = ["TwoPhaseTuner", "TwoPhaseScenario"]
