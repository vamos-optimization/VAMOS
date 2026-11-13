from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sampler import Sampler, UniformSampler
from .tuning_task import EvalContext, TuningTask, every_evaluation_failed


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class TrialResult:
    trial_id: int
    config: dict[str, Any]
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomSearchTuner:
    """
    Simple random-search tuner for VAMOS.
    """

    task: TuningTask
    max_trials: int
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    sampler: Sampler | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        if self.sampler is None:
            self.sampler = UniformSampler(self.task.param_space)

    def run(
        self,
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
        verbose: bool = True,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        """Sample ``max_trials`` configurations and return the best with the history.

        An evaluation that raises or returns a non-finite score counts as failed and scores
        ``task.failure_value()``; a trial records ``evaluations`` and ``failed_evaluations`` in
        its details, and only a trial with a successful evaluation can be the best. When every
        evaluation fails, ``TuningFailedError`` is raised.
        """
        history: list[TrialResult] = []
        best_score: float | None = None
        best_config: dict[str, Any] | None = None
        evaluations = failures = 0
        assert self.sampler is not None

        for trial_id in range(self.max_trials):
            config = self.sampler.sample(self.rng)

            try:
                self.task.param_space.validate(config)
            except ValueError as e:
                if verbose:
                    _logger().info("[trial %s] invalid config skipped: %s", trial_id, e)
                continue

            evaluation = self.task.evaluate(config, eval_fn)
            score = evaluation.score
            evaluations += evaluation.evaluations
            failures += evaluation.failed_evaluations

            if verbose:
                direction = "max" if self.task.maximize else "min"
                _logger().info("[trial %s] %s score=%.6f", trial_id, direction, score)

            details = {"evaluations": evaluation.evaluations, "failed_evaluations": evaluation.failed_evaluations}
            history.append(TrialResult(trial_id=trial_id, config=config, score=score, details=details))
            if evaluation.failed_evaluations == evaluation.evaluations:
                continue

            if best_score is None:
                best_score = score
                best_config = config
            else:
                if self.task.maximize and score > best_score:
                    best_score = score
                    best_config = config
                elif not self.task.maximize and score < best_score:
                    best_score = score
                    best_config = config

        if evaluations and failures == evaluations:
            raise every_evaluation_failed(evaluations, failures, tuner="RandomSearchTuner")
        if best_config is None:
            raise RuntimeError("Tuner finished without a valid configuration")

        if verbose:
            _logger().info("[tuner] Best score=%.6f", best_score)

        return best_config, history


__all__ = ["RandomSearchTuner", "TrialResult"]
