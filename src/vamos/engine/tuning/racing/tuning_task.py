from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .param_space import ParamSpace


@dataclass
class Instance:
    """
    Represents a single problem instance used during tuning.
    """

    name: str
    n_var: int
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalContext:
    """
    Information passed to the evaluation function.

    For multi-fidelity warm-starting:
    - `fidelity_level`: Current fidelity level (0-indexed)
    - `previous_budget`: Budget used in previous fidelity level (None if first level)
    - `checkpoint`: Algorithm state from previous fidelity level (None if first level)

    The eval_fn can use `checkpoint` to warm-start the algorithm instead of
    starting from scratch.
    """

    instance: Instance
    seed: int
    budget: int
    fidelity_level: int = 0
    previous_budget: int | None = None
    checkpoint: Any | None = None


class TuningFailedError(RuntimeError):
    """Every evaluation of a tuning run failed, so no configuration can be chosen."""


@dataclass(frozen=True)
class TaskEvaluation:
    """The aggregated score of one configuration and how many of its evaluations failed."""

    score: float
    evaluations: int
    failed_evaluations: int


def every_evaluation_failed(evaluations: int, failed: int, *, tuner: str) -> TuningFailedError:
    """The error a tuner raises when every one of its ``evaluations`` failed."""
    return TuningFailedError(
        f"{tuner}: every one of the {evaluations} evaluations failed (an exception or a non-finite score), "
        "so no configuration can be chosen. Check the evaluation function on one configuration."
    )


@dataclass
class TuningTask:
    """
    Describes a tuning task for VAMOS:
      - parameter space
      - set of instances
      - seeds
      - budget per run
      - how to aggregate scores across (instance, seed).
    """

    name: str
    param_space: ParamSpace
    instances: Sequence[Instance]
    seeds: Sequence[int]
    budget_per_run: int
    maximize: bool = True
    aggregator: Callable[[list[float]], float] = np.mean
    # Score given to a failed evaluation; ``None`` is the worst finite value for the direction.
    failure_score: float | None = None

    def failure_value(self) -> float:
        """The score of a failed evaluation: ``failure_score``, or -1e30 (maximize) or 1e30."""
        if self.failure_score is not None:
            return float(self.failure_score)
        return -1e30 if self.maximize else 1e30

    def checked_score(self, value: float) -> tuple[float, bool]:
        """``(score, failed)``: a non-finite score is a failed evaluation and gets ``failure_value()``."""
        score = float(value)
        if not math.isfinite(score):
            return self.failure_value(), True
        return score, False

    def evaluate(
        self,
        config: dict[str, Any],
        eval_fn: Callable[[dict[str, Any], EvalContext], Any],
    ) -> TaskEvaluation:
        """Evaluate ``config`` on every (instance, seed) pair and aggregate the scores.

        An evaluation that raises or returns a non-finite score counts as failed and
        contributes ``failure_value()``; the result says how many failed.
        """
        scores: list[float] = []
        failed = 0
        for inst in self.instances:
            for seed in self.seeds:
                ctx = EvalContext(instance=inst, seed=seed, budget=self.budget_per_run)
                try:
                    result = eval_fn(config, ctx)
                    score, failure = self.checked_score(result[0] if isinstance(result, tuple) else result)
                except Exception:
                    logging.getLogger(__name__).warning("[tuning] eval_fn failed; assigning the failure score.", exc_info=True)
                    score, failure = self.failure_value(), True
                scores.append(score)
                failed += int(failure)
        if not scores:
            raise RuntimeError("No scores computed for configuration")
        return TaskEvaluation(float(self.aggregator(scores)), len(scores), failed)

    def eval_config(
        self,
        config: dict[str, Any],
        eval_fn: Callable[[dict[str, Any], EvalContext], Any],
    ) -> float:
        """
        Evaluate a configuration across all (instance, seed) combinations
        and aggregate the scores into a single scalar (see ``evaluate``).
        """
        return self.evaluate(config, eval_fn).score


__all__ = ["EvalContext", "Instance", "TaskEvaluation", "TuningFailedError", "TuningTask", "every_evaluation_failed"]
