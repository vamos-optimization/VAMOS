from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.observer import Observer, RunContext


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ConsoleObserver(Observer):
    """
    Observer that prints run status and results to the console/log.
    Migrated from vamos.experiment.runner_output.
    """

    def on_start(self, ctx: RunContext) -> None:
        spec = getattr(ctx.selection, "spec", None)
        label = getattr(spec, "label", None) or getattr(spec, "key", "unknown")
        description = getattr(spec, "description", None) or ""

        _logger().info("%s", "=" * 80)
        _logger().info("%s", getattr(ctx.config, "title", "Run"))
        _logger().info("%s", "=" * 80)
        _logger().info("Problem: %s", label)
        if description:
            _logger().info("Description: %s", description)

        # Safe access to n_var/n_obj if problem is a selection or instance
        n_var = getattr(ctx.problem, "n_var", getattr(ctx.selection, "n_var", "?"))
        n_obj = getattr(ctx.problem, "n_obj", getattr(ctx.selection, "n_obj", "?"))
        _logger().info("Decision variables: %s", n_var)
        _logger().info("Objectives: %s", n_obj)

        encoding = getattr(ctx.problem, "encoding", None)
        if encoding is None and spec:
            encoding = getattr(spec, "encoding", None)
        encoding = normalize_encoding(encoding, default="real")
        _logger().info("Encoding: %s", encoding)

        _logger().info("Algorithm: %s", ctx.algorithm_name.upper())
        _logger().info("Backend: %s", ctx.engine_name)

        pop_s = getattr(ctx.config, "population_size", "?")
        max_evals = getattr(ctx.config, "max_evaluations", "?")
        _logger().info("Population size: %s", pop_s)
        try:
            # Try to get offspring size if method usually
            off_s = ctx.config.offspring_size()
            _logger().info("Offspring population size: %s", off_s)
        except Exception:
            _logger().debug("Could not retrieve offspring size from config", exc_info=True)

        _logger().info("Max evaluations: %s", max_evals)
        _logger().info("%s", "-" * 80)

    def on_generation(
        self,
        generation: int,
        F: np.ndarray[Any, Any] | None = None,
        X: np.ndarray[Any, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        # Console observer is typically quiet during generations to avoid spam,
        # unless we want a progress bar. For now, keep it equivalent to old runner (quiet).
        pass

    def on_end(
        self,
        final_F: np.ndarray[Any, Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        if final_stats is None:
            return

        algo = final_stats.get("algorithm", "unknown")
        time_ms = final_stats.get("time_ms", 0.0)
        evals_sec = final_stats.get("evals_per_sec", 0.0)

        # The run's own metrics have no hypervolume: the CLI reports it after all runs, with one
        # reference point for all of them.
        _logger().info("%s -> Time: %.2f ms | Eval/s: %.1f", algo, time_ms, evals_sec)

        spread = final_stats.get("spread")
        if spread is not None:
            _logger().info("Objective 1 spread: %.6f", spread)
