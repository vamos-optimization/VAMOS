from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.config.spec import ExperimentSpec
from vamos.experiment.artifacts import save_result
from vamos.experiment.artifacts.specs import RunSpecInputs, build_run_specs
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.observer import Observer, RunContext


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class _PersistableResult:
    F: np.ndarray[Any, Any] | None
    X: np.ndarray[Any, Any] | None
    data: dict[str, Any]
    meta: dict[str, Any]


class StorageObserver(Observer):
    """Publish each completed CLI run through the canonical v1 writer."""

    def __init__(
        self,
        output_dir: str,
        *,
        config_source: str | None = None,
        config_spec: ExperimentSpec | None = None,
        problem_override: Mapping[str, Any] | None = None,
        hv_stop_config: Mapping[str, Any] | None = None,
        selection_pressure: int = 2,
        external_archive: ExternalArchiveConfig | None = None,
        variations: Mapping[str, Any] | None = None,
        termination: tuple[str, Any],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config_source = config_source
        self.config_spec = config_spec
        self.problem_override = problem_override
        self.hv_stop_config = hv_stop_config
        self.selection_pressure = selection_pressure
        self.external_archive = external_archive
        self.variations = dict(variations or {})
        self.termination = termination
        self._ctx: RunContext | None = None
        self._started_at: str | None = None
        self._started_monotonic: float | None = None

    def on_start(self, ctx: RunContext) -> None:
        self._ctx = ctx
        self._started_at = _utc_now()
        self._started_monotonic = time.perf_counter()

    def on_generation(
        self,
        generation: int,
        F: np.ndarray[Any, Any] | None = None,
        X: np.ndarray[Any, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        return None

    def on_end(
        self,
        final_F: np.ndarray[Any, Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        if final_stats is None or self._ctx is None or final_F is None:
            return
        payload = dict(final_stats.get("payload", {}))
        payload["F"] = final_F
        metrics = self._metrics(final_stats)
        genealogy = payload.get("genealogy")
        if isinstance(genealogy, Mapping):
            metrics["genealogy"] = dict(genealogy)
        payload["metrics"] = metrics
        requested_spec, resolved_spec = self._run_specs(final_stats)
        completed_at = _utc_now()
        runtime_ms = float(final_stats.get("time_ms", 0.0))
        if self._started_monotonic is not None and runtime_ms <= 0:
            runtime_ms = (time.perf_counter() - self._started_monotonic) * 1000.0
        x_value = payload.get("X")
        result = _PersistableResult(
            F=final_F,
            X=x_value if isinstance(x_value, np.ndarray) else None,
            data=payload,
            meta={
                "run_artifact_requested_spec": requested_spec,
                "run_artifact_resolved_spec": resolved_spec,
                "run_artifact_timestamps": {
                    "started_at": self._started_at or completed_at,
                    "completed_at": completed_at,
                    "runtime_ms": runtime_ms,
                },
                "run_artifact_entry_point": {
                    "kind": "cli",
                    "arguments_source": "requested_spec",
                },
                "seed": self._ctx.config.seed,
            },
        )
        save_result(result, self.output_dir)

    def _run_specs(self, final_stats: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._ctx is None:
            raise RuntimeError("StorageObserver requires on_start before on_end.")
        ctx = self._ctx
        cfg_value = final_stats.get("config")
        algorithm_config = dict(cfg_value.to_dict()) if isinstance(cfg_value, AlgorithmConfigProtocol) else {}
        encoding = normalize_encoding(getattr(ctx.problem, "encoding", "real"))
        population_size = getattr(ctx.config, "population_size", None)
        defaults = {
            "algorithm": "explicit",
            "population_size": "explicit",
            "max_evaluations": "explicit",
            "engine": "explicit",
            "algorithm_config": "explicit",
        }
        # Only n_var and n_obj of a problems.<name> block build the problem; the rest configures the run.
        override = self.problem_override or {}
        problem_kwargs = {key: override[key] for key in ("n_var", "n_obj") if override.get(key) is not None}
        generated_requested, resolved = build_run_specs(
            RunSpecInputs(
                problem_built_in=ctx.problem.__class__.__module__.startswith("vamos."),
                problem_label=ctx.selection.spec.key,
                problem_kwargs=problem_kwargs,
                n_var_requested=None,
                n_obj_requested=None,
                n_var=ctx.selection.n_var,
                n_obj=ctx.selection.n_obj,
                encoding=encoding,
                algorithm_requested=ctx.algorithm_name,
                algorithm=ctx.algorithm_name,
                algorithm_config=algorithm_config,
                algorithm_config_explicit=True,
                max_evaluations_requested=ctx.config.max_evaluations,
                termination=self.termination,
                pop_size_requested=population_size,
                resolved_pop_size=population_size,
                engine_requested=ctx.engine_name,
                engine=ctx.engine_name,
                eval_strategy=ctx.config.eval_strategy,
                seed_requested=ctx.config.seed,
                seed=ctx.config.seed,
                default_sources=defaults,
            )
        )
        # The request is what the run effectively asked for, with CLI and problem-block overrides; the
        # config file it came from is kept as provenance.
        requested = dict(generated_requested)
        if self.config_spec is not None:
            requested["config_file"] = dict(self.config_spec)
        return requested, resolved

    def _metrics(self, final_stats: Mapping[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "termination": final_stats.get("termination"),
            "evals_per_sec": final_stats.get("evals_per_sec"),
            "spread": final_stats.get("spread"),
            "backend_device": final_stats.get("backend_device"),
            "backend_capabilities": final_stats.get("backend_capabilities", []),
        }
        hook_metadata = final_stats.get("hook_metadata")
        if isinstance(hook_metadata, Mapping):
            metrics["hooks"] = dict(hook_metadata)
        return metrics


__all__ = ["StorageObserver"]
