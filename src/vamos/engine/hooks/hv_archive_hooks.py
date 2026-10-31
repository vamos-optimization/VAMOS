from __future__ import annotations

import logging as _logging
import warnings as _warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.archive.factory import ResultArchiveManager, setup_result_archive
from vamos.engine.hooks.hv_convergence import HVConvergenceConfig, HVConvergenceMonitor, HVDecision
from vamos.foundation.core.hv_stop import near_nadir_reference
from vamos.foundation.observer import RunContext
from vamos.foundation.quality_indicators.hypervolume import compute_hypervolume

_TRACE_LIMIT = 10_000


def _append_trace(trace: list[dict[str, Any]], row: dict[str, Any]) -> None:
    if len(trace) >= _TRACE_LIMIT:
        del trace[0]
    trace.append(row)


def _hv_logger() -> _logging.Logger:
    return _logging.getLogger(__name__)


def _try_compute_hv(F: np.ndarray[Any, Any], ref: list[float] | None = None) -> float | None:
    """
    Best-effort HV computation:
      - For 2D minimization: exact via compute_hypervolume
      - Otherwise: returns None
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0
    if F.ndim != 2 or F.shape[1] != 2:
        n_obj = F.shape[1] if F.ndim == 2 else F.ndim
        _warnings.warn(
            f"HV-based termination/archiving requires exactly 2 objectives; "
            f"this problem has {n_obj}. HV computation is disabled for this run.",
            UserWarning,
            stacklevel=3,
        )
        return None
    if ref is None:
        mx = np.max(F, axis=0)
        ref = [float(mx[0] + 1.0), float(mx[1] + 1.0)]
    # Points outside the reference box contribute nothing, as in the HV termination.
    inside = F[np.all(F <= np.asarray(ref, dtype=float), axis=1)]
    if inside.size == 0:
        return 0.0
    try:
        return float(compute_hypervolume(inside, ref))
    except Exception:
        return None


@dataclass
class HookManagerConfig:
    stopping_enabled: bool
    stop_cfg: HVConvergenceConfig
    archive_enabled: bool
    archive_cfg: ExternalArchiveConfig
    hv_ref_point: list[float] | None = None


class HookManager:
    """
    Orchestrates:
      - external archive updates
      - HV trace computation
      - HV convergence stopping
      - bounded in-memory trace capture for the canonical run manifest
    The runner calls `on_generation(...)` with eval counts in stats.
    """

    def __init__(self, cfg: HookManagerConfig):
        self.cfg = cfg
        self._hv_trace: list[dict[str, Any]] = []
        self._archive_stats: list[dict[str, Any]] = []

        self.monitor = HVConvergenceMonitor(cfg.stop_cfg) if cfg.stopping_enabled else None
        self.archive: ResultArchiveManager | None = None
        self._archive_F: np.ndarray[Any, Any] | None = None
        self._archive_total_inserted = 0
        self._archive_total_pruned = 0

        self._last_hv: float | None = None
        # With an automatic reference, the first sample fixes it so the HV series stays comparable.
        self._auto_ref: list[float] | None = None
        self._last_sample_evals: int | None = None
        self._stop_decision: HVDecision | None = None

    def _ensure_archive(self, F: np.ndarray[Any, Any]) -> None:
        if not self.cfg.archive_enabled or self.archive is not None:
            return
        archive = setup_result_archive(self.cfg.archive_cfg, n_var=0, n_obj=F.shape[1], dtype=np.dtype(float))
        if archive is None:
            raise ValueError("Hook archive requires a finite archive.external.capacity.")
        self.archive = archive

    def on_start(self, ctx: RunContext) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: np.ndarray[Any, Any] | None = None,
        X: np.ndarray[Any, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        if F is None:
            return
        evals = None
        if isinstance(stats, dict):
            evals = stats.get("evals")
        if evals is None:
            evals = generation
        self.on_checkpoint(evals=int(evals), F=np.asarray(F), X=None if X is None else np.asarray(X))

    def on_end(
        self,
        final_F: np.ndarray[Any, Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        return None

    def _should_sample(self, evals: int) -> bool:
        if self.monitor is None:
            return True
        every_k = int(self.cfg.stop_cfg.every_k)
        if every_k <= 0:
            return True
        if self._last_sample_evals is None:
            return True
        return (evals - self._last_sample_evals) >= every_k

    def on_checkpoint(
        self,
        evals: int,
        F: np.ndarray[Any, Any],
        X: np.ndarray[Any, Any] | None = None,
    ) -> None:
        del X
        # Update archive (optional)
        if self.cfg.archive_enabled:
            self._ensure_archive(F)
        if self.archive is not None:
            before = 0 if self._archive_F is None else int(self._archive_F.shape[0])
            dummy_X = np.empty((F.shape[0], 0), dtype=float)
            _, archive_F = self.archive.update(dummy_X, np.asarray(F, dtype=float), None)
            after = int(archive_F.shape[0])
            inserted = int(F.shape[0])
            pruned = max(0, before + inserted - after)
            self._archive_total_inserted += inserted
            self._archive_total_pruned += pruned
            self._archive_F = archive_F
            _append_trace(
                self._archive_stats,
                {
                    "evals": int(evals),
                    "archive_size": after,
                    "inserted": inserted,
                    "pruned": pruned,
                    "prune_reason": self.cfg.archive_cfg.pruning if pruned > 0 else "none",
                },
            )

        if not self._should_sample(evals):
            return
        self._last_sample_evals = int(evals)

        # Choose set to compute HV on: archive if enabled else provided F
        hv_set = F
        if self._archive_F is not None and self._archive_F.size > 0:
            hv_set = self._archive_F

        ref = self.cfg.hv_ref_point
        if ref is None:
            if self._auto_ref is None and hv_set.ndim == 2 and hv_set.shape[0] > 0:
                self._auto_ref = [float(value) for value in near_nadir_reference(hv_set)]
            ref = self._auto_ref
        hv = _try_compute_hv(hv_set, ref=ref)
        hv_delta = None
        stop_flag = False
        stop_reason = ""

        if hv is not None and self.monitor is not None:
            dec = self.monitor.add_sample(int(evals), float(hv))
            self._stop_decision = dec
            hv_delta = dec.hv_delta
            stop_flag = bool(dec.stop)
            stop_reason = str(dec.reason)
        elif hv is None and self.monitor is not None:
            stop_reason = "monitor_enabled_but_hv_unavailable"

        if hv_delta is None and hv is not None and self._last_hv is not None:
            hv_delta = float(hv - self._last_hv)

        reason = "hv_ok" if hv is not None else "hv_unavailable"
        if stop_reason:
            reason = stop_reason

        _append_trace(
            self._hv_trace,
            {
                "evals": int(evals),
                "hv": None if hv is None else float(hv),
                "hv_delta": None if hv_delta is None else float(hv_delta),
                "stop_flag": bool(stop_flag),
                "reason": reason,
            },
        )

        if hv is not None:
            self._last_hv = float(hv)

    def should_stop(self) -> bool:
        if self._stop_decision is None:
            return False
        return bool(self._stop_decision.stop)

    def metadata_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"stopping": {}, "archive": {}}

        if self.monitor is None:
            payload["stopping"] = {"enabled": False}
        else:
            dec = self._stop_decision
            payload["stopping"] = {
                "enabled": True,
                "monitor_type": "hv_convergence",
                "params": self.cfg.stop_cfg.__dict__,
                "triggered": bool(dec.stop) if dec else False,
                "evals_stop": int(dec.evals) if (dec and dec.stop) else None,
                "reason": str(dec.reason) if dec else None,
                "reference_point": list(self.cfg.hv_ref_point) if self.cfg.hv_ref_point is not None else self._auto_ref,
                "trace": list(self._hv_trace),
            }

        if self.archive is None:
            payload["archive"] = {"enabled": False}
        else:
            final_size = 0 if self._archive_F is None else int(self._archive_F.shape[0])
            payload["archive"] = {
                "enabled": True,
                "params": self.cfg.archive_cfg.__dict__,
                "final_size": final_size,
                "total_inserted": int(self._archive_total_inserted),
                "total_pruned": int(self._archive_total_pruned),
                "trace": list(self._archive_stats),
            }

        return payload
