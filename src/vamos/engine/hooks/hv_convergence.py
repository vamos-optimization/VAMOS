from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

Statistic = Literal["mean", "median", "min"]
EpsilonMode = Literal["abs", "rel"]


@dataclass(frozen=True)
class HVConvergenceConfig:
    every_k: int = 200
    window: int = 10
    patience: int = 5
    epsilon: float = 1e-4
    epsilon_mode: EpsilonMode = "rel"
    statistic: Statistic = "median"
    min_points: int = 25
    confidence: float | None = None  # e.g., 0.95
    bootstrap_samples: int = 300
    rng_seed: int = 0


@dataclass
class HVDecision:
    stop: bool
    reason: str
    evals: int
    hv: float
    hv_delta: float | None
    extra: dict[str, Any] = field(default_factory=dict)


class HVConvergenceMonitor:
    """
    Algorithm-agnostic monitor: you feed (evals, hv) samples; it returns stop decisions.
    Logging is done by caller using `trace_rows()`.
    """

    def __init__(self, cfg: HVConvergenceConfig):
        if cfg.every_k <= 0:
            raise ValueError("every_k must be > 0")
        if cfg.window <= 1:
            raise ValueError("window must be > 1")
        if cfg.patience <= 0:
            raise ValueError("patience must be > 0")
        if cfg.min_points < cfg.window + 2:
            raise ValueError("min_points must be at least window+2")
        self.cfg = cfg
        self._evals: list[int] = []
        self._hv: list[float] = []
        self._bad_streak: int = 0
        self._stopped: bool = False
        self._rng = random.Random(cfg.rng_seed)

    def add_sample(self, evals: int, hv: float) -> HVDecision:
        if self._stopped:
            return HVDecision(True, "already_stopped", evals, hv, None)

        self._evals.append(int(evals))
        self._hv.append(float(hv))

        # Only check every_k rhythm in terms of evals increments when caller follows the contract;
        # we still allow decisions whenever enough points exist.
        idx = len(self._hv) - 1
        if len(self._hv) < self.cfg.min_points:
            return HVDecision(False, "warming_up", evals, hv, None, {"points": len(self._hv)})

        delta = self._window_delta(idx)

        eps_abs = self._epsilon_abs(hv_ref=self._hv[idx])
        stat_delta = self._statistic_over_recent_deltas(idx)

        # Optional bootstrap confidence check
        ci_ok = True
        ci_upper = None
        if self.cfg.confidence is not None:
            ci_upper = self._bootstrap_ci_upper_recent(idx, conf=self.cfg.confidence)
            # Conservative: if upper bound is still <= eps_abs, we consider "no meaningful improvement"
            ci_ok = (ci_upper is not None) and (ci_upper <= eps_abs)

        insufficient = (stat_delta is not None) and (stat_delta <= eps_abs) and ci_ok
        if insufficient:
            self._bad_streak += 1
        else:
            self._bad_streak = 0

        if self._bad_streak >= self.cfg.patience:
            self._stopped = True
            return HVDecision(
                True,
                "converged_hv",
                evals,
                hv,
                delta,
                {
                    "eps_abs": eps_abs,
                    "stat_delta": stat_delta,
                    "ci_upper": ci_upper,
                    "bad_streak": self._bad_streak,
                    "window": self.cfg.window,
                    "patience": self.cfg.patience,
                },
            )

        return HVDecision(
            False,
            "continue",
            evals,
            hv,
            delta,
            {
                "eps_abs": eps_abs,
                "stat_delta": stat_delta,
                "ci_upper": ci_upper,
                "bad_streak": self._bad_streak,
            },
        )

    def stopped(self) -> bool:
        return self._stopped

    def trace_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i in range(len(self._hv)):
            delta = None
            if i >= self.cfg.window:
                delta = self._hv[i] - self._hv[i - self.cfg.window]
            rows.append(
                {
                    "evals": self._evals[i],
                    "hv": self._hv[i],
                    "hv_delta": delta,
                }
            )
        return rows

    # ---- internals ----

    def _epsilon_abs(self, hv_ref: float) -> float:
        if self.cfg.epsilon_mode == "abs":
            return float(self.cfg.epsilon)
        # relative to scale: max(|hv_ref|, tiny)
        scale = max(abs(hv_ref), 1e-12)
        return float(self.cfg.epsilon) * scale

    def _window_delta(self, idx: int) -> float | None:
        if idx < self.cfg.window:
            return None
        return self._hv[idx] - self._hv[idx - self.cfg.window]

    def _recent_window_deltas(self, idx: int) -> list[float]:
        # collect deltas over last 'window' checks
        W = self.cfg.window
        deltas = []
        # We define a delta at j as hv[j] - hv[j-W]
        for j in range(max(W, idx - W + 1), idx + 1):
            d = self._hv[j] - self._hv[j - W]
            if not math.isnan(d):
                deltas.append(d)
        return deltas

    def _statistic_over_recent_deltas(self, idx: int) -> float | None:
        deltas = self._recent_window_deltas(idx)
        if not deltas:
            return None
        if self.cfg.statistic == "mean":
            return sum(deltas) / len(deltas)
        if self.cfg.statistic == "min":
            return min(deltas)
        # median
        s = sorted(deltas)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])

    def _bootstrap_ci_upper_recent(self, idx: int, conf: float) -> float | None:
        deltas = self._recent_window_deltas(idx)
        if len(deltas) < 5:
            return None
        B = int(self.cfg.bootstrap_samples)
        if B <= 0:
            return None
        stats = []
        for _ in range(B):
            sample = [self._rng.choice(deltas) for _ in range(len(deltas))]
            if self.cfg.statistic == "mean":
                stats.append(sum(sample) / len(sample))
            elif self.cfg.statistic == "min":
                stats.append(min(sample))
            else:
                s = sorted(sample)
                mid = len(s) // 2
                stats.append(s[mid] if len(s) % 2 == 1 else 0.5 * (s[mid - 1] + s[mid]))
        stats.sort()
        # upper bound at conf
        q = min(max(conf, 0.0), 1.0)
        k = int(math.ceil(q * (len(stats) - 1)))
        return float(stats[k])
