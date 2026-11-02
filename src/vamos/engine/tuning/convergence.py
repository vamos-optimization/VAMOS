"""
Convergence-aware evaluation for tuning workflows.

Provides a TuningCallback that captures per-generation metrics during an
algorithm run and produces a ConvergenceProfile summarizing convergence
behavior, diversity dynamics, and stagnation characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.quality_indicators.hypervolume import hypervolume


@dataclass
class ConvergenceProfile:
    """Summary of convergence behavior from a single algorithm run.

    Attributes:
        hv_curve: HV value at each sampled generation.
        hv_velocity: Rate of HV change at the end of the run (dHV/dgen).
        convergence_gen: Generation index where HV plateaued (None if never).
        stagnation_ratio: Fraction of generations with no HV improvement.
        diversity_curve: Spread metric per sampled generation (if available).
        final_hv: HV at the last generation.
        total_generations: Number of generations observed.
    """

    hv_curve: NDArray[np.float64]
    hv_velocity: float
    convergence_gen: int | None
    stagnation_ratio: float
    final_hv: float
    total_generations: int
    diversity_curve: NDArray[np.float64] | None = None


def compute_profile(
    hv_curve: NDArray[np.float64],
    *,
    plateau_window: int = 10,
    plateau_threshold: float = 1e-4,
    diversity_curve: NDArray[np.float64] | None = None,
) -> ConvergenceProfile:
    """Build a ConvergenceProfile from raw HV trace.

    Parameters
    ----------
    hv_curve : array of HV values per generation
    plateau_window : number of consecutive gens with < threshold improvement
                     to declare convergence
    plateau_threshold : minimum HV improvement to count as "progressing"
    diversity_curve : optional spread/spacing values per generation
    """
    n = len(hv_curve)
    if n == 0:
        return ConvergenceProfile(
            hv_curve=np.array([]),
            hv_velocity=0.0,
            convergence_gen=None,
            stagnation_ratio=1.0,
            final_hv=0.0,
            total_generations=0,
            diversity_curve=diversity_curve,
        )

    hv = np.asarray(hv_curve, dtype=float)

    # HV velocity: slope over the last min(10, n) generations
    tail = min(10, n)
    if tail >= 2:
        x = np.arange(tail, dtype=float)
        y = hv[-tail:]
        # Simple linear regression slope
        hv_velocity = float(np.polyfit(x, y, 1)[0])
    else:
        hv_velocity = 0.0

    # Stagnation: count generations where HV didn't improve
    if n >= 2:
        deltas = np.diff(hv)
        stagnant_gens = int(np.sum(deltas < plateau_threshold))
        stagnation_ratio = stagnant_gens / (n - 1)
    else:
        stagnation_ratio = 0.0

    # Convergence detection: first window of consecutive stagnant gens
    convergence_gen: int | None = None
    if n >= plateau_window + 1:
        deltas = np.diff(hv)
        consecutive = 0
        for i, d in enumerate(deltas):
            if d < plateau_threshold:
                consecutive += 1
                if consecutive >= plateau_window:
                    convergence_gen = i - plateau_window + 2  # gen where plateau started
                    break
            else:
                consecutive = 0

    return ConvergenceProfile(
        hv_curve=hv,
        hv_velocity=hv_velocity,
        convergence_gen=convergence_gen,
        stagnation_ratio=stagnation_ratio,
        final_hv=float(hv[-1]),
        total_generations=n,
        diversity_curve=diversity_curve,
    )


class TuningCallback:
    """LiveVisualization callback that collects convergence metrics during a run.

    Computes HV at each generation using the provided reference point, and
    optionally computes a spread/diversity metric.

    After the run, call ``build_profile()`` to get a ConvergenceProfile.
    """

    def __init__(
        self,
        ref_point: list[float] | NDArray[np.float64],
        *,
        compute_diversity: bool = False,
        early_stop_stagnation: float = 0.0,
    ) -> None:
        self._ref_point = list(ref_point)
        self._compute_diversity = compute_diversity
        self._early_stop_stagnation = early_stop_stagnation

        self._hv_values: list[float] = []
        self._diversity_values: list[float] = []
        self._generation_count: int = 0
        self._stagnant_streak: int = 0
        self._should_stop: bool = False

    def on_start(self, ctx: Any = None) -> None:
        self._hv_values.clear()
        self._diversity_values.clear()
        self._generation_count = 0
        self._stagnant_streak = 0
        self._should_stop = False

    def on_generation(
        self,
        generation: int,
        F: NDArray[np.float64] | None = None,
        X: NDArray[np.float64] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self._generation_count = generation

        if F is None or len(F) == 0:
            return

        F_arr = np.asarray(F, dtype=float)

        # Record HV once a point lies inside the reference box: before that the front has no
        # measurable volume, and a run of zeros would look like stagnation.
        hv_available = False
        try:
            ref = np.asarray(self._ref_point, dtype=float)
            if bool(np.all(F_arr <= ref, axis=1).any()):
                self._hv_values.append(float(hypervolume(F_arr, ref)))
                hv_available = True
        except ValueError:
            hv_available = False

        # Compute diversity (spread) if requested
        if self._compute_diversity and F_arr.shape[0] >= 2:
            spread = _compute_spread(F_arr)
            self._diversity_values.append(spread)

        # Adaptive early stopping based on stagnation
        if hv_available and self._early_stop_stagnation > 0 and len(self._hv_values) >= 2:
            if self._hv_values[-1] - self._hv_values[-2] < 1e-6:
                self._stagnant_streak += 1
            else:
                self._stagnant_streak = 0

            total = len(self._hv_values)
            if total > 10 and self._stagnant_streak / total > self._early_stop_stagnation:
                self._should_stop = True

    def on_end(
        self,
        final_F: NDArray[np.float64] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        pass

    def should_stop(self) -> bool:
        return self._should_stop

    def build_profile(self) -> ConvergenceProfile:
        """Build a ConvergenceProfile from the collected metrics."""
        div_curve = np.array(self._diversity_values) if self._diversity_values else None
        return compute_profile(
            np.array(self._hv_values),
            diversity_curve=div_curve,
        )

    @property
    def hv_values(self) -> list[float]:
        return list(self._hv_values)


def score_with_profile(
    final_hv: float,
    profile: ConvergenceProfile,
    *,
    stagnation_penalty: float = 0.1,
    velocity_bonus: float = 0.05,
) -> float:
    """Compute a composite score that goes beyond final HV.

    Penalizes configs that stagnate early and rewards configs still improving.
    """
    score = final_hv
    score -= stagnation_penalty * profile.stagnation_ratio
    score += velocity_bonus * max(0.0, profile.hv_velocity)
    return score


@dataclass
class ConvergenceDiagnostic:
    """Diagnostic summary from a convergence profile.

    Attributes:
        issue: What went wrong ("stagnation", "low_diversity", "slow_convergence", "good")
        suggestion: Parameter role to bias in next sampling ("population", "operator", etc.)
        confidence: How confident the diagnostic is (0.0 to 1.0)
    """

    issue: str
    suggestion: str
    confidence: float


def diagnose(profile: ConvergenceProfile) -> ConvergenceDiagnostic:
    """Diagnose convergence behavior and suggest parameter adjustments.

    Returns a diagnostic with a suggested parameter role to bias in the next
    sampling round. This enables the tuner to make MOEA-informed decisions
    about where to search next.
    """
    if profile.total_generations == 0:
        return ConvergenceDiagnostic("no_data", "structural", 0.0)

    # Early stagnation: converged too fast, likely stuck in local optimum
    # → increase diversity via mutation / larger population
    if profile.stagnation_ratio > 0.6 and profile.convergence_gen is not None:
        early_ratio = profile.convergence_gen / profile.total_generations
        if early_ratio < 0.3:
            return ConvergenceDiagnostic(
                issue="early_stagnation",
                suggestion="population",  # bigger pop or different selection
                confidence=min(1.0, profile.stagnation_ratio),
            )

    # Low diversity with good convergence → exploitation too strong
    if profile.diversity_curve is not None and len(profile.diversity_curve) >= 2 and profile.stagnation_ratio > 0.4:
        final_div = float(profile.diversity_curve[-1])
        initial_div = float(profile.diversity_curve[0]) if profile.diversity_curve[0] > 0 else 1.0
        div_ratio = final_div / initial_div
        if div_ratio < 0.3:
            return ConvergenceDiagnostic(
                issue="low_diversity",
                suggestion="operator_rate",  # higher mutation prob/eta
                confidence=0.8,
            )

    # Still improving at the end → might benefit from more budget or same operator
    if profile.hv_velocity > 0.001:
        return ConvergenceDiagnostic(
            issue="still_improving",
            suggestion="operator",  # keep exploring operator combos
            confidence=0.6,
        )

    # High stagnation but no early convergence → wrong operator family
    if profile.stagnation_ratio > 0.5:
        return ConvergenceDiagnostic(
            issue="stagnation",
            suggestion="operator",  # try different operators
            confidence=0.7,
        )

    return ConvergenceDiagnostic("good", "operator_rate", 0.3)


def _compute_spread(F: NDArray[np.float64]) -> float:
    """Compute a simple spread metric: max extent of the Pareto front."""
    if F.shape[0] < 2:
        return 0.0
    ranges = np.ptp(F, axis=0)
    return float(np.sqrt(np.sum(ranges**2)))


__all__ = [
    "ConvergenceProfile",
    "TuningCallback",
    "compute_profile",
    "score_with_profile",
    "ConvergenceDiagnostic",
    "diagnose",
]
