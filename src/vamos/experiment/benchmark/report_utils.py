"""
Shared helpers for benchmark reporting (IO, stats, formatting).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Benchmark reporting requires pandas. Install via 'pip install pandas' or the 'analysis'/'examples' extras."
        ) from exc
    return pd


def higher_is_better(metric: str) -> bool:
    """Return whether larger values of ``metric`` are better.

    Only the hypervolume family is maximized. Every other indicator a benchmark reports
    (IGD, IGD+, the epsilon indicators, the Hausdorff distances, time) is minimized, under
    any of its aliases, its ``indicator_`` column name or its ``lab/`` name. The LaTeX
    tables, the rank tests and ``lab/`` all use this one orientation.
    """
    key = metric.strip().lower().removeprefix("indicator_")
    return key in {"hv", "hypervolume", "hv_approx", "whv_rect", "weighted_hv"}


def min_rank_sum_p_value(n_a: int, n_b: int) -> float:
    """Smallest two-sided p-value of an exact rank-sum test with samples of these sizes.

    It is reached when the samples do not overlap: ``2 / C(n_a + n_b, n_a)``, so 0.1 with
    three runs per algorithm and 0.0079 with five.
    """
    if n_a < 1 or n_b < 1:
        return 1.0
    return min(1.0, 2.0 / math.comb(n_a + n_b, n_a))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order.

    A NaN p-value (a test that could not be computed) stays NaN and does not count as a
    hypothesis.
    """
    ordered = sorted((p, index) for index, p in enumerate(p_values) if not math.isnan(p))
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, (p, index) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[index] = running
    return adjusted


def format_cell(fmt: str, mean: float, std: float, is_best: bool, marker: str) -> str:
    if fmt.startswith("%"):
        fmt = fmt.lstrip("%")
    # An infinite mean (a run without feasible solutions) has no meaningful spread.
    cell = f"{mean:{fmt}}" if math.isinf(mean) else f"{mean:{fmt}} +/- {std:{fmt}}"
    if is_best:
        cell = f"\\textbf{{{cell}}}"
    if marker:
        cell = f"{cell}$^{{{marker}}}$"
    return cell


def dump_stats_summary(stats: dict[str, Any], path: Path) -> None:
    serializable = {}
    for metric, payload in stats.items():
        fried = payload.get("friedman")
        serializable[metric] = {
            "friedman": {
                "statistic": _json_number(getattr(fried, "statistic", None)),
                "p_value": _json_number(getattr(fried, "p_value", None)),
            }
            if fried
            else None,
            "pairwise": [{key: _json_number(value) for key, value in entry.items()} for entry in payload.get("pairwise", [])],
        }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _json_number(value: Any) -> Any:
    """Return ``value`` with NaN as ``None``, so the summary stays strict JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


__all__ = [
    "ensure_dir",
    "import_pandas",
    "higher_is_better",
    "holm_adjust",
    "min_rank_sum_p_value",
    "format_cell",
    "dump_stats_summary",
]
