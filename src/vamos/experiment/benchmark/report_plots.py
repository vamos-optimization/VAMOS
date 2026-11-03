"""
Plot helpers for benchmark reporting.

Figures are built with ``matplotlib.figure.Figure`` and never through ``pyplot``, so writing a
report leaves the process-wide matplotlib backend and figure registry untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from vamos.ux.analysis.stats import plot_critical_distance


def generate_plots(
    tidy: Any,
    stats: dict[str, dict[str, Any]],
    metrics: list[str],
    alpha: float,
    suite_name: str,
    output_dir: Path,
    higher_is_better: Callable[[str], bool],
) -> dict[str, list[Path]]:
    try:  # pragma: no cover - optional heavy dep
        from matplotlib.figure import Figure
    except ImportError:  # pragma: no cover
        return {}
    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: dict[str, list[Path]] = {}
    for metric in metrics:
        metric_paths: list[Path] = []
        payload = stats.get(metric, {})
        fried = payload.get("friedman")
        if fried and getattr(fried, "p_value", 1.0) is not None and fried.p_value < alpha:
            pivot_mean = payload.get("pivot_mean")
            if pivot_mean is not None:
                fig = Figure()
                ax = fig.add_subplot(111)
                plot_critical_distance(
                    avg_ranks=fried.avg_ranks,
                    algo_names=list(pivot_mean.columns),
                    alpha=alpha,
                    n_problems=pivot_mean.shape[0],
                    higher_is_better=higher_is_better(metric),
                    ax=ax,
                    show=False,
                )
                cd_path = plots_dir / f"cd_{metric}.pdf"
                fig.tight_layout()
                fig.savefig(cd_path)
                metric_paths.append(cd_path)
        dfm = tidy[tidy["metric"] == metric]
        problems = sorted(dfm["problem"].unique())
        for problem in problems:
            subset = dfm[dfm["problem"] == problem]
            fig = Figure()
            ax = fig.add_subplot(111)
            labels = list(subset["algorithm"].unique())
            data = [subset[subset["algorithm"] == alg]["value"].dropna().values for alg in labels]
            ax.boxplot(data, patch_artist=True)
            # Label the ticks separately: the boxplot keyword changed name in matplotlib 3.9.
            ax.set_xticks(range(1, len(labels) + 1), labels)
            ax.set_title(f"{metric} | {problem}")
            ax.set_ylabel(metric)
            fig.tight_layout()
            path = plots_dir / f"boxplot_{metric}_{problem}.pdf"
            fig.savefig(path)
            metric_paths.append(path)
        if metric_paths:
            created[metric] = metric_paths
    return created


__all__ = ["generate_plots"]
