from __future__ import annotations

from pathlib import Path
from typing import Any

from vamos.experiment.benchmark.report_utils import ensure_dir, import_pandas


def generate_boxplots(summary: Any, output_dir: Path) -> dict[str, list[Path]]:
    pd = import_pandas()
    try:  # pragma: no cover - optional heavy dep
        from matplotlib.figure import Figure
    except ImportError:  # pragma: no cover
        return {}
    output_dir = ensure_dir(output_dir)
    created: dict[str, list[Path]] = {}

    algorithms = pd.unique(summary["Algorithm"])
    problems = pd.unique(summary["Problem"])
    indicators = pd.unique(summary["IndicatorName"])

    for indicator_name in indicators:
        data = summary[summary["IndicatorName"] == indicator_name]
        paths: list[Path] = []
        for problem in problems:
            data_to_plot = []
            for alg in algorithms:
                subset = data[(data["Algorithm"] == alg) & (data["Problem"] == problem)]
                data_to_plot.append(subset["IndicatorValue"].values)

            fig = Figure(figsize=(9, 6))
            fig.suptitle(problem, y=0.95, fontsize=14)
            ax = fig.add_subplot(111)
            ax.boxplot(data_to_plot)
            ax.set_xticks(range(1, len(algorithms) + 1), list(algorithms), rotation=45, ha="right")
            ax.tick_params(labelsize=10)

            png_path = output_dir / f"boxplot-{problem}-{indicator_name}.png"
            pdf_path = output_dir / f"boxplot-{problem}-{indicator_name}.pdf"
            fig.savefig(png_path, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            paths.extend([png_path, pdf_path])
        if paths:
            created[indicator_name] = paths
    return created


__all__ = ["generate_boxplots"]
