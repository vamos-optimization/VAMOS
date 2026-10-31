from __future__ import annotations

from pathlib import Path
from typing import Any

from vamos.experiment.benchmark.report_utils import ensure_dir, higher_is_better, import_pandas


def _group_stats(summary: Any, agg: str) -> Any:
    grouped = summary.groupby(["Algorithm", "Problem", "IndicatorName"])["IndicatorValue"]
    if agg == "median":
        return grouped.median()
    if agg == "mean":
        return grouped.mean()
    if agg == "std":
        return grouped.std()
    if agg == "iqr":
        q75 = grouped.quantile(0.75)
        q25 = grouped.quantile(0.25)
        return q75 - q25
    raise ValueError(f"Unknown aggregation '{agg}'")


def write_summary_tables(summary: Any, output_dir: Path) -> dict[str, Path]:
    import_pandas()
    output_dir = ensure_dir(output_dir)
    created: dict[str, Path] = {}

    stats = {
        "Median": _group_stats(summary, "median"),
        "IQR": _group_stats(summary, "iqr"),
        "Mean": _group_stats(summary, "mean"),
        "Std": _group_stats(summary, "std"),
    }

    for indicator_name in summary["IndicatorName"].unique():
        for label, series in stats.items():
            subset = series.xs(indicator_name, level="IndicatorName")
            table = subset.unstack("Algorithm")
            table.index.name = "Problem"
            path = output_dir / f"{label}-{indicator_name}.csv"
            table.to_csv(path, sep="\t", encoding="utf-8")
            created[f"{label}-{indicator_name}"] = path

    return created


def write_wilcoxon_tables(summary: Any, output_dir: Path, *, alpha: float = 0.05) -> dict[str, Path]:
    pd = import_pandas()
    try:
        from scipy import stats as spstats  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - optional
        return {}

    output_dir = ensure_dir(output_dir)
    created: dict[str, Path] = {}
    algorithms = pd.unique(summary["Algorithm"])
    problems = pd.unique(summary["Problem"])
    indicators = pd.unique(summary["IndicatorName"])

    for indicator in indicators:
        table = pd.DataFrame(index=algorithms[:-1], columns=algorithms[1:])
        for i, row_algorithm in enumerate(algorithms[:-1]):
            wilcoxon_rows = []
            for j, col_algorithm in enumerate(algorithms[1:]):
                if i <= j:
                    line = []
                    for problem in problems:
                        df1 = summary[
                            (summary["Algorithm"] == row_algorithm)
                            & (summary["Problem"] == problem)
                            & (summary["IndicatorName"] == indicator)
                        ]
                        df2 = summary[
                            (summary["Algorithm"] == col_algorithm)
                            & (summary["Problem"] == problem)
                            & (summary["IndicatorName"] == indicator)
                        ]
                        data1 = df1["IndicatorValue"]
                        data2 = df2["IndicatorValue"]
                        if data1.empty or data2.empty:
                            line.append("")
                            continue
                        median1 = data1.median()
                        median2 = data2.median()
                        stat, p = spstats.mannwhitneyu(data1, data2)
                        if p <= alpha:
                            if not higher_is_better(indicator):
                                line.append("+" if median1 <= median2 else "o")
                            else:
                                line.append("+" if median1 >= median2 else "o")
                        else:
                            line.append("-")
                    wilcoxon_rows.append("".join(line))
            if len(wilcoxon_rows) < len(algorithms):
                wilcoxon_rows = [""] * (len(algorithms) - len(wilcoxon_rows) - 1) + wilcoxon_rows
            table.loc[row_algorithm] = wilcoxon_rows

        path = output_dir / f"Wilcoxon-{indicator}.csv"
        table.to_csv(path, sep="\t", encoding="utf-8")
        created[f"Wilcoxon-{indicator}"] = path

    return created


__all__ = ["write_summary_tables", "write_wilcoxon_tables"]
