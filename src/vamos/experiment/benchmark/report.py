from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vamos.experiment.benchmark.lab_plots import generate_boxplots
from vamos.experiment.benchmark.lab_stats import write_summary_tables, write_wilcoxon_tables
from vamos.experiment.benchmark.lab_summary import build_quality_indicator_summary
from vamos.experiment.benchmark.report_plots import generate_plots
from vamos.experiment.benchmark.report_utils import (
    dump_stats_summary,
    ensure_dir,
    format_cell,
    higher_is_better,
    import_pandas,
)
from vamos.experiment.benchmark.runner import BenchmarkResult
from vamos.ux.analysis.stats import friedman_test, pairwise_wilcoxon


@dataclass
class BenchmarkReportConfig:
    metrics: list[str]
    alpha: float = 0.05
    latex_float_format: str = "%.3f"
    table_caption_prefix: str = ""
    table_label_prefix: str = ""
    emit_lab_outputs: bool = True
    lab_include_time: bool = True


class BenchmarkReport:
    def __init__(self, result: BenchmarkResult, config: BenchmarkReportConfig, output_dir: Path):
        self.result = result
        self.config = config
        self.output_dir = ensure_dir(output_dir)
        self._tidy: Any | None = None
        self._stats_cache: dict[str, Any] | None = None

    def aggregate_metrics(self) -> Any:
        if self._tidy is not None:
            return self._tidy
        pd = import_pandas()
        raw = pd.DataFrame.from_records(self.result.summary_rows())
        records: list[dict[str, Any]] = []
        for _, row in raw.iterrows():
            for metric in self.config.metrics:
                col = metric if metric in row else None
                if col is None and f"indicator_{metric}" in row:
                    col = f"indicator_{metric}"
                if col is None:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                records.append(
                    {
                        "suite": self.result.suite.name,
                        "problem": row.get("problem"),
                        "algorithm": row.get("algorithm"),
                        "engine": row.get("engine"),
                        "seed": row.get("seed"),
                        "metric": metric,
                        "value": val,
                        "n_var": row.get("n_var"),
                        "n_obj": row.get("n_obj"),
                    }
                )
        tidy = pd.DataFrame.from_records(records)
        self._tidy = tidy
        tidy_path = self.output_dir / "metrics_tidy.csv"
        tidy.to_csv(tidy_path, index=False)
        return tidy

    def compute_statistics(self) -> dict[str, Any]:
        if self._stats_cache is not None:
            return self._stats_cache
        import_pandas()  # Ensure pandas is available
        tidy = self.aggregate_metrics()
        stats: dict[str, Any] = {}
        for metric in self.config.metrics:
            dfm = tidy[tidy["metric"] == metric]
            if dfm.empty:
                continue
            grouped = dfm.groupby(["problem", "algorithm"])["value"].agg(["mean", "std"])
            pivot_mean = grouped["mean"].unstack("algorithm")
            fried = None
            wilc = []
            if pivot_mean.shape[1] > 1 and pivot_mean.shape[0] > 1:
                scores = pivot_mean.values
                # Friedman requires at least 3 algorithms; skip when not applicable.
                if scores.shape[1] >= 3:
                    fried = friedman_test(scores, higher_is_better=higher_is_better(metric))
                    wilc = pairwise_wilcoxon(
                        scores,
                        list(pivot_mean.columns),
                        higher_is_better=higher_is_better(metric),
                        alpha=self.config.alpha,
                    )
            stats[metric] = {"grouped": grouped, "pivot_mean": pivot_mean, "friedman": fried, "wilcoxon": wilc}
        self._stats_cache = stats
        dump_stats_summary(stats, self.output_dir / "statistics.json")
        return stats

    def _best_algorithms(self, mean_table: Any, metric: str) -> dict[str, str]:
        best: dict[str, str] = {}
        higher = higher_is_better(metric)
        for problem in mean_table.index:
            row = mean_table.loc[problem]
            if higher:
                best_alg = row.idxmax()
            else:
                best_alg = row.idxmin()
            best[problem] = best_alg
        return best

    def _per_problem_markers(self, dfm: Any, problem: str, best_alg: str, metric: str) -> dict[str, str]:
        markers: dict[str, str] = {}
        subset = dfm[dfm["problem"] == problem]
        pivot = subset.pivot_table(index="seed", columns="algorithm", values="value")
        if best_alg not in pivot.columns:
            return markers
        best_values = pivot[best_alg].dropna()
        for alg in pivot.columns:
            if alg == best_alg:
                continue
            other = pivot[alg].dropna()
            if best_values.empty or other.empty:
                continue
            try:
                # Seeds pair the runs; runs that both found no feasible solution are ties.
                pair = np.column_stack([best_values.to_numpy(dtype=float), other.to_numpy(dtype=float)])
                (result,) = pairwise_wilcoxon(pair, [best_alg, alg], higher_is_better=higher_is_better(metric))
            except ValueError:
                continue
            p = result.p_value
            if p < self.config.alpha:
                higher = higher_is_better(metric)
                best_mean = best_values.mean()
                other_mean = other.mean()
                if (higher and other_mean > best_mean) or ((not higher) and other_mean < best_mean):
                    markers[alg] = "+"
                else:
                    markers[alg] = "-"
        return markers

    def generate_latex_tables(self) -> dict[str, Path]:
        pd = import_pandas()
        tidy = self.aggregate_metrics()
        _ = self.compute_statistics()  # Populate cache for later use
        tables_dir = ensure_dir(self.output_dir / "tables")
        created: dict[str, Path] = {}
        for metric in self.config.metrics:
            dfm = tidy[tidy["metric"] == metric]
            if dfm.empty:
                continue
            grouped = dfm.groupby(["problem", "algorithm"])["value"].agg(["mean", "std"])
            mean_table = grouped["mean"].unstack("algorithm")
            std_table = grouped["std"].unstack("algorithm")
            best_map = self._best_algorithms(mean_table, metric)
            rows = []
            problems = mean_table.index.tolist()
            algorithms = mean_table.columns.tolist()
            for problem in problems:
                row_cells = {"Problem": problem}
                markers = self._per_problem_markers(dfm, problem, best_map[problem], metric)
                for alg in algorithms:
                    mean = mean_table.loc[problem, alg]
                    std = std_table.loc[problem, alg]
                    is_best = alg == best_map[problem]
                    marker = markers.get(alg, "")
                    row_cells[alg] = format_cell(self.config.latex_float_format, mean, std, is_best, marker)
                rows.append(row_cells)
            table_df = pd.DataFrame(rows)
            latex_path = tables_dir / f"{self.result.suite.name}_{metric}.tex"
            caption = f"{self.config.table_caption_prefix}{metric} results for {self.result.suite.name}"
            label = f"{self.config.table_label_prefix}{self.result.suite.name}_{metric}"
            table_df.to_latex(
                latex_path,
                index=False,
                escape=False,
                caption=caption,
                label=label,
            )
            created[metric] = latex_path
        if created:
            master = tables_dir / "all_tables.tex"
            master.write_text(
                "\n".join(f"\\input{{{path.name}}}" for path in created.values()),
                encoding="utf-8",
            )
            created["all"] = master
        return created

    def generate_plots(self) -> dict[str, list[Path]]:
        tidy = self.aggregate_metrics()
        stats = self.compute_statistics()
        plots_dir = ensure_dir(self.output_dir / "plots")
        return generate_plots(tidy, stats, self.config.metrics, self.config.alpha, self.result.suite.name, plots_dir, higher_is_better)

    def generate_lab_outputs(self) -> dict[str, Path | dict[str, Path]]:
        if not self.config.emit_lab_outputs:
            return {}
        pd = import_pandas()
        raw = pd.DataFrame.from_records(self.result.summary_rows())
        if raw.empty:
            return {}
        lab_dir = ensure_dir(self.output_dir / "lab")
        summary_df = build_quality_indicator_summary(
            raw,
            self.config.metrics,
            include_time=self.config.lab_include_time,
        )
        if summary_df.empty:
            return {}
        summary_path = lab_dir / "QualityIndicatorSummary.csv"
        summary_df.to_csv(summary_path, index=False)
        tables_dir = ensure_dir(lab_dir / "tables")
        wilcoxon_dir = ensure_dir(lab_dir / "wilcoxon")
        boxplot_dir = ensure_dir(lab_dir / "boxplots")
        _ = write_summary_tables(summary_df, tables_dir)
        _ = write_wilcoxon_tables(summary_df, wilcoxon_dir, alpha=self.config.alpha)
        _ = generate_boxplots(summary_df, boxplot_dir)
        return {
            "summary": summary_path,
            "tables": tables_dir,
            "wilcoxon": wilcoxon_dir,
            "boxplots": boxplot_dir,
        }
