"""Publication tables and data files for the Studio statistical comparison."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vamos.ux.studio.batch import csv_text, latex_escape
from vamos.ux.studio.comparison import INDICATOR_DIRECTIONS, Comparison, ComparisonData, Ranking
from vamos.ux.studio.comparison_stats import CORRECTIONS

MARK_TEXT = {"+": "better than", "-": "worse than", "=": "not significantly different from"}


def ordered_algorithms(data: ComparisonData, pivot: str) -> list[str]:
    """Contenders first and the pivot last, as in the usual comparison tables."""
    return [name for name in data.algorithms if name != pivot] + ([pivot] if pivot in data.algorithms else [])


@dataclass(frozen=True)
class TableRow:
    """The median, IQR and mark of every algorithm on one problem, with the two best medians."""

    problem: str
    cells: dict[str, tuple[float, float, str]]
    best: float | None
    second: float | None


def table_cells(data: ComparisonData, indicator: str, comparisons: Sequence[Comparison], pivot: str) -> list[TableRow]:
    """One row per problem: the median, IQR and mark of every algorithm."""
    marks = {(item.problem, item.contender): item.verdict for item in comparisons}
    higher = INDICATOR_DIRECTIONS[indicator]
    rows: list[TableRow] = []
    for problem in data.problems:
        cells: dict[str, tuple[float, float, str]] = {}
        for algorithm in ordered_algorithms(data, pivot):
            sample = data.sample(indicator, problem, algorithm)
            if sample.size:
                q25, q50, q75 = np.percentile(sample, [25, 50, 75])
                cells[algorithm] = (float(q50), float(q75 - q25), marks.get((problem, algorithm), ""))
        medians = sorted({value[0] for value in cells.values()}, reverse=higher)
        rows.append(TableRow(problem, cells, medians[0] if medians else None, medians[1] if len(medians) > 1 else None))
    return rows


def comparison_rows(comparisons: Sequence[Comparison]) -> list[dict[str, object]]:
    return [
        {
            "Problem": item.problem,
            "Algorithm": item.contender,
            "Pivot": item.pivot,
            "Median": item.median,
            "Pivot median": item.pivot_median,
            "p-value": item.p_value,
            "Adjusted p-value": item.p_adjusted,
            "A12": item.effect,
            "Magnitude": item.magnitude,
            "Verdict": item.verdict,
        }
        for item in comparisons
    ]


def comparison_html(data: ComparisonData, indicator: str, comparisons: Sequence[Comparison], pivot: str) -> str:
    """The median (IQR) table, shading the best and second-best median of every problem."""
    algorithms = ordered_algorithms(data, pivot)
    head = "".join(f"<th>{html.escape(name)}{' (pivot)' if name == pivot else ''}</th>" for name in algorithms)
    body = []
    for row in table_cells(data, indicator, comparisons, pivot):
        cells = []
        for algorithm in algorithms:
            value = row.cells.get(algorithm)
            if value is None:
                cells.append("<td>—</td>")
                continue
            median, iqr, mark = value
            shade = "#b7e1cd" if median == row.best else "#e3f2ea" if median == row.second else "transparent"
            cells.append(
                f'<td style="background:{shade};text-align:right">{median:.4g} <small>({iqr:.2g})</small> <b>{html.escape(mark)}</b></td>'
            )
        body.append(f"<tr><th style='text-align:left'>{html.escape(row.problem)}</th>{''.join(cells)}</tr>")
    return (
        "<table style='border-collapse:collapse;width:100%' border='1' cellpadding='6'>"
        f"<thead><tr><th>Problem</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def comparison_latex(
    data: ComparisonData, indicator: str, comparisons: Sequence[Comparison], pivot: str, *, correction: str, alpha: float
) -> str:
    """A booktabs table of medians with IQR subscripts and marks against the pivot."""
    algorithms = ordered_algorithms(data, pivot)
    direction = "higher" if INDICATOR_DIRECTIONS[indicator] else "lower"
    caption = (
        f"Median and interquartile range of {indicator} ({direction} is better). Marks compare each algorithm with "
        f"{pivot} by a two-sided Mann-Whitney test at $\\alpha={alpha:g}$ with {CORRECTIONS.get(correction, correction)} "
        "correction and the Vargha-Delaney A12 effect size: $+$ better, $-$ worse, $=$ no significant difference. "
        "Bold marks the best median of each problem and underline the second best."
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:vamos-comparison}",
        r"\begin{tabular}{l" + "r" * len(algorithms) + "}",
        r"\toprule",
        "Problem & " + " & ".join(latex_escape(name) for name in algorithms) + r" \\",
        r"\midrule",
    ]
    marks = {"+": "$+$", "-": "$-$", "=": "$=$", "": ""}
    for row in table_cells(data, indicator, comparisons, pivot):
        cells = []
        for algorithm in algorithms:
            value = row.cells.get(algorithm)
            if value is None:
                cells.append("--")
                continue
            median, iqr, mark = value
            text = f"{median:.4f}$_{{{iqr:.2e}}}$"
            text = f"\\textbf{{{text}}}" if median == row.best else f"\\underline{{{text}}}" if median == row.second else text
            cells.append(f"{text} {marks[mark]}".strip())
        lines.append(latex_escape(row.problem) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def ranking_latex(ranking: Ranking, pivot: str, indicator: str) -> str:
    friedman = f"Friedman $p={ranking.friedman[1]:.3g}$" if ranking.friedman else "Friedman test not applicable"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Average ranks of the median "
        + latex_escape(indicator)
        + f" over {ranking.problems} problems (1 is best); {friedman}; Holm post-hoc against {latex_escape(pivot)}."
        + "}",
        r"\label{tab:vamos-ranking}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Algorithm & Average rank & $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    order = np.argsort(ranking.ranks)
    for index in order:
        name = ranking.algorithms[index]
        z_p_adj = ranking.against_pivot[index] if ranking.against_pivot else (0.0, 1.0, 1.0)
        p_text = "--" if name == pivot else f"{z_p_adj[1]:.3g}"
        adj_text = "--" if name == pivot else f"{z_p_adj[2]:.3g}"
        lines.append(f"{latex_escape(name)} & {ranking.ranks[index]:.2f} & {p_text} & {adj_text} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def saes_csv(data: ComparisonData) -> str:
    """Long-format results as the SAES statistical tool reads them."""
    rows: list[dict[str, object]] = []
    for indicator, by_key in data.values.items():
        for (problem, algorithm), values in by_key.items():
            for execution, value in enumerate(values):
                if np.isfinite(value):
                    rows.append(
                        {
                            "Algorithm": algorithm,
                            "Problem": problem,
                            "MetricName": indicator,
                            "ExecutionId": execution,
                            "MetricValue": float(value),
                        }
                    )
    return csv_text(rows)


def saes_metrics_csv(data: ComparisonData) -> str:
    return csv_text([{"MetricName": name, "Maximize": INDICATOR_DIRECTIONS[name]} for name in data.indicators])


__all__ = [
    "MARK_TEXT",
    "TableRow",
    "comparison_html",
    "comparison_latex",
    "comparison_rows",
    "ordered_algorithms",
    "ranking_latex",
    "saes_csv",
    "saes_metrics_csv",
    "table_cells",
]
