import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from vamos.experiment.benchmark.report import BenchmarkReport, BenchmarkReportConfig
from vamos.experiment.benchmark.runner import load_benchmark_result, run_benchmark_suite
from vamos.experiment.benchmark.suites import BenchmarkExperiment, BenchmarkSuite, get_benchmark_suite, list_benchmark_suites


def test_suite_registry_contains_defaults():
    names = list_benchmark_suites()
    assert "ZDT_small" in names
    assert "CEC2009_UF_CF_curved" in names
    assert "LSMOP_large" in names
    assert "Constrained_CDTLZ_MW_DCDTLZ" in names
    suite = get_benchmark_suite("ZDT_small")
    assert suite.experiments
    assert suite.default_algorithms


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_report_pipeline_on_fake_csv(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    suite = get_benchmark_suite("ZDT_small")
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for problem in ["zdt1", "zdt2"]:
        for alg in ["a1", "a2"]:
            for seed in [0, 1]:
                rows.append(
                    {
                        "problem": problem,
                        "algorithm": alg,
                        "engine": "numpy",
                        "seed": seed,
                        "n_var": 30,
                        "n_obj": 2,
                        "hv": 0.8 + 0.01 * rng.random(),
                        "indicator_igd_plus": 0.2 + 0.01 * rng.random(),
                    }
                )
    df = pd.DataFrame(rows)

    class _SummaryResult:
        def __init__(self):
            self.suite = suite

        def summary_rows(self):
            return tuple(df.to_dict(orient="records"))

    result = _SummaryResult()
    report = BenchmarkReport(
        result=result,
        config=BenchmarkReportConfig(metrics=["hv", "igd_plus"], alpha=0.1),
        output_dir=summary_dir,
    )

    tidy = report.aggregate_metrics()
    assert not tidy.empty
    assert set(tidy["metric"].unique()) == {"hv", "igd_plus"}

    stats = report.compute_statistics()
    assert "hv" in stats

    tables = report.generate_latex_tables()
    hv_table = tables.get("hv")
    assert hv_table is not None and hv_table.exists()
    content = hv_table.read_text(encoding="utf-8")
    assert "\\textbf" in content

    # Plots are optional; ensure the function does not raise.
    report.generate_plots()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_report_ranks_runs_without_feasible_solutions_last(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    rows = []
    for problem in ["mw2", "mw3"]:
        for algorithm, offset in [("a1", 0.1), ("a2", 0.2), ("never_feasible", None)]:
            for seed in range(5):
                rows.append(
                    {
                        "problem": problem,
                        "algorithm": algorithm,
                        "engine": "numpy",
                        "seed": seed,
                        "n_obj": 2,
                        "n_feasible": 20 if offset is not None else 0,
                        # derive_summary_rows gives a run without feasible solutions HV 0 and IGD+ +inf.
                        "hv": 1.0 - offset - 0.01 * seed if offset is not None else 0.0,
                        "indicator_igd_plus": offset + 0.01 * seed if offset is not None else math.inf,
                    }
                )

    class _SummaryResult:
        suite = get_benchmark_suite("Constrained_CDTLZ_MW_DCDTLZ")

        def summary_rows(self):
            return tuple(rows)

    report = BenchmarkReport(
        result=_SummaryResult(),
        config=BenchmarkReportConfig(metrics=["hv", "igd_plus"], alpha=0.05),
        output_dir=tmp_path / "summary",
    )

    stats = report.compute_statistics()
    for metric in ("hv", "igd_plus"):
        friedman = stats[metric]["friedman"]
        assert friedman is not None
        assert dict(zip(stats[metric]["pivot_mean"].columns, friedman.avg_ranks)) == {"a1": 1.0, "a2": 2.0, "never_feasible": 3.0}
    assert report.generate_latex_tables()
    report.generate_plots()
    lab = report.generate_lab_outputs()
    for indicator in ("HV", "IGD+"):
        table = pd.read_csv(lab["wilcoxon"] / f"Wilcoxon-{indicator}.csv", sep="\t", index_col=0)
        assert table.loc["a1", "never_feasible"] == "++"
        assert table.loc["a2", "never_feasible"] == "++"


def _igd_plus_report(output_dir: Path, values: dict[tuple[str, str], list[float]]) -> BenchmarkReport:
    rows = tuple(
        {"problem": problem, "algorithm": algorithm, "engine": "numpy", "seed": seed, "indicator_igd_plus": value}
        for (problem, algorithm), seed_values in values.items()
        for seed, value in enumerate(seed_values)
    )

    class _SummaryResult:
        suite = get_benchmark_suite("Constrained_CDTLZ_MW_DCDTLZ")

        def summary_rows(self):
            return rows

    return BenchmarkReport(
        result=_SummaryResult(),
        config=BenchmarkReportConfig(metrics=["igd_plus"], alpha=0.05),
        output_dir=output_dir,
    )


def _runtime_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(warning.message) for warning in caught if issubclass(warning.category, RuntimeWarning)]


def test_report_ties_infeasible_means_shared_by_two_algorithms(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")

    # Mean IGD+ of a1 minus a2 on p2..p10: a1 is better on eight problems and worse on one.
    gaps = [-0.09, -0.015, -0.034, -0.137, 0.133, -0.127, -0.083, -0.199, -0.196]

    def values(penalty: float) -> dict[tuple[str, str], list[float]]:
        # On p0 and p1, a1 and a2 never find a feasible solution: both means are the penalty.
        means: dict[tuple[str, str], float] = {}
        for index, gap in enumerate([None, None, *gaps]):
            a2 = 0.5 + 0.01 * index
            means[(f"p{index}", "a1")] = penalty if gap is None else a2 + gap
            means[(f"p{index}", "a2")] = penalty if gap is None else a2
            # Constant, so the penalty minus a3 is the same gap on p0 and p1, as +inf is.
            means[(f"p{index}", "a3")] = 0.8
        return {key: [mean + 0.001 * seed for seed in range(5)] for key, mean in means.items()}

    def mean_tests(report: BenchmarkReport) -> list[tuple[object, ...]]:
        return [(item.algo_i, item.algo_j, item.statistic, item.p_value) for item in report.compute_statistics()["igd_plus"]["wilcoxon"]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shared = mean_tests(_igd_plus_report(tmp_path / "inf", values(math.inf)))

    # Equal finite penalties, worse than every feasible value, are plain ties: with the
    # two tied problems a1 beats a2 (p = 0.035); dropping them would give p = 0.055.
    assert shared == mean_tests(_igd_plus_report(tmp_path / "tied", values(9.0)))
    assert all(math.isfinite(p_value) for *_, p_value in shared)
    assert not _runtime_warnings(caught)


def test_report_latex_marks_tie_seeds_where_no_algorithm_is_feasible(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    # a1 - a2 per seed: a1 is better in eight seeds and worse in one.
    gaps = [-0.09, -0.015, -0.034, -0.137, 0.133, -0.127, -0.083, -0.199, -0.196]

    def values(penalty: float) -> dict[tuple[str, str], list[float]]:
        # Neither algorithm finds a feasible solution in the first two seeds.
        a2 = [0.5 + 0.01 * seed for seed in range(len(gaps))]
        return {("p1", "a1"): [penalty, penalty, *(x + gap for x, gap in zip(a2, gaps))], ("p1", "a2"): [penalty, penalty, *a2]}

    def marks(report: BenchmarkReport) -> dict[str, list[str]]:
        table = report.generate_latex_tables()["igd_plus"].read_text(encoding="utf-8")
        return {line.split(" &")[0]: re.findall(r"\$\^\{(.)\}\$", line) for line in table.splitlines() if line.startswith("p1 &")}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shared = marks(_igd_plus_report(tmp_path / "inf", values(math.inf)))

    # Two tied seeds plus nine paired seeds make a2 significantly worse (p = 0.035);
    # dropping the tied seeds instead would leave p = 0.055 and no mark.
    assert shared == marks(_igd_plus_report(tmp_path / "tied", values(9.0))) == {"p1": ["-"]}
    assert not _runtime_warnings(caught)


def test_run_benchmark_suite_returns_traceable_canonical_studies(tmp_path: Path):
    suite = BenchmarkSuite(
        name="tiny",
        experiments=[BenchmarkExperiment("zdt1", {"n_var": 3, "n_obj": 2}, evaluation_budget=8, seeds=[3])],
        default_algorithms=["nsgaii"],
        default_metrics=[],
    )

    result = run_benchmark_suite(
        suite,
        algorithms=None,
        metrics=None,
        base_output_dir=tmp_path / "benchmark",
        global_config_overrides={"population_size": 4, "engine": "numpy"},
    )

    assert len(result.studies) == 1
    execution = result.studies[0]
    assert result.study_roots == (execution.study.root,)
    assert execution.study.status == "completed"
    assert execution.report.study_id == execution.summary.study_id == result.study_ids[0]
    rows = result.summary_rows()
    assert len(rows) == 1
    assert rows[0]["task_id"] == execution.summary.rows[0].task_id
    assert rows[0]["selected_run_id"] is not None
    assert rows[0]["run_manifest_sha256"] is not None
    assert isinstance(rows[0]["hv"], float)
    assert rows[0]["hv_reference"]
    assert result.summary_path is not None and result.summary_path.exists()

    before = {path.relative_to(result.base_output_dir): path.read_bytes() for path in result.base_output_dir.rglob("*") if path.is_file()}
    loaded = load_benchmark_result(
        suite,
        algorithms=result.algorithms,
        metrics=result.metrics,
        base_output_dir=result.base_output_dir,
    )
    after = {path.relative_to(result.base_output_dir): path.read_bytes() for path in result.base_output_dir.rglob("*") if path.is_file()}
    assert loaded.study_ids == result.study_ids
    assert before == after
