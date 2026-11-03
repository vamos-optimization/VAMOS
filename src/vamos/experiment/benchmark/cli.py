from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

from vamos.experiment.benchmark.report import BenchmarkReport, BenchmarkReportConfig
from vamos.experiment.benchmark.runner import (
    load_benchmark_result,
    run_benchmark_suite,
    validate_config_overrides,
    write_benchmark_summary,
)
from vamos.experiment.benchmark.suites import BenchmarkSuite, get_benchmark_suite
from vamos.experiment.study.errors import InvalidStudySpecError, StudyOutputCollisionError, UnresolvedStudyTaskError

_SMOKE_POP_SIZE = 8
_SMOKE_BUDGET = 24
_SMOKE_SEED = 0
# Exit codes: 2 for a usage or configuration error, 5 for an occupied output path (as `vamos study create`).
_USAGE_EXIT = 2
_COLLISION_EXIT = 5


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _configure_cli_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file '{cfg_path}' not found.")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("PyYAML is required to read YAML configs.") from exc
        with cfg_path.open("r", encoding="utf-8") as fh:
            try:
                data = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                raise ValueError(f"Config file '{cfg_path}' is not valid YAML: {exc}") from exc
    else:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{cfg_path}' must hold a mapping of overrides, not {type(data).__name__}.")
    return data


def _apply_smoke_mode(suite: BenchmarkSuite, overrides: dict[str, Any]) -> tuple[BenchmarkSuite, dict[str, Any]]:
    if not suite.experiments:
        return suite, dict(overrides)
    smoke_overrides = dict(overrides)
    smoke_overrides.setdefault("population_size", _SMOKE_POP_SIZE)
    smoke_overrides.setdefault("offspring_population_size", smoke_overrides["population_size"])
    smoke_budget = max(int(smoke_overrides["population_size"]) * 2, _SMOKE_BUDGET)
    smoke_experiment = replace(
        suite.experiments[0],
        evaluation_budget=smoke_budget,
        max_generations=None,
        seeds=[_SMOKE_SEED],
    )
    smoke_suite = replace(
        suite,
        experiments=[smoke_experiment],
        default_seeds=[_SMOKE_SEED],
        description=f"{suite.description} [smoke mode]",
    )
    return smoke_suite, smoke_overrides


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vamos bench", description="Run benchmark suites and generate paper-ready reports.")
    parser.add_argument("suite", help="Benchmark suite name (see list with --list).", nargs="?")
    parser.add_argument("--list", action="store_true", help="List available benchmark suites.")
    parser.add_argument("--algorithms", nargs="+", help="Algorithms to run (default: suite defaults).")
    parser.add_argument("--metrics", nargs="+", help="Metrics to compute (default: suite defaults).")
    parser.add_argument("--output", required=False, help="Output directory for reports and runs.")
    parser.add_argument(
        "--config",
        help=(
            "JSON/YAML mapping of overrides: population_size, offspring_population_size, max_evaluations, seeds, "
            "engine and eval_strategy. Other keys are rejected."
        ),
    )
    parser.add_argument(
        "--only-report",
        action="store_true",
        help="Skip execution and regenerate summary/metrics.csv and the reports from existing outputs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny benchmark slice (first experiment, one seed, small population/budget) for fast verification.",
    )
    return parser


def _fail(parser: argparse.ArgumentParser, code: int, message: str) -> NoReturn:
    parser.exit(code, f"{parser.prog}: error: {message}\n")


def main(argv: list[str] | None = None) -> int:
    _configure_cli_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    from vamos.experiment.benchmark.suites import list_benchmark_suites

    if args.list or not args.suite:
        _logger().info("Available benchmark suites: %s", ", ".join(list_benchmark_suites()))
        return 0

    try:
        suite = get_benchmark_suite(args.suite)
    except KeyError as exc:
        parser.error(str(exc.args[0]) if exc.args else f"Unknown benchmark suite '{args.suite}'.")
    algorithms = args.algorithms or suite.default_algorithms
    metrics = args.metrics or suite.default_metrics
    if not args.output:
        parser.error("--output is required unless --list is used.")
    base_output_dir = Path(args.output).expanduser().resolve()
    try:
        overrides = validate_config_overrides(_load_config(args.config))
    except (OSError, ValueError) as exc:
        parser.error(f"--config: {exc}")
    if args.smoke:
        suite, overrides = _apply_smoke_mode(suite, overrides)
        _logger().info(
            "[Benchmark] Smoke mode enabled: suite=%s, experiment=%s, seeds=%s, population_size=%s.",
            suite.name,
            suite.experiments[0].problem_name if suite.experiments else "-",
            overrides.get("seeds", suite.experiments[0].seeds if suite.experiments else []),
            overrides.get("population_size"),
        )

    if args.only_report:
        result = load_benchmark_result(
            suite=suite,
            algorithms=algorithms,
            metrics=metrics,
            base_output_dir=base_output_dir,
        )
        if not result.studies:
            _fail(parser, _USAGE_EXIT, f"No benchmark studies of suite '{suite.name}' under {base_output_dir}; run the suite first.")
        write_benchmark_summary(result)
    else:
        try:
            result = run_benchmark_suite(
                suite=suite,
                algorithms=algorithms,
                metrics=metrics,
                base_output_dir=base_output_dir,
                global_config_overrides=overrides,
            )
        except StudyOutputCollisionError as exc:
            _fail(parser, _COLLISION_EXIT, str(exc))
        except (InvalidStudySpecError, UnresolvedStudyTaskError) as exc:
            _fail(parser, _USAGE_EXIT, str(exc))

    report_output = base_output_dir / "summary"
    report = BenchmarkReport(
        result=result,
        config=BenchmarkReportConfig(metrics=list(metrics)),
        output_dir=report_output,
    )
    _ = report.aggregate_metrics()  # Writes tidy CSV
    _ = report.compute_statistics()  # Populates internal cache
    tables = report.generate_latex_tables()
    plots = report.generate_plots()
    lab_outputs = report.generate_lab_outputs()

    _logger().info("[Benchmark] Suite '%s' completed.", suite.name)
    _logger().info("[Benchmark] Summary CSV: %s", result.summary_path)
    _logger().info("[Benchmark] Tidy metrics: %s", report_output / "metrics_tidy.csv")
    if tables:
        _logger().info("[Benchmark] LaTeX tables in %s", report_output / "tables")
    if plots:
        _logger().info("[Benchmark] Plots in %s", report_output / "plots")
    if lab_outputs:
        _logger().info("[Benchmark] Lab outputs in %s", report_output / "lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
