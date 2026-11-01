"""
CLI for ``vamos profile``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def _float_list(text: str) -> list[float]:
    try:
        return [float(item) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``vamos profile`` CLI.

    Returns 0 when every engine ran and 1 when one failed; invalid options exit with 2 before any run.
    """
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(prog="vamos profile", description="Profile VAMOS optimization across different backends")
    parser.add_argument("--problem", "-p", default="zdt1", help="Problem to benchmark (default: zdt1)")
    parser.add_argument("--engines", "-e", default="numpy", help="Comma-separated list of engines (default: numpy)")
    parser.add_argument("--budget", "-b", type=_positive_int, default=2000, help="Evaluation budget (default: 2000)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--repeats",
        "-r",
        type=_positive_int,
        default=3,
        help="Timed runs per engine after one untimed warm-up run; the table reports their median (default: 3)",
    )
    parser.add_argument(
        "--hv-reference",
        type=_float_list,
        help="Comma-separated hypervolume reference point (default: the problem's reference front nadir plus 10%% of its range)",
    )
    parser.add_argument("--output", "-o", help="Output CSV file path (optional)")
    parser.add_argument("--no-hv", action="store_true", help="Skip hypervolume computation")

    args = parser.parse_args(argv)

    from vamos.experiment.profiler.runner import check_engines, run_profile
    from vamos.foundation.problem.registry import available_problem_names

    engines = [e.strip() for e in args.engines.split(",")]
    try:
        check_engines(engines)
    except ValueError as exc:
        parser.error(f"--engines: {exc}")
    if args.problem not in available_problem_names():
        parser.error(f"--problem: unknown problem '{args.problem}'.")

    try:
        report = run_profile(
            problem=args.problem,
            engines=engines,
            budget=args.budget,
            seed=args.seed,
            compute_hv=not args.no_hv,
            repeats=args.repeats,
            hv_reference=args.hv_reference,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Output
    report.print_summary()

    if args.output:
        report.to_csv(args.output)
        logger.info("Results saved to: %s", args.output)

    if report.failed:
        logger.error("Profiling failed for: %s", ", ".join(result.engine for result in report.failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
