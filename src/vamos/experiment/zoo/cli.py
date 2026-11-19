from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vamos.experiment.runner import run_experiment
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem import modact_fronts
from vamos.foundation.problem.registry_info import get_problem_info, list_problems
from vamos.resources import user_reference_fronts_dir


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


def _list_cmd(args: argparse.Namespace) -> None:
    infos = list_problems()
    for info in sorted(infos, key=lambda x: x.name):
        cats = ",".join(info.categories)
        _logger().info(
            "%-15s | %-12s | n_var=%s n_obj=%s | %s",
            info.name,
            cats,
            info.default_n_variables,
            info.default_n_objectives,
            info.description,
        )


def _info_cmd(args: argparse.Namespace) -> int:
    info = get_problem_info(args.name)
    if info is None:
        _logger().warning("Problem '%s' not found.", args.name)
        return 1
    _logger().info("Name: %s", info.name)
    _logger().info("Description: %s", info.description)
    _logger().info("Categories: %s", ", ".join(info.categories))
    _logger().info(
        "Defaults: n_var=%s, n_obj=%s",
        info.default_n_variables,
        info.default_n_objectives,
    )
    _logger().info("Tags: %s", ", ".join(info.tags))
    return 0


def _run_cmd(args: argparse.Namespace) -> None:
    config = ExperimentConfig(
        output_root=args.output,
        population_size=args.pop_size,
        max_evaluations=args.budget,
        seed=args.seed,
    )
    metrics = run_experiment(
        algorithm=args.algorithm,
        problem=args.problem,
        engine=args.engine,
        config=config,
        selection_pressure=2,
    )
    # The run's metrics carry no hypervolume: it needs a reference point, which zoo does not choose.
    _logger().info("Run finished. Output: %s", metrics.get("output_dir"))


def _fetch_fronts_cmd(args: argparse.Namespace) -> int:
    """Download the requested MODAct fronts; exit code 1 for an unknown or packaged problem, or a failed download."""
    if args.all:
        keys = [f"modact_{name}" for name in modact_fronts.FETCHABLE]
    else:
        keys = args.problems or [f"modact_{name}" for name in modact_fronts.DEFAULT_FETCH]
    directory = Path(args.output) if args.output else user_reference_fronts_dir()
    try:
        modact_fronts.fetch_fronts(keys, directory, force=args.force, progress=_logger().info)
    except ValueError as exc:
        _logger().error("%s", exc)
        return 1
    except OSError as exc:
        _logger().error("Could not download the fronts from %s: %s", modact_fronts.ZENODO_DOI, exc)
        return 1
    _logger().info("The fronts and their CC BY 4.0 notice, %s, are in %s.", modact_fronts.NOTICE_FILE, directory)
    if args.output and directory.resolve() != user_reference_fronts_dir().resolve():
        _logger().info(
            "VAMOS looks for downloaded fronts in %s, not here: pass a front with --hv-reference-front PATH.", user_reference_fronts_dir()
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VAMOS problem zoo browser.")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List available problems")

    info_p = sub.add_parser("info", help="Show detailed info for a problem")
    info_p.add_argument("name")

    run_p = sub.add_parser("run", help="Quick run a problem with defaults")
    run_p.add_argument("problem")
    run_p.add_argument("--algorithm", default="nsgaii")
    run_p.add_argument("--engine", default=None)
    run_p.add_argument("--budget", type=int, default=5000)
    run_p.add_argument("--pop-size", type=int, default=50)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--output", default="zoo_runs")

    fetchable = ", ".join(f"modact_{name}" for name in modact_fronts.FETCHABLE)
    fetch_p = sub.add_parser(
        "fetch-fronts",
        help="Download the MODAct reference fronts that VAMOS does not package",
        description=(
            "Download Picard and Schiffmann's best-known fronts of MODAct problems from Zenodo "
            f"({modact_fronts.ZENODO_DOI}, version {modact_fronts.ZENODO_VERSION}, CC BY 4.0), check their MD5 "
            "checksums, convert them to VAMOS's signs, and write them, with their attribution, in the directory "
            "where VAMOS looks for downloaded reference fronts."
        ),
    )
    fetch_p.add_argument(
        "problems",
        nargs="*",
        metavar="PROBLEM",
        help=f"problem keys among {fetchable} (default: modact_cts1 to modact_cts4)",
    )
    fetch_p.add_argument("--all", action="store_true", help="download the twelve fronts, about 730 MB")
    fetch_p.add_argument("--force", action="store_true", help="download a front again even if it is there")
    fetch_p.add_argument("--output", default=None, help="write here instead of VAMOS's directory of downloaded fronts")
    return p


def main(argv: list[str] | None = None) -> int:
    _configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd = "list"
    if args.cmd == "list":
        _list_cmd(args)
    elif args.cmd == "info":
        return _info_cmd(args)
    elif args.cmd == "run":
        _run_cmd(args)
    elif args.cmd == "fetch-fronts":
        return _fetch_fronts_cmd(args)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
