"""irace handling for ``vamos tune``: options, availability, smoke profile and work directory."""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vamos.engine.tuning import available_model_based_backends

# irace needs dozens of experiments even for one iteration (36-48 for the built-in spaces
# on one instance with two seeds), so the smoke profile keeps one iteration and this budget.
SMOKE_TUNE_BUDGET = 64


def add_irace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--irace-rscript", type=str, default="Rscript", help="Rscript executable for --backend irace.")
    parser.add_argument(
        "--irace-instance-seeds",
        type=str,
        default="task",
        choices=["task", "irace"],
        help="'task': each training instance/seed pair is a deterministic irace instance; 'irace': irace draws the seeds.",
    )
    parser.add_argument(
        "--irace-work-dir",
        type=str,
        default="",
        help="Directory for the irace scenario, log and irace.Rdata (default: <output>/irace).",
    )
    parser.add_argument(
        "--irace-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra irace scenario option, e.g. nbIterations=3 (repeatable).",
    )


def _option_value(text: str) -> bool | int | float | str:
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text  # not numeric: irace receives it as a string


def validate_irace_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Parse --irace-option into ``args.irace_options`` and reject settings irace cannot honor."""
    options: dict[str, Any] = {}
    for item in args.irace_option:
        key, sep, value = str(item).partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", key) or key in options:
            parser.error(f"--irace-option expects distinct KEY=VALUE pairs with an irace option name, got {item!r}.")
        options[key] = _option_value(value)
    args.irace_options = options
    if str(args.backend) != "irace":
        return
    if float(args.timeout_seconds) > 0.0:
        parser.error("--backend irace has no wall-clock limit; bound it with --tune-budget (irace experiments).")
    if args.fidelity_levels:
        parser.error("--backend irace evaluates one budget per run; omit --fidelity-levels.")


def backend_availability(args: argparse.Namespace, base: Callable[[], dict[str, bool]] = available_model_based_backends) -> dict[str, bool]:
    """Optional backend flags from ``base``, with irace judged by the configured Rscript."""
    flags = dict(base())
    flags["irace"] = shutil.which(str(getattr(args, "irace_rscript", "Rscript"))) is not None
    return flags


def print_backend_table(model_backends: Sequence[str], flags: dict[str, bool]) -> None:
    print("Backend availability:")
    print("  racing       : True")
    print("  random       : True")
    for name in model_backends:
        print(f"  {name:12s}: {bool(flags.get(name, False))}")


def smoke_tune_budget(args: argparse.Namespace, clamped: int) -> int:
    """Smoke budget for the requested backend; irace keeps one iteration with enough experiments."""
    if str(args.backend) != "irace":
        return clamped
    args.irace_options.setdefault("nbIterations", 1)
    return SMOKE_TUNE_BUDGET


def resolve_irace_work_dir(args: argparse.Namespace, out_dir: Path) -> None:
    if str(args.backend) == "irace" and not str(args.irace_work_dir).strip():
        args.irace_work_dir = str(out_dir / "irace")


__all__ = [
    "SMOKE_TUNE_BUDGET",
    "add_irace_arguments",
    "backend_availability",
    "print_backend_table",
    "resolve_irace_work_dir",
    "smoke_tune_budget",
    "validate_irace_arguments",
]
