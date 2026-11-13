"""
Lightweight self-check to verify a VAMOS installation.

Runs tiny NSGA-II jobs on ZDT1 across available backends and reports results.
Intended for quick sanity checks; not a benchmark.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from vamos.experiment.runner import run_single
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection


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


@dataclass
class CheckResult:
    name: str
    status: str  # "ok", "skipped", "failed"
    detail: str | None = None


@contextmanager
def _quiet_runs() -> Iterator[None]:
    """Silence the per-run summaries of the observers, so that each check prints one line."""
    observers = logging.getLogger("vamos.experiment.observers")
    level = observers.level
    observers.setLevel(logging.WARNING)
    try:
        yield
    finally:
        observers.setLevel(level)


def _report(result: CheckResult) -> None:
    detail = f" {result.detail}" if result.detail else ""
    _logger().info("[self-check] %s: %s%s", result.name, result.status.upper(), detail)


def _run_backend_check(
    engine: str,
    *,
    output_root: str,
    pop_size: int = 6,
    max_eval: int = 20,
) -> CheckResult:
    selection = make_problem_selection("zdt1", n_var=6)
    cfg = ExperimentConfig(
        population_size=pop_size,
        offspring_population_size=pop_size,
        max_evaluations=max_eval,
        seed=1,
        output_root=output_root,
    )
    try:
        run_single(engine, "nsgaii", selection, cfg, selection_pressure=2)
        return CheckResult(name=f"nsgaii-{engine}", status="ok")
    except ImportError as exc:
        return CheckResult(name=f"nsgaii-{engine}", status="skipped", detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(name=f"nsgaii-{engine}", status="failed", detail=str(exc))


def run_self_check(verbose: bool = False) -> list[CheckResult]:
    """Run a minimal set of smoke checks for each compute backend.

    Always exercises NumPy; Numba and MooCore are optional and reported
    as ``"skipped"`` when the dependency is missing.

    Experimental: a development diagnostic exported by ``vamos`` and
    ``vamos.api``, not covered by the 1.x stability policy.

    Parameters
    ----------
    verbose : bool, default False
        If ``True``, log the result of each check at INFO level.

    Returns
    -------
    list[CheckResult]
        One entry per backend / encoding variant with ``status`` in
        ``{"ok", "skipped", "failed"}``.

    Raises
    ------
    RuntimeError
        If the mandatory NumPy backend check fails.
    """
    checks: list[CheckResult] = []
    with tempfile.TemporaryDirectory(prefix="vamos-self-check-") as output_root, _quiet_runs():
        for engine in ("numpy", "numba", "moocore"):
            result = _run_backend_check(engine, output_root=output_root)
            checks.append(result)
            if verbose:
                _report(result)

        # Binary and mixed smoke on NumPy only
        for name, label in (("bin_knapsack", "binary"), ("mixed_design", "mixed")):
            selection = make_problem_selection(name)
            cfg = ExperimentConfig(
                population_size=8,
                offspring_population_size=8,
                max_evaluations=20,
                seed=2,
                output_root=output_root,
            )
            try:
                run_single("numpy", "nsgaii", selection, cfg, selection_pressure=2)
                result = CheckResult(name=f"{name}", status="ok")
            except Exception as exc:  # pragma: no cover - quick smoke only
                result = CheckResult(name=f"{name}", status="failed", detail=str(exc))
            checks.append(result)
            if verbose:
                _report(result)

    numpy_ok = next((c for c in checks if c.name == "nsgaii-numpy"), None)
    if numpy_ok is None or numpy_ok.status != "ok":
        detail = numpy_ok.detail if numpy_ok else "NumPy check missing"
        raise RuntimeError(f"VAMOS self-check failed for NumPy backend: {detail}")
    return checks


def main(argv: list[str] | None = None) -> int:
    """Entry-point for `python -m vamos.experiment.diagnostics.self_check`."""
    parser = argparse.ArgumentParser(description="Run VAMOS installation and backend smoke checks.")
    parser.add_argument("--quiet", action="store_true", help="Only use the exit code; suppress informational output.")
    args = parser.parse_args(argv)
    _configure_cli_logging()
    try:
        checks = run_self_check(verbose=not args.quiet)
    except RuntimeError as exc:
        # The NumPy check failed; its line above says why.
        _logger().error("[self-check] %s", exc)
        return 1
    failed = [c for c in checks if c.status == "failed"]
    skipped = [c for c in checks if c.status == "skipped"]
    if failed:
        return 1
    if skipped and not args.quiet:
        _logger().info("[self-check] Skipped: %s", ", ".join(c.name for c in skipped))
    if not args.quiet:
        _logger().info("[self-check] Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
