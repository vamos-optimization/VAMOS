"""Files, run destinations and the closing summary of ``vamos quickstart``."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from vamos.engine.config.spec import EXPERIMENT_SPEC_VERSION
from vamos.experiment.runner_utils import run_output_dir
from vamos.experiment.runtime.catalog import resolve_engine
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.io_utils import ensure_dir
from vamos.foundation.problem.registry import available_problem_names, make_problem_selection

from .quickstart_templates import QuickstartTemplate


def run_destination(problem: str, algorithm: str, engine: str, seed: int, *, output_root: str) -> str | None:
    """The run directory a quickstart run writes, or None when the problem is unknown (parsing reports it)."""
    if problem not in available_problem_names():
        return None
    selection = make_problem_selection(problem)
    return run_output_dir(
        selection, algorithm, resolve_engine(engine, algorithm=algorithm), seed, ExperimentConfig(output_root=output_root)
    )


def free_seed(destination: Callable[[int], str | None], start: int) -> int:
    """The first seed from ``start`` whose run directory does not exist yet."""
    seed = start
    while (path := destination(seed)) is not None and os.path.lexists(path):
        seed += 1
    return seed


def _write_spec(
    *,
    title: str,
    problem: str,
    algorithm: str,
    engine: str,
    budget: int,
    pop_size: int,
    seed: int,
    output_root: str,
    plot: bool,
    config_path: str | None,
) -> Path:
    spec: dict[str, object] = {
        "version": EXPERIMENT_SPEC_VERSION,
        "defaults": {
            "title": title,
            "problem": problem,
            "algorithm": algorithm,
            "engine": engine,
            "max_evaluations": budget,
            "population_size": pop_size,
            "seed": seed,
            "output_root": output_root,
            "plot": plot,
        },
    }
    return _write_spec_data(spec=spec, output_root=output_root, config_path=config_path)


def _write_spec_data(*, spec: Mapping[str, object], output_root: str, config_path: str | None) -> Path:
    ensure_dir(output_root)
    if config_path:
        path = Path(config_path)
        ensure_dir(path.parent)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(output_root) / f"quickstart_{timestamp}.json"
        # Two quickstarts within the same second keep both configs.
        suffix = 1
        while path.exists():
            suffix += 1
            path = Path(output_root) / f"quickstart_{timestamp}_{suffix}.json"
    path.write_text(json.dumps(dict(spec), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _latest_plot_path(problem_dir: str) -> str | None:
    paths = sorted(Path(problem_dir).glob("pareto_front_*.png"))
    if not paths:
        return None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _summary(
    *,
    output_dir: str,
    config_path: Path,
    problem_dir: str,
    plot_path: str | None,
    plot_note: str,
    template: QuickstartTemplate,
    seed: int,
    rerun_seed: int,
) -> None:
    print("\nQuickstart complete.")
    print(f"Template: {template.label}")
    print(f"Results folder: {output_dir}")
    print(f"Config saved: {config_path}")
    print(f"Run manifest: {Path(output_dir) / 'manifest.json'}")
    print(f"Plot: {plot_path or plot_note}")
    # The same seed would write the same run directory, and a run is never overwritten. The command
    # has its own line, so that it can be pasted into a shell.
    print(f"Seed {seed} now has a run; the command below uses a free seed.")
    print(f"Re-run: vamos --config {config_path} --seed {rerun_seed}")
    print(f"Browse outputs: {problem_dir}")


__all__ = ["free_seed", "run_destination"]
