"""Run directories of a CLI series, checked before its first run."""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Sequence

from vamos.engine.algorithm.registry import get_algorithms_registry
from vamos.experiment.artifacts.errors import OutputCollisionError
from vamos.experiment.execution import planned_engines_and_algorithms
from vamos.experiment.runner_utils import run_output_dir
from vamos.experiment.runtime.catalog import resolve_engine
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import ProblemSelection


def planned_run_dirs(args: Namespace, problem_selection: ProblemSelection, config: ExperimentConfig) -> list[str]:
    """The run directories that ``execute_problem_suite`` writes for one problem, in run order.

    External baselines write no run artifact, so they have none.
    """
    engines, algorithms = planned_engines_and_algorithms(args)
    registered = set(get_algorithms_registry().keys())
    return [
        run_output_dir(problem_selection, name, resolve_engine(engine, algorithm=name), config.seed, config)
        for engine in engines
        for name in algorithms
        if name in registered
    ]


def check_run_destinations(paths: Sequence[str]) -> None:
    """Reject, before any run starts, a run directory that exists or that two runs of the series share.

    Raises
    ------
    OutputCollisionError
        For the first such directory; the CLI exits with code 8, as ``vamos reproduce`` does.
    """
    seen: set[str] = set()
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen or os.path.lexists(path):
            raise OutputCollisionError(
                operation="run the experiment",
                path=path,
                reason="is the destination of two runs of this command" if key in seen else "already exists",
                expected="a run directory that does not exist",
                actual="occupied output path",
                action="Pass another --seed (seed in a config file) or --output-root; VAMOS never overwrites run artifacts.",
            )
        seen.add(key)


__all__ = ["check_run_destinations", "planned_run_dirs"]
