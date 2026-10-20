"""Settings that open Run Experiment: a problem from the catalog, or a stored run to run again."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vamos.ux.studio.algorithm_catalog import algorithm_label
from vamos.ux.studio.data import RunRecord
from vamos.ux.studio.environment import DISTRIBUTION, installed_version

_DEFAULT_SOURCES = {"auto", "default"}


@dataclass(frozen=True)
class ExperimentSetup:
    """What Run Experiment loads; a ``None`` field keeps the page's current value."""

    problem: str
    n_var: int | None = None
    n_obj: int | None = None
    algorithm: str | None = None
    pop_size: int | None = None
    max_evaluations: int | None = None
    seed: int | None = None
    source: str = ""
    notes: tuple[str, ...] = ()


def setup_from_run(run: RunRecord) -> ExperimentSetup:
    """The settings of a stored run, and what Run Experiment cannot carry over from it."""
    resolved = _mapping(run.metadata.get("resolved_spec"))
    problem = _mapping(resolved.get("problem"))
    problem_config = _mapping(problem.get("config"))
    algorithm = _mapping(resolved.get("algorithm"))
    algorithm_config = _mapping(algorithm.get("config"))
    population = _mapping(resolved.get("population"))
    termination = _mapping(resolved.get("termination"))
    algorithm_key = _component(algorithm.get("component_id"), "vamos.algorithm") or run.algorithm_name
    run_dir = str(run.root) if run.root is not None else "<run directory>"
    notes: list[str] = []

    if _algorithm_config_was_default(resolved):
        notes.append(f"{algorithm_label(algorithm_key)} runs with its VAMOS default configuration, as that run did.")
        stored_version = _mapping(_mapping(run.metadata.get("provenance")).get("implementation")).get("vamos_version")
        current_version = installed_version(DISTRIBUTION)
        if isinstance(stored_version, str) and current_version is not None and stored_version != current_version:
            notes.append(f"That run used VAMOS {stored_version} and this is {current_version}; their defaults may differ.")
    else:
        notes.append(
            "That run was given its own algorithm configuration, which Studio cannot load into the form, so this run "
            f"starts from the VAMOS default configuration. To repeat that run exactly, use `vamos reproduce {run_dir}`."
        )
    max_evaluations: int | None = None
    termination_name = _component(termination.get("component_id"), "vamos.termination")
    if termination_name == "max_evaluations":
        max_evaluations = _integer(_mapping(termination.get("config")).get("max_evaluations"))
    elif termination_name is not None:
        notes.append(f"That run stopped by `{termination_name}`; Run Experiment stops at an evaluation budget.")
    engine = _mapping(_mapping(_mapping(resolved.get("backend")).get("kernel")).get("resolution")).get("name")
    if isinstance(engine, str) and engine != "numpy":
        notes.append(f"That run used the {engine} engine; Run Experiment uses NumPy.")

    return ExperimentSetup(
        problem=_component(problem.get("component_id"), "vamos.problem") or run.problem_name,
        n_var=_integer(problem_config.get("n_var")),
        n_obj=_integer(problem_config.get("n_obj")),
        algorithm=algorithm_key,
        pop_size=_integer(algorithm_config.get("pop_size")) or _integer(population.get("initial_size")),
        max_evaluations=max_evaluations,
        seed=_integer(resolved.get("seed")),
        source=f"run {run.experiment_id[:8]} ({algorithm_label(algorithm_key)}, seed {run.seed})",
        notes=tuple(notes),
    )


def _algorithm_config_was_default(resolved: Mapping[str, Any]) -> bool:
    defaults = resolved.get("defaults_applied")
    if not isinstance(defaults, list):
        return False
    return any(
        isinstance(item, Mapping) and item.get("field") == "/algorithm_config" and item.get("source") in _DEFAULT_SOURCES
        for item in defaults
    )


def _component(component_id: object, namespace: str) -> str | None:
    prefix = f"{namespace}:"
    if not isinstance(component_id, str) or not component_id.startswith(prefix):
        return None
    return component_id[len(prefix) :].split("@", 1)[0]


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["ExperimentSetup", "setup_from_run"]
