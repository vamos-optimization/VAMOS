"""The top-level command line and ``vamos.optimize`` resolve the same run, except for known differences.

The two paths build a run separately: the command line through ``engine.algorithm.factory``, ``optimize``
through ``engine.algorithm.config.defaults``. Both are stable, so their defaults stay until 2.0, when one
execution path replaces them. Until then, this test lists every difference between the resolved
specifications the two paths record for a default run of each built-in algorithm, and fails on any other.
``docs/reference/algorithms.md`` describes them. Nothing runs: the test stops each path once its
configuration is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import vamos
from vamos.experiment import unified
from vamos.experiment.artifacts.specs import RunSpecInputs, build_run_specs
from vamos.experiment.cli.main import _run_standard_cli
from vamos.experiment.services import orchestrator
from vamos.foundation.problem.registry import make_problem_selection

BUDGET = 600
SEED = 3
# A population each lattice-based algorithm accepts for the problem's objectives.
POPULATIONS = {"zdt1": 40, "dtlz2": 91}
ALGORITHMS = sorted(vamos.algorithms.available_algorithms())

# A value that depends on the problem, such as a lattice's divisions, or on the installation.
VARIES = "varies"
ABSENT = "absent"
_RESULT_MODE = {"algorithm.config.result_mode": ("non_dominated", "population")}

# Key of the resolved specification -> (command line, optimize).
KNOWN_DIFFERENCES: dict[str, dict[str, tuple[object, object]]] = {
    **{name: dict(_RESULT_MODE) for name in ("agemoea", "ccmo", "cmoeams", "ctaea", "ibea", "rvea", "spea2", "top")},
    # The command line records the offspring size, which optimize leaves to default to pop_size.
    "nsgaii": {"algorithm.config.offspring_size": (VARIES, None)},
    "smpso": {},
    # SMS-EMOA's command line leaves out the reference point vector that optimize records as null.
    "smsemoa": {**_RESULT_MODE, "algorithm.config.reference_point.vector": (ABSENT, None)},
    # The command line reads the packaged weight files and records their directory, a path of the
    # installation; optimize records the divisions of the lattice.
    "moead": {
        **_RESULT_MODE,
        "algorithm.config.weight_vectors.divisions": (None, VARIES),
        "algorithm.config.weight_vectors.path": (VARIES, None),
    },
    # The command line derives the reference directions from the population when the run starts;
    # optimize records the lattice.
    "nsgaiii": {
        "algorithm.config.pop_size_auto": (False, True),
        "algorithm.config.reference_directions.divisions": (ABSENT, VARIES),
        "algorithm.config.reference_directions.path": (ABSENT, None),
    },
}
# Differences of one problem only: (algorithm, problem) -> key -> (command line, optimize). With three
# objectives MOEA/D runs a packaged jMetal file on the command line and the lattice with optimize, which
# the resolved weight vectors record; with two objectives both run uniform weights.
PROBLEM_DIFFERENCES: dict[tuple[str, str], dict[str, tuple[object, object]]] = {
    ("moead", "dtlz2"): {
        # Development after 1.4.0: the command line records the packaged file it read and the digest of its vectors.
        "algorithm.config.weight_vectors.sha256": (VARIES, ABSENT),
        "algorithm.resolution.weight_vectors.source": ("file", "das_dennis"),
        "algorithm.resolution.weight_vectors.file": (VARIES, ABSENT),
        "algorithm.resolution.weight_vectors.divisions": (ABSENT, VARIES),
    },
}


class _Built(Exception):
    """Raised by the patched runners once the configuration is built, so nothing runs."""

    def __init__(self, config: Any) -> None:
        super().__init__("configuration built")
        self.config = config


def _cli_config(problem: str, algorithm: str, output: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    def built(*_args: Any, cfg_data: Any, **_kwargs: Any) -> None:
        raise _Built(cfg_data)

    monkeypatch.setattr(orchestrator, "execute_run_single", built)
    argv = [
        f"--problem={problem}",
        f"--algorithm={algorithm}",
        f"--population-size={POPULATIONS[problem]}",
        f"--max-evaluations={BUDGET}",
        f"--seed={SEED}",
        "--engine=numpy",
        f"--output-root={output}",
    ]
    with pytest.raises(_Built) as captured:
        _run_standard_cli(argv)
    return dict(captured.value.config.to_dict())


def _api_config(problem: str, algorithm: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    def built(config: Any, *_args: Any, **_kwargs: Any) -> None:
        raise _Built(config.algorithm_config)

    monkeypatch.setattr(unified, "_run_config", built)
    with pytest.raises(_Built) as captured:
        vamos.optimize(problem, algorithm=algorithm, max_evaluations=BUDGET, pop_size=POPULATIONS[problem], seed=SEED, engine="numpy")
    return dict(captured.value.config.to_dict())


def _resolved_spec(problem: str, algorithm: str, config: Mapping[str, object]) -> dict[str, Any]:
    """The resolved specification of a run with this configuration; every other fact is the same for both paths."""
    selection = make_problem_selection(problem)
    _, resolved = build_run_specs(
        RunSpecInputs(
            problem_built_in=True,
            problem_label=problem,
            problem_kwargs=None,
            n_var_requested=None,
            n_obj_requested=None,
            n_var=selection.n_var,
            n_obj=selection.n_obj,
            encoding="real",
            algorithm_requested=algorithm,
            algorithm=algorithm,
            algorithm_config=config,
            algorithm_config_explicit=False,
            max_evaluations_requested=BUDGET,
            termination=("max_evaluations", BUDGET),
            pop_size_requested=POPULATIONS[problem],
            resolved_pop_size=POPULATIONS[problem],
            engine_requested="numpy",
            engine="numpy",
            eval_strategy="serial",
            seed_requested=SEED,
            seed=SEED,
            default_sources={},
        )
    )
    return resolved


def _flat(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        flat: dict[str, object] = {}
        for key, item in value.items():
            flat.update(_flat(item, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: value}


def _differences(cli: Mapping[str, object], api: Mapping[str, object]) -> dict[str, tuple[object, object]]:
    return {
        key: (cli.get(key, ABSENT), api.get(key, ABSENT))
        for key in sorted(set(cli) | set(api))
        if cli.get(key, ABSENT) != api.get(key, ABSENT)
    }


def test_every_built_in_algorithm_has_an_entry() -> None:
    assert set(KNOWN_DIFFERENCES) == set(ALGORITHMS)


@pytest.mark.parametrize("problem", sorted(POPULATIONS))
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_cli_and_optimize_resolve_the_same_config(problem: str, algorithm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _flat(_resolved_spec(problem, algorithm, _cli_config(problem, algorithm, tmp_path, monkeypatch)))
    api = _flat(_resolved_spec(problem, algorithm, _api_config(problem, algorithm, monkeypatch)))

    differences = _differences(cli, api)
    known = {**KNOWN_DIFFERENCES[algorithm], **PROBLEM_DIFFERENCES.get((algorithm, problem), {})}
    assert set(differences) == set(known), f"new or resolved differences: {differences}"
    for key, (cli_value, api_value) in differences.items():
        expected_cli, expected_api = known[key]
        assert expected_cli in (VARIES, cli_value), (key, cli_value)
        assert expected_api in (VARIES, api_value), (key, api_value)
