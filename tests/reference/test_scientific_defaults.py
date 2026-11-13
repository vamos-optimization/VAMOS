"""The scientific default values of the built-in algorithms, pinned with their sources (VR-271).

Tests elsewhere pass these values explicitly (``kappa=0.05``, explicit ``divisions``), so a default
could drift without any test noticing; IBEA's kappa did (VR-013). Here the ``default()``
configuration of each built-in algorithm is pinned, and ``optimize``, the ``vamos`` CLI and a
study, none given an algorithm configuration, must resolve the same values.

Sources, for 30 variables, 2 objectives and 100 individuals:

- NSGA-II: Deb, Pratap, Agarwal and Meyarivan (2002), SBX p_c = 0.9, eta_c = 20, polynomial
  mutation p_m = 1/n, eta_m = 20, binary tournament.
- NSGA-III: Deb and Jain (2014), eta_c = 30, eta_m = 20, p_m = 1/n, and a complete Das-Dennis
  lattice of the population's size. The paper uses p_c = 1; VAMOS uses 0.9 in every default.
- MOEA/D: MOEA/D-DE of Li and Zhang (2009), CR = 1, F = 0.5, eta_m = 20, p_m = 1/n,
  T = 20, delta = 0.9, n_r = 2. PBI with theta = 5 (Zhang and Li, 2007) is the VAMOS choice of
  aggregation (``docs/algorithms/moead.md``).
- SPEA2: Zitzler, Laumanns and Thiele (2001), archive as large as the population. k = 1, as in
  jMetal; the paper uses sqrt(N + archive size) (``SPEA2`` docstring).
- SMS-EMOA: Beume, Naujoks and Emmerich (2007), steady state with random parents; the adaptive
  reference point is the nadir plus 1 (``docs/algorithms/smsemoa.md``).
- IBEA: Zitzler and Kuenzli (2004), additive epsilon indicator, kappa = 0.05.
- AGE-MOEA: SBX p_c = 0.9, eta_c = 15 and polynomial mutation eta_m = 20, as pymoo's AGEMOEA.
- RVEA: Cheng, Jin, Olhofer and Sendhoff (2016), alpha = 2, f_r = 0.1, eta_c = 30, eta_m = 20.
- SMPSO: Nebro et al. (2009), C1 and C2 drawn from [1.5, 2.5], w = 0.1, velocity bounded by half
  the variable range, polynomial mutation p_m = 1/n, eta_m = 20.
- CCMO: Tian, Zhang, Xiao, Zhang and Jin (2021) and jMetal's port, SBX p_c = 1.0, eta_c = 20,
  polynomial mutation p_m = 1/n, eta_m = 20.
- C-TAEA: Li, Chen, Fu and Yao (2019) and jMetal's port, SBX p_c = 1.0, eta_c = 20, polynomial
  mutation p_m = 1/n, eta_m = 20.
- ToP: Liu and Wang (2019) as PlatEMO's implementation and jMetal's port set it, DE with CR = 1.0
  and F = 0.5, polynomial mutation p_m = 1/n, eta_m = 20, and phase 1 while (delta >= 0.2 or
  Pf <= 1/3) within 90% of the budget.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import vamos
from vamos.engine.algorithm.config.defaults import build_default_algorithm_config

_N_VAR, _N_OBJ, _POP = 30, 2, 100
_SBX_20 = ["sbx", {"prob": 0.9, "eta": 20.0}]
_SBX_30 = ["sbx", {"prob": 0.9, "eta": 30.0}]
_PM = ["pm", {"prob": 1.0 / _N_VAR, "eta": 20.0}]

PINNED: dict[str, dict[str, Any]] = {
    "nsgaii": {"crossover": _SBX_20, "mutation": _PM, "selection": ["tournament", {}]},
    "nsgaiii": {
        "crossover": _SBX_30,
        "mutation": _PM,
        "selection": ["tournament", {}],
        "reference_directions": {"divisions": 99, "path": None},
        "enforce_ref_dirs": True,
    },
    "moead": {
        "crossover": ["de", {"cr": 1.0, "f": 0.5}],
        "mutation": _PM,
        "neighbor_size": 20,
        "delta": 0.9,
        "replace_limit": 2,
        "aggregation": ["pbi", {"theta": 5.0}],
        "weight_vectors": {"divisions": 99, "path": None},
    },
    "spea2": {"crossover": _SBX_20, "mutation": _PM, "selection": ["tournament", {}], "archive_size": _POP, "k_neighbors": None},
    "smsemoa": {
        "crossover": _SBX_20,
        "mutation": _PM,
        "selection": ["random", {}],
        "reference_point": {"adaptive": True, "offset": 1.0, "vector": None},
    },
    "ibea": {"crossover": _SBX_20, "mutation": _PM, "selection": ["tournament", {}], "indicator": "eps", "kappa": 0.05},
    "agemoea": {"crossover": ["sbx", {"prob": 0.9, "eta": 15.0}], "mutation": _PM},
    "rvea": {"crossover": _SBX_30, "mutation": _PM, "alpha": 2.0, "adapt_freq": 0.1, "n_partitions": 99},
    "ccmo": {"crossover": ["sbx", {"prob": 1.0, "eta": 20.0}], "mutation": _PM},
    "ctaea": {"crossover": ["sbx", {"prob": 1.0, "eta": 20.0}], "mutation": _PM},
    "top": {
        "crossover": ["de", {"cr": 1.0, "f": 0.5}],
        "mutation": _PM,
        "delta_threshold": 0.2,
        "feasible_ratio_threshold": 1.0 / 3.0,
        "phase_one_evaluation_fraction": 0.9,
    },
    "smpso": {
        "mutation": ["polynomial", {"prob": 1.0 / _N_VAR, "eta": 20.0}],
        "inertia": 0.1,
        "c1": 1.5,
        "c2": 1.5,
        "vmax_fraction": 0.5,
        "archive_size": _POP,
    },
}
# Fields whose value depends on the population size, which each entry point sizes on its own.
_LATTICE_FIELDS = {"nsgaiii": "reference_directions", "moead": "weight_vectors", "rvea": "n_partitions"}
# CCMO's two initial populations take 2 * pop_size evaluations: 400 with the quick default population of ZDT1.
_BUDGET = {"ccmo": 400}
# Resolved run specs name operators by their component, not by the configuration's alias.
_COMPONENT_NAMES = {"pm": "polynomial", "de": "differential_evolution"}


def _plain(value: Any) -> Any:
    """``value`` with tuples as lists, as JSON records it, and floats rounded to 12 digits."""
    return json.loads(json.dumps(value), parse_float=lambda text: round(float(text), 12))


def _without_none(fields: dict[str, Any]) -> dict[str, Any]:
    """``fields`` without the keys whose value is None inside a field's mapping."""
    return {
        field: {k: v for k, v in value.items() if v is not None} if isinstance(value, dict) else value for field, value in fields.items()
    }


def _default(algorithm: str, pop_size: int | None = _POP) -> dict[str, Any]:
    config = build_default_algorithm_config(algorithm, pop_size=pop_size, n_var=_N_VAR, n_obj=_N_OBJ)
    assert config is not None
    return _plain(config.to_dict())


@pytest.mark.parametrize("algorithm", sorted(PINNED))
def test_default_configuration_keeps_its_scientific_values(algorithm: str) -> None:
    config = _default(algorithm)
    assert {field: config.get(field) for field in PINNED[algorithm]} == _plain(PINNED[algorithm])
    assert config["pop_size"] == _POP


def _compared_fields(algorithm: str) -> list[str]:
    # A run spec records every algorithm's parents as a selection operator, and sizes the archive and
    # the lattice with the population it uses.
    return [field for field in PINNED[algorithm] if field not in {"selection", "archive_size", _LATTICE_FIELDS.get(algorithm)}]


def _component_name(operator: dict[str, Any]) -> str:
    return str(operator["component_id"]).split(":", 1)[1].split("@", 1)[0]


def _assert_resolves_the_defaults(algorithm: str, resolved_spec: dict[str, Any]) -> None:
    recorded: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for field in _compared_fields(algorithm):
        value = PINNED[algorithm][field]
        if field in {"crossover", "mutation"}:
            operator = resolved_spec["operators"][field]
            recorded[field] = [_component_name(operator), operator["config"]]
            expected[field] = [_COMPONENT_NAMES.get(value[0], value[0]), value[1]]
        else:
            recorded[field] = resolved_spec["algorithm"]["config"][field]
            expected[field] = value
    # Some writers leave out the nested keys whose value is None (SMS-EMOA's reference ``vector``).
    assert _without_none(_plain(recorded)) == _without_none(_plain(expected))
    lattice = _LATTICE_FIELDS.get(algorithm)
    if lattice is not None:
        # Without explicit divisions, the lattice is the default one for the population actually used.
        pop_size = resolved_spec["population"]["initial_size"]
        assert resolved_spec["algorithm"]["config"][lattice] == _default(algorithm, pop_size)[lattice]


@pytest.mark.parametrize("algorithm", sorted(PINNED))
def test_optimize_without_a_configuration_resolves_the_defaults(algorithm: str) -> None:
    result = vamos.optimize("zdt1", algorithm=algorithm, max_evaluations=_BUDGET.get(algorithm, 200), seed=0)
    _assert_resolves_the_defaults(algorithm, result.explain_defaults()["resolved_spec"])


_CLI_DIVERGENCES = {
    "moead": "VR-329: the CLI gives MOEA/D the packaged weights directory instead of the default lattice",
    "nsgaiii": "VR-330: the CLI builds NSGA-III with SBX eta 20 instead of the default 30",
}


@pytest.mark.parametrize(
    "algorithm",
    [
        pytest.param(name, marks=[pytest.mark.xfail(strict=True, reason=_CLI_DIVERGENCES[name])] if name in _CLI_DIVERGENCES else [])
        for name in sorted(PINNED)
    ],
)
def test_cli_without_a_configuration_resolves_the_defaults(algorithm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.experiment.cli.main import main

    argv = ["--problem", "zdt1", "--algorithm", algorithm, "--max-evaluations", "200", "--output-root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["vamos", *argv])
    main()

    (manifest,) = tmp_path.rglob("manifest.json")
    _assert_resolves_the_defaults(algorithm, json.loads(manifest.read_text(encoding="utf-8"))["resolved_spec"])


@pytest.fixture(scope="module")
def study_plan(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    spec = vamos.StudySpec(problems=["zdt1"], algorithms=sorted(PINNED), seeds=[0], max_evaluations=400)
    study = vamos.create_study(spec, output=tmp_path_factory.mktemp("defaults") / "study")
    plan = json.loads((study.root / "plan.json").read_text(encoding="utf-8"))
    specs = (task["resolved_run_spec"] for task in plan["tasks"])
    return {_component_name(spec["algorithm"]): spec for spec in specs}


@pytest.mark.parametrize("algorithm", sorted(PINNED))
def test_study_without_a_configuration_resolves_the_defaults(algorithm: str, study_plan: dict[str, dict[str, Any]]) -> None:
    _assert_resolves_the_defaults(algorithm, study_plan[algorithm])
