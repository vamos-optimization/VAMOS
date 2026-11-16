"""DC-DTLZ's parameters: the authors' constants by default, and jMetal's (which STEP also uses) on request.

``data/dcdtlz_jmetal_values.json`` holds jMetal's values with ``a = 3``, ``b = 0.5`` and ``g`` unscaled in
the constraints; ``tools/problem_reference_values.py --dcdtlz-jmetal`` regenerates it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.problems import DCDTLZ

_FIXTURE: dict[str, Any] = json.loads((Path(__file__).parent / "data" / "dcdtlz_jmetal_values.json").read_text(encoding="utf-8"))
_VARIANTS = ("dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3", "dc3dtlz1", "dc3dtlz3")


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: f"{case['key']}-n{case['n_var']}-m{case['n_obj']}")
def test_dcdtlz_with_jmetal_constants_matches_jmetal(case: dict[str, Any]) -> None:
    problem = DCDTLZ(case["key"], n_var=case["n_var"], n_obj=case["n_obj"], **_FIXTURE["parameters"])

    F, G = evaluate_population_with_constraints(problem, np.asarray(case["X"], dtype=float))

    np.testing.assert_allclose(F, np.asarray(case["F"]), rtol=1e-9, atol=1e-12)
    assert G is not None
    np.testing.assert_allclose(G, np.asarray(case["G"]), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("key", _VARIANTS)
def test_dcdtlz_defaults_are_the_authors_constants(key: str) -> None:
    authors = {"dc1": (5.0, 0.95, 1.0), "dc2": (3.0, 0.9, 0.01), "dc3": (5.0, 0.5, 1.0)}[key[:3]]
    default, explicit = DCDTLZ(key), DCDTLZ(key, a=authors[0], b=authors[1], g_factor=100.0, constraint_g_scale=authors[2])
    X = np.random.default_rng(1).random((50, default.n_var))

    assert (default.a, default.b, default.g_factor, default.constraint_g_scale) == (*authors[:2], 100.0, authors[2])
    for got, expected in zip(evaluate_population_with_constraints(default, X), evaluate_population_with_constraints(explicit, X)):
        np.testing.assert_array_equal(got, expected)


# Table II of the supplementary document of Li et al. (2019, doi:10.1109/TEVC.2018.2855411): the share of
# 10^7 uniform samples of the decision space that is feasible, at M = 3, 5 and 8.
_AUTHORS_FEASIBLE_RATIOS = {"dc1": (0.1011, 0.1012, 0.1011), "dc2": (0.0, 0.0, 0.0), "dc3": (0.0370, 0.0041, 0.0002)}


def _feasible_ratio(problem: Any, n_samples: int) -> float:
    X = np.random.default_rng(3).random((n_samples, problem.n_var))
    _, G = evaluate_population_with_constraints(problem, X)
    assert G is not None
    return float(np.mean(np.all(G <= 0.0, axis=1)))


@pytest.mark.parametrize("key", _VARIANTS)
def test_dcdtlz_defaults_reproduce_the_feasible_ratios_of_the_authors_study(key: str) -> None:
    # DC1's b = 0.95 makes arccos(0.95) / pi = 10.11 % of x1 feasible. DC3 has M constraints, one on g and one
    # on each of the M - 1 position variables, each met by a third of the samples, so (1/3)^M is feasible.
    for n_obj, expected in zip((3, 5, 8), _AUTHORS_FEASIBLE_RATIOS[key[:3]]):
        problem = DCDTLZ(key, n_var=n_obj + 9, n_obj=n_obj)

        assert _feasible_ratio(problem, 200_000) == pytest.approx(expected, abs=0.003), n_obj


def test_dc1_with_the_constants_of_the_supplements_text_does_not_reproduce_its_table() -> None:
    # The supplement's text gives a = 3 and b = 0.5 for DC1, which jMetal, PlatEMO and STEP use: then a third
    # of the samples is feasible, not the 10.11 % that its Table II reports and the authors' code produces.
    problem = DCDTLZ("dc1dtlz1", a=3.0, b=0.5, constraint_g_scale=1.0)

    assert _feasible_ratio(problem, 200_000) == pytest.approx(1.0 / 3.0, abs=0.005)


@pytest.mark.parametrize("key", _VARIANTS)
def test_dcdtlz_without_distance_variables_has_g_zero(key: str) -> None:
    # STEP16 and STEP18 have three objectives and two variables: every design lies on DTLZ's front.
    problem = DCDTLZ(key, n_var=2, n_obj=3)
    X = np.random.default_rng(2).random((50, 2))

    F, _ = evaluate_population_with_constraints(problem, X)

    if key.endswith("dtlz1"):
        np.testing.assert_allclose(F.sum(axis=1), 0.5, rtol=0.0, atol=1e-12)
    else:
        np.testing.assert_allclose(np.linalg.norm(F, axis=1), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_var": 1, "n_obj": 3}, "n_var"),
        ({"a": 0.0}, "a must be positive"),
        ({"b": 1.5}, r"b must be in \[-1, 1\]"),
        ({"g_factor": 0.0}, "positive"),
        ({"constraint_g_scale": -1.0}, "positive"),
    ],
)
def test_dcdtlz_rejects_invalid_parameters(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DCDTLZ("dc2dtlz1", **kwargs)
