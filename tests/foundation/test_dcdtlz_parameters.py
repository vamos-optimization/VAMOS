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
