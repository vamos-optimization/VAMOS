"""C-DTLZ with as many variables as objectives, and with the dimensions and radius of the STEP suite.

``data/cdtlz_jmetal_values.json`` holds jMetal 7.10's values at three points of each case:
``tools/problem_reference_values.py --cdtlz-jmetal`` regenerates it. With ``n_var == n_obj`` a single
distance variable is left, as in STEP20, C3-DTLZ4 with three objectives and three variables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.constrained_many import CDTLZProblem

_FIXTURE: dict[str, Any] = json.loads((Path(__file__).parent / "data" / "cdtlz_jmetal_values.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: f"{case['key']}-n{case['n_var']}-m{case['n_obj']}-r{case.get('r')}")
def test_cdtlz_matches_jmetal(case: dict[str, Any]) -> None:
    kwargs = {"r": case["r"]} if "r" in case else {}
    problem = CDTLZProblem(case["key"], n_var=case["n_var"], n_obj=case["n_obj"], **kwargs)

    F, G = evaluate_population_with_constraints(problem, np.asarray(case["X"], dtype=float))

    np.testing.assert_allclose(F, np.asarray(case["F"]), rtol=1e-9, atol=1e-12)
    assert G is not None
    np.testing.assert_allclose(G, np.asarray(case["G"]), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("key", ["c1dtlz1", "c1dtlz3", "c2dtlz2", "c3dtlz1", "c3dtlz4"])
def test_cdtlz_needs_as_many_variables_as_objectives(key: str) -> None:
    CDTLZProblem(key, n_var=3, n_obj=3)

    with pytest.raises(ValueError, match="n_var >= n_obj"):
        CDTLZProblem(key, n_var=2, n_obj=3)


@pytest.mark.parametrize(("n_obj", "radius"), [(2, 0.5), (3, 0.4), (5, 0.5)])
def test_c2_dtlz2_default_radius_is_jain_and_debs(n_obj: int, radius: float) -> None:
    # Jain and Deb (2014) and jMetal's C2_DTLZ2: 0.4 with three objectives and 0.5 otherwise; VAMOS 1.1.0 used
    # 0.2 with two objectives.
    X = np.random.default_rng(3).uniform(size=(20, n_obj + 9))
    _, G = evaluate_population_with_constraints(CDTLZProblem("c2dtlz2", n_var=n_obj + 9, n_obj=n_obj), X)
    _, explicit = evaluate_population_with_constraints(CDTLZProblem("c2dtlz2", n_var=n_obj + 9, n_obj=n_obj, r=radius), X)

    assert G is not None and explicit is not None
    np.testing.assert_array_equal(G, explicit)


def test_c2_dtlz2_with_two_objectives_is_feasible_on_the_whole_quarter_circle() -> None:
    # With r = 0.5 every point of the unit circle in the first quadrant lies within r of (1, 0), (0, 1) or the centre.
    angles = np.linspace(0.0, 1.0, 91)
    X = np.column_stack([angles, np.full((91, 11), 0.5)])
    F, G = evaluate_population_with_constraints(CDTLZProblem("c2dtlz2", n_var=12, n_obj=2), X)

    np.testing.assert_allclose(np.sum(F * F, axis=1), 1.0)
    assert G is not None and np.all(G <= 0.0)
