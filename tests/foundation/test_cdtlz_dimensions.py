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
