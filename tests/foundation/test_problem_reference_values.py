"""The benchmark problems against independent implementations and their analytical fronts (VR-068).

``data/problem_reference_values.json`` holds, for each problem that pymoo 0.6.2, jMetal 7.8 or, for
LIR-CMOP and DAS-CMOP, jMetal's ``main`` also implements, three points inside the VAMOS bounds and that implementation's objective and constraint
values there; ``tools/problem_reference_values.py`` regenerates it. The fixture's ``excluded`` table
gives, for each family member left out, why its counterpart is not the same problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.registry import available_problem_names, make_problem_selection
from vamos.resources import reference_front_path

_FIXTURE: dict[str, Any] = json.loads((Path(__file__).parent / "data" / "problem_reference_values.json").read_text(encoding="utf-8"))
_PROBLEMS: dict[str, Any] = _FIXTURE["problems"]
_EXCLUDED: dict[str, str] = _FIXTURE["excluded"]
# Families that have an independent implementation; each member must be compared or excluded.
_REFERENCED_FAMILIES = (
    "zdt",
    "dtlz",
    "c1dtlz",
    "c2dtlz",
    "c3dtlz",
    "dc1dtlz",
    "dc2dtlz",
    "dc3dtlz",
    "lircmop",
    "dascmop",
    "binh2",
    "constrex",
    "srinivas",
    "tanaka",
    "osyczka2",
    "golinski",
    "water",
    "car_side_impact",
    "wfg",
    "mw",
    "cec2009_",
    "lz09_",
    "lsmop",
    "zcat",
    "re",
    "rwa",
)


@pytest.mark.parametrize("key", sorted(_PROBLEMS))
def test_problem_matches_its_reference_implementation(key: str) -> None:
    entry = _PROBLEMS[key]
    problem = make_problem_selection(key).instantiate()
    assert (int(problem.n_var), int(problem.n_obj)) == (entry["n_var"], entry["n_obj"])

    F, G = evaluate_population_with_constraints(problem, np.asarray(entry["X"], dtype=float))

    np.testing.assert_allclose(F, np.asarray(entry["F"]), rtol=1e-9, atol=1e-12, err_msg=f"{key} against {entry['source']}")
    if "G" in entry:
        assert G is not None, f"{key}: {entry['source']} has constraints, VAMOS has none"
        np.testing.assert_allclose(G, np.asarray(entry["G"]), rtol=1e-9, atol=1e-12, err_msg=f"{key} constraints")
    else:
        assert G is None, f"{key}: VAMOS has constraints, {entry['source']} has none"


def test_every_member_of_a_referenced_family_is_compared_or_excluded() -> None:
    names = {name for name in available_problem_names() if name.startswith(_REFERENCED_FAMILIES)}
    assert names - set(_PROBLEMS) - set(_EXCLUDED) == set()
    assert set(_EXCLUDED).isdisjoint(_PROBLEMS)
    assert all(reason.strip() for reason in _EXCLUDED.values())


def _pareto_set(key: str, problem: Any, n: int = 51) -> np.ndarray:
    """Points of the analytical Pareto set: the position variables on a grid, the others optimal."""
    t = np.linspace(0.0, 1.0, n)
    if key.startswith("zdt"):
        # x1 on a grid and every other variable at 0.
        X = np.zeros((n, int(problem.n_var)))
        X[:, 0] = t
        return X
    # DTLZ1-4 with three objectives: the two position variables on a grid, every distance variable at 0.5.
    grid = np.stack(np.meshgrid(t, t, indexing="ij"), axis=-1).reshape(-1, 2)
    X = np.full((grid.shape[0], int(problem.n_var)), 0.5)
    X[:, :2] = grid
    return X


_ANALYTICAL_FRONTS = {
    "zdt1": lambda F: F[:, 1] - (1.0 - np.sqrt(F[:, 0])),
    "zdt2": lambda F: F[:, 1] - (1.0 - F[:, 0] ** 2),
    "zdt3": lambda F: F[:, 1] - (1.0 - np.sqrt(F[:, 0]) - F[:, 0] * np.sin(10.0 * np.pi * F[:, 0])),
    "zdt4": lambda F: F[:, 1] - (1.0 - np.sqrt(F[:, 0])),
    "zdt6": lambda F: F[:, 1] - (1.0 - F[:, 0] ** 2),
    "dtlz1": lambda F: F.sum(axis=1) - 0.5,
    "dtlz2": lambda F: (F**2).sum(axis=1) - 1.0,
    "dtlz3": lambda F: (F**2).sum(axis=1) - 1.0,
    "dtlz4": lambda F: (F**2).sum(axis=1) - 1.0,
}


@pytest.mark.parametrize("key", sorted(_ANALYTICAL_FRONTS))
def test_pareto_set_points_lie_on_the_analytical_and_packaged_fronts(key: str) -> None:
    problem = make_problem_selection(key).instantiate()
    X = _pareto_set(key, problem)
    F, _ = evaluate_population_with_constraints(problem, X)

    np.testing.assert_allclose(_ANALYTICAL_FRONTS[key](F), 0.0, atol=1e-9)
    # The packaged lookup, which also finds ZDT1.csv on case-sensitive filesystems.
    front = np.loadtxt(reference_front_path(key), delimiter=",", ndmin=2)
    # ZDT3's Pareto set also maps to dominated parts of its curve; the packaged front holds only the
    # non-dominated ones, so only those points are checked against it.
    if key == "zdt3":
        nondominated = ~np.any(np.all(F[:, None, :] >= F[None, :, :], axis=2) & np.any(F[:, None, :] > F[None, :, :], axis=2), axis=1)
        F = F[nondominated]
    distance = np.min(np.linalg.norm(F[:, None, :] - front[None, :, :], axis=2), axis=1)
    assert distance.max() < 0.02


def test_qubo_energy_matches_its_definition() -> None:
    # bin_qubo has no external counterpart: f1 = x^T Q x + b^T x and f2 = -(selected bits).
    problem = make_problem_selection("bin_qubo").instantiate()
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(5, int(problem.n_var))).astype(float)
    F, G = evaluate_population_with_constraints(problem, X)
    Q, b = np.asarray(problem.Q), np.asarray(problem.bias)
    expected = [sum(Q[i, j] * x[i] * x[j] for i in range(x.size) for j in range(x.size)) + float(b @ x) for x in X]
    np.testing.assert_allclose(F[:, 0], expected, rtol=1e-12)
    np.testing.assert_array_equal(F[:, 1], -X.sum(axis=1))
    assert G is None
