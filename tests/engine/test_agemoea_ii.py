"""AGE-MOEA-II, the ``variant="agemoea_ii"`` of AGE-MOEA, against jMetal's own (tests/engine/data/agemoea2_jmetal_values.json)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos import optimize
from vamos.algorithms import AGEMOEAConfig
from vamos.engine.algorithm.agemoea.geometry import (
    _compute_geometry_ii,
    _geodesic_distances,
    _newton_exponent,
    _survival_score,
    age_survival,
)
from vamos.engine.algorithm.config.types import AGEMOEA_VARIANTS, resolve_agemoea_variant
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.racing.parameter_names import from_evolver, to_evolver
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.problems import DTLZ2, LIRCMOP

FIXTURE = Path(__file__).resolve().parent / "data" / "agemoea2_jmetal_values.json"
CASES: list[dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
GEOMETRY = [case for case in CASES if case["kind"] == "geometry"]
DISTANCES = [case for case in CASES if case["kind"] == "distances"]
SURVIVAL = [case for case in CASES if case["kind"] == "survival"]


def _scores(values: list[float | None]) -> np.ndarray:
    return np.array([np.inf if value is None else value for value in values], dtype=float)


def _same_scores(actual: np.ndarray, expected: np.ndarray) -> bool:
    infinite = np.isinf(expected)
    return bool(np.array_equal(np.isinf(actual), infinite) and np.allclose(actual[~infinite], expected[~infinite], rtol=1e-9, atol=1e-12))


def _vamos_scores(case: dict[str, Any], variant: str) -> np.ndarray:
    F = np.asarray(case["front"], dtype=float)
    scores, _, _ = _survival_score(F, F.min(axis=0), None, resolve_agemoea_variant(variant))
    return scores


def _config(variant: str, **changes: Any) -> AGEMOEAConfig:
    return dataclasses.replace(AGEMOEAConfig.default(pop_size=40, n_var=12), variant=variant, **changes)


# ---- The steps that AGE-MOEA-II overrides, against jMetal ---------------------


def test_the_fixture_spans_flat_convex_and_concave_fronts() -> None:
    exponents = [case["p"] for case in GEOMETRY]
    assert min(exponents) < 0.5 and max(exponents) > 5.0
    assert {case["q"] for case in SURVIVAL} >= {0.4, 1.0, 6.0}
    assert {len(case["front"][0]) for case in SURVIVAL} == {2, 3, 4, 5}


@pytest.mark.parametrize("case", GEOMETRY, ids=lambda case: f"m{len(case['front'])}-n{len(case['front'][0])}")
def test_exponent_matches_jmetal(case: dict[str, Any]) -> None:
    front = np.asarray(case["front"], dtype=float)
    expected = min(case["p"], 20.0)  # VAMOS caps p at 20, as in AGE-MOEA; no case reaches it
    assert _compute_geometry_ii(front, np.asarray(case["extremes"])) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("case", DISTANCES, ids=lambda case: f"p{case['p']}-n{len(case['front'][0])}")
def test_geodesic_distances_match_jmetal(case: dict[str, Any]) -> None:
    front = np.asarray(case["front"], dtype=float)
    np.testing.assert_allclose(_geodesic_distances(front, case["p"]), case["distances"], rtol=1e-12, atol=1e-15)


def test_survival_scores_match_jmetal_where_the_shared_steps_agree() -> None:
    # AGE-MOEA's extreme points and its tie-breaking follow PlatEMO and pymoo, not jMetal; on a front
    # where they agree with jMetal (AGE-MOEA's own scores match), AGE-MOEA-II's scores must match too.
    compared = 0
    for case in SURVIVAL:
        if case["q"] == 1.0 or not _same_scores(_vamos_scores(case, "agemoea"), _scores(case["agemoea_scores"])):
            continue
        compared += 1
        assert _same_scores(_vamos_scores(case, "agemoea_ii"), _scores(case["agemoea_ii_scores"])), case["front"]
    assert compared >= len(SURVIVAL) // 2


# ---- The exponent's bounds ------------------------------------------------------


def test_newton_exponent_is_the_norm_one_exponent() -> None:
    point = np.array([0.3, 0.5, 0.7])
    p = _newton_exponent(point, precision=1e-12)
    assert np.sum(point**p) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (np.array([0.99, 0.99, 0.99]), 20.0),  # a near-cubic front: p of about 109, capped
        (np.array([1e-6, 1e-6]), 1.0),  # p of about 0.05: a flat front
        (np.array([0.0, 0.0, 0.0]), 1.0),  # no positive coordinate: Newton diverges
    ],
)
def test_exponent_is_bounded_as_in_agemoea(point: np.ndarray, expected: float) -> None:
    front = np.vstack([np.eye(point.size), point])
    assert _compute_geometry_ii(front, np.arange(point.size)) == expected


def test_a_point_at_the_ideal_point_has_finite_distances() -> None:
    front = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.6, 0.6]])
    distances = _geodesic_distances(front, 2.0)
    assert np.all(np.isfinite(distances))
    np.testing.assert_allclose(distances, distances.T)


def test_numba_selection_kernel_matches_the_numpy_loop() -> None:
    pytest.importorskip("numba")
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    kernels = NumbaKernel().selection_kernels()
    rng = np.random.default_rng(3)
    for n_obj in (2, 3, 5):
        F = rng.random((40, n_obj))
        F = F / np.sum(F**2.5, axis=1, keepdims=True) ** (1 / 2.5)
        reference = _survival_score(F.copy(), F.min(axis=0), None, "agemoea_ii")
        compiled = _survival_score(F.copy(), F.min(axis=0), kernels, "agemoea_ii")
        np.testing.assert_array_equal(reference[0], compiled[0])
        assert reference[1] == compiled[1]


def test_variants_differ_on_a_curved_front() -> None:
    rng = np.random.default_rng(5)
    F = rng.random((30, 3))
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    kernel = NumPyKernel()
    first, _ = age_survival(F, 15, kernel, variant="agemoea")
    second, _ = age_survival(F, 15, kernel, variant="agemoea_ii")
    assert set(first.tolist()) != set(second.tolist())


# ---- Configuration -----------------------------------------------------------


def test_variant_defaults_to_agemoea_and_round_trips() -> None:
    assert AGEMOEAConfig.default().variant == "agemoea"
    config = AGEMOEAConfig.builder().pop_size(20).crossover("sbx", prob=0.9, eta=15.0).mutation("pm", prob=0.1, eta=20.0)
    built = config.variant("agemoea_ii").build()
    assert built.variant == "agemoea_ii"
    assert built.to_dict()["variant"] == "agemoea_ii"
    assert AGEMOEAConfig(**built.to_dict()) == built


@pytest.mark.parametrize("bad", ["agemoea2", "AGE-MOEA-II", ""])
def test_unknown_variant_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="agemoea_ii"):
        AGEMOEAConfig.builder().variant(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="agemoea_ii"):
        _config(bad)


def test_variant_names() -> None:
    assert AGEMOEA_VARIANTS == ("agemoea", "agemoea_ii")
    assert resolve_agemoea_variant(" AGEMOEA_II ") == "agemoea_ii"


# ---- Runs ----------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_agemoea_ii_runs_with_an_exact_budget_and_a_seed(engine: str) -> None:
    if engine == "numba":
        pytest.importorskip("numba")
    problem = DTLZ2(n_var=12, n_obj=3)
    first = optimize(problem, algorithm="agemoea", algorithm_config=_config("agemoea_ii"), max_evaluations=400, seed=7, engine=engine)
    again = optimize(problem, algorithm="agemoea", algorithm_config=_config("agemoea_ii"), max_evaluations=400, seed=7, engine=engine)
    other = optimize(problem, algorithm="agemoea", algorithm_config=_config("agemoea"), max_evaluations=400, seed=7, engine=engine)
    assert first.data["evaluations"] == 400
    np.testing.assert_array_equal(first.F, again.F)
    assert not np.array_equal(np.sort(first.F, axis=0), np.sort(other.F, axis=0))


EPSILON = ("epsilon_constrained", {"initial_quantile": 0.2, "exponent": 5.0, "decay_fraction": 0.8})


@pytest.mark.parametrize("technique", [None, EPSILON])
def test_agemoea_ii_runs_on_a_constrained_problem(technique: tuple[str, dict[str, float]] | None) -> None:
    config = _config("agemoea_ii", constraint_handling=technique)
    result = optimize(LIRCMOP(1, n_var=10), algorithm="agemoea", algorithm_config=config, max_evaluations=600, seed=2, engine="numpy")
    assert result.data["evaluations"] == 600
    assert result.F.shape[1] == 2 and np.all(np.isfinite(result.F))


# ---- Tuning ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "space", ["agemoea", "agemoea_constrained", "agemoea_binary", "agemoea_integer", "agemoea_permutation", "agemoea_mixed"]
)
def test_every_agemoea_space_chooses_the_variant(space: str) -> None:
    assert config_from_assignment(space, {"replacement": "agemoea_ii"}).variant == "agemoea_ii"  # type: ignore[attr-defined]
    assert config_from_assignment(space, {}).variant == "agemoea"  # type: ignore[attr-defined]


def test_replacement_is_exchanged_with_evolver() -> None:
    assert to_evolver({"replacement": "agemoea_ii"}) == {"replacement": "agemoeaII"}
    assert from_evolver({"replacement": "agemoeaII"}) == {"replacement": "agemoea_ii"}
    assert to_evolver({"replacement": "agemoea"}) == {"replacement": "agemoea"}
