"""Namura's HPA problems (EMO 2025): the authors' values, dimensions, bounds, penalties, the registry and the fronts.

``data/hpa_reference_values.json`` holds, for every registered HPA problem, the objective and constraint values of
the authors' reference code (https://github.com/Nobuo-Namura/hpa, commit ``3524115``, with NumPy 2.4.6, SciPy 1.17.1
and pandas 3.1.1) at two points of ``numpy.random.default_rng(1000 * int(index) + level).random((2, n_var))``.
On the platform that produced them, VAMOS's values were bit-identical, there and at 20 other points per problem.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.foundation.problem.hpa import _kernels
from vamos.foundation.problem.hpa.design import GRAVITY, WingConfig
from vamos.foundation.problem.hpa.problems import CONSTRAINTS, HPA_INDICES, HPA_TYPES, OBJECTIVES, HPAProblem, constrained_index, hpa_type
from vamos.foundation.problem.registry import available_problem_names, make_problem_selection
from vamos.foundation.problem.resolver import resolve_reference_front_path

# The paper's Table 2: the number of variables at levels 0, 1 and 2, with n = 4, by penalized index.
TABLE_2 = {
    "201": (18, 33, 109),
    "202": (16, 31, 187),
    "203": (19, 34, 190),
    "204": (18, 33, 189),
    "205": (18, 33, 189),
    "301": (18, 33, 109),
    "302": (16, 31, 187),
    "303": (19, 34, 190),
    "304": (18, 33, 189),
    "305": (18, 33, 189),
    "401": (18, 33, 189),
    "402": (16, 31, 187),
    "403": (18, 33, 189),
    "501": (18, 33, 189),
    "502": (19, 34, 190),
    "601": (18, 33, 189),
    "901": (19, 34, 190),
}
# The constrained problems of Table 2.
CONSTRAINED = {"241", "222", "233", "244", "245", "341", "322", "333", "344", "345", "441", "422", "443", "541", "542", "641", "941"}
_REFERENCE: dict[str, Any] = json.loads((Path(__file__).parent / "data" / "hpa_reference_values.json").read_text(encoding="utf-8"))


def test_every_problem_has_reference_values() -> None:
    assert set(_REFERENCE["values"]) == {name for name in available_problem_names() if name.startswith("hpa")}


@pytest.mark.parametrize("key", sorted(_REFERENCE["values"]))
def test_problem_matches_the_authors_code(key: str) -> None:
    entry = _REFERENCE["values"][key]
    problem = make_problem_selection(key).instantiate()
    index, level = key.removeprefix("hpa").split("_")
    X = np.random.default_rng(1000 * int(index) + int(level)).random((2, problem.n_var))
    out: dict[str, np.ndarray] = {}
    problem.evaluate(X, out)
    np.testing.assert_allclose(out["F"], np.asarray(entry["F"]), rtol=1e-9, atol=1e-12)
    if "G" in entry:
        np.testing.assert_allclose(out["G"], np.asarray(entry["G"]), rtol=1e-9, atol=1e-12)
    else:
        assert "G" not in out


def test_the_problems_are_those_of_table_2() -> None:
    assert set(HPA_TYPES) == set(TABLE_2)
    assert {constrained_index(index) for index in HPA_TYPES} == CONSTRAINED
    assert HPA_INDICES == {
        **{index: (index, False) for index in HPA_TYPES},
        **{constrained_index(index): (index, True) for index in HPA_TYPES},
    }
    assert {name for name in available_problem_names() if name.startswith("hpa")} == {
        f"hpa{index}_{level}" for index in HPA_INDICES for level in (0, 1, 2)
    }


@pytest.mark.parametrize("base", sorted(TABLE_2))
def test_dimensions_are_those_of_table_2(base: str) -> None:
    for level in (0, 1, 2):
        for index in (base, constrained_index(base)):
            problem = make_problem_selection(f"hpa{index}_{level}").instantiate()
            assert isinstance(problem, HPAProblem)
            assert problem.n_var == TABLE_2[base][level]
            assert problem.n_obj == int(index[0])
            assert problem.n_constraints == (int(index[1]) if index != base else 0)
            np.testing.assert_array_equal(problem.xl, np.zeros(problem.n_var))
            np.testing.assert_array_equal(problem.xu, np.ones(problem.n_var))


def test_bounds_are_those_of_table_5() -> None:
    n, m = 4, 20
    lower, upper = WingConfig(n, m, 2, wire=True, dihedral=True, payload=True).bounds()
    expected = [
        (n, 8 / n, 20 / n),  # segment lengths [m]
        (n + 1, 0.4, 1.5),  # chords [m]
        (1, 4.0, 7.0),  # angle of attack at the root [deg]
        (n, -8 / n, 0.0),  # washouts [deg]
        (n, 0.8, 1.0),  # beam diameter factors
        (m * n, 0.0, 1.0),  # flange length increments
        (m * n, 0.002, 0.26),  # flange width increments [m]
        (n, 0.0, 1.0),  # spanwise reference positions
        (1, 0.0, 135 * GRAVITY),  # wire tension [N]
        (1, 0.0, 4.0),  # dihedral angle at the root [deg]
        (1, 0.0, 60.0),  # payload [kg]
        (n + 1, 0.0, 3.0),  # airfoils
    ]
    start = 0
    for size, low, high in expected:
        np.testing.assert_allclose(lower[start : start + size], low)
        np.testing.assert_allclose(upper[start : start + size], high)
        start += size
    assert start == len(lower)
    lower, upper = WingConfig(n, m, 0).bounds()
    np.testing.assert_allclose(lower[3 * n + 2 : 3 * n + 4], 0.01)  # the strain reduction factors of level 0
    np.testing.assert_allclose(upper[3 * n + 2 : 3 * n + 4], 1.0)


@pytest.mark.parametrize("base", sorted(TABLE_2))
def test_penalized_objectives_add_the_penalties_of_eq_5(base: str) -> None:
    rng = np.random.default_rng(int(base))
    for level in (0, 2):
        penalized = HPAProblem(base, level)
        constrained = HPAProblem(constrained_index(base), level)
        X = rng.random((2, penalized.n_var))
        out_p: dict[str, np.ndarray] = {}
        out_c: dict[str, np.ndarray] = {}
        penalized.evaluate(X, out_p)
        constrained.evaluate(X, out_c)
        assert "G" not in out_p
        weights = [CONSTRAINTS[name][1] for name in hpa_type(base).constraints]
        penalty = np.zeros(len(X))
        for column, weight in enumerate(weights):
            penalty = penalty + weight * np.maximum(0.0, out_c["G"][:, column])
        v = np.array([OBJECTIVES[name][1] for name in hpa_type(base).objectives])
        np.testing.assert_allclose(out_p["F"], out_c["F"] + v * penalty[:, None], rtol=1e-12, atol=1e-12)


def test_rows_of_a_batch_are_evaluated_alone() -> None:
    problem = HPAProblem("241", 1)
    X = np.random.default_rng(5).random((3, problem.n_var))
    batch: dict[str, np.ndarray] = {}
    problem.evaluate(X, batch)
    for row in range(len(X)):
        alone: dict[str, np.ndarray] = {}
        problem.evaluate(X[row : row + 1], alone)
        np.testing.assert_array_equal(alone["F"][0], batch["F"][row])
        np.testing.assert_array_equal(alone["G"][0], batch["G"][row])
    assert np.all(np.isfinite(batch["F"])) and np.all(np.isfinite(batch["G"]))


def test_a_moderate_wing_is_feasible_and_physical() -> None:
    # The centre of HPA241-0's space: a 28 m span, about 260 W at about 8 m/s, within every constraint.
    problem = HPAProblem("241", 0)
    out: dict[str, np.ndarray] = {}
    problem.evaluate(np.full((1, problem.n_var), 0.5), out)
    power, minus_speed = out["F"][0]
    assert 200.0 < power < 400.0
    assert 7.3 < -minus_speed < 12.0
    assert np.all(out["G"][0] <= 0.0)


def test_invalid_problems_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown HPA problem"):
        HPAProblem("101", 0)  # single-objective: not included
    with pytest.raises(ValueError, match="level"):
        HPAProblem("201", 3)
    with pytest.raises(ValueError, match="n_div"):
        HPAProblem("201", 0, n_div=0)


def test_other_wing_segmentations_scale_the_dimension() -> None:
    # |x0| = 3n + 4 + k, |x1| = 7n + 3 + k and |x2| = (2m + 6)n + 3 + k (Section 2.4), here with k = 2 and m = 10.
    for n in (2, 6):
        assert [HPAProblem("201", level, n_div=n).n_var for level in (0, 1, 2)] == [3 * n + 6, 7 * n + 5, 26 * n + 5]


@pytest.mark.parametrize("index", sorted(HPA_INDICES))
def test_the_published_fronts_are_packaged_for_the_penalized_problems_only(index: str) -> None:
    base, constrained = HPA_INDICES[index]
    for level in (0, 1, 2):
        path = resolve_reference_front_path(f"hpa{index}_{level}", None)
        if constrained:
            assert path is None
            continue
        assert path is not None
        front = np.loadtxt(path, delimiter=",", ndmin=2)
        assert front.shape[1] == len(hpa_type(base).objectives)
        assert np.all(np.isfinite(front))


@pytest.fixture
def compiled_kernels() -> Iterator[None]:
    pytest.importorskip("numba")
    yield
    _kernels.set_enabled(True)


@pytest.mark.parametrize("index", sorted(HPA_INDICES))
def test_the_compiled_kernels_give_the_bits_of_the_numpy_code(index: str, compiled_kernels: None) -> None:
    for level in (0, 1, 2):
        problem = HPAProblem(index, level)
        X = np.random.default_rng(10 * int(index) + level).random((2, problem.n_var))
        outputs = []
        for enabled in (True, False):
            _kernels.set_enabled(enabled)
            out: dict[str, np.ndarray] = {}
            problem.evaluate(X, out)
            outputs.append(out)
        assert _kernels.kernels() is None
        np.testing.assert_array_equal(outputs[0]["F"], outputs[1]["F"])
        if problem.n_constraints:
            np.testing.assert_array_equal(outputs[0]["G"], outputs[1]["G"])
