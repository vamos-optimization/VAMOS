"""MODAct (Picard and Schiffmann, 2021): registration, sign conventions, the missing package and the values.

Development after 1.5.0. The ``modact`` package evaluates the problems; CI does not install it, so the tests that
evaluate the real problems skip without it. The others replace it with a stub, or need no evaluation.
"""

from __future__ import annotations

import sys
import types
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.foundation.constraints.utils import is_feasible
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.modact import CONSTRAINT_LEVELS, INSTALL_COMMAND, MODACT, MODActProblem
from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection

Array = np.ndarray[Any, Any]

# The paper's Table III: the objectives of each class (MODAct's weights, 1 maximized) and the constraints of
# each level.
WEIGHTS = {
    "cs": (-1, 1),
    "ct": (-1, 1),
    "cts": (-1, 1, 1),
    "ctse": (-1, 1, 1, 1),
    "ctsei": (-1, 1, 1, 1, -1),
}
N_CONSTRAINTS = {1: 7, 2: 8, 3: 10, 4: 9}
NAMES = [f"{name}{level}" for name in WEIGHTS for level in N_CONSTRAINTS]


def _weights(name: str) -> tuple[int, ...]:
    return WEIGHTS[name.rstrip("1234")]


def test_the_twenty_problems_are_registered_with_the_papers_dimensions() -> None:
    registered = {key for key in get_problem_specs() if key.startswith("modact_")}
    assert registered == {f"modact_{name}" for name in NAMES}
    assert set(MODACT) == set(NAMES) and CONSTRAINT_LEVELS == N_CONSTRAINTS
    for name in NAMES:
        spec = get_problem_specs()[f"modact_{name}"]
        n_obj, n_constraints = MODACT[name]
        assert (spec.default_n_var, spec.default_n_obj, spec.encoding) == (20, len(_weights(name)), "real")
        assert n_obj == len(_weights(name)) and n_constraints == N_CONSTRAINTS[int(name[-1])]
        assert not spec.allow_n_obj_override and spec.fixed_n_var
        assert spec.option_defaults() == {}
        assert "modact" in spec.description and "GPL-3.0" in spec.description
        assert f"{n_constraints} constraints" in spec.description


@pytest.mark.parametrize("name", ["cs1", "ctsei4"])
def test_dimensions_are_fixed(name: str) -> None:
    n_obj = len(_weights(name))
    with pytest.raises(ValueError, match="fixed number of variables"):
        make_problem_selection(f"modact_{name}", n_var=21)
    with pytest.raises(ValueError, match="fixed number of objectives"):
        make_problem_selection(f"modact_{name}", n_obj=n_obj + 1)


@pytest.mark.parametrize("name", ["cs5", "c1", "cts", "modact_cs1", "cs1s2", ""])
def test_unknown_names_are_rejected_before_importing_modact(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # C5 gathers every constraint "only for analysis", and "s2" changes the number of gear stages: neither is one
    # of the 20 problems.
    monkeypatch.setitem(sys.modules, "modact", None)
    with pytest.raises(ValueError, match="Unknown MODAct problem"):
        MODActProblem(name)


def test_a_missing_modact_package_raises_import_error_naming_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "modact", None)
    monkeypatch.setitem(sys.modules, "modact.problems", None)

    with pytest.raises(ImportError, match="GPL-3.0") as caught:
        MODActProblem("cs1")
    assert INSTALL_COMMAND in str(caught.value) and "vamos-optimization[modact]" in INSTALL_COMMAND
    # Through the registry and the public facade too: the registry-wide tests skip a problem on ImportError.
    with pytest.raises(ImportError, match=r"vamos-optimization\[modact\]"):
        make_problem_selection("modact_ct2").instantiate()

    import vamos

    with pytest.raises(ImportError, match=r"vamos-optimization\[modact\]"):
        vamos.optimize("modact_cts1", algorithm="nsgaii", max_evaluations=100, seed=1)


# ---------------------------------------------------------------------------------------------------------
# A stub of modact.problems: the conversion, without the package
# ---------------------------------------------------------------------------------------------------------


class _StubProblem:
    """A problem with modact's interface: ``bounds()``, ``weights``, ``c_weights`` and ``__call__(x) -> (f, g)``."""

    def __init__(self, n_var: int, weights: tuple[int, ...], c_weights: tuple[int, ...]) -> None:
        self.n_var = n_var
        self.weights = weights
        self.c_weights = c_weights
        self.calls = 0

    def bounds(self) -> tuple[Array, Array]:
        return -np.ones(self.n_var), np.ones(self.n_var)

    def __call__(self, x: Array) -> tuple[list[float], tuple[float, ...]]:
        self.calls += 1
        f = [float(x[j] + 10.0 * j) for j in range(len(self.weights))]
        g = tuple(float(x[k]) for k in range(len(self.c_weights)))
        x[:] = 0.0  # The wrapper passes a copy: the caller's population must not change.
        return f, g


def _install_stub(monkeypatch: pytest.MonkeyPatch, problems: dict[str, _StubProblem]) -> None:
    package = types.ModuleType("modact")
    module = types.ModuleType("modact.problems")
    module.get_problem = problems.__getitem__  # type: ignore[attr-defined]
    package.problems = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modact", package)
    monkeypatch.setitem(sys.modules, "modact.problems", module)


def test_objectives_and_constraints_take_vamos_signs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two constraints g >= 0 (c_weight -1) and five g <= 0 (c_weight 1); the second objective is maximized.
    c_weights = (-1, -1, 1, 1, 1, 1, 1)
    stub = _StubProblem(20, (-1, 1), c_weights)
    _install_stub(monkeypatch, {"ct1": stub})
    problem = MODActProblem("CT1")
    X = np.random.default_rng(0).uniform(-1.0, 1.0, (64, 20))
    X_before = X.copy()

    out: dict[str, Array] = {}
    problem.evaluate(X, out)

    np.testing.assert_array_equal(X, X_before)
    assert stub.calls == 64
    raw_f = np.column_stack([X[:, 0], X[:, 1] + 10.0])
    raw_g = X[:, :7]
    np.testing.assert_array_equal(out["F"], np.column_stack([raw_f[:, 0], -raw_f[:, 1]]))
    np.testing.assert_array_equal(out["G"], raw_g * np.array(c_weights))
    # modact's own rule: c_weight -1 means g >= 0 and 1 means g <= 0.
    modact_feasible = np.all(np.where(np.array(c_weights) < 0, raw_g >= 0.0, raw_g <= 0.0), axis=1)
    np.testing.assert_array_equal(is_feasible(out["G"]), modact_feasible)
    assert problem.key == "modact_ct1" and problem.modact_name == "ct1"
    np.testing.assert_array_equal(problem.objective_signs, [1.0, -1.0])
    np.testing.assert_array_equal(problem.constraint_signs, c_weights)
    np.testing.assert_array_equal(problem.xl, -np.ones(20))
    np.testing.assert_array_equal(problem.xu, np.ones(20))


def test_a_population_is_checked_at_the_problem_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch, {"cs1": _StubProblem(20, (-1, 1), (-1,) * 7)})
    problem = MODActProblem("cs1")
    with pytest.raises(ValueError, match=r"shape \(N, 20\)"):
        problem.evaluate(np.zeros(20), {})
    with pytest.raises(ValueError, match=r"shape \(N, 20\)"):
        problem.evaluate(np.zeros((3, 19)), {})


@pytest.mark.parametrize(
    ("stub", "message"),
    [
        (_StubProblem(14, (-1, 1), (-1,) * 7), "14 variables"),
        (_StubProblem(20, (-1, 1, 1), (-1,) * 7), "3 objectives"),
        (_StubProblem(20, (-1, 1), (-1,) * 8), "8 constraints"),
        (_StubProblem(20, (0, 1), (-1,) * 7), "weights"),
        (_StubProblem(20, (-1, 1), (-1, -1, -1, -1, -1, -1, 2)), "c_weights"),
    ],
)
def test_a_modact_that_defines_another_problem_is_rejected(stub: _StubProblem, message: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch, {"cs1": stub})
    with pytest.raises(ValueError, match=message):
        MODActProblem("cs1")


# ---------------------------------------------------------------------------------------------------------
# The real problems: VAMOS's values are modact's, with the documented signs
# ---------------------------------------------------------------------------------------------------------

# One design per constraint level that is feasible for CS<level>, found by NSGA-II with modact 1.0.1. Every
# constraint but the minimum torque excess is the same for the five classes, and that of CS is the strictest, so
# each design is feasible for the five problems of its level.
FEASIBLE: dict[int, list[float]] = {
    1: [
        0.3131018613781791,
        0.5156815449143909,
        12.85133696767489,
        76.7811593924199,
        0.5066856180328065,
        9.741893647161701,
        7.276819907437618,
        0.21074695458853188,
        22.11733467721581,
        61.3274951726781,
        0.4999936275952548,
        13.373319523010037,
        -10.10416689745418,
        0.15918349419308508,
        20.776320282325567,
        78.2096914243699,
        0.7161779675118337,
        10.275216725709283,
        8.945164277987864,
        2.6666886411580784,
    ],
    2: [
        1.582287921998699,
        0.8119865735638275,
        27.12949641698943,
        80.69403684007466,
        0.7135442100950694,
        12.46694585852952,
        8.772828354744668,
        -1.8233499327000873,
        25.092653435507163,
        73.09371267914617,
        0.9411193491013455,
        13.074327507891569,
        -6.52809316193986,
        1.3516701912107263,
        21.278213371560696,
        71.04992873168115,
        0.9926008949499178,
        14.037084189033408,
        -3.6188165502305205,
        1.0797549408576432,
    ],
    3: [
        2.637148477358137,
        0.6971720400589662,
        20.404229758066823,
        72.59972310403255,
        0.4139818674761613,
        11.186542607394369,
        0.7043413282132602,
        2.6474616532404966,
        26.993404935969476,
        51.31182622157317,
        0.42860090694404923,
        13.863134862352283,
        -16.868777638812382,
        1.869241503993951,
        17.54874647947844,
        69.99747758755065,
        0.6822527019698094,
        14.046735759376077,
        0.052343593490172355,
        -1.8446312738763089,
    ],
    4: [
        3.3390637418138422,
        1.5548886727694153,
        17.79492860051622,
        55.86217042003782,
        0.6437163456941235,
        9.850089429316835,
        1.297666007490067,
        -0.3967029551970824,
        28.717753257780075,
        70.66398690360874,
        0.5633730442111005,
        7.780720246048205,
        2.317592162220265,
        2.8767504584100734,
        32.90460213409245,
        69.05921378280617,
        0.8119219110539764,
        6.00903403525761,
        2.977829997345469,
        -2.672857629982656,
    ],
}


def _modact_problems() -> types.ModuleType:
    return pytest.importorskip("modact.problems", reason="the modact extra is not installed", exc_type=ImportError)


def _designs(name: str, problem: Any, n: int) -> Array:
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    X = problem.xl + rng.random((n, problem.n_var)) * (problem.xu - problem.xl)
    return np.vstack([X, np.asarray(FEASIBLE[int(name[-1])])])


@pytest.mark.parametrize("name", NAMES)
def test_values_are_modacts_with_the_documented_signs(name: str) -> None:
    modact = _modact_problems().get_problem(name)
    problem = make_problem_selection(f"modact_{name}").instantiate()
    assert isinstance(problem, MODActProblem)
    lower, upper = modact.bounds()
    np.testing.assert_array_equal(problem.xl, lower)
    np.testing.assert_array_equal(problem.xu, upper)
    assert tuple(modact.weights) == _weights(name)
    X = _designs(name, problem, 4)

    F, G = evaluate_population_with_constraints(problem, X)

    assert G is not None
    raw = [modact(x) for x in X]
    raw_f = np.array([f for f, _ in raw], dtype=float)
    raw_g = np.array([g for _, g in raw], dtype=float)
    np.testing.assert_array_equal(F, raw_f * -np.array(modact.weights))
    np.testing.assert_array_equal(G, raw_g * np.array(modact.c_weights))
    c_weights = np.array(modact.c_weights)
    modact_feasible = np.all(np.where(c_weights < 0, raw_g >= 0.0, raw_g <= 0.0), axis=1)
    np.testing.assert_array_equal(is_feasible(G), modact_feasible)
    # The design of the level is feasible.
    assert modact_feasible[-1]


def test_a_modact_run_is_stored_with_the_package_version_and_replays_exactly(tmp_path: Path) -> None:
    _modact_problems()
    import json
    from importlib import metadata

    import vamos

    result = vamos.optimize("modact_ct1", algorithm="nsgaii", max_evaluations=60, pop_size=20, seed=3)

    assert result.F.shape == (20, 2) and result.X.shape == (20, 20)
    assert np.isfinite(result.F).all()
    source = vamos.save_result(result, tmp_path / "source")
    environment = json.loads((Path(source.root) / "environment.json").read_text(encoding="utf-8"))
    assert environment["packages"]["modact"] == metadata.version("modact")
    assert vamos.verify_run(source.root).effective_replayability == "exact"
    assert vamos.reproduce(source.root, output=tmp_path / "replay").exact is True
