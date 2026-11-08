"""Packaged reference fronts: mutual non-dominance, analytic residuals, feasibility and sample quality."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.resources import reference_front_path

_FRONTS = resources.files("vamos.resources.reference_fronts")
_PACKAGED = sorted(entry.name[: -len(".csv")] for entry in _FRONTS.iterdir() if entry.name.endswith(".csv"))
# Minimum f1 of ZDT6's front: min over x of 1 - exp(-4x) sin(6 pi x)^6.
_ZDT6_F1_MIN = 0.28077531881536977


def _load(name: str) -> np.ndarray:
    return np.loadtxt(reference_front_path(name), delimiter=",", ndmin=2)


def _dominated_by(A: np.ndarray, B: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Mask of the rows of B that some row of A dominates (minimization)."""
    mask = np.zeros(B.shape[0], dtype=bool)
    for start in range(0, B.shape[0], chunk):
        block = B[start : start + chunk, None, :]
        mask[start : start + chunk] = np.any(np.all(A[None, :, :] <= block, axis=2) & np.any(A[None, :, :] < block, axis=2), axis=1)
    return mask


def _wfg_position(f1: np.ndarray) -> np.ndarray:
    # WFG1 and WFG2 with M = 2: f1 = 2 * (1 - cos(pi x / 2)).
    return np.arccos(np.clip(1.0 - f1 / 2.0, -1.0, 1.0)) * 2.0 / np.pi


def _wfg1(F: np.ndarray) -> np.ndarray:
    x = _wfg_position(F[:, 0])
    return F[:, 1] - 4.0 * (1.0 - x - np.cos(10.0 * np.pi * x + np.pi / 2.0) / (10.0 * np.pi))


def _wfg2(F: np.ndarray) -> np.ndarray:
    x = _wfg_position(F[:, 0])
    return F[:, 1] - 4.0 * (1.0 - x * np.cos(5.0 * np.pi * x) ** 2)


def _dtlz7(F: np.ndarray) -> np.ndarray:
    head = F[:, :-1]
    return F[:, -1] - 2.0 * (F.shape[1] - np.sum(head / 2.0 * (1.0 + np.sin(3.0 * np.pi * head)), axis=1))


def _uf9(F: np.ndarray) -> np.ndarray:
    # CEC2009 UF9: f = (x1 x2, (1 - x1) x2, 1 - x2) with x1 in [0, 1/4] or [3/4, 1].
    off_plane = np.abs(F.sum(axis=1) - 1.0)
    share = F[:, 0] / np.maximum(1.0 - F[:, 2], 1e-300)
    in_gap = (F[:, 2] < 1.0) & (share > 0.25 + 1e-12) & (share < 0.75 - 1e-12)
    return np.where(in_gap, 1.0, off_plane)


def _sum(target: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda F: F.sum(axis=1) - target


def _sphere(F: np.ndarray) -> np.ndarray:
    return np.sum(F * F, axis=1) - 1.0


def _zdt_convex(F: np.ndarray) -> np.ndarray:
    return F[:, 1] - (1.0 - np.sqrt(F[:, 0]))


def _zdt_concave(F: np.ndarray) -> np.ndarray:
    return F[:, 1] - (1.0 - F[:, 0] ** 2)


_RESIDUALS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "zdt1": _zdt_convex,
    "zdt2": _zdt_concave,
    "zdt3": lambda F: F[:, 1] - (1.0 - np.sqrt(F[:, 0]) - F[:, 0] * np.sin(10.0 * np.pi * F[:, 0])),
    "zdt4": _zdt_convex,
    "zdt6": lambda F: np.where(F[:, 0] >= _ZDT6_F1_MIN - 1e-12, _zdt_concave(F), 1.0),
    "dtlz1": _sum(0.5),
    "dtlz2": _sphere,
    "dtlz3": _sphere,
    "dtlz4": _sphere,
    "dtlz5": lambda F: np.maximum(np.abs(_sphere(F)), np.abs(F[:, 0] - F[:, 1])),
    "dtlz6": lambda F: np.maximum(np.abs(_sphere(F)), np.abs(F[:, 0] - F[:, 1])),
    "dtlz7": _dtlz7,
    "wfg1": _wfg1,
    "wfg2": _wfg2,
    "wfg3": lambda F: F[:, 0] / 2.0 + F[:, 1] / 4.0 - 1.0,
    **{f"wfg{i}": (lambda F: (F[:, 0] / 2.0) ** 2 + (F[:, 1] / 4.0) ** 2 - 1.0) for i in range(4, 10)},
    "cec2009_uf1": _zdt_convex,
    "cec2009_uf2": _zdt_convex,
    "cec2009_uf3": _zdt_convex,
    "cec2009_uf4": _zdt_concave,
    "cec2009_uf5": _sum(1.0),
    "cec2009_uf6": _sum(1.0),
    "cec2009_uf7": _sum(1.0),
    "cec2009_uf8": _sphere,
    "cec2009_uf9": _uf9,
    "cec2009_uf10": _sphere,
    **{f"lsmop{i}": _sum(1.0) for i in range(1, 5)},
    **{f"lsmop{i}": _sphere for i in range(5, 9)},
    "lsmop9": lambda F: F[:, 1] - 2.0 * (2.0 - F[:, 0] / 2.0 * (1.0 + np.sin(3.0 * np.pi * F[:, 0]))),
    # C-DTLZ and DC-DTLZ fronts are parts of DTLZ's fronts, except C3-DTLZ1's, which lies on its constraints.
    **{f"{name}.{m}d": _sum(0.5) for name in ("c1dtlz1", "dc1dtlz1", "dc2dtlz1", "dc3dtlz1") for m in (2, 3)},
    **{f"{name}.{m}d": _sphere for name in ("c1dtlz3", "c2dtlz2", "dc1dtlz3", "dc2dtlz3", "dc3dtlz3") for m in (2, 3)},
    **{f"c3dtlz1.{m}d": (lambda F: F.sum(axis=1) + F.min(axis=1) - 1.0) for m in (2, 3)},
    **{f"c3dtlz4.{m}d": (lambda F: np.sum(F * F, axis=1) - 0.75 * np.max(F * F, axis=1) - 1.0) for m in (2, 3)},
}


@pytest.mark.parametrize("name", _PACKAGED)
def test_packaged_reference_fronts_are_mutually_nondominated(name: str) -> None:
    front = _load(name)
    assert np.isfinite(front).all()
    dominated = _dominated_by(front, front)
    assert not dominated.any(), f"{dominated.mean():.1%} of {name} is dominated within the file"


@pytest.mark.parametrize("name", sorted(_RESIDUALS))
def test_analytic_reference_fronts_lie_on_their_fronts(name: str) -> None:
    front = _load(name)
    assert np.max(np.abs(_RESIDUALS[name](front))) <= 1e-9
    # UF5's front is 21 isolated points; every other analytic sample is dense.
    assert front.shape[0] >= (21 if name == "cec2009_uf5" else 1000)


def test_uf6_reference_front_includes_its_isolated_point() -> None:
    # CEC2009 UF6's front is {0} U [1/4, 1/2] U [3/4, 1] in f1, with f2 = 1 - f1.
    front = _load("cec2009_uf6")
    assert np.any(np.all(np.isclose(front, [0.0, 1.0], rtol=0.0, atol=1e-12), axis=1))


CONSTRAINED_DTLZ = (
    "c1dtlz1",
    "c1dtlz3",
    "c2dtlz2",
    "c3dtlz1",
    "c3dtlz4",
    "dc1dtlz1",
    "dc1dtlz3",
    "dc2dtlz1",
    "dc2dtlz3",
    "dc3dtlz1",
    "dc3dtlz3",
)
CONSTRAINED_DTLZ_FRONTS = [(name, m) for name in CONSTRAINED_DTLZ for m in (2, 3)]


def _dtlz_designs(name: str, F: np.ndarray, n_var: int) -> np.ndarray:
    """Designs whose objectives are ``F`` under DTLZ1's (``*dtlz1``) or DTLZ2/3's closed form.

    The distance variables sit at 0.5, where g = 0, except one that moves C3-DTLZ1 out to g = 2 sum(f) - 1.
    """
    n, m = F.shape
    linear = name.endswith("dtlz1")
    g = 2.0 * F.sum(axis=1) - 1.0 if linear else np.linalg.norm(F, axis=1) - 1.0
    X = np.full((n, n_var), 0.5)
    if linear:
        s = F / F.sum(axis=1, keepdims=True)
        rest = np.ones(n)
        for j in range(m - 1):
            X[:, j] = 1.0 - np.divide(s[:, m - 1 - j], rest, out=np.zeros(n), where=rest > 1e-15)
            rest = rest * X[:, j]
    else:
        # DTLZ4 raises its position variables to alpha = 100 before the trigonometric mapping.
        power = 0.01 if name == "c3dtlz4" else 1.0
        for j in range(m - 1):
            i = m - 1 - j
            X[:, j] = (2.0 / np.pi * np.arctan2(F[:, i], np.linalg.norm(F[:, :i], axis=1))) ** power
    if name == "c3dtlz1":
        # One distance variable at 0.5 + d gives g = 100 (d^2 - cos(20 pi d) + 1), increasing on [0, 0.05].
        lo, hi = np.zeros(n), np.full(n, 0.05)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = 100.0 * (mid * mid - np.cos(20.0 * np.pi * mid) + 1.0) < g
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        X[:, m - 1] = 0.5 + 0.5 * (lo + hi)
    elif name == "c3dtlz4":
        # DTLZ2's g, shared by every distance variable.
        X[:, m - 1 :] = 0.5 + np.sqrt(np.maximum(g, 0.0) / (n_var - m + 1))[:, None]
    return np.clip(X, 0.0, 1.0)


def _evaluate(problem: object, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F, G = evaluate_population_with_constraints(problem, X)
    assert G is not None
    return np.asarray(F), np.asarray(G)


@pytest.mark.parametrize(("name", "m"), CONSTRAINED_DTLZ_FRONTS)
def test_constrained_dtlz_fronts_are_reached_by_feasible_designs(name: str, m: int) -> None:
    problem = make_problem_selection(name, n_obj=m).instantiate()
    front = _load(f"{name}.{m}d")
    assert front.shape[1] == m

    F, G = _evaluate(problem, _dtlz_designs(name, front, problem.n_var))

    assert np.max(np.abs(F - front)) <= 1e-9
    # C3-DTLZ1's points lie on a constraint boundary, where G is 0 up to rounding.
    assert np.max(G) <= 1e-9


@pytest.mark.parametrize(("name", "m"), CONSTRAINED_DTLZ_FRONTS)
def test_constrained_dtlz_fronts_cover_their_feasible_pareto_sets(name: str, m: int) -> None:
    problem = make_problem_selection(name, n_obj=m).instantiate()
    front = _load(f"{name}.{m}d")
    rng = np.random.default_rng(m)
    X = np.full((4_000, problem.n_var), 0.5)
    X[:, : m - 1] = rng.random((4_000, m - 1))
    if name == "c3dtlz1":
        # Its Pareto set leaves g = 0: try every distance as well, through one distance variable.
        X[:, m - 1] = 0.5 + 0.05 * rng.random(4_000)
    elif name == "c3dtlz4":
        # Likewise, with DTLZ2's g shared by the distance variables.
        X[:, m - 1 :] = 0.5 + 0.35 * rng.random(4_000)[:, None]
    F, G = _evaluate(problem, X)
    F = F[np.all(G <= 0.0, axis=1)]
    assert F.shape[0] >= 100

    # No feasible design dominates the front, and the feasible Pareto-optimal designs lie next to it.
    assert not _dominated_by(F, front).any()
    if not name.startswith("c3"):
        gap, _ = cKDTree(front).query(F)
        assert np.max(gap) <= (0.005 if m == 2 else 0.03)


def _lircmop_designs(k: int, F: np.ndarray, n_var: int) -> np.ndarray:
    """Designs whose objectives are ``F`` on LIR-CMOP``k``: the position variables from the closed form, and
    the distance terms g1 and g2 (LIR-CMOP13 and 14: g) moved by one variable each."""
    n = F.shape[0]
    X = np.empty((n, n_var))
    odd, even = np.arange(2, n_var, 2), np.arange(1, n_var, 2)
    if k >= 13:
        radius = np.linalg.norm(F, axis=1)
        X[:, 0] = 2.0 / np.pi * np.arctan2(F[:, 2], np.linalg.norm(F[:, :2], axis=1))
        X[:, 1] = 2.0 / np.pi * np.arctan2(F[:, 1], F[:, 0])
        X[:, 2:] = 0.5
        X[:, 2] = 0.5 + np.sqrt(np.maximum(radius - 1.7057, 0.0) / 10.0)
        return X
    concave = k in {1, 3, 6, 8, 9, 12}

    def shape(x: np.ndarray) -> np.ndarray:
        return 1.0 - x * x if concave else 1.0 - np.sqrt(x)

    if k <= 4:
        x0 = F[:, 0] - 0.5
        g1, g2 = np.full(n, 0.5), F[:, 1] - shape(x0)
    elif k <= 8:
        x0 = np.clip(F[:, 0] - 0.7057, 0.0, 1.0)
        g1, g2 = (F[:, 0] - 0.7057 - x0) / 10.0, (F[:, 1] - 0.7057 - shape(x0)) / 10.0
    else:
        # f = 1.7057 * h * (10 g + 1): take g1 = 0 and solve x0 from f1, or, where that leaves g2 < 0 or
        # x0 > 1, take g2 = 0 and solve x0 from f2.

        def g_for(f: np.ndarray, h: np.ndarray) -> np.ndarray:
            return np.divide(f / np.where(h > 0.0, h, 1.0) - 1.0, 10.0, out=np.zeros(n), where=h > 0.0)

        x0 = np.clip(F[:, 0] / 1.7057, 0.0, 1.0)
        g1, g2 = g_for(F[:, 0], 1.7057 * x0), g_for(F[:, 1], 1.7057 * shape(x0))
        level = np.clip(1.0 - F[:, 1] / 1.7057, 0.0, 1.0)
        x0_b = np.sqrt(level) if concave else level**2
        use_b = (g1 < -1e-12) | (g2 < -1e-12) | (F[:, 0] > 1.7057)
        x0 = np.where(use_b, x0_b, x0)
        g1 = np.where(use_b, g_for(F[:, 0], 1.7057 * x0_b), g1)
        g2 = np.where(use_b, 0.0, g2)
    X[:, 0] = x0
    if k <= 4:
        sine, cosine = np.sin(0.5 * np.pi * x0)[:, None], np.cos(0.5 * np.pi * x0)[:, None]
        sine, cosine = np.broadcast_to(sine, (n, odd.size)), np.broadcast_to(cosine, (n, even.size))
    else:
        sine = np.sin(0.5 * (odd + 1) / n_var * np.pi * x0[:, None])
        cosine = np.cos(0.5 * (even + 1) / n_var * np.pi * x0[:, None])
    X[:, odd], X[:, even] = sine, cosine
    for columns, target, g in ((odd, sine, g1), (even, cosine, g2)):
        # One variable away from its target by sqrt(g), towards whichever bound leaves room.
        step = np.sqrt(np.maximum(g, 0.0))
        up = target[:, 0] + step <= 1.0
        X[:, columns[0]] = np.where(up, target[:, 0] + step, target[:, 0] - step)
    return X


@pytest.mark.parametrize("k", range(1, 15))
def test_lircmop_fronts_are_reached_by_feasible_designs(k: int) -> None:
    problem = make_problem_selection(f"lircmop{k}").instantiate()
    front = _load(f"lircmop{k}")

    F, G = _evaluate(problem, _lircmop_designs(k, front, problem.n_var))

    assert np.max(np.abs(F - front)) <= 1e-9
    assert G.max() <= 1e-9


@pytest.mark.parametrize("k", range(1, 10))
def test_dascmop_fronts_avoid_the_infeasible_regions_of_objective_space(k: int) -> None:
    # The constraints on the first variables and on the distance need the design; those on the objectives
    # (rotated ellipses with two objectives, spheres with three) can be checked on the front itself.
    from vamos.foundation.problem import dascmop

    problem = dascmop.DASCMOPProblem(k)
    front = _load(f"dascmop{k}")
    assert front.shape[1] == problem.n_obj
    if problem.n_obj == 2:
        dx, dy = front[:, :1] - np.asarray(dascmop._P), front[:, 1:2] - np.asarray(dascmop._Q)
        u = dx * np.cos(dascmop._THETA) - dy * np.sin(dascmop._THETA)
        v = dx * np.sin(dascmop._THETA) + dy * np.cos(dascmop._THETA)
        outside = u * u / dascmop._SQUARED_A + v * v / dascmop._SQUARED_B - problem._r
    else:
        outside = np.sum((front[:, None, :] - np.asarray(dascmop._CENTERS)[None]) ** 2, axis=2) - problem._r**2
    assert outside.min() >= -1e-8


CLASSICS = ("binh2", "constrex", "srinivas", "tanaka", "osyczka2", "golinski", "water", "car_side_impact")


@pytest.mark.parametrize("name", CLASSICS)
def test_classic_fronts_are_not_dominated_by_random_feasible_designs(name: str) -> None:
    problem = make_problem_selection(name).instantiate()
    rng = np.random.default_rng(len(name))
    X = problem.xl + rng.random((50_000, problem.n_var)) * (problem.xu - problem.xl)
    F, G = _evaluate(problem, X)
    feasible = F[np.all(G <= 0.0, axis=1)]
    front = _load(name)

    assert front.shape[1] == problem.n_obj and feasible.shape[0] >= 100
    assert not _dominated_by(feasible, front).any()


def test_srinivas_front_reaches_the_least_feasible_f1() -> None:
    # The textbook segment x1 = -2.5 starts at f1 = 24.5; the least feasible f1 is 10.1, at (1.1, 3.7).
    front = _load("srinivas")
    assert front[:, 0].min() == pytest.approx(10.1, abs=1e-3)


@pytest.mark.parametrize("name", [name for name, spec in sorted(get_problem_specs().items())])
def test_a_resolved_reference_front_has_the_requested_number_of_objectives(name: str) -> None:
    default = get_problem_specs()[name].default_n_obj
    for n_obj in (None, 2, 3):
        path = resolve_reference_front_path(name, None, n_obj=n_obj)
        if path is not None:
            assert np.loadtxt(path, delimiter=",", ndmin=2).shape[1] == (default if n_obj is None else n_obj)


def test_reference_fronts_are_chosen_by_number_of_objectives() -> None:
    def resolved(name: str, n_obj: int | None) -> str | None:
        path = resolve_reference_front_path(name, None, n_obj=n_obj)
        return None if path is None else Path(path).name

    assert resolved("c1dtlz1", None) == "c1dtlz1.3d.csv"
    assert resolved("c1dtlz1", 2) == "c1dtlz1.2d.csv"
    assert resolved("c1dtlz1", 5) is None
    assert resolved("zcat1", 4) == "zcat1.4d.csv"
    assert resolved("dtlz2", None) == "dtlz2.csv"
    # WFG's packaged front has two objectives, and the problem three by default.
    assert resolved("wfg4", None) is None
    assert resolved("wfg4", 2) == "wfg4.csv"
    assert resolve_reference_front_path("wfg4", "my_front.csv", n_obj=3) == "my_front.csv"


@pytest.mark.parametrize(("name", "seed"), [("rwa3", 3), ("rwa9", 9), ("rwa10", 10), ("re91", 91)])
def test_sampled_reference_fronts_are_not_dominated_by_random_designs(name: str, seed: int) -> None:
    problem = make_problem_selection(name).instantiate()
    xl = np.broadcast_to(np.asarray(problem.xl, dtype=float), (problem.n_var,))
    xu = np.broadcast_to(np.asarray(problem.xu, dtype=float), (problem.n_var,))
    X = xl + np.random.default_rng(seed).random((4_000, problem.n_var)) * (xu - xl)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    front = _load(name)
    assert front.shape[1] == problem.n_obj
    dominated = _dominated_by(out["F"], front)
    assert not dominated.any(), f"{dominated.mean():.1%} of {name} is dominated by random designs"


CEC2009_CF = tuple(range(2, 11))
# Only CF7's front from jMetal lies on the feasible front of ZXH-CF; the others are not packaged.
ZXH_CF = tuple(k for k in range(1, 17) if f"zxh_cf{k}.3d" in _PACKAGED)


def _cec2009_cf_designs(k: int, n_var: int, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Designs on the Pareto set of CEC2009 CF``k``: y_j = 0, except y2 and y4 where CF4-CF7 move the front."""
    X = np.empty((x1.shape[0], n_var))
    X[:, 0] = x1
    if k >= 8:
        X[:, 1] = x2
        j = np.arange(3, n_var + 1)
        X[:, 2:] = 2.0 * x2[:, None] * np.sin(2.0 * np.pi * x1[:, None] + j * np.pi / n_var)
        return X
    j = np.arange(2, n_var + 1)
    angle = 6.0 * np.pi * x1[:, None] + j * np.pi / n_var
    odd = j % 2 == 1
    if k == 2:
        X[:, 1:] = np.where(odd, np.sin(angle), np.cos(angle))
    elif k in (3, 4):
        X[:, 1:] = np.sin(angle)
    else:
        X[:, 1:] = (0.8 * x1[:, None] if k in (5, 6) else 1.0) * np.where(odd, np.cos(angle), np.sin(angle))
    if k in (4, 5):
        # y2 = 0, then on the constraint boundary 0.5 x1 - 0.25, then at the local minimum 1 of h2.
        X[:, 1] += np.where(x1 <= 0.5, 0.0, np.where(x1 <= 0.75, 0.5 * x1 - 0.25, 1.0))
    elif k in (6, 7):
        # y2 and y4 on the boundaries of their constraints where those leave 0 infeasible.
        X[:, 1] += np.sqrt(np.maximum((x1 - 0.5) * (1.0 - x1), 0.0))
        X[:, 3] += np.sqrt(np.maximum(0.25 * np.sqrt(1.0 - x1) - 0.5 * (1.0 - x1), 0.0))
    return X


def _cec2009_cf_x2(k: int, x2: np.ndarray) -> np.ndarray:
    """x2 on the Pareto set of CF8 ({0, 1/3, 1/2, 2/3, 1}) or CF9 and CF10 ([0, 1/3], [1/2, 2/3] and {1})."""
    pieces = np.array(
        [[0.0, 0.0], [1 / 3, 1 / 3], [0.5, 0.5], [2 / 3, 2 / 3], [1.0, 1.0]] if k == 8 else [[0.0, 1 / 3], [0.5, 2 / 3], [1.0, 1.0]]
    )
    nearest = np.clip(x2[:, None], pieces[:, 0], pieces[:, 1])
    return np.asarray(nearest[np.arange(x2.size), np.argmin(np.abs(nearest - x2[:, None]), axis=1)])


def _zxh_cf_designs(problem: Any, F: np.ndarray, n_var: int) -> np.ndarray:
    """Three-objective ZXH-CF designs on the Pareto set (S = 1, h = 0) whose front shape is ``F``.

    ``problem`` is Any: the registry returns a ProblemProtocol, which does not declare ZXH-CF's ``shape``.
    """
    if problem.shape == "linear":
        total = F[:, 1] + F[:, 2]
        theta1 = np.divide(F[:, 2], total, out=np.zeros_like(total), where=total > 0.0)
        a0, a1 = 0.5 * np.pi * (1.0 - F[:, 0]), 0.5 * np.pi * theta1
    else:
        G = F if problem.shape == "concave" else 1.0 - F
        # The mixed shape replaces f1, so the first angle comes from the other two objectives.
        a0 = (
            np.arcsin(np.clip(np.hypot(G[:, 1], G[:, 2]), 0.0, 1.0))
            if problem.shape == "mixed"
            else np.arctan2(np.hypot(G[:, 1], G[:, 2]), G[:, 0])
        )
        a1 = np.arctan2(G[:, 2], G[:, 1])
    X = np.full((F.shape[0], n_var), 0.2)
    X[:, :3] = np.column_stack([np.cos(a0), np.sin(a0) * np.cos(a1), np.sin(a0) * np.sin(a1)])
    return np.clip(X, problem.xl, problem.xu)


@pytest.mark.parametrize("k", CEC2009_CF)
def test_cec2009_cf_fronts_are_reached_by_feasible_designs(k: int) -> None:
    problem = make_problem_selection(f"cec2009_cf{k}").instantiate()
    front = _load(f"cec2009_cf{k}")
    assert front.shape[1] == problem.n_obj
    if k >= 8:
        x1 = 2.0 / np.pi * np.arctan2(front[:, 2], np.hypot(front[:, 0], front[:, 1]))
        x2 = _cec2009_cf_x2(k, 2.0 / np.pi * np.arctan2(front[:, 1], front[:, 0]))
    else:
        x1, x2 = front[:, 0], front[:, 0]

    F, G = _evaluate(problem, _cec2009_cf_designs(k, problem.n_var, x1, x2))

    # jMetal's files keep ten significant digits.
    assert np.max(np.abs(F - front)) <= 1e-9
    assert G.max() <= 1e-9


@pytest.mark.parametrize("k", ZXH_CF)
def test_zxh_cf_fronts_are_reached_by_feasible_designs(k: int) -> None:
    problem = make_problem_selection(f"zxh_cf{k}").instantiate()
    front = _load(f"zxh_cf{k}.3d")
    assert front.shape[1] == problem.n_obj == 3

    F, G = _evaluate(problem, _zxh_cf_designs(problem, front, problem.n_var))

    assert np.max(np.abs(F - front)) <= 1e-9
    assert G.max() <= 1e-9


def _cf_samples(name: str, problem: Any, rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform designs, and as many next to the Pareto set, where uniform designs are seldom feasible."""
    uniform = problem.xl + rng.random((n, problem.n_var)) * (problem.xu - problem.xl)
    near = problem.xl + rng.random((n, problem.n_var)) * (problem.xu - problem.xl)
    noise = 10.0 ** rng.uniform(-5.0, -1.5, size=(n, 1)) * rng.standard_normal((n, problem.n_var))
    if name.startswith("zxh"):
        # Directions of the first M variables scaled to S in [r - 0.05, 1.02], the others near 0.2.
        m, low = problem.n_obj, max(problem.radius - 0.05, 0.0)
        direction = rng.random((n, m)) + 1e-3
        scale = np.sqrt(low + rng.random((n, 1)) * (1.02 - low))
        near[:, :m] = scale * direction / np.linalg.norm(direction, axis=1, keepdims=True)
        near[:, m:] = 0.2 + noise[:, m:]
    else:
        # The Pareto-set designs of the position variables, plus noise on the others.
        k = int(name.removeprefix("cec2009_cf"))
        near = _cec2009_cf_designs(k, problem.n_var, near[:, 0], near[:, 1])
        near[:, problem.n_obj - 1 :] += noise[:, problem.n_obj - 1 :]
    return np.clip(np.vstack([uniform, near]), problem.xl, problem.xu)


@pytest.mark.parametrize("name", [f"cec2009_cf{k}" for k in CEC2009_CF] + [f"zxh_cf{k}" for k in ZXH_CF])
def test_cf_fronts_are_not_dominated_by_random_feasible_designs(name: str) -> None:
    problem = make_problem_selection(name).instantiate()
    F, G = _evaluate(problem, _cf_samples(name, problem, np.random.default_rng(len(name)), 25_000))
    feasible = F[np.all(G <= 0.0, axis=1)]
    front = _load(name if name.startswith("cec") else f"{name}.3d")

    assert front.shape[1] == problem.n_obj and feasible.shape[0] >= 100
    assert not _dominated_by(feasible, front).any()
