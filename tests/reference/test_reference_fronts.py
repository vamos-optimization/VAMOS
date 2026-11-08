"""Packaged reference fronts: mutual non-dominance, analytic residuals, feasibility and sample quality."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from pathlib import Path

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


CONSTRAINED_DTLZ = ("c1dtlz1", "c1dtlz3", "c2dtlz2", "c3dtlz1", "dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3", "dc3dtlz1", "dc3dtlz3")
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
        for j in range(m - 1):
            i = m - 1 - j
            X[:, j] = 2.0 / np.pi * np.arctan2(F[:, i], np.linalg.norm(F[:, :i], axis=1))
    if name == "c3dtlz1":
        # One distance variable at 0.5 + d gives g = 100 (d^2 - cos(20 pi d) + 1), increasing on [0, 0.05].
        lo, hi = np.zeros(n), np.full(n, 0.05)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = 100.0 * (mid * mid - np.cos(20.0 * np.pi * mid) + 1.0) < g
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        X[:, m - 1] = 0.5 + 0.5 * (lo + hi)
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
    F, G = _evaluate(problem, X)
    F = F[np.all(G <= 0.0, axis=1)]
    assert F.shape[0] >= 100

    # No feasible design dominates the front, and the feasible Pareto-optimal designs lie next to it.
    assert not _dominated_by(F, front).any()
    if name != "c3dtlz1":
        gap, _ = cKDTree(front).query(F)
        assert np.max(gap) <= (0.005 if m == 2 else 0.03)


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
