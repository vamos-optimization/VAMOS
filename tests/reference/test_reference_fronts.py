"""Packaged reference fronts: mutual non-dominance, analytic residuals, feasibility and sample quality."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources

import numpy as np
import pytest

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.registry import make_problem_selection
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
    "c1dtlz3": _sphere,
    "dc2dtlz1": _sum(0.5),
    "dc2dtlz3": _sphere,
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


@pytest.mark.parametrize("name", ["dc2dtlz1", "dc2dtlz3"])
def test_dc2_reference_fronts_are_feasible(name: str) -> None:
    # DC2's constraints depend only on g, and g = 0 is feasible (both constraints are -0.1 there), so its
    # front is the whole DTLZ front. Pareto-set designs: x_i = 0.5 for the distance variables.
    problem = make_problem_selection(name, n_obj=2).instantiate()
    X = np.full((201, problem.n_var), 0.5)
    X[:, 0] = np.linspace(0.0, 1.0, 201)
    F, G = evaluate_population_with_constraints(problem, X)
    assert G is not None and np.all(G <= 0.0)
    front = _load(name)
    assert front.shape[1] == F.shape[1]
    assert not _dominated_by(F, front).any()


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
