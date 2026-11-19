"""Regenerate VAMOS's packaged reference fronts in ``src/vamos/resources/reference_fronts``.

Run it from the repository root with this checkout's ``src`` importable and the ``compute`` extra installed
(the exact non-dominated filter uses Numba):

    python experiments/scripts/generate_reference_fronts.py --all --reproblems-dir PATH --jmetal-dir PATH --modact-dir PATH
    python experiments/scripts/generate_reference_fronts.py zdt3 dtlz7 wfg1

The script writes three groups of files; ``docs/reference/problems.md`` ("Reference fronts") lists the
provenance of every packaged file, including those this script leaves alone.

- **Analytic samples** of the Pareto front from its closed form: ZDT1-4 and ZDT6, DTLZ1-7 (M = 3, and M = 2 as
  ``dtlz1.2d`` ... ``dtlz7.2d``; VAMOS 1.5.0: DTLZ1, DTLZ2 and DTLZ7 at M = 4, 6 and 8 as ``dtlz1.4d``
  ... ``dtlz7.8d``, DTLZ2's also serving DTLZ3 and DTLZ4), IDTLZ1 and IDTLZ2 at M = 2 and 3 (VAMOS
  1.5.0: ``idtlz1.2d`` ... ``idtlz2.3d``), WFG1 and
  WFG2 (M = 2, from the WFG shape functions, so the bias transformations cannot shift them), LSMOP9 (M = 2),
  CEC2009 UF6 and UF9, and the ten C-DTLZ and DC-DTLZ problems at M = 2 and 3 (``c1dtlz1.2d`` ...
  ``dc3dtlz3.3d``), Srinivas, ConstrEx and Tanaka from their Pareto-set pieces and constraint boundaries, ZXH-CF1-16 at
  M = 3 from a grid of angles on their Pareto set, and CTP1-CTP7 (``ctp1`` ... ``ctp7``) from the unconstrained
  front at g = 1 and the constraint boundaries. Disconnected fronts are filtered for non-dominance on a fine grid and thinned by
  farthest-point selection. The C-DTLZ and DC-DTLZ samples keep the points of DTLZ's front (C3-DTLZ1: of its
  constraint boundary) at which VAMOS's problem, evaluated at a design rebuilt from the point, is feasible.
- **The RE authors' fronts:** Tanabe and Ishibuchi's ``reference_points_RE*.dat`` from
  https://github.com/ryojitanabe/reproblems (``approximated_Pareto_fronts``), with only the separator changed.
  ``--reproblems-dir`` points at a local copy of that directory. Their MIT notice is
  ``LICENSE-reproblems.txt`` next to the CSV files.
- **jMetal's fronts** for LIR-CMOP1-14, DAS-CMOP1-9, MW4, MW8-14, Binh2, Golinski, Osyczka2, Water, the car side
  impact problem, CEC2009 CF2-CF10, CRE21-CRE51 and RCM01-RCM50 (except RCM17, RCM23, RCM26 and RCM50), and the
  optional jMetal source of ZDT1-4, ZDT6 and DTLZ1-7 at M = 2 and 3 (``jmetal-zdt1.csv``, ``jmetal-dtlz2.3d.csv``,
  which also serves DTLZ3 and DTLZ4), from
  jMetal's ``resources/referenceFronts`` at commit ``JMETAL_COMMIT``, copied line by line without the rows that
  another row of the same file dominates. ``--jmetal-dir`` points at a local copy of that directory. Their MIT
  notice is ``LICENSE-jmetal.txt``.
- **MODAct's best-known fronts** (Development after 1.5.0) for the two-objective MODAct problems, CS1-CS4 and
  CT1-CT4 (``modact_cs1.csv`` ... ``modact_ct4.csv``): Picard and Schiffmann's ``<name>_PF.dat`` of the Zenodo
  record https://doi.org/10.5281/zenodo.3824302, version 1.1.0, whose MD5 checksums are checked. Each value keeps
  its digits, and the columns that MODAct maximizes change sign, so that every column is minimized as in VAMOS's
  problems. ``--modact-dir`` points at a local copy of the files. Their CC BY 4.0 notice is
  ``LICENSE-modact-fronts.txt``. The fronts of the problems with three to five objectives (1.4 MB to 193 MB
  each) are not packaged.
- **Sampled fronts** where no closed form or authors' front applies: RWA3, RWA9, RWA10 and the deterministic
  RE91. Dense sampling of the design box, local refinement around the non-dominated designs and an exact
  non-dominated filter over every evaluated design. A well-spread pool of 1.25 times the packaged size is then
  moved to the front by minimizing the augmented achievement function relative to each candidate (three
  passes), the polished designs that stay non-dominated are kept, and farthest-point selection thins them to
  the packaged size.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "src" / "vamos" / "resources" / "reference_fronts"
SEED = 20261030

# Files keep their historical names (ZDT1 is upper case).
FILE_NAMES = {"zdt1": "ZDT1"}

# Tanabe and Ishibuchi's approximated fronts that match VAMOS's RE formulations. RE91 is not here: VAMOS
# evaluates it at the mean of its stochastic parameters, so the authors' stochastic front does not apply.
AUTHORS_RE = (21, 24, 31, 32, 33, 34, 37, 41, 42, 61)

# jMetal's fronts (MIT, LICENSE-jmetal.txt): the file of jMetal's resources/referenceFronts for each problem.
JMETAL_COMMIT = "1e4b83cae6"
JMETAL_FRONTS = {
    **{f"lircmop{k}": f"LIRCMOP{k}" for k in range(1, 15)},
    **{f"dascmop{k}": f"DASCMOP{k}" for k in range(1, 7)},
    **{f"dascmop{k}": f"DASCMOP{k}.3D" for k in range(7, 10)},
    **{f"mw{k}": f"MW{k}" for k in range(9, 14)},
    **{f"mw{k}.{m}d": f"MW{k}.{m}D" for k in (4, 8, 14) for m in (2, 3)},
    "binh2": "Binh2",
    "golinski": "Golinski",
    "osyczka2": "Osyczka2",
    "water": "Water",
    "car_side_impact": "CarSideImpact",
    **{f"cec2009_cf{k}": f"CEC2009CF{k}" for k in range(2, 11)},
    **{f"cre{k}": f"CRE{k}" for k in (21, 22, 23, 24, 25, 31, 32, 51)},
    # RCM17 has no front (its first objective is unbounded below). Feasible designs dominate rows of jMetal's
    # RCM23, RCM26 and RCM50 fronts (docs/reference/problems.md, "Reference fronts"), so they are not packaged.
    **{f"rcm{k:02d}": f"RCM{k:02d}" for k in range(1, 51) if k not in (17, 23, 26, 50)},
    # The optional jMetal source of ZDT and DTLZ (resolve_reference_front_path(..., source="jmetal")); VAMOS's
    # analytic samples of these problems stay the default. jMetal's DTLZ3 and DTLZ4 files are its DTLZ2 file,
    # byte for byte, so only DTLZ2's is packaged and the resolver gives it for the three problems.
    **{f"jmetal-zdt{k}": f"ZDT{k}" for k in (1, 2, 3, 4, 6)},
    **{f"jmetal-dtlz{k}.{m}d": f"DTLZ{k}.{m}D" for k in (1, 2, 5, 6, 7) for m in (2, 3)},
}

# MODAct's best-known fronts (CC BY 4.0, LICENSE-modact-fronts.txt): the MD5 checksum of each file of the Zenodo
# record 10.5281/zenodo.3824302, version 1.1.0, as the record lists it.
MODACT_ZENODO = "10.5281/zenodo.3824302"
MODACT_FRONTS = {
    "modact_cs1": "4990ad4b4fdb1e69bcd8c783194c4786",
    "modact_cs2": "3ee656a1337a8741f3563d0870ad7333",
    "modact_cs3": "dd70ee28cc5f72c2561dba9c6d235e89",
    "modact_cs4": "bb5dc7ac5b66453a01a4a435de5d099d",
    "modact_ct1": "6e4550df32881f98cb15af16635afcab",
    "modact_ct2": "797cd321c1f3712811914d6fc3faef61",
    "modact_ct3": "6331fe3b000d5d3f3487b13998b3cacc",
    "modact_ct4": "791662a99be6222a8d7a4750dae84974",
}

# Sampled fronts: packaged size, number of uniform samples and grid points per variable (0 = no grid).
SAMPLED = {
    "rwa3": (1484, 200_000, 81),
    "rwa9": (1343, 500_000, 0),
    "rwa10": (1493, 100_000, 61),
    "re91": (1806, 300_000, 0),
}


# ---------------------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------------------


def nondominated_2d(F: np.ndarray) -> np.ndarray:
    """Non-dominated rows of a two-objective set, sorted by f1."""
    order = np.lexsort((F[:, 1], F[:, 0]))
    F = F[order]
    keep = np.zeros(F.shape[0], dtype=bool)
    best = np.inf
    for index, f2 in enumerate(F[:, 1]):
        if f2 < best:
            keep[index] = True
            best = f2
    return F[keep]


def farthest_points(F: np.ndarray, k: int, first: tuple[int, ...] = ()) -> np.ndarray:
    """Indices of ``k`` well-spread rows: greedy farthest-point selection in normalized objective space, from the
    rows ``first`` and the least row of each objective."""
    lo, hi = F.min(axis=0), F.max(axis=0)
    Z = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    chosen = list(dict.fromkeys([*first, *(int(np.argmin(F[:, j])) for j in range(F.shape[1]))]))
    distance = np.min(np.linalg.norm(Z[:, None, :] - Z[None, chosen, :], axis=2), axis=1)
    while len(chosen) < min(k, F.shape[0]):
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(Z - Z[nxt], axis=1))
    return np.sort(np.array(chosen))


def simplex_lattice(m: int, h: int) -> np.ndarray:
    """Das-Dennis lattice on the unit simplex: all points with coordinates i/h summing to 1."""
    rows = [c for c in itertools.product(range(h + 1), repeat=m - 1) if sum(c) <= h]
    counts = np.array([(*c, h - sum(c)) for c in rows], dtype=float)
    return counts / h


def thinned_2d(F: np.ndarray, k: int) -> np.ndarray:
    front = nondominated_2d(F)
    return front[farthest_points(front, k)]


# ---------------------------------------------------------------------------------------------------------
# Analytic samples
# ---------------------------------------------------------------------------------------------------------


def _zdt6_f1_min() -> float:
    """min over x of 1 - exp(-4x) sin(6 pi x)^6, by golden-section search on its first trough."""

    def f(x: float) -> float:
        return float(1.0 - np.exp(-4.0 * x) * np.sin(6.0 * np.pi * x) ** 6)

    a, b = 0.05, 0.12
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        if f(c) < f(d):
            b = d
        else:
            a = c
    return f(0.5 * (a + b))


def _zdt(name: str) -> np.ndarray:
    lower = _zdt6_f1_min() if name == "zdt6" else 0.0
    f1 = np.linspace(lower, 1.0, 200_001)
    if name == "zdt3":
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
    elif name in {"zdt2", "zdt6"}:
        f2 = 1.0 - f1**2
    else:
        f2 = 1.0 - np.sqrt(f1)
    return thinned_2d(np.column_stack([f1, f2]), 1000)


def _dtlz(name: str) -> np.ndarray:
    if name in {"dtlz5", "dtlz6"}:
        # M = 3: the front is a curve, f1 = f2 = cos(theta) / sqrt(2), f3 = sin(theta).
        theta = np.linspace(0.0, np.pi / 2.0, 1000)
        return np.column_stack([np.cos(theta) / np.sqrt(2.0), np.cos(theta) / np.sqrt(2.0), np.sin(theta)])
    if name == "dtlz7":
        grid = np.linspace(0.0, 1.0, 201)
        head = np.array(list(itertools.product(grid, repeat=2)))
        f3 = 2.0 * (3.0 - np.sum(head / 2.0 * (1.0 + np.sin(3.0 * np.pi * head)), axis=1))
        front = _nondominated(np.column_stack([head, f3]))
        return front[farthest_points(front, 5050)]
    w = simplex_lattice(3, 99)
    return 0.5 * w if name == "dtlz1" else w / np.linalg.norm(w, axis=1, keepdims=True)


def _dtlz_two_objective(name: str) -> np.ndarray:
    """DTLZ1-7 with M = 2, 1000 points spaced evenly along the front.

    DTLZ1's front is the segment f1 + f2 = 0.5, DTLZ7's the non-dominated pieces of f2 = 4 - f1 (1 + sin(3 pi f1)),
    and the others' the quarter circle: with M = 2, DTLZ5 and DTLZ6 do not degenerate.
    """
    if name == "dtlz1":
        f1 = np.linspace(0.0, 0.5, 1000)
        return np.column_stack([f1, 0.5 - f1])
    if name == "dtlz7":
        f1 = np.linspace(0.0, 1.0, 200_001)
        return thinned_2d(np.column_stack([f1, 4.0 - f1 * (1.0 + np.sin(3.0 * np.pi * f1))]), 1000)
    theta = np.linspace(0.0, np.pi / 2.0, 1000)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _idtlz(name: str, m: int) -> np.ndarray:
    """VAMOS 1.5.0: the inverted DTLZ1 and DTLZ2 of Jain and Deb (2014), as PlatEMO's ``GetOptimum``.

    IDTLZ1's front is ``(1 - w) / 2`` and IDTLZ2's ``1 - w / |w|``, for ``w`` on the unit simplex: with M = 2, 1000
    points spaced evenly along the front, and with M = 3 the 5050-point simplex lattice of DTLZ1-DTLZ4's samples.
    """
    if m == 2:
        if name == "idtlz1":
            f1 = np.linspace(0.0, 0.5, 1000)
            return np.column_stack([f1, 0.5 - f1])
        theta = np.linspace(0.0, np.pi / 2.0, 1000)
        return np.column_stack([1.0 - np.cos(theta), 1.0 - np.sin(theta)])
    w = simplex_lattice(m, 99)
    return (1.0 - w) / 2.0 if name == "idtlz1" else 1.0 - w / np.linalg.norm(w, axis=1, keepdims=True)


# VAMOS 1.5.0: DTLZ1, DTLZ2 (whose sample DTLZ3 and DTLZ4 share) and DTLZ7 with more objectives.
MANY_OBJECTIVES = (4, 6, 8)
# Per objective count, the values per position objective of DTLZ7's candidate grid: about 10^5 candidates in all.
_DTLZ7_GRID = {4: 48, 6: 10, 8: 6}


def lattice_divisions(m: int, limit: int = 10_000) -> int:
    """The divisions H of the largest Das-Dennis lattice with ``m`` objectives and at most ``limit`` points."""
    h = 1
    while math.comb(h + m, m - 1) <= limit:
        h += 1
    return h


def _dtlz_many_objective(name: str, m: int) -> np.ndarray:
    """DTLZ1, DTLZ2 and DTLZ7 with M = 4, 6 or 8.

    DTLZ1 and DTLZ2: the largest Das-Dennis lattice with at most 10 000 points (H = 37, 13 and 8: 9880, 8568 and
    6435 points), scaled to ``sum(f) = 0.5`` or projected onto the unit sphere. DTLZ7: on the front, the last
    objective is ``2 (M - sum_i psi(f_i))`` over the others, with ``psi(t) = t (1 + sin(3 pi t))``. It separates, so a
    point is Pareto-optimal when each other objective is a value that no smaller one beats in ``psi``: the front is
    the product of those values, which a grid of 200 001 points of [0, 1] gives. Evenly spaced values of them,
    combined, are thinned to 5050 points by farthest-point selection.
    """
    if name == "dtlz7":
        fine = np.linspace(0.0, 1.0, 200_001)
        psi = fine * (1.0 + np.sin(3.0 * np.pi * fine))
        efficient = fine[psi > np.maximum.accumulate(np.concatenate([[-np.inf], psi[:-1]]))]
        values = efficient[np.round(np.linspace(0.0, efficient.size - 1, _DTLZ7_GRID[m])).astype(int)]
        head = np.array(list(itertools.product(values, repeat=m - 1)))
        last = 2.0 * (m - np.sum(head / 2.0 * (1.0 + np.sin(3.0 * np.pi * head)), axis=1))
        front = np.column_stack([head, last])
        return front[farthest_points(front, 5050)]
    w = simplex_lattice(m, lattice_divisions(m))
    return 0.5 * w if name == "dtlz1" else w / np.linalg.norm(w, axis=1, keepdims=True)


def _wfg(name: str) -> np.ndarray:
    # M = 2, S = (2, 4); on the front the distance parameter is 0 and f = S * h(x).
    x = np.linspace(0.0, 1.0, 200_001)
    f1 = 2.0 * (1.0 - np.cos(0.5 * np.pi * x))
    if name == "wfg1":
        h2 = 1.0 - x - np.cos(10.0 * np.pi * x + 0.5 * np.pi) / (10.0 * np.pi)
    else:
        h2 = 1.0 - x * np.cos(5.0 * np.pi * x) ** 2
    return thinned_2d(np.column_stack([f1, 4.0 * h2]), 10_000)


def _lsmop9() -> np.ndarray:
    # M = 2: f1 = x, f2 = (1 + g) * (2 - x / (1 + g) * (1 + sin(3 pi x))) with 1 + g = 2 on the Pareto set.
    x = np.linspace(0.0, 1.0, 200_001)
    return thinned_2d(np.column_stack([x, 2.0 * (2.0 - x / 2.0 * (1.0 + np.sin(3.0 * np.pi * x)))]), 1000)


def _uf6() -> np.ndarray:
    # f2 = 1 - f1 on {0} U [1/4, 1/2] U [3/4, 1].
    f1 = np.concatenate([[0.0], np.linspace(0.25, 0.5, 500), np.linspace(0.75, 1.0, 500)])
    return np.column_stack([f1, 1.0 - f1])


def _uf9() -> np.ndarray:
    # f = (x1 x2, (1 - x1) x2, 1 - x2) with x1 in [0, 1/4] U [3/4, 1]: the plane sum(f) = 1 minus its middle.
    h = 140
    counts = np.round(simplex_lattice(3, h) * h).astype(int)
    i, j = counts[:, 0], counts[:, 1]
    keep = (4 * i <= i + j) | (4 * i >= 3 * (i + j))
    return counts[keep] / h


# The C-DTLZ problems of Jain and Deb (2014) and the DC-DTLZ problems of Li et al. (2019), at M = 2 and 3.
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


def _simplex_directions(m: int) -> np.ndarray:
    """Dense directions on the unit simplex: 200 001 for M = 2 and a lattice of 45 150 for M = 3."""
    if m == 2:
        u = np.linspace(0.0, 1.0, 200_001)
        return np.column_stack([u, 1.0 - u])
    return simplex_lattice(3, 299)


def _constrained_dtlz_candidates(name: str, m: int) -> np.ndarray:
    """Objective vectors that contain the front: DTLZ's own front, or C3-DTLZ1's constraint boundary."""
    s = _simplex_directions(m)
    if name == "c3dtlz1":
        # sum(f) - f_j + 2 f_j >= 1 for every j moves the front out to f = t s with t = 1 / (1 + min_j s_j).
        return s / (1.0 + s.min(axis=1, keepdims=True))
    if name == "c3dtlz4":
        # |f|^2 - 3 f_j^2 / 4 >= 1 for every j moves the unit sphere out to f = u / sqrt(1 - 3 max_j u_j^2 / 4).
        u = s / np.linalg.norm(s, axis=1, keepdims=True)
        return u / np.sqrt(1.0 - 0.75 * np.max(u * u, axis=1, keepdims=True))
    if name.endswith("dtlz1"):
        return 0.5 * s
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def _dtlz1_distance(g: np.ndarray) -> np.ndarray:
    """The offset d in [0, 0.05] at which DTLZ1's g, with one distance variable at 0.5 + d, equals ``g``."""
    lo, hi = np.zeros_like(g), np.full_like(g, 0.05)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = 100.0 * (mid * mid - np.cos(20.0 * np.pi * mid) + 1.0) < g
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def constrained_dtlz_designs(name: str, F: np.ndarray, n_var: int) -> np.ndarray:
    """Designs whose objectives are ``F``: the position variables invert DTLZ's closed form, and the distance
    variables sit at 0.5 (g = 0), except one that gives C3-DTLZ1 its g = 2 sum(f) - 1."""
    n, m = F.shape
    linear = name.endswith("dtlz1")
    if linear:
        g = 2.0 * F.sum(axis=1) - 1.0
        s = F / F.sum(axis=1, keepdims=True)
    else:
        g = np.linalg.norm(F, axis=1) - 1.0
        s = F / (1.0 + g)[:, None]
    # DTLZ4 raises its position variables to alpha = 100 before the trigonometric mapping.
    power = 1.0 / 100.0 if name == "c3dtlz4" else 1.0
    X = np.full((n, n_var), 0.5)
    # f_M depends on x_1 only, f_{M-1} on x_1 and x_2, and so on: peel one position variable at a time.
    rest = np.ones(n)
    for j in range(m - 1):
        i = m - 1 - j
        if linear:
            x = 1.0 - np.divide(s[:, i], rest, out=np.zeros(n), where=rest > 1e-15)
            rest = rest * x
        else:
            # tan(pi x / 2) = f_i / |(f_1, ..., f_{i-1})|, which atan2 inverts without losing precision.
            x = (2.0 / np.pi * np.arctan2(s[:, i], np.linalg.norm(s[:, :i], axis=1))) ** power
        X[:, j] = np.clip(x, 0.0, 1.0)
    if name == "c3dtlz1":
        X[:, m - 1] = 0.5 + _dtlz1_distance(np.maximum(g, 0.0))
    elif name == "c3dtlz4":
        # DTLZ2's g = sum((x - 0.5)^2), shared by every distance variable so that each stays in [0, 1].
        X[:, m - 1 :] = 0.5 + np.sqrt(np.maximum(g, 0.0) / (n_var - m + 1))[:, None]
    return X


def _constrained_dtlz(name: str, m: int) -> np.ndarray:
    """The feasible, non-dominated part of the candidates, as VAMOS's problem evaluates them: 1000 points
    for M = 2 and at most 5050 for M = 3."""
    from vamos.foundation.problem.registry import make_problem_selection

    problem = make_problem_selection(name, n_obj=m).instantiate()
    F = _constrained_dtlz_candidates(name, m)
    out = {"F": np.empty_like(F)}
    problem.evaluate(constrained_dtlz_designs(name, F, problem.n_var), out)
    if np.max(np.abs(out["F"] - F)) > 1e-12:
        raise RuntimeError(f"{name}: the rebuilt designs do not evaluate to the candidates")
    # C3-DTLZ1's front lies on its constraint boundary, where G is 0 up to rounding.
    feasible = np.all(np.asarray(out["G"]).reshape(F.shape[0], -1) <= 1e-12, axis=1)
    F = F[feasible]
    if m == 2:
        return thinned_2d(F, 1000)
    front = _nondominated(F)
    return front[farthest_points(front, 5050)] if front.shape[0] > 5050 else front


def _two_variable_front(name: str, curves: list[np.ndarray]) -> np.ndarray:
    """The feasible, non-dominated objectives of designs on ``curves`` (pieces of the Pareto set and the
    constraint boundaries that may hold it), as VAMOS evaluates them, thinned to 1000 points."""
    from vamos.foundation.problem.registry import make_problem_selection

    problem = make_problem_selection(name).instantiate()
    X = np.vstack(curves)
    X = X[np.all((X >= problem.xl) & (X <= problem.xu), axis=1)]
    out = {"F": np.empty((X.shape[0], 2))}
    problem.evaluate(X, out)
    feasible = np.all(np.asarray(out["G"]) <= 1e-9, axis=1)
    return thinned_2d(out["F"][feasible], 1000)


def _srinivas() -> np.ndarray:
    # Where f1 + f2 = x1^2 + 5 x1 + 6 is least, x1 = -2.5 (the segment in Deb 2001). The front goes on along the
    # boundary 3 x2 - x1 = 10 down to the least f1, 10.1 at (1.1, 3.7), and the circle closes it.
    t = np.linspace(0.0, 1.0, 200_001)
    segment = np.column_stack([np.full_like(t, -2.5), 2.5 + t * (np.sqrt(225.0 - 6.25) - 2.5)])
    x1 = -20.0 + 40.0 * t
    line = np.column_stack([x1, (10.0 + x1) / 3.0])
    angle = 2.0 * np.pi * t
    circle = np.column_stack([15.0 * np.cos(angle), 15.0 * np.sin(angle)])
    return _two_variable_front("srinivas", [segment, line, circle])


def _constrex() -> np.ndarray:
    # On x2 = 6 - 9 x1 for x1 in [7/18, 2/3], then on x2 = 0 up to x1 = 1 (Deb 2001); f = (x1, (1 + x2) / x1).
    x1 = 7.0 / 18.0 + (1.0 - 7.0 / 18.0) * np.linspace(0.0, 1.0, 200_001)
    return _two_variable_front("constrex", [np.column_stack([x1, np.maximum(6.0 - 9.0 * x1, 0.0)])])


def _tanaka() -> np.ndarray:
    # f = x on the boundary x1^2 + x2^2 = 1 + 0.1 cos(16 atan(x1 / x2)), where the second constraint holds.
    phi = np.linspace(1e-9, 0.5 * np.pi - 1e-9, 400_001)
    radius = np.sqrt(1.0 + 0.1 * np.cos(16.0 * phi))
    return _two_variable_front("tanaka", [np.column_stack([radius * np.sin(phi), radius * np.cos(phi)])])


def _zxh_cf(k: int) -> np.ndarray:
    """ZXH-CF``k`` at M = 3: on the Pareto set, S = 1 and h = 0, so the objectives are the front shape of the
    angles. Designs on a 301 x 301 grid of angles are evaluated by VAMOS's problem, and the feasible,
    non-dominated ones are kept, thinned to at most 5050."""
    from vamos.foundation.problem.registry import make_problem_selection

    problem = make_problem_selection(f"zxh_cf{k}").instantiate()
    grid = np.linspace(0.0, 1.0, 301)
    theta = np.array(list(itertools.product(grid, repeat=2)))
    a0, a1 = 0.5 * np.pi * theta[:, 0], 0.5 * np.pi * theta[:, 1]
    X = np.full((theta.shape[0], problem.n_var), 0.2)
    X[:, :3] = np.column_stack([np.cos(a0), np.sin(a0) * np.cos(a1), np.sin(a0) * np.sin(a1)])
    X = np.clip(X, problem.xl, problem.xu)
    out = {"F": np.empty((X.shape[0], 3))}
    problem.evaluate(X, out)
    feasible = np.all(np.asarray(out["G"]) <= 1e-9, axis=1)
    front = _nondominated(out["F"][feasible])
    return front[farthest_points(front, 5050)] if front.shape[0] > 5050 else front


def ctp_designs(problem: Any, F: np.ndarray) -> np.ndarray:
    """CTP designs whose objectives are ``F``: ``x1 = f1``, ``x2`` in [0, 1/4] for the g that ``f2`` needs, the
    others at 0. At a fixed f1, CTP1's ``f2 = g exp(-f1 / g)`` and CTP2-CTP7's ``f2 = g - f1`` grow with g, and
    ``g = 1 + x2^2 + 20 sin^2(2 pi x2)`` grows with x2 on [0, 1/4], up to 21.06."""
    f1, f2 = F[:, 0], F[:, 1]
    if problem.variant == 1:
        lo, hi = np.ones_like(f2), np.full_like(f2, 22.0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = mid * np.exp(-f1 / mid) < f2
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        g = 0.5 * (lo + hi)
    else:
        g = f1 + f2
    lo, hi = np.zeros_like(g), np.full_like(g, 0.25)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        below = mid * mid + 20.0 * np.sin(2.0 * np.pi * mid) ** 2 < g - 1.0
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    X = np.zeros((F.shape[0], problem.n_var))
    X[:, 0], X[:, 1] = f1, 0.5 * (lo + hi)
    return X


def _checked_ctp(problem: Any, front: np.ndarray) -> np.ndarray:
    """``front``, once VAMOS's problem, at the designs rebuilt from it, gives its points and finds them feasible.

    The points on the constraint boundary are feasible up to rounding. Where ``d < 1`` (CTP3-CTP5), the power
    turns the rounding of ``sin(b pi v)`` at its zeros, about 1e-14, into a ``G`` of up to about 1e-7."""
    out = {"F": np.empty_like(front)}
    problem.evaluate(ctp_designs(problem, front), out)
    if np.max(np.abs(out["F"] - front)) > 1e-12:
        raise RuntimeError(f"CTP{problem.variant}: the rebuilt designs do not evaluate to the front")
    tolerance = 1e-6 if problem.variant > 1 and problem.d < 1.0 else 1e-9
    if np.max(out["G"]) > tolerance:
        raise RuntimeError(f"CTP{problem.variant}: a front point is infeasible, G = {np.max(out['G'])}")
    return front


def _ctp1() -> np.ndarray:
    """CTP1: f2 grows with g at a fixed f1, so the least feasible f2 is the larger of the unconstrained front
    ``exp(-f1)`` (g = 1) and the constraint boundaries ``a_j exp(-b_j f1)``, reached with g > 1. That envelope
    decreases, so it is the front: 1000 of 200 001 points along f1 in [0, 1]."""
    from vamos.foundation.problem.registry import make_problem_selection

    problem = make_problem_selection("ctp1").instantiate()
    a, b = np.asarray(problem.a_j), np.asarray(problem.b_j)
    f1 = np.linspace(0.0, 1.0, 200_001)
    f2 = np.maximum(np.exp(-f1), np.max(a * np.exp(-b * f1[:, None]), axis=1))
    return _checked_ctp(problem, thinned_2d(np.column_stack([f1, f2]), 1000))


def _ctp_least_f2(constraint: Callable[[np.ndarray, np.ndarray], np.ndarray], f1: float) -> float:
    """The least ``f2 >= 1 - f1`` (g >= 1) where the constraint holds at ``f1``: a scan in steps of 1e-5 over the
    next 25 units, then bisection to the boundary."""
    f2 = (1.0 - f1) + 1e-5 * np.arange(2_500_001)
    feasible = constraint(np.full_like(f2, f1), f2) >= 0.0
    first = int(np.argmax(feasible))
    if not feasible[first]:
        raise RuntimeError(f"no feasible f2 at f1 = {f1}")
    if first == 0:
        return float(f2[0])
    lo, hi = float(f2[first - 1]), float(f2[first])
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if constraint(np.array([f1]), np.array([mid]))[0] >= 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def _ctp(k: int) -> np.ndarray:
    """CTP``k`` (2-7). With ``v = sin(theta) (f2 - e) + cos(theta) f1`` and ``u = cos(theta) (f2 - e) - sin(theta) f1``,
    the constraint is ``u >= w(v) = a |sin(b pi v^c)|^d``. A Pareto-optimal point lies on the boundary of the feasible
    set: on ``u = w(v)``, on the unconstrained front ``f2 = 1 - f1`` (g = 1), or at the least feasible f2 of f1 = 0 or
    1. The candidates are the boundary at 2 000 001 values of v, plus the zeros of the sine (where ``u = 0``, the
    paper's eq. 6 for CTP2-CTP5), the feasible points of 200 001 on the unconstrained front with the ends of its
    feasible pieces, and those two ends; the reachable (``0 <= f1 <= 1``, ``f2 >= 1 - f1``), non-dominated ones are
    thinned to at most 1000, keeping the zeros."""
    from vamos.foundation.problem.registry import make_problem_selection

    problem = make_problem_selection(f"ctp{k}").instantiate()
    theta, a, b, c, d, e = problem.theta, problem.a, problem.b, problem.c, problem.d, problem.e
    sine, cosine = np.sin(theta), np.cos(theta)

    def constraint(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        along = sine * (f2 - e) + cosine * f1
        return np.asarray(cosine * (f2 - e) - sine * f1 - a * np.abs(np.sin(b * np.pi * along**c)) ** d)

    # (0, m0) is feasible and dominates every point with f2 >= m0: the front lies where 1 - f1 <= f2 <= m0.
    ends = np.array([[0.0, _ctp_least_f2(constraint, 0.0)], [1.0, _ctp_least_f2(constraint, 1.0)]])
    corners = np.array([[0.0, 1.0], [0.0, ends[0, 1]], [1.0, 0.0], [1.0, max(ends[0, 1], 0.0)]])
    v_corners = sine * (corners[:, 1] - e) + cosine * corners[:, 0]
    along = np.linspace(v_corners.min(), v_corners.max(), 2_000_001)
    across = a * np.abs(np.sin(b * np.pi * along**c)) ** d
    # The zeros of the sine, v^c = n / b, exactly on u = 0 (c is a positive integer and b > 0 here).
    powers = along**c
    levels = np.arange(np.floor(b * powers.min()), np.ceil(b * powers.max()) + 1.0) / b
    roots = np.sign(levels) * np.abs(levels) ** (1.0 / c)
    zeros = np.concatenate([roots, -roots[levels >= 0.0]]) if c % 2 == 0 else roots
    zeros = zeros[(zeros >= along[0]) & (zeros <= along[-1])]
    along, across = np.concatenate([along, zeros]), np.concatenate([across, np.zeros_like(zeros)])
    boundary = np.column_stack([cosine * along - sine * across, e + sine * along + cosine * across])
    f1 = np.linspace(0.0, 1.0, 200_001)
    feasible = constraint(f1, 1.0 - f1) >= 0.0
    # Where feasibility changes between two points of the unconstrained front, bisection finds the feasible end.
    flips = np.flatnonzero(feasible[:-1] != feasible[1:])
    inside, outside = np.where(feasible[flips], f1[flips], f1[flips + 1]), np.where(feasible[flips], f1[flips + 1], f1[flips])
    for _ in range(100):
        mid = 0.5 * (inside + outside)
        holds = constraint(mid, 1.0 - mid) >= 0.0
        inside, outside = np.where(holds, mid, inside), np.where(holds, outside, mid)
    f1 = np.concatenate([f1[feasible], inside])
    F = np.vstack([boundary, np.column_stack([f1, 1.0 - f1]), ends])
    F = F[(F[:, 0] >= 0.0) & (F[:, 0] <= 1.0) & (F[:, 1] >= 1.0 - F[:, 0])]
    front = nondominated_2d(F)
    if front.shape[0] > 1000:
        # The zeros that are Pareto-optimal stay: they are the points of the paper's line.
        on_line = set(zip((cosine * zeros).tolist(), (e + sine * zeros).tolist()))
        first = tuple(index for index, (f1, f2) in enumerate(front.tolist()) if (f1, f2) in on_line)
        front = front[farthest_points(front, 1000, first)]
    return _checked_ctp(problem, front)


ANALYTIC: dict[str, Callable[[], np.ndarray]] = {
    "ctp1": _ctp1,
    **{f"ctp{k}": (lambda k=k: _ctp(k)) for k in range(2, 8)},
    **{f"zxh_cf{k}.3d": (lambda k=k: _zxh_cf(k)) for k in range(1, 17)},
    "srinivas": _srinivas,
    "constrex": _constrex,
    "tanaka": _tanaka,
    **{name: (lambda name=name: _zdt(name)) for name in ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")},
    **{f"dtlz{i}": (lambda i=i: _dtlz(f"dtlz{i}")) for i in range(1, 8)},
    **{f"dtlz{i}.2d": (lambda i=i: _dtlz_two_objective(f"dtlz{i}")) for i in range(1, 8)},
    **{f"idtlz{i}.{m}d": (lambda i=i, m=m: _idtlz(f"idtlz{i}", m)) for i in (1, 2) for m in (2, 3)},
    **{f"dtlz{i}.{m}d": (lambda i=i, m=m: _dtlz_many_objective(f"dtlz{i}", m)) for i in (1, 2, 7) for m in MANY_OBJECTIVES},
    "wfg1": lambda: _wfg("wfg1"),
    "wfg2": lambda: _wfg("wfg2"),
    "lsmop9": _lsmop9,
    "cec2009_uf6": _uf6,
    "cec2009_uf9": _uf9,
    **{f"{name}.{m}d": (lambda name=name, m=m: _constrained_dtlz(name, m)) for name in CONSTRAINED_DTLZ for m in (2, 3)},
}


# ---------------------------------------------------------------------------------------------------------
# Sampled fronts (Numba for the exact non-dominated filter)
# ---------------------------------------------------------------------------------------------------------


def _nondominated_kernel() -> Callable[[np.ndarray], np.ndarray]:
    import numba

    @numba.njit(cache=False)
    def mask(F: np.ndarray) -> np.ndarray:
        # F is sorted by objective sum without duplicates, so a row can only be dominated by an earlier one.
        n, m = F.shape
        keep = np.zeros(n, dtype=np.bool_)
        archive = np.empty(n, dtype=np.int64)
        size = 0
        last = -1
        for i in range(n):
            dominated = False
            if last >= 0:
                le = True
                lt = False
                for d in range(m):
                    if F[last, d] > F[i, d]:
                        le = False
                        break
                    if F[last, d] < F[i, d]:
                        lt = True
                dominated = le and lt
            if not dominated:
                for k in range(size):
                    j = archive[k]
                    le = True
                    lt = False
                    for d in range(m):
                        if F[j, d] > F[i, d]:
                            le = False
                            break
                        if F[j, d] < F[i, d]:
                            lt = True
                    if le and lt:
                        dominated = True
                        last = j
                        break
            if not dominated:
                keep[i] = True
                archive[size] = i
                size += 1
        return keep

    return mask


_ND_MASK: Callable[[np.ndarray], np.ndarray] | None = None


def _nondominated(F: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
    front, _ = _nondominated_with_designs(F, np.zeros((F.shape[0], 1)) if X is None else X)
    return front


def _nondominated_with_designs(F: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global _ND_MASK
    if _ND_MASK is None:
        _ND_MASK = _nondominated_kernel()
    F, index = np.unique(F, axis=0, return_index=True)
    X = X[index]
    order = np.argsort(F.sum(axis=1), kind="stable")
    F, X = np.ascontiguousarray(F[order]), X[order]
    keep = _ND_MASK(F)
    return F[keep], X[keep]


def _sampled(name: str) -> np.ndarray:
    from vamos.foundation.problem.registry import make_problem_selection

    rng = np.random.default_rng([SEED, list(SAMPLED).index(name)])
    size, n_uniform, grid_points = SAMPLED[name]
    problem = make_problem_selection(name).instantiate()
    xl = np.broadcast_to(np.asarray(problem.xl, dtype=float), (problem.n_var,))
    xu = np.broadcast_to(np.asarray(problem.xu, dtype=float), (problem.n_var,))

    def evaluate(X: np.ndarray) -> np.ndarray:
        out = {"F": np.empty((X.shape[0], problem.n_obj))}
        problem.evaluate(X, out)
        return out["F"]

    unit = rng.random((n_uniform, problem.n_var))
    if grid_points:
        unit = np.vstack([np.array(list(itertools.product(np.linspace(0.0, 1.0, grid_points), repeat=problem.n_var))), unit])
    X = xl + unit * (xu - xl)
    F, X = _nondominated_with_designs(evaluate(X), X)
    for sigma in (0.02, 0.005, 0.001):
        parents = X[farthest_points(F, 3000)]
        kids = np.repeat(parents, 20, axis=0) + rng.normal(0.0, sigma, (parents.shape[0] * 20, problem.n_var)) * (xu - xl)
        kids = np.clip(kids, xl, xu)
        F, X = _nondominated_with_designs(np.vstack([F, evaluate(kids)]), np.vstack([X, kids]))

    # The dense set is not converged far from the refined designs: move a well-spread pool of candidates to the
    # front with the augmented achievement function, then keep the polished designs that stay non-dominated.
    from scipy.spatial import cKDTree

    lo, hi = F.min(axis=0), F.max(axis=0)
    weights = 1.0 / np.where(hi > lo, hi - lo, 1.0)
    pool = farthest_points(F, int(1.25 * size))
    _, neighbours = cKDTree((F - lo) * weights).query((F[pool] - lo) * weights, k=3)
    random_starts = xl + rng.random((pool.size, problem.n_var)) * (xu - xl)
    Y = _polish(evaluate, F[pool], [X[pool], X[neighbours[:, 1]], X[neighbours[:, 2]], random_starts], xl, xu, weights, rng)
    for _ in range(POLISH_PASSES - 1):
        Y = _polish(evaluate, evaluate(Y), [Y, xl + rng.random(Y.shape) * (xu - xl)], xl, xu, weights, rng)
    polished = evaluate(Y)
    tags = np.concatenate([np.ones(pool.size), np.zeros(F.shape[0])])
    union_F, union_X = _nondominated_with_designs(np.vstack([polished, F]), np.hstack([np.vstack([Y, X]), tags[:, None]]))
    front = union_F[union_X[:, -1] == 1.0]
    return front[farthest_points(front, size)]


RHO = 1e-4
POLISH_PASSES = 3


def _polish(
    evaluate: Callable[[np.ndarray], np.ndarray],
    reference: np.ndarray,
    starts: list[np.ndarray],
    xl: np.ndarray,
    xu: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 150,
    candidates: int = 16,
) -> np.ndarray:
    """Minimize ``max_i w_i (f_i(y) - r_i) + RHO * sum_i w_i (f_i(y) - r_i)`` for each reference row ``r``.

    A vectorized stochastic local search from each start, with Gaussian steps shrinking from 10 % to 0.01 % of
    the box; the best start wins. A negative value means the design found dominates its reference.
    """

    def asf(F: np.ndarray, R: np.ndarray) -> np.ndarray:
        d = weights * (F - R)
        return np.asarray(d.max(axis=-1) + RHO * d.sum(axis=-1))

    k, n = starts[0].shape
    rows = np.arange(k)
    best_score = np.full(k, np.inf)
    best = starts[0].copy()
    for start in starts:
        Y = start.copy()
        score = asf(evaluate(Y), reference)
        for step in range(iterations):
            sigma = 0.1 * (1e-4 / 0.1) ** (step / (iterations - 1))
            trial = np.clip(Y[:, None, :] + rng.normal(0.0, sigma, (k, candidates, n)) * (xu - xl), xl, xu)
            trial_score = asf(evaluate(trial.reshape(-1, n)).reshape(k, candidates, -1), reference[:, None, :])
            pick = trial_score.argmin(axis=1)
            better = trial_score[rows, pick] < score
            Y[better] = trial[rows, pick][better]
            score[better] = trial_score[rows, pick][better]
        improved = score < best_score
        best[improved] = Y[improved]
        best_score[improved] = score[improved]
    # Keep a design that the search could not improve.
    return np.where((best_score < 0.0)[:, None], best, starts[0])


# ---------------------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------------------


def write_front(name: str, front: np.ndarray, out_dir: Path) -> Path:
    path = out_dir / f"{FILE_NAMES.get(name, name)}.csv"
    if front.shape[1] > 3:
        # VAMOS 1.5.0: with more objectives, the shortest repr of each value, which reads back exactly,
        # keeps the files at about half the size.
        path.write_text("".join(",".join(map(repr, row)) + "\n" for row in front.tolist()), encoding="ascii", newline="\n")
    else:
        np.savetxt(path, front, delimiter=",")
    return path


def write_authors_front(number: int, reproblems_dir: Path, out_dir: Path) -> Path:
    """Copy ``reference_points_RE<number>.dat`` with commas as separators; the values are unchanged."""
    rows = [line.split() for line in (reproblems_dir / f"reference_points_RE{number}.dat").read_text(encoding="ascii").splitlines()]
    rows = [row for row in rows if row]
    for row in rows:
        for value in row:
            float(value)
    path = out_dir / f"re{number}.csv"
    path.write_text("".join(",".join(row) + "\n" for row in rows), encoding="ascii", newline="\n")
    return path


def _dominated_within(F: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Mask of the rows of ``F`` that another row dominates (minimization)."""
    mask = np.zeros(F.shape[0], dtype=bool)
    for start in range(0, F.shape[0], chunk):
        block = F[start : start + chunk, None, :]
        mask[start : start + chunk] = np.any(np.all(F[None] <= block, axis=2) & np.any(F[None] < block, axis=2), axis=1)
    return mask


def write_jmetal_front(name: str, jmetal_dir: Path, out_dir: Path) -> Path:
    """Copy jMetal's front line by line, without the rows that another row of the file dominates.

    Only DAS-CMOP8 and DAS-CMOP9 lose rows: two each, which lie 1e-6 from the corner point that dominates them.
    """
    lines = [line for line in (jmetal_dir / f"{JMETAL_FRONTS[name]}.csv").read_text(encoding="ascii").splitlines() if line.strip()]
    F = np.array([[float(value) for value in line.split(",")] for line in lines])
    keep = ~_dominated_within(F)
    path = out_dir / f"{name}.csv"
    path.write_text("".join(line + "\n" for line, kept in zip(lines, keep) if kept), encoding="ascii", newline="\n")
    return path


def _negated(value: str) -> str:
    """``value`` with the opposite sign and the same digits."""
    negated = value[1:] if value.startswith("-") else f"-{value}"
    assert float(negated) == -float(value), value
    return negated


def write_modact_front(name: str, modact_dir: Path, out_dir: Path) -> Path:
    """Copy MODAct's ``<problem>_PF.dat`` with commas as separators, and the maximized columns with the opposite sign."""
    from vamos.foundation.problem.modact import MAXIMIZED, OBJECTIVES

    source = modact_dir / f"{name.removeprefix('modact_')}_PF.dat"
    digest = hashlib.md5(source.read_bytes()).hexdigest()
    if digest != MODACT_FRONTS[name]:
        raise ValueError(f"{source} is not the file of Zenodo record {MODACT_ZENODO}, version 1.1.0 (MD5 {digest}).")
    maximized = [objective in MAXIMIZED for objective in OBJECTIVES[name.removeprefix("modact_").rstrip("1234")]]
    rows = [line.split() for line in source.read_text(encoding="ascii").splitlines() if line.strip()]
    if any(len(row) != len(maximized) for row in rows):
        raise ValueError(f"{source} does not have {len(maximized)} columns on every row.")
    path = out_dir / f"{name}.csv"
    lines = (",".join(_negated(value) if flip else value for value, flip in zip(row, maximized)) for row in rows)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="ascii", newline="\n")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="fronts to write, e.g. zdt3 re31 rwa10 (default: none)")
    parser.add_argument("--all", action="store_true", help="write every front this script owns")
    parser.add_argument("--reproblems-dir", type=Path, help="local copy of reproblems' approximated_Pareto_fronts")
    parser.add_argument("--jmetal-dir", type=Path, help=f"local copy of jMetal's resources/referenceFronts at commit {JMETAL_COMMIT}")
    parser.add_argument("--modact-dir", type=Path, help=f"local copy of the MODAct fronts of Zenodo record {MODACT_ZENODO}, version 1.1.0")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    authors = [f"re{number}" for number in AUTHORS_RE]
    names = [*ANALYTIC, *authors, *JMETAL_FRONTS, *MODACT_FRONTS, *SAMPLED] if args.all else args.names
    unknown = sorted(set(names) - {*ANALYTIC, *authors, *JMETAL_FRONTS, *MODACT_FRONTS, *SAMPLED})
    if unknown:
        parser.error(f"unknown fronts: {unknown}")
    if any(name in authors for name in names) and args.reproblems_dir is None:
        parser.error("the RE authors' fronts need --reproblems-dir")
    if any(name in JMETAL_FRONTS for name in names) and args.jmetal_dir is None:
        parser.error("jMetal's fronts need --jmetal-dir")
    if any(name in MODACT_FRONTS for name in names) and args.modact_dir is None:
        parser.error("MODAct's fronts need --modact-dir")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name in ANALYTIC:
            path = write_front(name, ANALYTIC[name](), args.out)
        elif name in authors:
            path = write_authors_front(int(name[2:]), args.reproblems_dir, args.out)
        elif name in JMETAL_FRONTS:
            path = write_jmetal_front(name, args.jmetal_dir, args.out)
        elif name in MODACT_FRONTS:
            path = write_modact_front(name, args.modact_dir, args.out)
        else:
            path = write_front(name, _sampled(name), args.out)
        print(f"wrote {path.name}", flush=True)


if __name__ == "__main__":
    main()
