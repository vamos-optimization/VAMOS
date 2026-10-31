"""Regenerate VAMOS's packaged reference fronts in ``src/vamos/resources/reference_fronts``.

Run it from the repository root with this checkout's ``src`` importable and the ``compute`` extra installed
(the exact non-dominated filter uses Numba):

    python experiments/scripts/generate_reference_fronts.py --all --reproblems-dir PATH
    python experiments/scripts/generate_reference_fronts.py zdt3 dtlz7 wfg1

The script writes three groups of files; ``docs/reference/problems.md`` ("Reference fronts") lists the
provenance of every packaged file, including those this script leaves alone.

- **Analytic samples** of the Pareto front from its closed form: ZDT1-4 and ZDT6, DTLZ1-7 (M = 3), WFG1 and
  WFG2 (M = 2, from the WFG shape functions, so the bias transformations cannot shift them), LSMOP9 (M = 2),
  CEC2009 UF6 and UF9, C1-DTLZ3 (M = 2) and DC2-DTLZ1/3 (M = 2). Disconnected fronts are filtered for
  non-dominance on a fine grid and thinned by farthest-point selection.
- **The RE authors' fronts:** Tanabe and Ishibuchi's ``reference_points_RE*.dat`` from
  https://github.com/ryojitanabe/reproblems (``approximated_Pareto_fronts``), with only the separator changed.
  ``--reproblems-dir`` points at a local copy of that directory. Their MIT notice is
  ``LICENSE-reproblems.txt`` next to the CSV files.
- **Sampled fronts** where no closed form or authors' front applies: RWA3, RWA9, RWA10 and the deterministic
  RE91. Dense sampling of the design box, local refinement around the non-dominated designs and an exact
  non-dominated filter over every evaluated design. A well-spread pool of 1.25 times the packaged size is then
  moved to the front by minimizing the augmented achievement function relative to each candidate (three
  passes), the polished designs that stay non-dominated are kept, and farthest-point selection thins them to
  the packaged size.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "src" / "vamos" / "resources" / "reference_fronts"
SEED = 20261030

# Files keep their historical names (ZDT1 is upper case).
FILE_NAMES = {"zdt1": "ZDT1"}

# Tanabe and Ishibuchi's approximated fronts that match VAMOS's RE formulations. RE91 is not here: VAMOS
# evaluates it at the mean of its stochastic parameters, so the authors' stochastic front does not apply.
AUTHORS_RE = (21, 24, 31, 32, 33, 34, 37, 41, 42, 61)

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


def farthest_points(F: np.ndarray, k: int) -> np.ndarray:
    """Indices of ``k`` well-spread rows: greedy farthest-point selection in normalized objective space."""
    lo, hi = F.min(axis=0), F.max(axis=0)
    Z = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    chosen = list(dict.fromkeys(int(np.argmin(F[:, j])) for j in range(F.shape[1])))
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


def _quarter_circle(n: int = 1000) -> np.ndarray:
    theta = np.linspace(0.0, np.pi / 2.0, n)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _dc2(name: str) -> np.ndarray:
    # DC2's constraints depend only on g and hold at g = 0, so the front is the whole DTLZ front (M = 2).
    if name == "dc2dtlz1":
        f1 = np.linspace(0.0, 0.5, 1000)
        return np.column_stack([f1, 0.5 - f1])
    return _quarter_circle()


ANALYTIC: dict[str, Callable[[], np.ndarray]] = {
    **{name: (lambda name=name: _zdt(name)) for name in ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")},
    **{f"dtlz{i}": (lambda i=i: _dtlz(f"dtlz{i}")) for i in range(1, 8)},
    "wfg1": lambda: _wfg("wfg1"),
    "wfg2": lambda: _wfg("wfg2"),
    "lsmop9": _lsmop9,
    "cec2009_uf6": _uf6,
    "cec2009_uf9": _uf9,
    "c1dtlz3": _quarter_circle,
    "dc2dtlz1": lambda: _dc2("dc2dtlz1"),
    "dc2dtlz3": lambda: _dc2("dc2dtlz3"),
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="fronts to write, e.g. zdt3 re31 rwa10 (default: none)")
    parser.add_argument("--all", action="store_true", help="write every front this script owns")
    parser.add_argument("--reproblems-dir", type=Path, help="local copy of reproblems' approximated_Pareto_fronts")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    authors = [f"re{number}" for number in AUTHORS_RE]
    names = [*ANALYTIC, *authors, *SAMPLED] if args.all else args.names
    unknown = sorted(set(names) - {*ANALYTIC, *authors, *SAMPLED})
    if unknown:
        parser.error(f"unknown fronts: {unknown}")
    if any(name in authors for name in names) and args.reproblems_dir is None:
        parser.error("the RE authors' fronts need --reproblems-dir")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name in ANALYTIC:
            path = write_front(name, ANALYTIC[name](), args.out)
        elif name in authors:
            path = write_authors_front(int(name[2:]), args.reproblems_dir, args.out)
        else:
            path = write_front(name, _sampled(name), args.out)
        print(f"wrote {path.name}", flush=True)


if __name__ == "__main__":
    main()
