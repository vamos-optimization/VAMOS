"""Reference directions on the unit simplex: Das-Dennis lattices and Riesz s-energy sets.

This leaf module imports no other VAMOS module, so algorithms, result archives
and subset selectors share one lattice without creating import cycles.

Development after 1.5.2: :func:`riesz_energy`, any number of well-spaced directions.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb, lcm

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def lattice_size(n_obj: int, divisions: int) -> int:
    """Return the number of directions of a complete Das-Dennis lattice.

    A lattice with ``divisions`` divisions on the ``n_obj``-objective simplex has
    ``comb(divisions + n_obj - 1, n_obj - 1)`` directions.
    """
    if n_obj < 1:
        raise ValueError("n_obj must be >= 1")
    if divisions < 1:
        raise ValueError("divisions must be >= 1")
    return comb(divisions + n_obj - 1, n_obj - 1)


def largest_lattice_divisions(max_directions: int, n_obj: int) -> int:
    """Return the most divisions whose complete lattice has at most ``max_directions`` directions."""
    if n_obj < 2:
        raise ValueError("A Das-Dennis lattice needs n_obj >= 2.")
    if max_directions < n_obj:
        raise ValueError(f"The smallest Das-Dennis lattice for n_obj={n_obj} has {n_obj} directions, more than {max_directions}.")
    divisions = 1
    while lattice_size(n_obj, divisions + 1) <= max_directions:
        divisions += 1
    return divisions


def das_dennis(n_obj: int, divisions: int) -> FloatArray:
    """Return the complete Das-Dennis lattice, one direction per row.

    Each row has coordinates in ``{0, 1/divisions, ..., 1}`` summing to one, and
    every such point appears once, so the set is symmetric between objectives and
    contains each axis. Rows are in ascending lexicographic order of their
    integer coordinates.
    """
    count = lattice_size(n_obj, divisions)
    coords = np.empty((count, n_obj), dtype=float)
    current = [0] * n_obj
    row = 0

    def fill(depth: int, remaining: int) -> None:
        nonlocal row
        if depth == n_obj - 1:
            current[depth] = remaining
            coords[row] = current
            row += 1
            return
        for value in range(remaining + 1):
            current[depth] = value
            fill(depth + 1, remaining - value)

    fill(0, divisions)
    weights = np.clip(coords / divisions, 0.0, 1.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def two_layer_size(n_obj: int, divisions: int, inner_divisions: int) -> int:
    """Return the number of directions of a two-layer lattice: both complete lattices.

    VAMOS 1.5.0: new.
    """
    return lattice_size(n_obj, divisions) + lattice_size(n_obj, inner_divisions)


def two_layer_lattice(n_obj: int, divisions: int, inner_divisions: int) -> FloatArray:
    """Return Deb and Jain's two-layer reference directions, one per row.

    With many objectives a lattice of few divisions has every direction on the boundary of the
    simplex, and one more division multiplies the directions. Deb and Jain (2014) add an inside
    layer instead: the boundary layer is the complete lattice of ``divisions`` and the
    inside layer the complete lattice of ``inner_divisions`` shrunk by half towards the centre of
    the simplex, ``w / 2 + 1 / (2 n_obj)``, as pymoo's multi-layer directions with scaling 0.5 and
    PlatEMO's ``UniformPoint`` do. The boundary rows come first, each layer in the order of
    :func:`das_dennis`. Eight objectives with ``divisions=3`` and ``inner_divisions=2`` give
    120 + 36 = 156 directions.

    The inside layer must add directions that the boundary layer does not have: two objectives with
    ``divisions=4`` and ``inner_divisions=2`` would repeat all three inside directions, and a repeated
    direction is a niche that never gets its own individual. Such settings raise ``ValueError`` (see
    :func:`two_layer_shared`).

    VAMOS 1.5.0: new.
    """
    shared = two_layer_shared(n_obj, divisions, inner_divisions)
    if shared:
        raise ValueError(
            f"The inside layer of inner_divisions={inner_divisions} repeats {shared} of the directions of the boundary layer "
            f"of divisions={divisions} for n_obj={n_obj}; choose layers that share no direction."
        )
    return np.asarray(np.vstack(_two_layers(n_obj, divisions, inner_divisions)), dtype=float)


def two_layer_shared(n_obj: int, divisions: int, inner_divisions: int) -> int:
    """Return how many directions of the inside layer the boundary layer already has.

    Every coordinate of either layer is a multiple of ``1 / lcm(divisions, 2 inner_divisions n_obj)``,
    so the layers are compared exactly on that grid. A two-layer lattice needs 0.

    VAMOS 1.5.0: new.
    """
    boundary, inside = _two_layers(n_obj, divisions, inner_divisions)
    grid = lcm(divisions, 2 * inner_divisions * n_obj)
    boundary_points = {tuple(row) for row in np.rint(boundary * grid).astype(np.int64).tolist()}
    return sum(tuple(row) in boundary_points for row in np.rint(inside * grid).astype(np.int64).tolist())


def _two_layers(n_obj: int, divisions: int, inner_divisions: int) -> tuple[FloatArray, FloatArray]:
    """The boundary layer and the inside layer, shrunk by half towards the centre of the simplex."""
    boundary = das_dennis(n_obj, divisions)
    inside = np.asarray(das_dennis(n_obj, inner_divisions) * 0.5 + 0.5 / n_obj, dtype=float)
    return boundary, inside


#: The most directions :func:`riesz_energy` places: its memory and time grow with the square of their number
#: (three arrays of ``n_points ** 2`` floats, 600 MB for 5000 directions).
RIESZ_ENERGY_MAX_POINTS = 5000


def riesz_energy(n_obj: int, n_points: int, *, seed: int, s: int | None = None, max_iter: int = 1000) -> FloatArray:
    """Return ``n_points`` well-spaced directions on the unit simplex, one per row: a Riesz s-energy set.

    Blank, Deb, Dhebar, Bandaru and Seada (2021) place any number of points on the simplex by minimizing their
    Riesz s-energy, ``sum over pairs i < j of 1 / ||z_i - z_j|| ** s``, which grows without bound as two
    points meet, so a set of low energy has no close pair. As in the paper, ``s`` is ``n_obj ** 2`` by default,
    the logarithm of the energy is minimized by gradient descent with Adam, and each gradient is projected onto
    the plane of the simplex; every step is then followed by the Euclidean projection onto the simplex, so each
    row is non-negative and sums to 1.

    The start, the restarts and the stopping rule follow the authors' implementation in pymoo
    (``RieszEnergyReferenceDirectionFactory``), written anew here. The start is the boundary of the largest
    Das-Dennis lattice with at most ``n_points`` directions and the remaining points, taken one at a time,
    farthest from those chosen, from ``max(20 n_points, 2000)`` uniform random points of the simplex (pymoo
    then runs k-means, which is left out). After 30 steps without a lower energy the descent returns to the
    best set and halves Adam's step, which starts at 5% of the mean distance between nearest points. It stops
    after ``max_iter`` steps (1000, as pymoo) or when no point moves by more than 1e-4 of the first step, and
    returns the set of lowest energy, its rows in ascending lexicographic order. Unlike pymoo, the ``n_obj``
    corners of the simplex stay fixed, so the set always holds the extreme directions; the other points,
    boundary ones included, move.

    The result depends only on the arguments: ``seed`` seeds the random start, and the same arguments give the
    same array bit for bit. The computation uses only correctly rounded arithmetic, sorting and NumPy's own
    sums (no BLAS, logarithm or power function), so that the result does not depend on the BLAS library or on
    the vector instructions of the processor. Each step costs ``O(n_points ** 2 n_obj)`` time and
    ``O(n_points ** 2)`` memory, so at most :data:`RIESZ_ENERGY_MAX_POINTS` directions are placed: 100
    directions of three objectives take about 0.1 s, 1000 about half a minute. ``n_obj`` must be at least 2
    and ``n_points`` at least ``n_obj``; ``n_points == n_obj`` gives the corners.

    Blank, J., Deb, K., Dhebar, Y., Bandaru, S., and Seada, H. (2021). Generating well-spaced points on a unit
    simplex for evolutionary many-objective optimization. *IEEE Transactions on Evolutionary Computation*,
    25(1), 48-60. https://doi.org/10.1109/TEVC.2020.2992387

    Development after 1.5.2: new.
    """
    n_obj = _count(n_obj, "n_obj", 2)
    n_points = _count(n_points, "n_points", n_obj)
    if n_points > RIESZ_ENERGY_MAX_POINTS:
        raise ValueError(f"riesz_energy places at most {RIESZ_ENERGY_MAX_POINTS} directions; got n_points={n_points}.")
    exponent = n_obj * n_obj if s is None else _count(s, "s", 1)
    points = _riesz_energy(n_obj, n_points, _count(seed, "seed", 0), exponent, _count(max_iter, "max_iter", 0))
    return np.array(points, dtype=float)


def _count(value: object, name: str, minimum: int) -> int:
    """``value`` as an integer of at least ``minimum``, or ``ValueError``; a bool is not a count."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")
    return int(value)


@lru_cache(maxsize=16)
def _riesz_energy(n_obj: int, n_points: int, seed: int, s: int, max_iter: int) -> FloatArray:
    """The set of :func:`riesz_energy`, read-only: MOEA/D's run and its resolved spec both ask for it."""
    points = _riesz_start(n_obj, n_points, np.random.default_rng(seed))
    points = _riesz_descent(points, np.any(points == 1.0, axis=1), s, max_iter)
    points = np.ascontiguousarray(points[np.lexsort(points.T[::-1])])
    points.flags.writeable = False
    return points


def _riesz_start(n_obj: int, n_points: int, rng: np.random.Generator) -> FloatArray:
    """The boundary of the largest lattice that fits, then the farthest of many uniform random points, one at a time."""
    lattice = das_dennis(n_obj, largest_lattice_divisions(n_points, n_obj))
    boundary = lattice[np.any(lattice == 0.0, axis=1)]
    missing = n_points - boundary.shape[0]
    if missing == 0:
        return boundary
    count = max(20 * n_points, 2000)
    # Sorted uniform numbers split [0, 1] into n_obj spacings: a uniform random point of the simplex.
    cuts = np.sort(rng.random((count, n_obj - 1)), axis=1)
    pool = np.diff(np.hstack([np.zeros((count, 1)), cuts, np.ones((count, 1))]), axis=1)
    pool /= pool.sum(axis=1, keepdims=True)
    nearest = np.full(count, np.inf)
    for point in boundary:
        nearest = np.minimum(nearest, _squared_distances_to(pool, point))
    chosen = np.empty(missing, dtype=np.intp)
    for k in range(missing):
        chosen[k] = np.argmax(nearest)
        nearest = np.minimum(nearest, _squared_distances_to(pool, pool[chosen[k]]))
    return np.asarray(np.vstack([boundary, pool[chosen]]), dtype=float)


def _riesz_descent(points: FloatArray, frozen: npt.NDArray[np.bool_], s: int, max_iter: int) -> FloatArray:
    """Adam on the logarithm of the energy, projected onto the simplex; the ``frozen`` rows (the corners) stay."""
    free = ~frozen
    if max_iter == 0 or not free.any():
        return points
    beta1, beta2, epsilon, patience = 0.9, 0.999, 1e-8, 30
    energy = _RieszEnergy(points.shape[0], s)
    energy.evaluate(points)
    first_step = 0.05 * float(np.sqrt(energy.squared.min(axis=1)).mean())
    step_size = first_step
    best, best_closest, best_total = points, energy.closest, energy.total
    moment, second = np.zeros_like(points), np.zeros_like(points)
    power1 = power2 = 1.0
    stalled = 0
    for _ in range(max_iter):
        gradient = _row_centred(energy.gradient(points))
        power1, power2 = power1 * beta1, power2 * beta2
        moment = beta1 * moment + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient * gradient
        step = _row_centred(step_size * (moment / (1.0 - power1)) / (np.sqrt(second / (1.0 - power2)) + epsilon))
        moved = points.copy()
        moved[free] = _onto_simplex(points[free] - step[free])
        movement = float(np.abs(moved - points).max())
        points = moved
        energy.evaluate(points)
        # U = closest ** (-s / 2) * total, so U < U_best when total * (best_closest / closest) ** (s / 2) < best_total.
        scale = float(_ratio_power(np.array([best_closest / energy.closest]), s, np.empty(1))[0])
        if energy.total * scale < best_total:
            best, best_closest, best_total, stalled = points, energy.closest, energy.total, 0
        else:
            stalled += 1
        if stalled >= patience:
            points, step_size, stalled = best, step_size / 2.0, 0
            energy.evaluate(points)
            moment, second, power1, power2 = np.zeros_like(points), np.zeros_like(points), 1.0, 1.0
        if movement < 1e-4 * first_step:
            break
    return best


class _RieszEnergy:
    """The Riesz s-energy of a set of points and the gradient of its logarithm, in arrays reused from step to step.

    The energy is scaled by the closest pair: every term ``(d_min / d_ij) ** s`` is at most 1 and the energy is
    ``d_min ** -s`` times their sum, so a large ``s`` never overflows.
    """

    def __init__(self, n_points: int, s: int) -> None:
        self.s = s
        self.squared = np.empty((n_points, n_points))
        self.weights = np.empty((n_points, n_points))
        self._work = np.empty((n_points, n_points))
        self.closest = self.total = 0.0

    def evaluate(self, points: FloatArray) -> None:
        """The squared distances (infinite on the diagonal), the smallest, the terms and their sum over pairs."""
        squared, work = self.squared, self._work
        for k in range(points.shape[1]):
            column = points[:, k]
            difference = np.subtract.outer(column, column, out=work)
            if k == 0:
                np.multiply(difference, difference, out=squared)
            else:
                squared += np.multiply(difference, difference, out=work)
        np.fill_diagonal(squared, np.inf)
        self.closest = float(squared.min())
        _ratio_power(np.divide(self.closest, squared, out=work), self.s, self.weights)
        self.total = float(self.weights.sum()) / 2.0

    def gradient(self, points: FloatArray) -> FloatArray:
        """``-s / total * sum_j (w_ij / d_ij ** 2) (z_i - z_j)`` at the evaluated points; it uses up the terms."""
        pull = np.divide(self.weights, self.squared, out=self._work)
        weighted = np.empty_like(points)
        for k in range(points.shape[1]):
            weighted[:, k] = np.multiply(pull, points[:, k], out=self.weights).sum(axis=1)
        return np.asarray(-(self.s / self.total) * (points * pull.sum(axis=1)[:, None] - weighted), dtype=float)


def _ratio_power(ratio: FloatArray, s: int, out: FloatArray) -> FloatArray:
    """``ratio ** (s / 2)`` into ``out`` for squared ratios, by repeated squaring and a square root: no power function.

    ``ratio`` is overwritten.
    """
    if s % 2:
        np.sqrt(ratio, out=out)
    else:
        out.fill(1.0)
    exponent = s // 2
    while exponent:
        if exponent & 1:
            np.multiply(out, ratio, out=out)
        exponent >>= 1
        if exponent:
            np.multiply(ratio, ratio, out=ratio)
    return out


def _row_centred(values: FloatArray) -> FloatArray:
    """``values`` projected onto the plane of the simplex: each row less its mean, so that it sums to 0."""
    return np.asarray(values - values.sum(axis=1, keepdims=True) / values.shape[1], dtype=float)


def _onto_simplex(points: FloatArray) -> FloatArray:
    """The Euclidean projection of each row onto the unit simplex (Held, Wolfe and Crowder's sorting method)."""
    n_rows, n_obj = points.shape
    descending = -np.sort(-points, axis=1)
    excess = np.cumsum(descending, axis=1) - 1.0
    positive = descending - excess / np.arange(1, n_obj + 1) > 0.0
    last = n_obj - 1 - np.argmax(positive[:, ::-1], axis=1)
    projected = np.maximum(points - (excess[np.arange(n_rows), last] / (last + 1))[:, None], 0.0)
    return np.asarray(projected / projected.sum(axis=1, keepdims=True), dtype=float)


def _squared_distances_to(points: FloatArray, point: FloatArray) -> FloatArray:
    squared = np.zeros(points.shape[0])
    for k in range(points.shape[1]):
        difference = points[:, k] - point[k]
        squared += difference * difference
    return squared


__all__ = [
    "RIESZ_ENERGY_MAX_POINTS",
    "das_dennis",
    "largest_lattice_divisions",
    "lattice_size",
    "riesz_energy",
    "two_layer_lattice",
    "two_layer_shared",
    "two_layer_size",
]
