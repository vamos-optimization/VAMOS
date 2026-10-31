"""Multi-parent real-valued crossover operators."""

from __future__ import annotations

import numpy as np

from ._crossover_base import Crossover
from .utils import ArrayLike, _ensure_bounds

_TINY = 1e-32


def _remove_component(vectors: ArrayLike, direction: ArrayLike) -> ArrayLike:
    """Project each row of ``vectors`` onto the complement of the matching row of ``direction``."""
    norm_sq = np.maximum(np.sum(direction * direction, axis=-1, keepdims=True), _TINY)
    coefficient = np.sum(vectors * direction, axis=-1, keepdims=True) / norm_sq
    return np.asarray(vectors - coefficient * direction, dtype=np.float64)


class PCXCrossover(Crossover):
    """Parent-centric crossover (PCX) of Deb, Anand and Joshi (2002).

    Each group of three parents yields three children, and child ``j`` is centred on
    parent ``j``, its index parent. With ``g`` the centroid of the three parents and
    ``d = g - x_j``:

        child = x_j + w * d + noise,   w ~ N(0, sigma_zeta**2)

    ``noise`` is Gaussian with standard deviation ``sigma_eta * D`` in every direction
    orthogonal to ``d``, where ``D`` is the mean distance of the other two parents to
    the line through ``x_j`` along ``d``. As in Deb's C code and pymoo, the noise is an
    isotropic draw with its component along ``d`` removed. ``prob_crossover`` is
    ignored: every group is recombined.

    jMetal's ``ParentCentricCrossover`` is a different variant, with uniform draws per
    coordinate and the second and third parents defining the search line.

    Reference: Deb, K., Anand, A. & Joshi, D. (2002). A computationally efficient
    evolutionary algorithm for real-parameter optimization. Evolutionary Computation
    10(4), 371-395.
    """

    def __init__(
        self,
        sigma_eta: float = 0.1,
        sigma_zeta: float = 0.1,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        prob_crossover: float | None = None,
        workspace: object | None = None,
        allow_inplace: bool = False,
    ) -> None:
        _ = prob_crossover, workspace, allow_inplace
        self.sigma_eta = float(sigma_eta)
        self.sigma_zeta = float(sigma_zeta)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        groups = self._as_matings(parents, expected_parents=3, copy=False)
        n_groups, n_parents, n_vars = groups.shape
        offspring = np.empty_like(groups)
        centroid = groups.mean(axis=1)
        for j in range(n_parents):
            index_parent = groups[:, j, :]
            d = centroid - index_parent
            others = np.delete(groups, j, axis=1) - index_parent[:, None, :]
            perpendicular = _remove_component(others, d[:, None, :])
            mean_distance = np.linalg.norm(perpendicular, axis=-1).mean(axis=1)
            noise = rng.normal(0.0, 1.0, size=(n_groups, n_vars)) * (self.sigma_eta * mean_distance)[:, None]
            noise = _remove_component(noise, d)
            w = rng.normal(0.0, self.sigma_zeta, size=(n_groups, 1))
            offspring[:, j, :] = index_parent + w * d + noise
        return offspring


class UNDXCrossover(Crossover):
    """Unimodal normal distribution crossover (UNDX) of Ono and Kobayashi (1997).

    The first two parents of a group define the middle point ``m = (p1 + p2) / 2`` and
    ``d = p2 - p1``; the third parent sets the distance ``D`` to the line p1-p2. The two
    children are symmetric about ``m``:

        c = m +/- (xi * d + D * sum_i eta_i * e_i)

    with one ``xi ~ N(0, zeta**2)`` per pair, ``eta_i ~ N(0, eta**2 / n)`` and ``e_i`` an
    orthonormal basis of the complement of ``d`` (Kita, Ono and Kobayashi, 1999,
    recommend ``zeta = 0.5`` and ``eta = 0.35``). A group whose first two parents
    coincide is copied unchanged.

    jMetal's ``UnimodalNormalDistributionCrossover`` is a different variant: it draws
    uniform ``alpha`` and ``beta`` per coordinate and does not project the third
    parent's direction onto the complement of ``d``.

    References: Ono, I. & Kobayashi, S. (1997). A real-coded genetic algorithm for
    function optimization using unimodal normal distribution crossover. ICGA 1997.
    Kita, H., Ono, I. & Kobayashi, S. (1999). Multi-parental extension of the unimodal
    normal distribution crossover for real-coded genetic algorithms. CEC 1999.
    """

    def __init__(
        self,
        prob_crossover: float = 0.9,
        zeta: float = 0.5,
        eta: float = 0.35,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: object | None = None,
        allow_inplace: bool = False,
    ) -> None:
        _ = workspace, allow_inplace
        self.prob = float(prob_crossover)
        self.zeta = float(zeta)
        self.eta = float(eta)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        groups = self._as_matings(parents, expected_parents=3, copy=False)
        n_groups, _, n_vars = groups.shape
        offspring = np.array(groups[:, :2, :], dtype=groups.dtype, copy=True)
        if n_groups == 0:
            return offspring
        p1, p2, p3 = groups[:, 0, :], groups[:, 1, :], groups[:, 2, :]
        d = p2 - p1
        active = (rng.random(n_groups) <= self.prob) & (np.linalg.norm(d, axis=1) >= 1.0e-10)
        if not np.any(active):
            return offspring
        d = d[active]
        middle = 0.5 * (p1[active] + p2[active])
        distance = np.linalg.norm(_remove_component(p3[active] - middle, d), axis=1)
        n_active = d.shape[0]
        xi = rng.normal(0.0, self.zeta, size=(n_active, 1))
        orthogonal = rng.normal(0.0, self.eta / np.sqrt(n_vars), size=(n_active, n_vars)) * distance[:, None]
        step = xi * d + _remove_component(orthogonal, d)
        offspring[active, 0, :] = middle + step
        offspring[active, 1, :] = middle - step
        return offspring


class SPXCrossover(Crossover):
    """Simplex crossover (SPX) of Tsutsui, Yamamura and Higuchi (1999).

    The simplex of the ``m`` parents of a group is expanded about their centroid ``O``
    by ``epsilon`` and each child is drawn uniformly inside it, with Dirichlet(1)
    weights ``w``: ``child = O + epsilon * (sum_k w_k x_k - O)``. The default
    ``epsilon = sqrt(m + 1)`` preserves the mean and the covariance of the parents.
    Each group of ``m`` parents yields ``m`` children. ``prob_crossover`` is ignored:
    every group is recombined.

    Reference: Tsutsui, S., Yamamura, M. & Higuchi, T. (1999). Multi-parent
    recombination with simplex crossover in real coded genetic algorithms. GECCO 1999.
    """

    def __init__(
        self,
        epsilon: float | None = None,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        prob_crossover: float | None = None,
        workspace: object | None = None,
        allow_inplace: bool = False,
    ) -> None:
        _ = prob_crossover, workspace, allow_inplace
        self.epsilon = None if epsilon is None else float(epsilon)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        groups = self._as_matings(parents, expected_parents=3, copy=False)
        n_groups, k, _ = groups.shape
        epsilon = np.sqrt(k + 1.0) if self.epsilon is None else self.epsilon
        centroid = groups.mean(axis=1, keepdims=True)
        weights = rng.dirichlet(np.ones(k), size=(n_groups, k))
        points = np.einsum("gck,gkv->gcv", weights, groups)
        return np.asarray(centroid + epsilon * (points - centroid), dtype=np.float64)


__all__ = ["PCXCrossover", "SPXCrossover", "UNDXCrossover"]
