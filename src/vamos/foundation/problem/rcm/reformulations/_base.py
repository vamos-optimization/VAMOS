"""The shared contract of the RCM reformulations: decode the genotype, then evaluate the original problem."""

from __future__ import annotations

import numpy as np

from vamos.foundation.problem.base import Problem
from vamos.foundation.problem.rcm import RCMProblem
from vamos.foundation.problem.rcm._common import Array

# The failure measure of a decoding whose solver did not converge (the scripts' FAILED and NOT_CONVERGED).
NOT_CONVERGED = 1e3
# The objectives and inequalities of a failed row that are not finite at its fallback point.
FALLBACK_VALUE = 1e6
# Up to this many products, ordered_product_sum forms them all at once; beyond, it adds them term by term.
_PRODUCTS_AT_ONCE = 4096


def ordered_product_sum(a: Array, b: Array) -> Array:
    """``sum_k a[..., k] * b[..., k]`` over the last axis, added from the first term to the last as a loop does.

    ``a`` and ``b`` broadcast to the shape of the products. Both ways give the same sums, bit for bit: a
    running sum (``np.cumsum``) of every product, cheaper for a few rows, and one product at a time.
    """
    shape = np.broadcast_shapes(a.shape, b.shape)
    if int(np.prod(shape)) <= _PRODUCTS_AT_ONCE:
        return np.asarray(np.cumsum(a * b, axis=-1)[..., -1])
    total = np.zeros(shape[:-1], dtype=np.result_type(a, b))
    for k in range(shape[-1]):
        total = total + a[..., k] * b[..., k]
    return total


class RCMReformulation(Problem):
    """An RCM problem searched through a decoder: the variables are the reformulation's, the values the original's.

    ``decode(U)`` maps each row of ``U`` to a complete decision vector of the original problem, the attribute
    ``original`` (an :class:`~vamos.foundation.problem.rcm.RCMProblem`). ``evaluate`` writes the original's
    ``F`` at the decoded vectors and, in ``G``, the original's inequality constraints followed by one
    decoding-failure constraint:

    - where the decoding succeeds, the decoded vector lies within the original bounds, its equalities
      ``|h| - 1e-4 <= 0`` hold by construction, and the last constraint is the largest of them (0 for a
      problem without equalities), so a row is feasible here exactly when its decoded vector is feasible on
      the original problem;
    - where it fails (the solver does not converge, or a decoded variable leaves its bounds), ``decode``
      gives a row of NaN, the last constraint is the subclass's failure measure, which is positive, and ``F``
      and the inequalities are the original's at the fallback point: the decoded vector clipped to the
      original bounds. A value that is not finite there becomes ``1e6``.

    Subclasses define ``_solve(U)``: the decoded vectors (finite; a solver's starting point where it
    diverged) and the failure measure of each row, positive where the decoding failed.
    """

    original: RCMProblem
    key: str
    n_inequalities: int

    def __init__(self, number: int, lower: Array, upper: Array, n_inequalities: int) -> None:
        self.original = RCMProblem(number)
        self.key = f"rcm{int(number):02d}_reformulated"
        self.xl = np.asarray(lower, dtype=float)
        self.xu = np.asarray(upper, dtype=float)
        self.n_var = int(self.xl.size)
        self.n_obj = self.original.n_obj
        self.n_inequalities = n_inequalities
        self.n_constraints = n_inequalities + 1

    def _solve(self, U: Array) -> tuple[Array, Array]:
        raise NotImplementedError

    def _decoded(self, U: Array) -> tuple[Array, Array, Array]:
        """The decoded vectors, the rows whose decoding failed, and the failure measure."""
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        X, measure = self._solve(U)
        lower, upper = self.original.xl, self.original.xu
        finite = np.isfinite(X).all(axis=1)
        with np.errstate(invalid="ignore"):
            excess = (np.maximum(lower - X, 0.0) + np.maximum(X - upper, 0.0)).sum(axis=1)
        failed = ~finite | ~(measure <= 0.0) | (excess > 0.0)
        # A subclass's measure is positive on its failed rows; the bounds and finiteness are checked again here.
        fallback = np.where(finite, np.where(excess > 0.0, excess, np.finfo(float).tiny), NOT_CONVERGED)
        measure = np.where(failed, np.where(measure > 0.0, measure, fallback), measure)
        return X, failed, measure

    def decode(self, U: Array) -> Array:
        """The original decision vectors of the rows of ``U``; a row of NaN where the decoding fails."""
        X, failed, _ = self._decoded(U)
        X = np.array(X, dtype=float)
        X[failed] = np.nan
        return X

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        decoded, failed, measure = self._decoded(X)
        lower, upper = self.original.xl, self.original.xu
        point = np.where(failed[:, None], np.clip(np.nan_to_num(decoded, nan=0.0), lower, upper), decoded)
        values: dict[str, Array] = {}
        self.original.evaluate(point, values)
        F = np.asarray(values["F"], dtype=float)
        G = np.asarray(values["G"], dtype=float)
        inequalities = G[:, : self.n_inequalities]
        equalities = G[:, self.n_inequalities :]
        largest = equalities.max(axis=1) if equalities.shape[1] else np.zeros(point.shape[0])
        if failed.any():
            F = np.where(failed[:, None] & ~np.isfinite(F), FALLBACK_VALUE, F)
            inequalities = np.where(failed[:, None] & ~np.isfinite(inequalities), FALLBACK_VALUE, inequalities)
        out["F"] = F
        out["G"] = np.column_stack([inequalities, np.where(failed, measure, largest)])


__all__ = ["FALLBACK_VALUE", "NOT_CONVERGED", "RCMReformulation", "ordered_product_sum"]
