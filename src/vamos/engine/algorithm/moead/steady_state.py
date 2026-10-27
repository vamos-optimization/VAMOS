"""Compiled one-offspring MOEA/D step for the numba engine.

MOEA/D's default steady-state mode (``batch_size=1``) creates one child per
subproblem and updates its neighbourhood at once. With the numba engine and a
covered configuration, :class:`SteadyStateStep` replaces the per-step Python
operator stack with compiled calls. Python still draws every random value from
the run's Generator, in the order, shape and conditions of the reference
``MOEAD.ask``/``MOEAD.tell``, so populations, results and Generator states are
bit-identical to the NumPy engine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.archive.factory import update_archive
from vamos.engine.operators.policies.moead import OneOffspringRealVariation, one_offspring_real_variation
from vamos.foundation.constraints.utils import compute_violation

from .neighborhood_kernels import dummy_buffers, get_update_neighborhood_numba

if TYPE_CHECKING:
    from vamos.foundation.eval import EvaluationBackend, EvaluationResult
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol

    from .state import MOEADState

# Arguments of the compiled replacement: X, F, G, cv, weights, ideal, children, F_child, G_child, child_cv, order, constrained.
ReplaceArgs = tuple[Any, ...]


def _float_matrix(value: object, rows: int) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.ndim == 2
        and value.shape[0] == rows
        and value.flags.c_contiguous
    )


def _supported_state(st: MOEADState) -> bool:
    pop_size, n_obj = st.pop_size, st.F.shape[1] if st.F.ndim == 2 else -1
    if not (_float_matrix(st.X, pop_size) and _float_matrix(st.F, pop_size)):
        return False
    if st.G is not None and not _float_matrix(st.G, pop_size):
        return False
    if st.cv is not None and not (isinstance(st.cv, np.ndarray) and st.cv.dtype == np.float64 and st.cv.shape == (pop_size,)):
        return False
    if not (isinstance(st.ideal, np.ndarray) and st.ideal.dtype == np.float64 and st.ideal.shape == (n_obj,)):
        return False
    neighbors = st.neighbors
    if not (isinstance(neighbors, np.ndarray) and neighbors.dtype == np.int64 and neighbors.ndim == 2 and neighbors.shape[0] == pop_size):
        return False
    return all(_float_matrix(w, pop_size) and w.shape[1] == n_obj for w in (st.weights, st.weights_safe, st.weights_unit))


def _compiled_kernels_available() -> bool:
    # The kernels call the compiled neighbourhood update, so they follow its switch (VAMOS_USE_NUMBA_MOEAD).
    if get_update_neighborhood_numba() is None:
        return False
    try:
        from . import steady_state_kernels  # noqa: F401
    except (ImportError, OSError, RuntimeError, AttributeError):
        return False
    return True


class SteadyStateStep:
    """One-offspring ask/tell that keeps MOEA/D's reference draws and arithmetic.

    Build it with :meth:`build`; ``None`` means the configuration is not covered
    and MOEA/D keeps its reference step. Covered: real encoding with float64
    arrays, DE or SBX crossover with polynomial mutation and the clip repair
    (the defaults), every aggregation, constraints, external archives, live
    callbacks and HV termination. Genealogy tracking and ``batch_size > 1``
    keep the reference step. A call that meets anything unexpected returns
    ``None`` before touching the state, and MOEA/D runs its reference code.
    """

    def __init__(self, st: MOEADState, variation: OneOffspringRealVariation) -> None:
        from vamos.foundation.kernel._numba_variation import numpy_power_squares

        from . import steady_state_kernels as kernels

        n_var = st.X.shape[1]
        # Compiled dispatchers; several calls spread a shared argument tuple, so they are typed loosely.
        self._de_child: Callable[..., None] = kernels.de_child
        self._sbx_parents: Callable[..., bool] = kernels.sbx_parents
        self._sbx_child: Callable[..., None] = kernels.sbx_child
        self._replace: Callable[..., int] = kernels.replace_neighbors
        self._replace_then_de_child: Callable[..., int] = kernels.replace_then_de_child
        self._replace_then_sbx_parents: Callable[..., int] = kernels.replace_then_sbx_parents
        self._squares = numpy_power_squares()
        self._f64 = np.dtype(np.float64)
        self._is_de = variation.crossover == "de"
        self._n_var = n_var
        self._child_shape = (1, n_var)
        self._f_shape = (1, st.F.shape[1])
        self._sbx_prob = float(variation.sbx_prob)
        self._use_var_draws = float(variation.sbx_prob_var) < 1.0
        bounds = (
            variation.clip_lower,
            variation.clip_upper,
            variation.pm_lower,
            variation.pm_upper,
            variation.sbx_lower,
            variation.sbx_upper,
        )
        self._bounds = np.ascontiguousarray(np.stack(bounds), dtype=np.float64)
        eta = float(variation.sbx_eta)
        params = (
            variation.de_f,
            variation.de_cr,
            variation.pm_prob,
            variation.pm_eta,
            variation.sbx_prob_var,
            -(eta + 1.0),
            1.0 / (eta + 1.0),
        )
        self._params = np.array([*params, st.aggregation_theta, st.aggregation_rho], dtype=np.float64)
        self._draws = np.empty((kernels.N_DRAWS, n_var))
        rows = [self._draws[i : i + 1] for i in range(kernels.N_DRAWS)]
        self._rand, self._pm_mask, self._pm_delta = rows[kernels.RAND], rows[kernels.PM_MASK], rows[kernels.PM_DELTA]
        self._var_draws, self._swap = rows[kernels.VAR_DRAWS], rows[kernels.SWAP]
        self._mating = np.empty((1, 3, n_var))
        self._sbx_active = np.zeros((1, n_var), dtype=np.bool_)
        self._slots = np.zeros(kernels.N_SLOTS, dtype=np.int64)
        self._slots[5:7] = (int(st.replace_limit), int(st.aggregation_id))
        self._active = self._slots[0:1]
        self._pairs = self._slots[1:3].reshape(1, 2)
        self._flags = np.zeros(1, dtype=np.bool_)
        self._weights = np.ascontiguousarray(np.stack((st.weights, st.weights_safe, st.weights_unit)), dtype=np.float64)
        self._dummy_g, self._dummy_cv, _ = dummy_buffers()
        self._dummy_children_g = np.empty((1, 0), dtype=np.float64)
        # Ideal updates happen in place; never write into a caller's checkpoint array.
        st.ideal = st.ideal.copy()

    @classmethod
    def build(cls, st: MOEADState, kernel: KernelBackend, cross_is_de: bool) -> SteadyStateStep | None:
        """Return the compiled step for ``st``, or ``None`` when the configuration is not covered."""
        variation_kernels = kernel.variation_kernels()
        if variation_kernels is None or getattr(variation_kernels, "name", None) != "numba":
            return None
        if int(st.batch_size) != 1 or st.track_genealogy or not _supported_state(st):
            return None
        variation = one_offspring_real_variation(st.crossover_fn, st.mutation_fn, st.X.shape[1])
        if variation is None or (variation.crossover == "de") != cross_is_de:
            return None
        if not _compiled_kernels_available():
            return None
        return cls(st, variation)

    def _draw(self, st: MOEADState) -> tuple[int, bool, int, int, int, bool]:
        """Draw the next step's subproblem, parents and variation grids in the reference order.

        Grids land in ``self._draws``. SBX stops before its spread and swap grids,
        which depend on the parents (see :meth:`_finish_sbx`).
        """
        rng = st.rng
        pop_size = st.pop_size
        cursor = st.subproblem_cursor
        if st.subproblem_order.size != pop_size or cursor >= pop_size:
            st.subproblem_order = rng.permutation(pop_size).astype(int, copy=False)
            cursor = 0
        index = int(st.subproblem_order[cursor])
        st.subproblem_cursor = cursor + 1
        use_neighbors = rng.random() < st.delta
        high = st.neighbors.shape[1] if use_neighbors else pop_size
        first = int(rng.integers(0, high, dtype=np.int64))
        second = int(rng.integers(0, high - 1, dtype=np.int64))
        if self._is_de:
            rng.random(out=self._rand)
            j_forced = int(rng.integers(0, self._n_var, dtype=np.int64))
            rng.random(out=self._pm_mask)
            rng.random(out=self._pm_delta)
            return index, use_neighbors, first, second, j_forced, False
        crossed = rng.random() <= self._sbx_prob
        if crossed and self._use_var_draws:
            rng.random(out=self._var_draws)
        return index, use_neighbors, first, second, 0, crossed

    def _finish_sbx(self, rng: np.random.Generator, any_active: bool, child: np.ndarray[Any, Any]) -> None:
        # SBX draws its spread and swap grids only when some variable crosses; mutation draws follow.
        if any_active:
            rng.random(out=self._rand)
            rng.random(out=self._swap)
        rng.random(out=self._pm_mask)
        rng.random(out=self._pm_delta)
        self._sbx_child(
            any_active, self._draws, self._bounds, self._params, self._squares, self._slots, self._mating, self._sbx_active, child
        )

    def _set_pending(self, st: MOEADState, child: np.ndarray[Any, Any]) -> None:
        st.pending_offspring = child
        st.pending_active_indices = self._active
        st.pending_parent_pairs = self._pairs
        st.pending_use_neighbors = self._flags

    def ask(self, st: MOEADState) -> np.ndarray[Any, Any] | None:
        """Create the next child, or return ``None`` to let the reference ``ask`` handle the call."""
        if st.n_eval >= st.max_evals or st.batch_size != 1:
            return None
        index, use_neighbors, first, second, j_forced, crossed = self._draw(st)
        child = np.empty(self._child_shape)
        if self._is_de:
            self._de_child(
                st.X,
                st.neighbors,
                index,
                use_neighbors,
                first,
                second,
                j_forced,
                self._draws,
                self._bounds,
                self._params,
                self._squares,
                self._slots,
                self._flags,
                self._mating,
                child,
            )
        else:
            any_active = self._sbx_parents(
                st.X,
                st.neighbors,
                index,
                use_neighbors,
                first,
                second,
                crossed,
                self._use_var_draws,
                self._draws,
                self._params,
                self._slots,
                self._flags,
                self._mating,
                self._sbx_active,
            )
            self._finish_sbx(st.rng, any_active, child)
        self._set_pending(st, child)
        return child

    def _replace_args(self, st: MOEADState, eval_result: EvaluationResult) -> ReplaceArgs | None:
        """Start the reference tell (budget, pending state, candidate order) and return the replacement arguments.

        Returns ``None``, with the state untouched, when the pending ask or the
        evaluation result is not one the compiled replacement takes.
        """
        children = st.pending_offspring
        active = st.pending_active_indices
        use_neighbors = st.pending_use_neighbors
        if children is None or active is None or use_neighbors is None or children.shape != self._child_shape:
            return None
        F_child = eval_result.F
        f64 = self._f64
        if F_child.dtype is not f64 or F_child.shape != self._f_shape:
            return None
        G_child = eval_result.G if st.constraint_mode != "none" else None
        if G_child is not None and (G_child.dtype is not f64 or G_child.ndim != 2 or G_child.shape[0] != 1):
            return None
        st.n_eval += 1
        st.pending_offspring = None
        st.pending_active_indices = None
        st.pending_parent_pairs = None
        st.pending_use_neighbors = None

        order = st.neighbors[active[0]] if use_neighbors[0] else st.rng.permutation(st.pop_size)
        G = st.G
        if G is None or G_child is None:
            return (
                st.X,
                st.F,
                self._dummy_g,
                self._dummy_cv,
                self._weights,
                st.ideal,
                children,
                F_child,
                self._dummy_children_g,
                0.0,
                order,
                0,
            )
        if st.cv is None:
            st.cv = compute_violation(G)
        child_cv = float(compute_violation(G_child)[0])
        return st.X, st.F, G, st.cv, self._weights, st.ideal, children, F_child, G_child, child_cv, order, 1

    def _replace_with_numpy_ideal(self, st: MOEADState, args: ReplaceArgs) -> None:
        # The compiled ideal update declined (NaN or signed-zero tie): use the reference np.minimum.
        st.ideal = np.minimum(st.ideal, args[7].min(axis=0))
        self._replace(*args[:5], st.ideal, *args[6:], self._slots, self._params, False)

    def tell(self, st: MOEADState, eval_result: EvaluationResult) -> bool | None:
        """Apply one evaluated child, or return ``None`` (state untouched) for the reference ``tell``."""
        args = self._replace_args(st, eval_result)
        if args is None:
            return None
        if self._replace(*args, self._slots, self._params, True) < 0:
            self._replace_with_numpy_ideal(st, args)
        if st.archive_manager is not None:
            update_archive(st, st.X, st.F, st.G)
        hv_tracker = st.hv_tracker
        return hv_tracker is not None and hv_tracker.enabled and hv_tracker.reached(st.hv_points())

    def _replace_then_ask(self, st: MOEADState, args: ReplaceArgs) -> np.ndarray[Any, Any] | None:
        """Finish the pending tell and, while budget remains, ask for the next child in the same compiled call."""
        if st.n_eval >= st.max_evals:
            if self._replace(*args, self._slots, self._params, True) < 0:
                self._replace_with_numpy_ideal(st, args)
            return None
        index, use_neighbors, first, second, j_forced, crossed = self._draw(st)
        child = np.empty(self._child_shape)
        neighbors = st.neighbors
        if self._is_de:
            status = self._replace_then_de_child(
                *args,
                neighbors,
                index,
                use_neighbors,
                first,
                second,
                j_forced,
                self._draws,
                self._bounds,
                self._params,
                self._squares,
                self._slots,
                self._flags,
                self._mating,
                child,
            )
            if status < 0:
                self._replace_with_numpy_ideal(st, args)
                self._de_child(
                    st.X,
                    neighbors,
                    index,
                    use_neighbors,
                    first,
                    second,
                    j_forced,
                    self._draws,
                    self._bounds,
                    self._params,
                    self._squares,
                    self._slots,
                    self._flags,
                    self._mating,
                    child,
                )
        else:
            sbx_args = (
                index,
                use_neighbors,
                first,
                second,
                crossed,
                self._use_var_draws,
                self._draws,
                self._params,
                self._slots,
                self._flags,
                self._mating,
                self._sbx_active,
            )
            any_active = self._replace_then_sbx_parents(*args, neighbors, *sbx_args)
            if any_active < 0:
                self._replace_with_numpy_ideal(st, args)
                any_active = self._sbx_parents(st.X, neighbors, *sbx_args)
            self._finish_sbx(st.rng, any_active > 0, child)
        self._set_pending(st, child)
        return child

    def run(
        self,
        st: MOEADState,
        problem: ProblemProtocol,
        eval_strategy: EvaluationBackend,
        generation: int,
        reference_tell: Callable[[EvaluationResult], object],
    ) -> int:
        """Run the remaining steps of a run that no callback observes and return the generation counter.

        ``MOEAD.run`` uses this without live callbacks, an HV target or an external
        archive: the replacement for one child and the creation of the next then
        share a compiled call. Draws, updates, budget and generation counter follow
        the reference loop exactly; ``reference_tell`` takes any evaluation result
        that the compiled replacement does not.
        """
        child = self.ask(st)
        while child is not None:
            st.generation = generation
            eval_result = eval_strategy.evaluate(child, problem)
            args = self._replace_args(st, eval_result)
            if args is None:
                reference_tell(eval_result)
                child = self.ask(st)
            else:
                child = self._replace_then_ask(st, args)
            generation += 1
        st.generation = generation
        return generation


__all__ = ["SteadyStateStep"]
