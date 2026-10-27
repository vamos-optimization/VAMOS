"""Compiled one-offspring step for steady-state NSGA-II.

When the kernel backend provides ``nsga2_steady_state_kernels`` (the Numba
engine) and the run uses the supported configuration, each ask/tell step runs in
a few compiled calls on buffers owned by the run instead of the reference
NumPy/Python path. The step reproduces that path exactly:

- every random value is still drawn here from ``state.rng`` with the same calls,
  in the same order, sizes and conditions; consecutive uniform draws are
  requested together, which yields the same values and generator state;
- tournament selection, SBX, clipping, polynomial mutation, front insertion,
  crowding and truncation reuse the compiled reference arithmetic and ties;
- the truncated front is still ordered with ``np.argsort``.

Supported configuration: one offspring per step, unconstrained populations
(``G is None``), binary tournament selection, real-coded SBX with polynomial
mutation and the default clip repair, without genealogy, immigration, parent
filters or non-breeding indices. Any other configuration, or a step whose
inputs fall outside the kernels' contract (for example non-finite objective
values), keeps the reference path. External archives and hypervolume
termination are updated after every step exactly as on the reference path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.termination import capped_offspring_size
from vamos.engine.operators.impl.real import ClampRepair, PolynomialMutation, SBXCrossover
from vamos.foundation.kernel.backend import STEADY_STATE_REJECTED, STEADY_STATE_UNSUPPORTED, NSGA2SteadyStateKernels

if TYPE_CHECKING:
    from vamos.engine.variation import VariationPipeline

    from .state import NSGAIIState


class SteadyStateStep:
    """Settings and reusable buffers of the compiled steady-state step of one run."""

    __slots__ = (
        "active",
        "comb_F",
        "comb_crowd",
        "comb_orders",
        "comb_ranks",
        "head",
        "head_size",
        "index_work",
        "kernels",
        "n_obj",
        "n_var",
        "no_coins",
        "orders",
        "orders_for",
        "params",
        "bounds",
        "pipeline",
        "pop_size",
        "single",
        "spare_orders",
        "split_crowd",
        "stream",
        "winners",
        "work",
    )

    def __init__(
        self,
        kernels: NSGA2SteadyStateKernels,
        pipeline: VariationPipeline,
        pop_size: int,
        n_var: int,
        n_obj: int,
        params: np.ndarray[Any, Any],
        bounds: np.ndarray[Any, Any],
    ) -> None:
        self.kernels = kernels
        self.pipeline = pipeline
        self.pop_size = pop_size
        self.n_var = n_var
        self.n_obj = n_obj
        self.params = params
        self.bounds = bounds
        size = pop_size + 1
        self.winners = np.empty(2, dtype=np.int64)
        self.no_coins = np.zeros(0, dtype=int)
        # SBX pair draw, variable draws, spread and swap draws, mutation mask and deltas.
        self.stream = np.empty(1 + 7 * n_var, dtype=np.float64)
        # Draws every step consumes whatever SBX does: the pair draw and both mutation grids.
        self.head_size = 1 + 4 * n_var
        self.head = self.stream[: self.head_size]
        self.work = np.empty((2, n_var), dtype=np.float64)
        self.active = np.empty((1, n_var), dtype=np.bool_)
        self.comb_F = np.empty((size, n_obj), dtype=np.float64)
        self.comb_ranks = np.empty(size, dtype=np.int64)
        self.comb_crowd = np.empty(size, dtype=np.float64)
        self.comb_orders = np.empty((n_obj, size), dtype=np.int64)
        self.index_work = np.empty((3, size), dtype=np.int64)
        self.split_crowd = np.empty(size, dtype=np.float64)
        self.single = np.zeros(1, dtype=np.int64)
        # Rows sorted by each objective (ties by row), valid for the objective array ``orders_for``.
        self.orders = np.empty((n_obj, pop_size), dtype=np.int64)
        self.spare_orders = np.empty((n_obj, pop_size), dtype=np.int64)
        self.orders_for: np.ndarray[Any, Any] | None = None


def build_steady_state_step(kernel: Any, st: NSGAIIState) -> SteadyStateStep | None:
    """Return the compiled step for a supported steady-state run, else ``None``."""
    factory = getattr(kernel, "nsga2_steady_state_kernels", None)
    kernels = factory() if callable(factory) else None
    if kernels is None:
        return None
    if not (st.incremental_enabled and st.offspring_size == 1 and st.replacement_size == 1):
        return None
    if st.track_genealogy or st.immigration_manager is not None or len(st.operator_pool) != 1:
        return None
    if st.sel_method != "tournament" or st.pressure != 2:
        return None
    pipeline = st.variation
    if pipeline.encoding != "real" or pipeline.parents_per_group != 2 or pipeline.children_per_group != 2:
        return None
    crossover = getattr(pipeline.crossover_op, "operator", None)
    mutation = getattr(pipeline.mutation_op, "operator", None)
    repair: object = pipeline.repair_op
    if type(crossover) is not SBXCrossover or type(mutation) is not PolynomialMutation or type(repair) is not ClampRepair:
        return None
    X, F = st.X, st.F
    if X.ndim != 2 or F.ndim != 2 or X.dtype != np.float64 or X.shape[0] != st.pop_size or X.shape[0] < 2:
        return None
    n_var = X.shape[1]
    repair_bounds = (np.asarray(pipeline.xl, dtype=float), np.asarray(pipeline.xu, dtype=float))
    for bound in (crossover.lower, crossover.upper, mutation.lower, mutation.upper, *repair_bounds):
        if bound.shape != (n_var,):
            return None
    params, bounds = kernels.pack_variation(
        prob_crossover=crossover.prob,
        eta_crossover=crossover.eta,
        prob_var=crossover.prob_var,
        prob_mutation=mutation.prob,
        eta_mutation=mutation.eta,
        crossover_bounds=(crossover.lower, crossover.upper),
        repair_bounds=repair_bounds,
        mutation_bounds=(mutation.lower, mutation.upper),
    )
    return SteadyStateStep(kernels, pipeline, int(st.pop_size), n_var, int(F.shape[1]), params, bounds)


def _supported(st: NSGAIIState, step: SteadyStateStep) -> bool:
    """Whether this step's state is inside the compiled path's contract."""
    return (
        st.incremental_enabled
        and st.G is None
        and st.ranks is not None
        and st.crowding is not None
        and st.variation is step.pipeline
        and st.sel_method == "tournament"
        and st.pressure == 2
        and st.parent_selection_filter is None
        and st.non_breeding_indices.size == 0
        and st.immigration_manager is None
        and not st.track_genealogy
        and st.X.shape == (step.pop_size, step.n_var)
        and st.X.dtype == np.float64
    )


def ask_steady_state(st: NSGAIIState, step: SteadyStateStep) -> np.ndarray[Any, Any] | None:
    """Produce the step's offspring like the reference ``ask``; ``None`` keeps the reference path."""
    if not _supported(st, step):
        return None
    if capped_offspring_size(st.n_eval, st.max_evals, st.offspring_size, "NSGA-II") != 1:
        return None
    ranks, crowding = st.ranks, st.crowding
    assert ranks is not None and crowding is not None
    rng = st.rng
    kernels = step.kernels
    n = step.pop_size
    # Reference binary tournament: two candidate vectors, then one coin per full tie.
    first = rng.integers(0, n, size=2, dtype=int)
    second = rng.integers(0, n - 1, size=2, dtype=int)
    n_ties = kernels.tournament(ranks, crowding, first, second, step.winners)
    coins = rng.integers(0, 2, size=n_ties, dtype=int) if n_ties else step.no_coins
    rng.random(out=step.head)
    X_off = np.empty((1, step.n_var), dtype=np.float64)
    args = (step.params, step.bounds, step.work, step.active, X_off)
    needed = kernels.offspring(st.X, step.winners, first, second, coins, step.stream, step.head_size, *args)
    if needed:
        # SBX changed variables: its variable/spread/swap draws precede the mutation grids.
        rng.random(out=step.stream[step.head_size : needed])
        kernels.offspring(st.X, step.winners, first, second, coins, step.stream, needed, *args)
    st.pending_offspring = X_off
    return X_off


def tell_steady_state(
    st: NSGAIIState,
    step: SteadyStateStep,
    X_off: np.ndarray[Any, Any],
    F_off: Any,
    G_off: Any,
) -> bool | None:
    """Apply the reference steady-state survival to the evaluated offspring.

    Returns ``True`` when the offspring entered the population, ``False`` when
    the reference early rejection kept the population unchanged, and ``None``
    (without changing the state) when the reference path must handle the step.
    The caller updates archives and checks hypervolume termination.
    """
    if G_off is not None or not _supported(st, step):
        return None
    if not isinstance(F_off, np.ndarray) or F_off.shape != (1, step.n_obj) or F_off.dtype != np.float64:
        return None
    if X_off.shape != (1, step.n_var) or X_off.dtype != np.float64:
        return None
    kernels = step.kernels
    F, ranks, crowding = st.F, st.ranks, st.crowding
    assert ranks is not None and crowding is not None
    if step.orders_for is not F:
        # Stable sorts give the reference tie order; afterwards the kernels maintain the orders.
        step.orders[:] = np.argsort(F, axis=0, kind="stable").T
        step.orders_for = F
    status = kernels.insert(
        F,
        ranks,
        crowding,
        F_off,
        step.orders,
        step.comb_F,
        step.comb_ranks,
        step.comb_crowd,
        step.comb_orders,
        step.index_work,
        step.split_crowd,
    )
    if status == STEADY_STATE_UNSUPPORTED:
        return None
    st.n_eval += 1
    st.pending_offspring_ids = None
    if status == STEADY_STATE_REJECTED:
        return False
    # Reference ordering of the truncated front: reversed np.argsort of its crowding.
    order = step.split_crowd[:status].argsort() if status > 1 else step.single
    n = step.pop_size
    out_X = np.empty((n, step.n_var), dtype=np.float64)
    out_F = np.empty((n, step.n_obj), dtype=np.float64)
    out_ranks = np.empty(n, dtype=np.int64)
    out_crowd = np.empty(n, dtype=np.float64)
    out_orders = step.spare_orders
    kernels.replace(
        st.X,
        X_off,
        step.comb_F,
        step.comb_ranks,
        step.comb_crowd,
        step.comb_orders,
        order,
        status,
        out_X,
        out_F,
        out_ranks,
        out_crowd,
        out_orders,
        step.index_work,
    )
    step.spare_orders, step.orders, step.orders_for = step.orders, out_orders, out_F
    st.X, st.F, st.G = out_X, out_F, None
    # Fronts are rebuilt from the ranks only if a later step takes the reference path.
    st.ranks, st.crowding, st.fronts = out_ranks, out_crowd, None
    return True


__all__ = ["SteadyStateStep", "ask_steady_state", "build_steady_state_step", "tell_steady_state"]
