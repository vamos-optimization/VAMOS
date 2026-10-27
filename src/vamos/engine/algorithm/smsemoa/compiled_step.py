"""Compiled steady-state step for SMS-EMOA.

SMS-EMOA creates and evaluates one child per step, so per-step Python overhead
dominates its run time. When the kernel offers compiled single-child variation
and SMS-EMOA survival (``engine="numba"``), ``CompiledSteadyStep`` replaces
``SMSEMOA._generate_offspring`` and ``survival_selection`` and reproduces them
bit for bit: parents and every variation draw come from the run generator in
the reference order, shape and conditions, and the population, cached ranks,
crowding distances and reference point evolve exactly as in the reference.

Supported configurations: real encoding with SBX, polynomial mutation and the
clip repair (``repair="auto"``); random or tournament parent selection;
duplicate elimination; constraints; adaptive or fixed reference points. The run
loop still updates archives, live callbacks and hypervolume termination after
each step. Two-objective contributions are compiled when they match the
installed reference (MooCore or the NumPy fallback); otherwise, and for three
or more objectives, the reference contribution function is called on the worst
front. Feasibility-first removal is compiled when the compiled violation sums
match NumPy's; non-finite objective values and anything else the kernels do not
accept run the reference survival. Other configurations, genealogy tracking
and kernels without the compiled hooks keep the reference path.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.operators.impl.real import ClampRepair, PolynomialMutation, SBXCrossover
from vamos.engine.operators.policies._real_operator_builder import RepairedRealOperator
from vamos.foundation.constraints.utils import is_feasible
from vamos.foundation.kernel.backend import SMSEMOA_CONTRIBUTIONS, SMSEMOA_DELEGATE, KernelBackend

from .helpers import fronts_from_ranks, survival_selection

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import SingleChildVariation, SMSEMOASurvival
    from vamos.foundation.problem.types import ProblemProtocol

    from .state import SMSEMOAState

__all__ = ["CompiledSteadyStep", "build_compiled_step"]


def _float_matrix(values: object, rows: int) -> bool:
    return (
        isinstance(values, np.ndarray)
        and values.ndim == 2
        and values.shape[0] == rows
        and values.dtype == np.float64
        and values.flags.c_contiguous
        and values.flags.writeable
    )


def _vector(values: object, size: int, kind: str) -> bool:
    return (
        isinstance(values, np.ndarray)
        and values.shape == (size,)
        and values.dtype.kind == kind
        and values.flags.c_contiguous
        and values.flags.writeable
    )


def _bounds(values: object, n_var: int) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    arr = np.asarray(values, dtype=np.float64)
    return arr if arr.shape == (n_var,) else None


def _reference_hooks(kernel: KernelBackend) -> bool:
    """Whether the kernel keeps the ranking and contribution hooks its compiled survival reproduces."""
    cls = type(kernel)
    owner = next(base for base in cls.__mro__ if "selection_kernels" in vars(base))
    return getattr(cls, "nsga2_ranking", None) is getattr(owner, "nsga2_ranking", None) and (
        getattr(cls, "hypervolume_contributions", None) is KernelBackend.hypervolume_contributions
    )


def _is_exactly(obj: object, cls: type) -> bool:
    return type(obj) is cls


def build_compiled_step(st: SMSEMOAState, kernel: KernelBackend) -> CompiledSteadyStep | None:
    """Return the compiled step for a supported state, or ``None`` to keep the reference path."""
    variation = kernel.variation_kernels()
    selection = kernel.selection_kernels()
    if variation is None or selection is None or not _reference_hooks(kernel):
        return None
    if st.track_genealogy or st.genealogy_tracker is not None or st.pending_offspring_ids is not None:
        return None
    if st.selection_method not in ("random", "tournament"):
        return None
    if not _float_matrix(st.X, st.X.shape[0]) or st.X.shape[0] < 2:
        return None
    n, n_var = st.X.shape
    if not _float_matrix(st.F, n) or not bool(np.isfinite(st.F).all()):
        return None
    n_obj = st.F.shape[1]
    if st.G is not None and not _float_matrix(st.G, n):
        return None
    if st.ranks is None or st.crowding is None:
        return None
    if not _vector(st.ranks, n, "i") or not _vector(st.crowding, n, "f") or st.crowding.dtype != np.float64:
        return None
    offset = float(st.ref_offset)
    if not math.isfinite(offset) or (offset == 0.0 and math.copysign(1.0, offset) < 0.0):
        return None

    crossover, mutation = st.crossover_fn, st.mutation_fn
    if not isinstance(crossover, RepairedRealOperator) or not isinstance(mutation, RepairedRealOperator):
        return None
    if not (_is_exactly(crossover.operator, SBXCrossover) and _is_exactly(mutation.operator, PolynomialMutation)):
        return None
    if not (_is_exactly(crossover.repair, ClampRepair) and _is_exactly(mutation.repair, ClampRepair)):
        return None
    sbx = cast(SBXCrossover, crossover.operator)
    pm = cast(PolynomialMutation, mutation.operator)
    if sbx.variation_kernels is not variation or pm.variation_kernels is not variation:
        return None
    raw_bounds = (sbx.lower, sbx.upper, pm.lower, pm.upper, crossover.xl, crossover.xu, mutation.xl, mutation.xu)
    bounds = [b for b in (_bounds(raw, n_var) for raw in raw_bounds) if b is not None]
    if len(bounds) != len(raw_bounds):
        return None
    sbx_lower, sbx_upper, pm_lower, pm_upper, clip_lower, clip_upper, mut_clip_lower, mut_clip_upper = bounds
    if mut_clip_lower.tobytes() != clip_lower.tobytes() or mut_clip_upper.tobytes() != clip_upper.tobytes():
        return None

    child = variation.single_child_sbx_polynomial(
        sbx_prob=sbx.prob,
        sbx_eta=sbx.eta,
        sbx_prob_var=sbx.prob_var,
        sbx_lower=sbx_lower,
        sbx_upper=sbx_upper,
        pm_prob=pm.prob,
        pm_eta=pm.eta,
        pm_lower=pm_lower,
        pm_upper=pm_upper,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    survival = selection.smsemoa_survival(
        pop_size=n,
        n_obj=n_obj,
        n_constraints=0 if st.G is None else int(st.G.shape[1]),
        ref_offset=offset,
        ref_adaptive=bool(st.ref_adaptive),
        eliminate_duplicates=bool(st.eliminate_duplicates),
        check_feasibility=st.G is not None and st.constraint_mode != "none" and st.G.size > 0,
    )
    st.ref_point = np.array(st.ref_point, dtype=np.float64)
    return CompiledSteadyStep(st, kernel, child, survival)


class CompiledSteadyStep:
    """One SMS-EMOA generation (select, vary, evaluate, survive) with compiled kernels."""

    def __init__(self, st: SMSEMOAState, kernel: KernelBackend, child: SingleChildVariation, survival: SMSEMOASurvival) -> None:
        self._st = st
        self._kernel = kernel
        self._child = child
        self._survival = survival
        self._n, self._n_var = st.X.shape
        self._tournament = st.selection_method == "tournament"
        self._no_G = np.empty((self._n, 0), dtype=np.float64)
        self._no_g = np.empty(0, dtype=np.float64)
        self._constrained = st.G is not None and st.constraint_mode != "none" and st.G.size > 0
        # The kernels handle finite populations; a non-finite row keeps the reference survival.
        self._finite = True
        survival.reset(st.F)

    def step(self, problem: ProblemProtocol, eval_strategy: EvaluationBackend) -> None:
        """Generate, evaluate and insert one child; counts one evaluation."""
        st = self._st
        if self._tournament:
            assert st.ranks is not None and st.crowding is not None
            parents = self._kernel.tournament_selection(st.ranks, st.crowding, st.pressure, st.rng, n_parents=2)
        else:
            parents = st.rng.choice(self._n, size=2, replace=True)
        X_child = np.empty((1, self._n_var), dtype=np.float64)
        self._child(st.X, int(parents[0]), int(parents[1]), X_child, st.rng)
        evaluated = eval_strategy.evaluate(X_child, problem)
        F_child, G_child = evaluated.F, evaluated.G
        if st.constraint_mode == "none":
            G_child = None
        st.n_eval += 1
        self._survive(X_child, F_child, G_child)

    def finish(self) -> None:
        """Restore the cached front lists, which the reference keeps unless infeasible rows remain."""
        st = self._st
        if st.fronts is None and st.ranks is not None:
            if not self._constrained or bool(np.all(is_feasible(st.G, n=st.F.shape[0]))):
                st.fronts = fronts_from_ranks(st.ranks)

    def _survive(self, X_child: np.ndarray[Any, Any], F_child: np.ndarray[Any, Any], G_child: np.ndarray[Any, Any] | None) -> None:
        st = self._st
        if st.G is None:
            G, g_child = self._no_G, self._no_g
        elif G_child is not None and G_child.shape == (1, st.G.shape[1]):
            G, g_child = st.G, G_child[0]
        else:
            self._reference(X_child, F_child, G_child)
            return
        if not self._finite or F_child.shape != (1, st.F.shape[1]):
            self._reference(X_child, F_child, G_child)
            return
        assert st.ranks is not None and st.crowding is not None
        st.fronts = None
        status, size = self._survival.insert(st.X, st.F, G, X_child[0], F_child[0], g_child, st.ranks, st.crowding, st.ref_point)
        if status == SMSEMOA_DELEGATE:
            self._reference(X_child, F_child, G_child)
        elif status == SMSEMOA_CONTRIBUTIONS:
            worst = self._survival.worst[:size]
            contributions = self._kernel.hypervolume_contributions(self._survival.F_work[worst], st.ref_point)
            remove = int(worst[int(np.argmin(contributions))])
            if remove != self._n:
                self._survival.commit(st.X, st.F, G, X_child[0], F_child[0], g_child, st.ranks, st.crowding, remove)

    def _reference(self, X_child: np.ndarray[Any, Any], F_child: np.ndarray[Any, Any], G_child: np.ndarray[Any, Any] | None) -> None:
        st = self._st
        ranks = st.ranks
        survival_selection(st, X_child, F_child, G_child, self._kernel)
        if st.ranks is not ranks:
            self._finite = bool(np.isfinite(st.F).all())
            self._survival.reset(st.F)
