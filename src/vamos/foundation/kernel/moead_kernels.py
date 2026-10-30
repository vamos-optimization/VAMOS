"""Kernel contract of MOEA/D's compiled one-offspring step and neighbourhood replacement.

**Development after 1.1.0.** ``KernelBackend.moead_steady_state_kernels`` returns
an implementation of :class:`MOEADSteadyStateKernels` (the Numba backend does)
or ``None``. The kernels never draw random numbers: MOEA/D draws every value
from the run generator in the order and shape of its reference ask/tell and
writes the draws into the named views of :class:`MOEADStepWorkspace`. This
module imports no optional dependency; the compiled code lives in
``_numba_moead_steady_state`` and ``_numba_moead_neighborhood``.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

#: Aggregation ids taken by the compiled neighbourhood replacement (``agg_id``).
AGG_TCHEBYCHEFF = 0
AGG_WEIGHTED_SUM = 1
AGG_PBI = 2
AGG_MODIFIED_TCHEBYCHEFF = 3

#: Replacement status: the compiled ideal-point update declined and nothing changed.
#: ``np.minimum`` could pick between two bit patterns (a NaN, or ``0.0`` against ``-0.0``);
#: the caller updates the ideal point with NumPy and replaces with ``update_ideal=False``.
IDEAL_DECLINED = -1


class MOEADStepWorkspace(NamedTuple):
    """Caller-owned buffers of the compiled step, reused across steps.

    Fill the draw views in draw order: ``var_draws`` (SBX per-variable draws),
    ``rand`` (DE grid or SBX spread), then ``sbx_draws`` (SBX spread, swap and
    mutation grids) or ``mutation_draws`` (the polynomial-mutation mask and delta
    grids). Consecutive views share memory, so one ``Generator.random(out=...)``
    fill draws several grids exactly as consecutive fills would. ``active``,
    ``pairs`` and ``flags`` receive the subproblem, the parents and the
    neighbourhood flag of the last child.
    """

    draws: FloatArray
    var_draws: FloatArray
    rand: FloatArray
    mutation_draws: FloatArray
    sbx_draws: FloatArray
    slots: IntArray
    active: IntArray
    pairs: IntArray
    flags: BoolArray
    mating: FloatArray
    sbx_active: BoolArray


class MOEADSteadyStateKernels(Protocol):
    """Compiled one-offspring MOEA/D step that reproduces the reference ask/tell path.

    ``replace`` and the ``replace_then_*`` kernels share the leading arguments
    ``X, F, G, cv, weights, ideal, children, children_f, children_g, child_cv,
    order, use_constraints``: ``weights`` stacks the plain, zero-safe and unit
    weight matrices; ``G``, ``cv`` and ``children_g`` are empty when the step is
    unconstrained. ``params`` and ``bounds`` come from :meth:`pack_variation`.
    """

    name: str

    def workspace(self, n_var: int, replace_limit: int, aggregation_id: int) -> MOEADStepWorkspace:
        """Return the buffers of one run."""
        ...

    def pack_variation(
        self,
        *,
        de_f: float,
        de_cr: float,
        sbx_prob_var: float,
        sbx_eta: float,
        pm_prob: float,
        pm_eta: float,
        theta: float,
        rho: float,
        clip_bounds: tuple[FloatArray, FloatArray],
        mutation_bounds: tuple[FloatArray, FloatArray],
        crossover_bounds: tuple[FloatArray, FloatArray],
    ) -> tuple[FloatArray, FloatArray]:
        """Pack operator and aggregation settings into ``(params, bounds)``."""
        ...

    def de_child(
        self,
        X: FloatArray,
        neighbors: IntArray,
        index: int,
        use_neighbors: bool,
        first: int,
        second: int,
        j_forced: int,
        draws: FloatArray,
        bounds: FloatArray,
        params: FloatArray,
        slots: IntArray,
        flags: BoolArray,
        mating: FloatArray,
        child: FloatArray,
    ) -> None:
        """Write the repaired, mutated DE child of subproblem ``index``.

        ``first`` and ``second`` are the two parent draws (the second skips the
        first), taken from the neighbourhood row when ``use_neighbors``.
        """
        ...

    def sbx_parents(
        self,
        X: FloatArray,
        neighbors: IntArray,
        index: int,
        use_neighbors: bool,
        first: int,
        second: int,
        crossed: bool,
        use_var_draws: bool,
        draws: FloatArray,
        params: FloatArray,
        slots: IntArray,
        flags: BoolArray,
        mating: FloatArray,
        sbx_active: BoolArray,
    ) -> bool:
        """Gather the SBX parents; return whether a variable crosses (the spread and swap draws follow)."""
        ...

    def sbx_child(
        self,
        any_active: bool,
        draws: FloatArray,
        bounds: FloatArray,
        params: FloatArray,
        slots: IntArray,
        mating: FloatArray,
        sbx_active: BoolArray,
        child: FloatArray,
    ) -> None:
        """Write the repaired, mutated first SBX child."""
        ...

    def replace(
        self,
        X: FloatArray,
        F: FloatArray,
        G: FloatArray,
        cv: FloatArray,
        weights: FloatArray,
        ideal: FloatArray,
        children: FloatArray,
        children_f: FloatArray,
        children_g: FloatArray,
        child_cv: float,
        order: IntArray,
        use_constraints: int,
        slots: IntArray,
        params: FloatArray,
        update_ideal: bool,
    ) -> int:
        """Update the ideal point (when ``update_ideal``) and replace neighbours.

        Returns the number of replacements, or :data:`IDEAL_DECLINED`.
        """
        ...

    def replace_then_de_child(
        self,
        X: FloatArray,
        F: FloatArray,
        G: FloatArray,
        cv: FloatArray,
        weights: FloatArray,
        ideal: FloatArray,
        children: FloatArray,
        children_f: FloatArray,
        children_g: FloatArray,
        child_cv: float,
        order: IntArray,
        use_constraints: int,
        neighbors: IntArray,
        index: int,
        use_neighbors: bool,
        first: int,
        second: int,
        j_forced: int,
        draws: FloatArray,
        bounds: FloatArray,
        params: FloatArray,
        slots: IntArray,
        flags: BoolArray,
        mating: FloatArray,
        child: FloatArray,
    ) -> int:
        """:meth:`replace` for the evaluated child, then :meth:`de_child`; :data:`IDEAL_DECLINED` means neither ran."""
        ...

    def replace_then_sbx_parents(
        self,
        X: FloatArray,
        F: FloatArray,
        G: FloatArray,
        cv: FloatArray,
        weights: FloatArray,
        ideal: FloatArray,
        children: FloatArray,
        children_f: FloatArray,
        children_g: FloatArray,
        child_cv: float,
        order: IntArray,
        use_constraints: int,
        neighbors: IntArray,
        index: int,
        use_neighbors: bool,
        first: int,
        second: int,
        crossed: bool,
        use_var_draws: bool,
        draws: FloatArray,
        params: FloatArray,
        slots: IntArray,
        flags: BoolArray,
        mating: FloatArray,
        sbx_active: BoolArray,
    ) -> int:
        """:meth:`replace`, then :meth:`sbx_parents`: :data:`IDEAL_DECLINED`, or ``1``/``0`` for whether a variable crosses."""
        ...


def available_moead_neighborhood_kernels() -> tuple[Callable[..., int], Callable[..., None]] | None:
    """Return the compiled neighbourhood updates ``(single, batch)`` when numba is installed.

    MOEA/D has used them whenever numba is importable, independently of the
    engine; they reproduce the Python update, so they change speed, not results.
    An installed but unloadable numba falls back to ``None``.
    """
    if find_spec("numba") is None:
        return None
    try:
        from ._numba_moead_neighborhood import update_neighborhood_batch_numba, update_neighborhood_numba
    except (ImportError, OSError, RuntimeError, AttributeError):
        return None
    return update_neighborhood_numba, update_neighborhood_batch_numba


__all__ = [
    "AGG_MODIFIED_TCHEBYCHEFF",
    "AGG_PBI",
    "AGG_TCHEBYCHEFF",
    "AGG_WEIGHTED_SUM",
    "IDEAL_DECLINED",
    "MOEADStepWorkspace",
    "MOEADSteadyStateKernels",
    "available_moead_neighborhood_kernels",
]
