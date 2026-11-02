from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .moead_kernels import MOEADSteadyStateKernels

FloatArray: TypeAlias = NDArray[np.float64]
RealArray: TypeAlias = NDArray[np.floating[Any]]
NumericArray: TypeAlias = NDArray[np.number[Any]]
IndexArray: TypeAlias = NDArray[np.int_]
DecisionArray: TypeAlias = NDArray[np.generic]


class SingleChildVariation(Protocol):
    """Compiled steady-state child generation bound to one real-coded operator chain."""

    def __call__(self, X: FloatArray, first: int, second: int, out: FloatArray, rng: np.random.Generator) -> None:
        """Write one child of rows ``first`` and ``second`` of ``X`` into ``out`` shaped ``(1, n_var)``."""
        ...


class VariationKernels(Protocol):
    """Compiled variation arithmetic that preserves the NumPy operator contracts.

    Implementations draw from the supplied generator in the same order and shape
    as the reference real-coded operators and modify arrays in place.
    """

    name: str

    def sbx_crossover_pairs(
        self,
        offspring: FloatArray,
        *,
        rng: np.random.Generator,
        lower: FloatArray,
        upper: FloatArray,
        prob_crossover: float,
        eta: float,
        prob_var: float,
    ) -> FloatArray:
        """Apply SBX in place to a float64 tensor shaped ``(n_pairs, 2, n_var)``."""
        ...

    def polynomial_mutation(
        self,
        X: FloatArray,
        *,
        lower: FloatArray,
        upper: FloatArray,
        prob_mutation: float,
        eta: float,
        rnd_mask: FloatArray,
        rnd_delta: FloatArray,
    ) -> None:
        """Apply polynomial mutation in place from caller-provided uniform draws."""
        ...

    def differential_crossover(
        self,
        parents: FloatArray,
        f: float,
        cr: float,
        rand: FloatArray,
        j_rand: NDArray[np.integer[Any]],
    ) -> FloatArray:
        """DE binomial children ``base + f * (p1 - p2)`` from ``parents`` shaped ``(n, 3, n_var)``.

        ``parents[:, 2]`` is the base vector; a variable takes the mutant value when
        ``rand < cr`` or it is the forced index ``j_rand`` of its row.
        """
        ...

    def single_child_sbx_polynomial(
        self,
        *,
        sbx_prob: float,
        sbx_eta: float,
        sbx_prob_var: float,
        sbx_lower: FloatArray,
        sbx_upper: FloatArray,
        pm_prob: float,
        pm_eta: float,
        pm_lower: FloatArray,
        pm_upper: FloatArray,
        clip_lower: FloatArray,
        clip_upper: FloatArray,
    ) -> SingleChildVariation:
        """Steady-state child: the first SBX child of one pair, clipped, mutated and clipped.

        The returned callable draws from the given generator exactly like the
        reference ``SBXCrossover``, ``ClampRepair``, ``PolynomialMutation`` and
        ``ClampRepair`` chain applied to one mating pair.
        """
        ...


#: ``SMSEMOASurvival.insert`` status: the child duplicates a row and is discarded.
SMSEMOA_DISCARDED = -1
#: ``SMSEMOASurvival.insert`` status: infeasible rows take part, run the reference survival.
SMSEMOA_DELEGATE = -2
#: ``SMSEMOASurvival.insert`` status: compute the worst front's contributions, then commit.
SMSEMOA_CONTRIBUTIONS = -3


class SMSEMOASurvival(Protocol):
    """Compiled steady-state SMS-EMOA survival over cached Pareto ranks and crowding.

    ``insert`` returns ``(status, size)``. A non-negative status is the removed
    row, already committed (the population size means the child was rejected).
    With ``SMSEMOA_CONTRIBUTIONS`` the worst front is ``worst[:size]`` (rows of
    ``F_work``, the population followed by the child) and the caller commits the
    least contributor that it computes. The population arrays, ranks, crowding
    and the adaptive reference point (``nadir + ref_offset * (nadir - ideal)`` of
    ``F_work``) are updated in place.
    """

    F_work: FloatArray
    worst: NDArray[np.int64]

    def reset(self, F: FloatArray) -> None:
        """Rebuild the cached objective orders after the population changed outside ``insert``/``commit``."""
        ...

    def insert(
        self,
        X: FloatArray,
        F: FloatArray,
        G: FloatArray,
        x_child: FloatArray,
        f_child: FloatArray,
        g_child: FloatArray,
        ranks: NDArray[np.integer[Any]],
        crowding: FloatArray,
        ref_point: FloatArray,
    ) -> tuple[int, int]:
        """Insert the child, choose the row to remove and commit it when possible."""
        ...

    def commit(
        self,
        X: FloatArray,
        F: FloatArray,
        G: FloatArray,
        x_child: FloatArray,
        f_child: FloatArray,
        g_child: FloatArray,
        ranks: NDArray[np.integer[Any]],
        crowding: FloatArray,
        remove: int,
    ) -> None:
        """Remove population row ``remove`` after an ``SMSEMOA_CONTRIBUTIONS`` insertion."""
        ...


class SelectionKernels(Protocol):
    """Compiled survival and truncation loops that reproduce NumPy reference loops.

    The kernels never draw random numbers. Callers use them only for inputs
    the kernels accept (finite values) and keep their NumPy reference otherwise.
    """

    name: str

    def iterative_crowding_truncation(self, F: FloatArray, target_size: int) -> NDArray[np.int64]:
        """Repeatedly drop the first least-crowded row of a single front."""
        ...

    def greedy_two_nearest_selection(
        self,
        distances: FloatArray,
        remaining: NDArray[np.integer[Any]],
        best1: FloatArray,
        best2: FloatArray,
        scores: FloatArray,
        selected_count: int,
        crowd_dist: FloatArray,
    ) -> None:
        """AGE-MOEA greedy diversity scores written into ``crowd_dist``."""
        ...

    def lexicographic_truncation(self, dist: FloatArray, keep: int) -> NDArray[np.int64]:
        """SPEA2 truncation operator: drop the lexicographically closest row until ``keep`` remain."""
        ...

    def indicator_removal(self, fitness: FloatArray, contrib: FloatArray, target_size: int) -> NDArray[np.int64]:
        """IBEA iterative removal of the lowest fitness, updating ``fitness`` in place."""
        ...

    def smsemoa_survival(
        self,
        *,
        pop_size: int,
        n_obj: int,
        n_constraints: int,
        ref_offset: float,
        ref_adaptive: bool,
        eliminate_duplicates: bool,
        check_feasibility: bool,
    ) -> SMSEMOASurvival:
        """Steady-state SMS-EMOA survival for a population of ``pop_size`` rows."""
        ...


#: ``NSGA2SteadyStateKernels.insert`` result: a member of the worst front dominates the child.
STEADY_STATE_REJECTED = -1
#: ``NSGA2SteadyStateKernels.insert`` result: the child is outside the kernel contract (non-finite).
STEADY_STATE_UNSUPPORTED = -2


class NSGA2SteadyStateKernels(Protocol):
    """Compiled one-offspring NSGA-II step that reproduces the reference ask/tell path.

    The caller draws every random value from the run generator in the reference
    order and shape and passes the draws; the kernels never draw. Buffers are
    caller-owned and reused across steps.
    """

    name: str

    def pack_variation(
        self,
        *,
        prob_crossover: float,
        eta_crossover: float,
        prob_var: float,
        prob_mutation: float,
        eta_mutation: float,
        crossover_bounds: tuple[FloatArray, FloatArray],
        repair_bounds: tuple[FloatArray, FloatArray],
        mutation_bounds: tuple[FloatArray, FloatArray],
    ) -> tuple[FloatArray, FloatArray]:
        """Pack SBX, clip-repair and polynomial-mutation settings for ``offspring``."""
        ...

    def tournament(
        self,
        ranks: NDArray[np.integer[Any]],
        crowding: FloatArray,
        first: NDArray[np.int64],
        second: NDArray[np.int64],
        X: FloatArray,
        winners: NDArray[np.int64],
        differing: NDArray[np.int64],
    ) -> int:
        """Resolve the reference binary tournaments.

        Returns minus the number of tied tournaments, or, without ties, the
        number of variables SBX may change (their stream offsets in ``differing``).
        """
        ...

    def draws(self, stream: FloatArray, differing: NDArray[np.int64], count: int, params: FloatArray) -> int:
        """Return how many uniform draws the step consumes after a tie-free tournament."""
        ...

    def offspring(
        self,
        X: FloatArray,
        winners: NDArray[np.int64],
        first: NDArray[np.int64],
        second: NDArray[np.int64],
        coins: NDArray[np.integer[Any]],
        stream: FloatArray,
        available: int,
        params: FloatArray,
        bounds: FloatArray,
        work: FloatArray,
        active: NDArray[np.bool_],
        child: FloatArray,
    ) -> int:
        """Write the first SBX/polynomial-mutation child, or return the number of draws it needs."""
        ...

    def survival_workspace(self, pop_size: int, n_obj: int) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
        """Return ``(int_work, float_work, split_crowd)`` buffers for ``insert`` and ``replace``."""
        ...

    def insert(
        self,
        F: FloatArray,
        ranks: NDArray[np.integer[Any]],
        crowding: FloatArray,
        f_child: FloatArray,
        orders: NDArray[np.int64],
        comb_F: FloatArray,
        int_work: NDArray[np.int64],
        float_work: FloatArray,
    ) -> int:
        """Insert the child into the cached fronts; return the size of the truncated front.

        ``orders[m]`` lists the population rows sorted by objective ``m`` with
        ties in row order. The truncated front's crowding distances are left in
        the ``split_crowd`` view of ``float_work``. Returns
        ``STEADY_STATE_REJECTED`` or ``STEADY_STATE_UNSUPPORTED`` instead when
        the population stays unchanged or the input is not supported.
        """
        ...

    def replace(
        self,
        X: FloatArray,
        x_child: FloatArray,
        comb_F: FloatArray,
        int_work: NDArray[np.int64],
        float_work: FloatArray,
        order: NDArray[np.integer[Any]],
        out_X: FloatArray,
        out_F: FloatArray,
        out_ranks: NDArray[np.int64],
        out_crowd: FloatArray,
        out_orders: NDArray[np.int64],
    ) -> None:
        """Write the survivors, their ranks, crowding distances and per-objective orders.

        ``order`` is ``np.argsort`` of the truncated front's crowding distances.
        """
        ...


class KernelBackend(ABC):
    """
    Interface for all kernel backends used by the evolutionary algorithms.
    Backends implement heavy numeric primitives on vectorized populations.
    Performance-sensitive: implementations should be vectorized/SoA and avoid Python loops.
    """

    # Stable backend identifier (e.g., "numpy", "numba", "moocore").
    name: str = "unknown"

    # -------- Computation device / capability metadata --------

    def device(self) -> str:
        """
        Return a short label describing the primary execution device.
        Examples: "cpu", "gpu", "tpu".
        """
        return "cpu"

    def capabilities(self) -> Iterable[str]:
        """
        Optional backend capability tags (e.g., {"gpu", "indicator:hypervolume"}).
        """
        return ()

    def quality_indicators(self) -> Iterable[str]:
        """
        Quality indicators this backend can compute natively (e.g., {"hypervolume"}).
        """
        return ()

    def supports_quality_indicator(self, name: str) -> bool:
        normalized = name.lower()
        return any(ind.lower() == normalized for ind in self.quality_indicators())

    def variation_kernels(self) -> VariationKernels | None:
        """
        Optional compiled SBX/polynomial-mutation arithmetic.

        ``None`` means callers keep the reference NumPy operator code. A backend
        that returns kernels guarantees the reference random-draw contract.
        """
        return None

    def selection_kernels(self) -> SelectionKernels | None:
        """
        Optional compiled survival/truncation loops.

        ``None`` means callers keep their NumPy reference loops. A backend that
        returns kernels guarantees identical selections for finite inputs.
        """
        return None

    def nsga2_steady_state_kernels(self) -> NSGA2SteadyStateKernels | None:
        """
        Optional compiled one-offspring NSGA-II step.

        ``None`` means NSGA-II keeps its reference steady-state ask/tell path. A
        backend that returns kernels guarantees the same offspring, survivors,
        ranks and crowding distances as that path for the same draws.
        """
        return None

    def moead_steady_state_kernels(self) -> MOEADSteadyStateKernels | None:
        """
        Optional compiled one-offspring MOEA/D step (see :mod:`.moead_kernels`).

        ``None`` means MOEA/D keeps its reference ``batch_size=1`` ask/tell
        path. A backend that returns kernels guarantees the same children,
        ideal point and replacements as that path for the same draws.
        """
        return None

    # -------- Evolutionary kernels --------

    @abstractmethod
    def nsga2_ranking(self, F: RealArray) -> tuple[IndexArray, FloatArray]:
        """
        Return ranks and crowding distances for objective matrix F.
        """

    @abstractmethod
    def tournament_selection(
        self,
        ranks: NumericArray,
        crowding: RealArray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> IndexArray:
        """
        Select n_parents indices using tournament selection.
        """

    @abstractmethod
    def sbx_crossover(
        self,
        X_parents: RealArray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> FloatArray:
        """
        Apply SBX crossover to parent array and return offspring.
        """

    @abstractmethod
    def polynomial_mutation(
        self,
        X: RealArray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        """
        Apply polynomial mutation in-place to array X.
        """

    @overload
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: Literal[False] = False,
    ) -> tuple[DecisionArray, RealArray]: ...

    @overload
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: Literal[True],
    ) -> tuple[DecisionArray, RealArray, IndexArray]: ...

    @abstractmethod
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[DecisionArray, RealArray] | tuple[DecisionArray, RealArray, IndexArray]:
        """
        Perform NSGA-II elitist replacement returning new (X, F).
        When return_indices is True, also return selected indices in the combined population.
        """

    # -------- Quality indicator hooks --------

    def update_archive(
        self,
        archive_X: DecisionArray | None,
        archive_F: FloatArray | None,
        population_X: DecisionArray,
        population_F: FloatArray,
        archive_size: int,
    ) -> tuple[DecisionArray | None, FloatArray | None]:
        """
        Optional external archive update implementation.

        Backends that implement this method can be used by algorithms to maintain a
        non-dominated archive inside the kernel for performance.
        """
        raise NotImplementedError(f"Backend '{self.__class__.__name__}' does not support update_archive.")

    def hypervolume(self, points: FloatArray, reference_point: FloatArray) -> float:
        """
        Optional hypervolume implementation. Backends that override this method
        should also list "hypervolume" in quality_indicators().
        """
        raise NotImplementedError(f"Backend '{self.__class__.__name__}' does not support hypervolume.")

    def hypervolume_contributions(self, points: FloatArray, reference_point: FloatArray) -> FloatArray:
        """
        Optional least-contributor support for hypervolume-based survival.

        The default implementation delegates to the shared indicator module so
        algorithms can rely on a backend-owned hook without requiring every
        backend to provide a native implementation.
        """
        from vamos.foundation.quality_indicators.hypervolume import hypervolume_contributions as hv_contrib_fn

        return hv_contrib_fn(points, reference_point)
