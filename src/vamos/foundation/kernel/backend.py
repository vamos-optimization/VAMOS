from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
RealArray: TypeAlias = NDArray[np.floating[Any]]
NumericArray: TypeAlias = NDArray[np.number[Any]]
IndexArray: TypeAlias = NDArray[np.int_]
DecisionArray: TypeAlias = NDArray[np.generic]


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
    ) -> tuple[DecisionArray, FloatArray]: ...

    @overload
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: Literal[True],
    ) -> tuple[DecisionArray, FloatArray, IndexArray]: ...

    @abstractmethod
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[DecisionArray, FloatArray] | tuple[DecisionArray, FloatArray, IndexArray]:
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
