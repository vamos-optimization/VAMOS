"""
Algorithm protocol and type definitions.

Defines the interface that all multi-objective optimization algorithms should follow.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol


class SelectionMethod(str, Enum):
    """Parent selection methods."""

    TOURNAMENT = "tournament"
    RANDOM = "random"
    BOLTZMANN = "boltzmann"
    RANKING = "ranking"
    SUS = "sus"

    def __str__(self) -> str:
        return self.value


class SurvivalMethod(str, Enum):
    """Survival selection methods."""

    RANK_CROWDING = "rank_crowding"
    RANK_ONLY = "rank_only"
    TRUNCATION = "truncation"

    def __str__(self) -> str:
        return self.value


class ConstraintMode(str, Enum):
    """Constraint handling modes that the built-in algorithms implement."""

    NONE = "none"
    FEASIBILITY = "feasibility"

    def __str__(self) -> str:
        return self.value


@runtime_checkable
class AlgorithmProtocol(Protocol):
    """
    Protocol defining the interface for multi-objective optimization algorithms.

    All algorithms should implement at minimum the `run()` method.
    The `ask()` and `tell()` methods are optional for algorithms that support
    incremental/interactive optimization.

    Attributes:
        cfg: Algorithm configuration dictionary
        kernel: Kernel backend for vectorized operations
    """

    cfg: dict[str, Any]
    kernel: KernelBackend

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: Any | None = None,
        live_viz: Any | None = None,
    ) -> dict[str, Any]:
        """
        Run the optimization algorithm.

        Parameters
        ----------
        problem : ProblemProtocol
            The optimization problem to solve.
        termination : tuple[str, Any]
            Termination criterion, e.g., ("max_evaluations", 10000) or ("hv", {...}).
            Defaults to ("max_evaluations", 25000).
        seed : int
            Random seed for reproducibility (default: 0).
        eval_strategy : Any | None
            Evaluation backend for problem evaluations.
        live_viz : Any | None
            Live visualization callback.

        Returns
        -------
        dict[str, Any]
            Result dictionary containing at minimum:
            - "X": Final population decision variables (n_pop, n_var)
            - "F": Final population objective values (n_pop, n_obj)
            - "evaluations": Total number of function evaluations
            Optional keys: "G" (constraints), "archive", "hv_reached", etc.
        """
        ...


@runtime_checkable
class InteractiveAlgorithmProtocol(AlgorithmProtocol, Protocol):
    """
    Extended protocol for algorithms supporting ask/tell interaction.

    This allows external control of the optimization loop, useful for:
    - Custom evaluation backends
    - Interactive optimization
    - Integration with external simulators
    """

    def ask(self) -> np.ndarray[Any, Any]:
        """
        Generate offspring solutions to be evaluated.

        Returns
        -------
        np.ndarray[Any, Any]
            Offspring decision variables to evaluate, shape (n_offspring, n_var).

        Raises
        ------
        RuntimeError
            If called before initialization (run not started).
        """
        ...

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """
        Receive evaluated offspring and update algorithm state.

        Parameters
        ----------
        eval_result : Any
            Evaluation result — accepts:
            - ``np.ndarray[Any, Any]`` of objectives, shape ``(n_offspring, n_obj)``
            - ``dict`` with ``"F"`` key (and optionally ``"G"``)
            - Object with ``.F`` attribute (and optionally ``.G``)
        problem : ProblemProtocol, optional
            Problem instance (unused by most algorithms, kept for interface
            consistency).

        Returns
        -------
        bool
            True if termination criterion reached (e.g., HV threshold).

        Raises
        ------
        RuntimeError
            If called before initialization or without pending ask().
        """
        ...


__all__ = [
    "AlgorithmProtocol",
    "InteractiveAlgorithmProtocol",
    "SelectionMethod",
    "SurvivalMethod",
    "ConstraintMode",
]
