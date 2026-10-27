# algorithm/ibea/core.py
"""
IBEA evolutionary algorithm core.

This module contains the main IBEA class with the evolutionary loop (run/ask/tell).
- Setup logic: setup.py
- Operator building: operators/policies/ibea.py
- State and results: state.py
- Helper functions: helpers.py

References:
    E. Zitzler and S. Künzli, "Indicator-Based Selection in Multiobjective
    Search," in Proc. PPSN VIII, 2004, pp. 832-842.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    live_should_stop,
    track_offspring_genealogy,
)
from vamos.engine.algorithm.components.lifecycle import evaluate_batch
from vamos.engine.algorithm.components.termination import HVTracker, capped_offspring_size
from vamos.engine.algorithm.components.utils import variation_operator_label
from vamos.engine.algorithm.nsgaii.helpers import build_mating_pool
from vamos.foundation.kernel import default_kernel

from .helpers import combine_constraints, environmental_selection
from .initialization import initialize_ibea_run
from .state import IBEAState, build_ibea_result

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.engine.hooks.live_viz import LiveVisualization


class IBEA:
    """Indicator-Based Evolutionary Algorithm.

    IBEA uses quality indicators (epsilon or hypervolume) to compute fitness
    from pairwise comparisons. Solutions are selected based on their contribution
    to the indicator value, promoting both convergence and diversity.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size
        - crossover (tuple): Crossover operator config
        - mutation (tuple): Mutation operator config
        - indicator (str, optional): "epsilon" (default) or "hypervolume"
        - kappa (float, optional): Scaling factor (default: 0.05)
        - constraint_mode (str, optional): "none" or "penalty"
    kernel : KernelBackend
        Backend for vectorized operations.

    Examples
    --------
    >>> from vamos.algorithms import IBEAConfig
    >>> config = IBEAConfig.builder().pop_size(100).indicator("epsilon").kappa(0.05).build()
    >>> ibea = IBEA(config, kernel)
    >>> result = ibea.run(problem, ("max_evaluations", 10000), seed=42)

    Ask/tell interface:
    >>> ibea.initialize(problem, ("max_evaluations", 10000), seed=42)
    >>> while not ibea.should_terminate():
    ...     X = ibea.ask()
    ...     F = evaluate(X)
    ...     ibea.tell(F)
    >>> result = ibea.result()
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: IBEAState | None = None
        self._live_cb: LiveVisualization | None = None
        self._eval_strategy: EvaluationBackend | None = None
        self._max_eval: int = 0
        self._hv_tracker: HVTracker | None = None
        self._problem: ProblemProtocol | None = None

    # -------------------------------------------------------------------------
    # Main run method (batch mode)
    # -------------------------------------------------------------------------

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> dict[str, Any]:
        """Run IBEA optimization loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion, e.g., ("max_evaluations", 10000).
        seed : int
            Random seed for reproducibility.
        eval_strategy : EvaluationBackend, optional
            Evaluation backend for parallel evaluation.
        live_viz : LiveVisualization, optional
            Live visualization callback.

        Returns
        -------
        dict
            Result dictionary with X, F, G, archive data.
        """
        live_cb, eval_strategy, max_eval, hv_tracker = self._initialize_run(problem, termination, seed, eval_strategy, live_viz)
        self._problem = problem

        st = self._st
        if st is None:
            raise RuntimeError("State not initialized")

        hv_reached = False
        stop_requested = False
        while st.n_eval < max_eval and not stop_requested:
            # Generate offspring
            X_off = self._generate_offspring(st)

            # Evaluate offspring
            F_off, G_off = self._evaluate_offspring(problem, X_off, eval_strategy, st.constraint_mode)
            st.n_eval += X_off.shape[0]

            # Environmental selection
            X_comb = np.vstack([st.X, X_off])
            F_comb = np.vstack([st.F, F_off])
            G_comb = combine_constraints(st.G, G_off)

            st.X, st.F, st.G, st.fitness = environmental_selection(
                X_comb, F_comb, G_comb, st.pop_size, st.indicator, st.kappa, self.kernel.selection_kernels()
            )

            st.generation += 1

            # Update archive
            if st.archive_manager is not None:
                st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)

            # Live callback
            live_cb.on_generation(st.generation, F=st.F, stats={"evals": st.n_eval})
            stop_requested = live_should_stop(live_cb)

            # Check HV threshold
            if hv_tracker is not None and hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

        live_cb.on_end(final_F=st.F)
        result = build_ibea_result(st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, st, self.kernel)
        return result

    def _initialize_run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> tuple[Any, Any, int, Any]:
        """Initialize the algorithm run."""
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_ibea_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._live_cb = live_cb
        self._eval_strategy = eval_strategy
        self._max_eval = max_eval
        self._hv_tracker = cast(HVTracker, hv_tracker)
        return live_cb, eval_strategy, max_eval, hv_tracker

    def _generate_offspring(self, st: IBEAState) -> np.ndarray[Any, Any]:
        """Generate offspring using tournament selection and variation."""
        if st.variation is None:
            raise RuntimeError("IBEA variation pipeline is not initialized.")
        variation = st.variation
        # Higher fitness is better in IBEA (less negative).
        ranks = np.argsort(np.argsort(-st.fitness))
        crowd = np.zeros_like(st.fitness, dtype=float)
        parents_per_group = variation.parents_per_group
        children_per_group = variation.children_per_group
        request_size = capped_offspring_size(st.n_eval, self._max_eval, st.offspring_size, "IBEA")
        parent_count = int(np.ceil(request_size / children_per_group) * parents_per_group)

        sel_method, _ = self.cfg["selection"]
        mating_pairs = build_mating_pool(self.kernel, ranks, crowd, st.pressure, st.rng, parent_count, parents_per_group, sel_method)
        parent_idx = mating_pairs.reshape(-1)
        X_parents = variation.gather_parents(st.X, parent_idx)
        X_off = variation.produce_offspring(X_parents, st.rng)
        if X_off.shape[0] > request_size:
            X_off = X_off[:request_size]

        # Track genealogy
        track_offspring_genealogy(st, parent_idx, X_off.shape[0], variation_operator_label(self.cfg, "sbx+pm"), "ibea")

        return X_off

    def _evaluate_offspring(
        self,
        problem: ProblemProtocol,
        X: np.ndarray[Any, Any],
        eval_strategy: EvaluationBackend,
        constraint_mode: str,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
        """Evaluate offspring and compute constraints."""
        return evaluate_batch(problem, eval_strategy, X, constraint_mode)

    # -------------------------------------------------------------------------
    # Ask/Tell Interface
    # -------------------------------------------------------------------------

    def initialize(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> None:
        """Initialize algorithm for ask/tell loop."""
        self._live_cb, self._eval_strategy, self._max_eval, self._hv_tracker = self._initialize_run(
            problem, termination, seed, eval_strategy, live_viz
        )
        self._problem = problem
        if self._st is not None:
            self._st.pending_offspring = None

    def ask(self) -> np.ndarray[Any, Any]:
        """Generate offspring for evaluation.

        Returns
        -------
        np.ndarray[Any, Any]
            Offspring decision vectors to evaluate.

        Raises
        ------
        RuntimeError
            If not initialized or previous offspring not consumed.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        offspring = self._generate_offspring(self._st)
        self._st.pending_offspring = offspring
        return offspring.copy()

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive evaluated offspring and update population.

        Parameters
        ----------
        eval_result : Any
            Objective values as ``np.ndarray[Any, Any]``, or an object with ``.F``
            attribute, or a dict with ``"F"`` key.
        problem : ProblemProtocol | None
            Unused, kept for interface consistency.

        Returns
        -------
        bool
            Always ``False`` (IBEA has no early-stop criterion via tell).

        Raises
        ------
        RuntimeError
            If not initialized or no pending offspring.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is None:
            raise RuntimeError("No pending offspring. Call ask() first.")

        st = self._st
        X_off = st.pending_offspring
        assert X_off is not None
        st.pending_offspring = None

        if hasattr(eval_result, "F"):
            F = np.asarray(eval_result.F, dtype=float)
            G = getattr(eval_result, "G", None)
        elif isinstance(eval_result, dict):
            F = np.asarray(eval_result["F"], dtype=float)
            G = eval_result.get("G")
        else:
            F = np.asarray(eval_result, dtype=float)
            G = None

        # Environmental selection
        X_comb = np.vstack([st.X, X_off])
        F_comb = np.vstack([st.F, F])
        G_comb = combine_constraints(st.G, G)

        st.X, st.F, st.G, st.fitness = environmental_selection(
            X_comb, F_comb, G_comb, st.pop_size, st.indicator, st.kappa, self.kernel.selection_kernels()
        )

        st.n_eval += X_off.shape[0]
        st.generation += 1

        # Update archive
        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)

        # Live callback
        if self._live_cb is not None:
            self._live_cb.on_generation(st.generation, F=st.F)

        # Check HV tracker
        if st.hv_tracker is not None and st.hv_tracker.enabled:
            st.hv_tracker.reached(st.hv_points())

        return False

    def should_terminate(self) -> bool:
        """Check if termination criterion is met."""
        if self._st is None:
            return True
        if self._st.n_eval >= self._max_eval:
            return True
        if self._st.hv_tracker is not None and self._st.hv_tracker.enabled and self._st.hv_tracker.reached(self._st.hv_points()):
            return True
        return False

    def result(self) -> dict[str, Any]:
        """Get current result."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")

        hv_reached = self._st.hv_tracker is not None and self._st.hv_tracker.enabled and self._st.hv_tracker.reached(self._st.hv_points())

        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)

        return build_ibea_result(self._st, hv_reached, kernel=self.kernel)

    @property
    def state(self) -> IBEAState | None:
        """Access current algorithm state."""
        return self._st


__all__ = ["IBEA"]
