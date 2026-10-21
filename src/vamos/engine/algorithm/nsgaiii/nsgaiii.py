"""NSGA-III core algorithm implementation.

Non-dominated Sorting Genetic Algorithm III uses reference points for
many-objective optimization (3+ objectives).

References:
    K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm
    Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving
    Problems With Box Constraints," IEEE Trans. Evolutionary Computation,
    vol. 18, no. 4, 2014.
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
from vamos.engine.algorithm.nsgaii.state import compute_selection_metrics
from vamos.foundation.kernel import default_kernel
from vamos.foundation.kernel.backend import DecisionArray, FloatArray

from .helpers import (
    nsgaiii_survival,
)
from .initialization import initialize_nsgaiii_run
from .state import NSGAIIIState, build_nsgaiii_result

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.observer import RunContext

__all__ = ["NSGAIII"]


def _offspring_constraints(G: Any, F: FloatArray, population_G: FloatArray | None, constraint_mode: str) -> FloatArray | None:
    """Offspring constraint values for survival; required while the population carries constraints."""
    if constraint_mode == "none" or population_G is None:
        return None
    if G is None:
        raise ValueError(
            "NSGA-III requires constraint values G when constraint handling is active "
            f"for a population with {population_G.shape[1]} constraint(s)."
        )
    offspring_G = np.asarray(G, dtype=float)
    if offspring_G.ndim == 1:
        offspring_G = offspring_G.reshape(-1, 1)
    if offspring_G.shape != (F.shape[0], population_G.shape[1]):
        raise ValueError(
            f"NSGA-III constraint values G must align row-wise with objective values F and have "
            f"{population_G.shape[1]} column(s); got shape {offspring_G.shape} for {F.shape[0]} offspring."
        )
    return offspring_G


class NSGAIII:
    """Non-dominated Sorting Genetic Algorithm III for many-objective optimization."""

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: NSGAIIIState | None = None
        self._live_cb: LiveVisualization | None = None
        self._eval_strategy: EvaluationBackend | None = None
        self._max_eval: int = 0
        self._hv_tracker: HVTracker | None = None
        self._problem: ProblemProtocol | None = None

    def _refresh_selection_metrics(self, st: NSGAIIIState) -> None:
        ranks, crowd = compute_selection_metrics(self.kernel, st.F, st.G, st.constraint_mode)
        st.selection_ranks = np.asarray(ranks, dtype=int)
        st.selection_crowding = np.asarray(crowd, dtype=float)

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
        """Run NSGA-III until the termination criterion is met."""
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_nsgaiii_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._live_cb = live_cb
        self._eval_strategy = eval_strategy
        self._max_eval = max_eval
        self._hv_tracker = cast(HVTracker, hv_tracker)
        self._problem = problem

        st = self._st
        ctx = RunContext(
            problem=problem,
            algorithm=self,
            config=self.cfg,
            algorithm_name="nsgaiii",
            engine_name=str(self.kernel.name),
        )
        live_cb.on_start(ctx)

        hv_reached = False
        stop_requested = False
        while st.n_eval < max_eval and not stop_requested:
            X_off = self._generate_offspring(st)
            F_off, G_off = self._evaluate_offspring(problem, X_off, eval_strategy, st.constraint_mode)
            st.n_eval += X_off.shape[0]

            ids_combined = None
            if st.ids is not None and st.pending_offspring_ids is not None:
                ids_combined = np.concatenate([st.ids, st.pending_offspring_ids])

            (
                st.X,
                st.F,
                st.G,
                survivor_indices,
                st.ideal_point,
                st.extreme_points,
                st.worst_point,
            ) = nsgaiii_survival(
                st.X,
                st.F,
                st.G,
                X_off,
                F_off,
                G_off,
                st.pop_size,
                st.ref_dirs_norm,
                st.rng,
                st.ideal_point,
                st.extreme_points,
                st.worst_point,
                kernel=self.kernel,
            )
            self._refresh_selection_metrics(st)

            if ids_combined is not None:
                st.ids = ids_combined[survivor_indices]

            st.generation += 1

            if st.archive_manager is not None:
                st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)

            live_cb.on_generation(st.generation, F=st.F, stats={"evals": st.n_eval})
            stop_requested = live_should_stop(live_cb)

            if hv_tracker is not None and hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

        live_cb.on_end(final_F=st.F)
        result = build_nsgaiii_result(st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, st, self.kernel)
        return result

    # -------------------------------------------------------------------------
    # Generation logic
    # -------------------------------------------------------------------------

    def _generate_offspring(self, st: NSGAIIIState) -> DecisionArray:
        """Generate offspring using tournament selection and variation."""
        n_var = st.X.shape[1]
        request_size = capped_offspring_size(st.n_eval, self._max_eval, st.pop_size, "NSGA-III")
        n_parents = 2 * int(np.ceil(request_size / 2))

        if st.selection_ranks is None or st.selection_crowding is None or st.selection_ranks.shape[0] != st.F.shape[0]:
            self._refresh_selection_metrics(st)
        assert st.selection_ranks is not None
        assert st.selection_crowding is not None
        parents_idx = self.kernel.tournament_selection(
            st.selection_ranks,
            st.selection_crowding,
            st.pressure,
            st.rng,
            n_parents=n_parents,
        )

        X_parents = st.X[parents_idx].reshape(-1, 2, n_var)
        if st.crossover_fn is None or st.mutation_fn is None:
            raise RuntimeError("NSGA-III variation operators are not initialized.")
        offspring_pairs = st.crossover_fn(X_parents)
        X_off: DecisionArray = offspring_pairs.reshape(-1, n_var)
        X_off = st.mutation_fn(X_off)
        if X_off.shape[0] > request_size:
            X_off = X_off[:request_size]

        if st.genealogy_tracker is not None:
            parent_pairs = parents_idx.flatten()
            track_offspring_genealogy(
                st,
                parent_pairs,
                X_off.shape[0],
                variation_operator_label(self.cfg, "sbx+pm"),
                "nsgaiii",
            )

        return X_off

    def _evaluate_offspring(
        self,
        problem: ProblemProtocol,
        X: DecisionArray,
        eval_strategy: EvaluationBackend,
        constraint_mode: str,
    ) -> tuple[FloatArray, FloatArray | None]:
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
        """Initialize the algorithm for an ask/tell loop."""
        self._st, self._live_cb, self._eval_strategy, self._max_eval, self._hv_tracker = initialize_nsgaiii_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._hv_tracker = cast(HVTracker, self._hv_tracker)
        self._problem = problem
        if self._st is not None:
            self._st.pending_offspring = None
        if self._live_cb is not None:
            ctx = RunContext(
                problem=problem,
                algorithm=self,
                config=self.cfg,
                algorithm_name="nsgaiii",
                engine_name=str(self.kernel.name),
            )
            self._live_cb.on_start(ctx)

    def ask(self) -> DecisionArray:
        """Generate offspring for evaluation."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        offspring = self._generate_offspring(self._st)
        self._st.pending_offspring = offspring
        return offspring.copy()

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive evaluated offspring and update the population."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is None:
            raise RuntimeError("No pending offspring. Call ask() first.")

        st = self._st
        X_off = st.pending_offspring
        assert X_off is not None

        if hasattr(eval_result, "F"):
            F = np.asarray(eval_result.F, dtype=float)
            G = getattr(eval_result, "G", None)
        elif isinstance(eval_result, dict):
            F = np.asarray(eval_result["F"], dtype=float)
            G = eval_result.get("G")
        else:
            F = np.asarray(eval_result, dtype=float)
            G = None
        G = _offspring_constraints(G, F, st.G, st.constraint_mode)
        st.pending_offspring = None

        ids_combined = None
        if st.ids is not None and st.pending_offspring_ids is not None:
            ids_combined = np.concatenate([st.ids, st.pending_offspring_ids])

        (
            st.X,
            st.F,
            st.G,
            survivor_indices,
            st.ideal_point,
            st.extreme_points,
            st.worst_point,
        ) = nsgaiii_survival(
            st.X,
            st.F,
            st.G,
            X_off,
            F,
            G,
            st.pop_size,
            st.ref_dirs_norm,
            st.rng,
            st.ideal_point,
            st.extreme_points,
            st.worst_point,
            kernel=self.kernel,
        )
        self._refresh_selection_metrics(st)

        if ids_combined is not None:
            st.ids = ids_combined[survivor_indices]

        st.n_eval += X_off.shape[0]
        st.generation += 1

        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)

        if self._live_cb is not None:
            self._live_cb.on_generation(st.generation, F=st.F)

        if st.hv_tracker is not None and st.hv_tracker.enabled:
            st.hv_tracker.reached(st.hv_points())

        return False

    def should_terminate(self) -> bool:
        """Check whether the current run should terminate."""
        if self._st is None:
            return True
        if self._st.n_eval >= self._max_eval:
            return True
        if self._st.hv_tracker is not None and self._st.hv_tracker.enabled and self._st.hv_tracker.reached(self._st.hv_points()):
            return True
        return False

    def result(self) -> dict[str, Any]:
        """Get the current result snapshot."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")

        hv_reached = self._st.hv_tracker is not None and self._st.hv_tracker.enabled and self._st.hv_tracker.reached(self._st.hv_points())

        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)

        result = build_nsgaiii_result(self._st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, self._st, self.kernel)
        return result

    @property
    def state(self) -> NSGAIIIState | None:
        """Access current algorithm state."""
        return self._st
