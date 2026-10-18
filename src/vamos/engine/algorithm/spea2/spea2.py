# algorithm/spea2/core.py
"""
SPEA2 evolutionary algorithm core.

This module contains the main SPEA2 class with the evolutionary loop (run/ask/tell).
- Setup logic: setup.py
- Operator building: operators/policies/spea2.py
- State and results: state.py
- Helper functions: helpers.py

References:
    E. Zitzler, M. Laumanns, and L. Thiele, "SPEA2: Improving the Strength
    Pareto Evolutionary Algorithm," TIK-Report 103, ETH Zurich, 2001.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    live_should_stop,
    notify_generation,
    track_offspring_genealogy,
)
from vamos.engine.algorithm.components.termination import HVTracker, capped_offspring_size
from vamos.engine.algorithm.components.utils import variation_operator_label
from vamos.engine.archive.factory import update_archive
from vamos.foundation.kernel import default_kernel

from .helpers import environmental_selection, selection_fitness_density
from .initialization import initialize_spea2_run
from .state import SPEA2State, build_spea2_result

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.engine.hooks.live_viz import LiveVisualization


def _tournament_by_strength(
    raw_fitness: np.ndarray[Any, Any],
    density: np.ndarray[Any, Any],
    rng: np.random.Generator,
    n_pairs: int,
) -> np.ndarray[Any, Any]:
    """Binary tournament selection using strength ranking + kNN density."""
    n = raw_fitness.size
    if n == 0:
        raise ValueError("Cannot select parents from an empty archive.")
    if n == 1:
        return np.zeros((n_pairs, 2), dtype=int)

    n_parents = n_pairs * 2
    first = rng.integers(0, n, size=n_parents, dtype=int)
    second = rng.integers(0, n - 1, size=n_parents, dtype=int)
    second += second >= first

    fa = raw_fitness[first]
    fb = raw_fitness[second]
    da = density[first]
    db = density[second]

    winners = np.empty(n_parents, dtype=int)
    better_first = fa < fb
    better_second = fb < fa
    winners[better_first] = first[better_first]
    winners[better_second] = second[better_second]

    tied = ~(better_first | better_second)
    if tied.any():
        tied_first = first[tied]
        tied_second = second[tied]
        tied_density_first = da[tied]
        tied_density_second = db[tied]

        density_first = tied_density_first > tied_density_second
        density_second = tied_density_second > tied_density_first
        tied_winners = np.empty(tied_first.shape[0], dtype=int)
        tied_winners[density_first] = tied_first[density_first]
        tied_winners[density_second] = tied_second[density_second]

        density_tied = ~(density_first | density_second)
        if density_tied.any():
            flip = rng.integers(0, 2, size=int(density_tied.sum()), dtype=int)
            tied_winners[density_tied] = np.where(flip == 0, tied_first[density_tied], tied_second[density_tied])

        winners[tied] = tied_winners

    return winners.reshape(n_pairs, 2)


class SPEA2:
    """SPEA2 (Strength Pareto Evolutionary Algorithm 2).

    Enhanced implementation with ask/tell interface, HV termination,
    and live visualization support.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size
        - archive_size (int, optional): Internal archive size (default: pop_size)
        - crossover (dict): Crossover operator config
        - mutation (dict): Mutation operator config
        - k_neighbors (int, optional): k for density estimation (default: sqrt(N))
        - constraint_mode (str, optional): "none" or "feasibility"
    kernel : KernelBackend
        Backend for vectorized operations.

    Examples
    --------
    >>> from vamos.algorithms import SPEA2Config
    >>> config = SPEA2Config.builder().pop_size(100).archive_size(100).build()
    >>> spea2 = SPEA2(config, kernel)
    >>> result = spea2.run(problem, ("max_evaluations", 10000), seed=42)

    Using ask/tell for external evaluation:
    >>> spea2.initialize(problem, ("max_evaluations", 10000), seed=42)
    >>> while not spea2.should_terminate():
    ...     X_off = spea2.ask()
    ...     F_off = problem.evaluate(X_off)
    ...     spea2.tell(EvalResult(F=F_off, G=None))
    >>> result = spea2.result()
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: SPEA2State | None = None
        self._live_cb: LiveVisualization | None = None
        self._eval_strategy: EvaluationBackend | None = None
        self._max_eval: int = 0
        self._hv_tracker: HVTracker | None = None
        self._problem: ProblemProtocol | None = None

    def _refresh_selection_metrics(self, st: SPEA2State) -> None:
        if st.env_F is None:
            st.selection_raw_fitness = None
            st.selection_density = None
            return
        raw_fitness, density = selection_fitness_density(st.env_F, st.env_G, st.constraint_mode, st.k_neighbors)
        st.selection_raw_fitness = raw_fitness
        st.selection_density = density

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> dict[str, Any]:
        """Run SPEA2 optimization.

        Parameters
        ----------
        problem : ProblemProtocol
            The optimization problem to solve.
        termination : tuple[str, Any]
            Termination criterion: ("max_evaluations", N) or ("hv", {...}).
        seed : int
            Random seed for reproducibility.
        eval_strategy : EvaluationBackend | None
            Optional evaluation backend for parallel evaluation.
        live_viz : LiveVisualization | None
            Optional live visualization callback.

        Returns
        -------
        dict[str, Any]
            Result dictionary with X, F, evaluations, and archive.
        """
        live_cb, eval_strategy, max_eval, hv_tracker = self._initialize_run(problem, termination, seed, eval_strategy, live_viz)
        hv_tracker = cast(HVTracker, hv_tracker)
        if eval_strategy is None:
            raise RuntimeError("SPEA2 requires an evaluation backend; initialize_spea2_run() returned None.")
        st = self._st
        assert st is not None, "State not initialized"
        assert st.env_F is not None

        generation = 0
        live_cb.on_generation(generation, F=st.env_F, stats={"evals": st.n_eval})
        stop_requested = live_should_stop(live_cb)
        hv_reached = hv_tracker.enabled and hv_tracker.reached(st.env_F)

        while st.n_eval < max_eval and not hv_reached and not stop_requested:
            st.generation = generation
            X_off = self.ask()

            # Evaluate offspring
            eval_result = eval_strategy.evaluate(X_off, problem)
            hv_reached = self.tell(eval_result, problem)

            assert st.env_F is not None
            if hv_tracker.enabled and hv_tracker.reached(st.env_F):
                hv_reached = True
                break

            generation += 1
            st.generation = generation
            assert st.env_F is not None
            stop_requested = notify_generation(live_cb, self.kernel, generation, st.env_F, stats={"evals": st.n_eval})

        result = build_spea2_result(st, hv_reached)
        finalize_genealogy(result, st, self.kernel)
        live_cb.on_end(final_F=st.env_F)
        return result

    def _initialize_run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> tuple[Any, Any, int, Any]:
        """Initialize algorithm state for a run."""
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_spea2_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._live_cb = live_cb
        self._eval_strategy = eval_strategy
        self._max_eval = max_eval
        self._hv_tracker = hv_tracker
        return live_cb, eval_strategy, max_eval, hv_tracker

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
        if self._st is not None and self._live_cb is not None and self._st.env_F is not None:
            self._live_cb.on_generation(0, F=self._st.env_F)

    def ask(self) -> np.ndarray[Any, Any]:
        """Generate offspring for external evaluation.

        Returns
        -------
        np.ndarray[Any, Any]
            Offspring decision variables to evaluate.

        Raises
        ------
        RuntimeError
            If called before initialization.
        """
        if self._st is None:
            raise RuntimeError("Call initialize() or run() before ask()")

        st = self._st
        if st.env_X is None or st.env_F is None:
            raise RuntimeError("SPEA2 internal archive is not initialized.")
        if st.crossover_fn is None or st.mutation_fn is None:
            raise RuntimeError("SPEA2 variation operators are not initialized.")
        if st.xl is None or st.xu is None:
            raise RuntimeError("SPEA2 bounds are not initialized.")
        n_pairs = capped_offspring_size(st.n_eval, self._max_eval, st.offspring_size, "SPEA2")

        # Fitness-based binary tournament selection (SPEA2)
        if st.selection_raw_fitness is None or st.selection_density is None or st.selection_raw_fitness.shape[0] != st.env_F.shape[0]:
            self._refresh_selection_metrics(st)
        assert st.selection_raw_fitness is not None
        assert st.selection_density is not None
        parent_idx = _tournament_by_strength(st.selection_raw_fitness, st.selection_density, st.rng, n_pairs)

        # Gather parents in shape (n_pairs, 2, n_var)
        n_var = st.env_X.shape[1]
        parents = st.env_X[parent_idx.reshape(-1)].reshape(n_pairs, 2, n_var)

        # Apply crossover
        offspring = st.crossover_fn(parents, st.rng)

        # Take first child from each pair
        offspring_X = offspring[:, 0, :].copy()

        # Apply mutation
        offspring_X = st.mutation_fn(offspring_X, st.rng)

        st.pending_offspring = offspring_X

        # Track genealogy
        track_offspring_genealogy(
            st,
            parent_idx.reshape(-1),
            offspring_X.shape[0],
            variation_operator_label(self.cfg, "sbx+pm"),
            "spea2",
        )

        return offspring_X

    def tell(
        self,
        eval_result: Any,
        problem: ProblemProtocol | None = None,
    ) -> bool:
        """Process evaluated offspring.

        Parameters
        ----------
        eval_result : Any
            Evaluation result with F and optionally G.
        problem : ProblemProtocol | None
            Problem instance (optional).

        Returns
        -------
        bool
            True if HV threshold reached.

        Raises
        ------
        RuntimeError
            If called before ask().
        """
        if self._st is None or self._st.pending_offspring is None:
            raise RuntimeError("Call ask() before tell()")

        st = self._st
        offspring_X = st.pending_offspring
        assert offspring_X is not None
        if st.env_X is None or st.env_F is None:
            raise RuntimeError("SPEA2 internal archive is not initialized.")

        # Extract F and G from result
        if hasattr(eval_result, "F"):
            F = eval_result.F
            G = getattr(eval_result, "G", None)
        elif isinstance(eval_result, dict):
            F = eval_result.get("F")
            G = eval_result.get("G")
        else:
            F = eval_result
            G = None

        # Update evaluation count
        st.n_eval += len(F)

        # Combine archive and offspring for environmental selection
        X_union = np.vstack([st.env_X, offspring_X])
        F_union = np.vstack([st.env_F, F])

        if G is not None:
            if st.env_G is not None:
                G_union = np.vstack([st.env_G, G])
            else:
                G_union = G
        elif st.env_G is not None:
            G_union = np.vstack([st.env_G, np.zeros((len(F), st.env_G.shape[1]))])
        else:
            G_union = None

        # Environmental selection
        st.env_X, st.env_F, st.env_G = environmental_selection(
            X_union,
            F_union,
            G_union,
            st.env_archive_size,
            st.k_neighbors,
            st.constraint_mode,
            self.kernel.selection_kernels(),
        )
        self._refresh_selection_metrics(st)

        # Update population reference
        st.X = st.env_X
        st.F = st.env_F
        st.G = st.env_G

        # Update external archive
        update_archive(st, st.env_X, st.env_F, st.env_G)

        # Clear pending
        st.pending_offspring = None

        # Check HV termination
        if st.hv_tracker is not None and st.hv_tracker.enabled:
            return st.hv_tracker.reached(st.env_F)

        return False

    def should_terminate(self) -> bool:
        """Check if termination criterion is met."""
        if self._st is None:
            return True

        if self._max_eval is not None:
            if self._st.n_eval >= self._max_eval:
                return True

        if self._hv_tracker is not None and self._hv_tracker.enabled and self._st.env_F is not None:
            return self._hv_tracker.reached(self._st.env_F)

        return False

    def result(self) -> dict[str, Any]:
        """Get final optimization result."""
        if self._st is None:
            raise RuntimeError("Call initialize() and run before result()")

        hv_reached = (
            self._hv_tracker is not None
            and self._hv_tracker.enabled
            and self._st.env_F is not None
            and self._hv_tracker.reached(self._st.env_F)
        )
        return build_spea2_result(self._st, hv_reached)


__all__ = ["SPEA2"]
