"""SMS-EMOA core algorithm implementation.

S-Metric Selection Evolutionary Multiobjective Optimization Algorithm uses
hypervolume contribution for survival selection.

References:
    N. Beume, B. Naujoks, and M. Emmerich, "SMS-EMOA: Multiobjective Selection
    Based on Dominated Hypervolume," European Journal of Operational Research,
    vol. 181, no. 3, 2007.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    live_should_stop,
    track_offspring_genealogy,
)
from vamos.engine.algorithm.components.termination import HVTracker, remaining_evaluations
from vamos.engine.algorithm.components.utils import variation_operator_label
from vamos.foundation.kernel import default_kernel

from .compiled_step import build_compiled_step
from .helpers import (
    refresh_selection_cache,
    survival_selection,
)
from .initialization import initialize_smsemoa_run
from .state import SMSEMOAState, build_smsemoa_result

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.observer import RunContext

__all__ = ["SMSEMOA"]


def _archive_child(st: SMSEMOAState, X_child: NDArray[Any], F_child: NDArray[np.float64], G_child: Any) -> None:
    """Offer an evaluated child to the external archive, whether or not it survives.

    The archive starts from the initial population and then receives every child
    once, so a child removed as the least hypervolume contributor is archived when
    no evaluated row dominates it.
    """
    if st.archive_manager is not None:
        # The archive holds constraint values exactly when the population does.
        G_rows = np.asarray(G_child, dtype=float).reshape(X_child.shape[0], -1) if st.G is not None and G_child is not None else None
        st.archive_X, st.archive_F = st.archive_manager.update(X_child, F_child, G_rows)


def _check_hv_target(st: SMSEMOAState) -> bool:
    """Measure an ``("hv", ...)`` target at a generation boundary and remember the outcome.

    The HV is measured only here, once per generation, in ``run`` and in ``tell`` alike;
    ``should_terminate`` and ``result`` read the stored outcome.
    """
    tracker = st.hv_tracker
    st.hv_reached = bool(tracker is not None and tracker.enabled and tracker.reached(st.hv_points()))
    return st.hv_reached


def _advance_step(st: SMSEMOAState) -> bool:
    """Count one child; return whether it completes a generation of ``pop_size`` children."""
    st.steps += 1
    if st.steps % st.pop_size != 0:
        return False
    st.generation = st.steps // st.pop_size
    return True


class SMSEMOA:
    """S-Metric Selection Evolutionary Multiobjective Optimization Algorithm.

    SMS-EMOA uses hypervolume contribution for survival selection, removing
    solutions that contribute least to the hypervolume indicator each generation.
    This provides strong convergence toward the Pareto front.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size
        - crossover (tuple): Crossover operator config
        - mutation (tuple): Mutation operator config
        - selection (tuple): Selection config, e.g., ("tournament", {"size": 2})
        - reference_point (dict, optional): {"offset": float} or {"point": list}
        - external_archive (dict, optional): {"capacity": int | None, "pruning": str}
        - hv_threshold (float, optional): HV threshold for early termination
    kernel : KernelBackend
        Backend for vectorized operations.

    Examples
    --------
    Basic usage:

    >>> from vamos.algorithms import SMSEMOAConfig
    >>> config = SMSEMOAConfig.builder().pop_size(100).crossover("sbx", prob=0.9).build()
    >>> smsemoa = SMSEMOA(config, kernel)
    >>> result = smsemoa.run(problem, ("max_evaluations", 10000), seed=42)

    Ask/tell interface:

    >>> smsemoa = SMSEMOA(config, kernel)
    >>> smsemoa.initialize(problem, ("max_evaluations", 10000), seed=42)
    >>> while not smsemoa.should_terminate():
    ...     X = smsemoa.ask()
    ...     F = evaluate(X)
    ...     smsemoa.tell(F)
    >>> result = smsemoa.result()
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: SMSEMOAState | None = None
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
        """Run SMS-EMOA optimization loop.

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
            Result dictionary with X, F, G, reference_point, archive data.
        """
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_smsemoa_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._problem = problem
        self._live_cb = live_cb
        self._eval_strategy = eval_strategy
        self._max_eval = max_eval
        self._hv_tracker = cast(HVTracker, hv_tracker)

        st = self._st
        ctx = RunContext(
            problem=problem,
            algorithm=self,
            config=self.cfg,
            algorithm_name="smsemoa",
            engine_name=str(self.kernel.name),
        )
        live_cb.on_start(ctx)
        # Generation 0 is the evaluated initial population, as in every algorithm.
        live_cb.on_generation(0, F=st.F, stats={"evals": st.n_eval})

        # Compiled kernels reproduce the step below when the configuration allows it.
        compiled = build_compiled_step(st, self.kernel)
        st.hv_reached = False
        stop_requested = live_should_stop(live_cb)
        try:
            while st.n_eval < max_eval and not stop_requested:
                if compiled is not None:
                    X_child, F_child, G_child = compiled.step(problem, eval_strategy)
                else:
                    X_child, F_child, G_child = self._reference_step(st, problem, eval_strategy)
                _archive_child(st, X_child, F_child, G_child)

                new_generation = _advance_step(st)
                if new_generation:
                    live_cb.on_generation(st.generation, F=st.F, stats={"evals": st.n_eval})
                stop_requested = live_should_stop(live_cb)

                # The HV target is checked once per generation, as in steady-state NSGA-II.
                if new_generation and _check_hv_target(st):
                    break
        finally:
            if compiled is not None:
                compiled.finish()

        live_cb.on_end(final_F=st.F)
        result = build_smsemoa_result(st, st.hv_reached, kernel=self.kernel)
        finalize_genealogy(result, st, self.kernel)
        return result

    # -------------------------------------------------------------------------
    # Generation logic
    # -------------------------------------------------------------------------

    def _reference_step(
        self, st: SMSEMOAState, problem: ProblemProtocol, eval_strategy: EvaluationBackend
    ) -> tuple[NDArray[Any], NDArray[np.float64], NDArray[np.float64] | None]:
        """Generate, evaluate and insert one child (the reference for ``CompiledSteadyStep``).

        Returns the evaluated child as ``(X, F, G)`` rows.
        """
        X_child = self._generate_offspring(st)
        F_child, G_child = self._evaluate_offspring(problem, X_child, eval_strategy, st.constraint_mode)
        st.n_eval += X_child.shape[0]
        survival_selection(st, X_child, F_child, G_child, self.kernel)
        return X_child, F_child, G_child

    def _generate_offspring(self, st: SMSEMOAState) -> NDArray[Any]:
        """Generate offspring using parent selection and variation."""
        sel_method = st.selection_method

        if sel_method == "tournament":
            if st.ranks is None or st.crowding is None:
                refresh_selection_cache(st, self.kernel)
            if st.ranks is None or st.crowding is None:
                raise RuntimeError("SMS-EMOA selection cache is unavailable.")
            parents_idx = self.kernel.tournament_selection(st.ranks, st.crowding, st.pressure, st.rng, n_parents=2)
        else:
            if st.X.shape[0] == 0:
                raise ValueError("Cannot select parents from an empty population.")
            parents_idx = st.rng.choice(st.X.shape[0], size=2, replace=True)

        parents = st.X[parents_idx]
        if parents.ndim == 2:
            parents = parents.reshape(1, 2, -1)

        # Apply crossover and mutation
        if st.crossover_fn is None or st.mutation_fn is None:
            raise RuntimeError("SMS-EMOA variation operators are not initialized.")
        offspring = st.crossover_fn(parents)
        child_vec = offspring.reshape(-1, st.X.shape[1])[0:1]  # first child as (1, n_var)
        child = st.mutation_fn(child_vec)

        # Track genealogy - store parent indices for later assignment
        if st.genealogy_tracker is not None:
            track_offspring_genealogy(
                st,
                parents_idx.reshape(-1),
                1,
                variation_operator_label(self.cfg, "sbx+pm"),
                "smsemoa",
            )

        return child

    def _evaluate_offspring(
        self,
        problem: ProblemProtocol,
        X: NDArray[Any],
        eval_strategy: EvaluationBackend,
        constraint_mode: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Evaluate offspring and compute constraints."""
        eval_result = eval_strategy.evaluate(X, problem)
        F, G = eval_result.F, eval_result.G
        if constraint_mode == "none":
            G = None
        return F, G

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
        """Initialize algorithm for ask/tell loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion.
        seed : int
            Random seed.
        eval_strategy : EvaluationBackend, optional
            Evaluation backend.
        live_viz : LiveVisualization, optional
            Live visualization callback.
        """
        self._st, self._live_cb, self._eval_strategy, self._max_eval, self._hv_tracker = initialize_smsemoa_run(
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
                algorithm_name="smsemoa",
                engine_name=str(self.kernel.name),
            )
            self._live_cb.on_start(ctx)
            self._live_cb.on_generation(0, F=self._st.F, stats={"evals": self._st.n_eval})

    def ask(self) -> NDArray[Any]:
        """Generate offspring for evaluation.

        Returns
        -------
        np.ndarray
            Offspring decision vectors to evaluate.

        Raises
        ------
        RuntimeError
            If algorithm not initialized or previous offspring not consumed.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        remaining_evaluations(self._st.n_eval, self._max_eval, "SMS-EMOA")
        offspring = self._generate_offspring(self._st)
        self._st.pending_offspring = offspring
        return offspring.copy()

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive evaluated offspring and update population.

        Parameters
        ----------
        eval_result : Any
            Objective values as ``np.ndarray``, or an object with ``.F``
            attribute, or a dict with ``"F"`` key.
        problem : ProblemProtocol | None
            Unused, kept for interface consistency.

        Returns
        -------
        bool
            Always ``False`` (SMS-EMOA has no early-stop criterion via tell).

        Raises
        ------
        RuntimeError
            If algorithm not initialized or no pending offspring.
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

        # ``ask`` creates one child per call.
        survival_selection(st, X_off, F, G, self.kernel)
        st.n_eval += X_off.shape[0]
        _archive_child(st, X_off, F, G)

        if _advance_step(st):
            if self._live_cb is not None:
                self._live_cb.on_generation(st.generation, F=st.F, stats={"evals": st.n_eval})
            _check_hv_target(st)

        # True once the HV target of an ("hv", ...) termination is reached, as in the other algorithms;
        # it is measured at generation boundaries only.
        return st.hv_reached

    def should_terminate(self) -> bool:
        """Check if termination criterion is met.

        Returns
        -------
        bool
            True if algorithm should stop.
        """
        if self._st is None:
            return True
        if self._st.n_eval >= self._max_eval:
            return True
        # The HV target is checked at generation boundaries, as in run(); no HV is computed here.
        if self._st.hv_reached:
            return True
        return False

    def result(self) -> dict[str, Any]:
        """Get current result.

        Returns
        -------
        dict
            Result dictionary with X, F, G, reference_point, archive data.

        Raises
        ------
        RuntimeError
            If algorithm not initialized.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")

        hv_reached = self._st.hv_reached

        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)

        result = build_smsemoa_result(self._st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, self._st, self.kernel)
        return result

    @property
    def state(self) -> SMSEMOAState | None:
        """Access current algorithm state."""
        return self._st
