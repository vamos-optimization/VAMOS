"""
AGE-MOEA: Adaptive Geometry Estimation MOEA.

Reference:
    Panichella, A. (2019). An Adaptive Evolutionary Algorithm based on
    Non-Euclidean Geometry for Many-objective Optimization. GECCO 2019.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.evaluation import combine_constraints, extract_evaluation_arrays
from vamos.engine.algorithm.components.hooks import get_live_viz, live_should_stop
from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.engine.algorithm.components.termination import capped_offspring_size, validate_initial_budget
from vamos.engine.algorithm.config.base import normalize_constraint_mode
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.config.variation import (
    ensure_operator_tuple,
    resolve_default_variation_config,
)
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.engine.variation.helpers import (
    ensure_supported_operator_names,
    ensure_supported_repair_name,
)
from vamos.engine.variation.pipeline import VariationPipeline
from vamos.engine.variation.protocol import RepairConfigValue
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.kernel import default_kernel
from vamos.foundation.kernel.backend import KernelBackend, VariationKernels
from vamos.foundation.observer import RunContext
from vamos.foundation.problem.types import ProblemProtocol

from .state import AGEMOEAState, build_agemoea_result
from .survival import constraint_aware_age_survival, population_selection_metrics, selection_metrics


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _build_variation(
    config: dict[str, Any],
    encoding: Any,
    xl: Any,
    xu: Any,
    problem: ProblemProtocol,
    variation_kernels: VariationKernels | None = None,
) -> VariationPipeline:
    explicit_overrides: dict[str, Any] = {}
    if "crossover" in config:
        explicit_overrides["crossover"] = config["crossover"]
    if "mutation" in config:
        explicit_overrides["mutation"] = config["mutation"]
    if "repair" in config and config["repair"] != "auto":
        explicit_overrides["repair"] = config["repair"]

    # The configuration already holds the resolved operators: they replace the encoding defaults whole.
    var_cfg = {**resolve_default_variation_config(encoding, None), **explicit_overrides}
    c_name, c_kwargs = ensure_operator_tuple(var_cfg.get("crossover", ("sbx", {})), key="crossover")
    m_name, m_kwargs = ensure_operator_tuple(var_cfg.get("mutation", ("polynomial", {})), key="mutation")
    repair_cfg: RepairConfigValue = "auto"
    cross_name, mut_name = ensure_supported_operator_names(encoding, c_name, m_name)
    if "repair" in var_cfg:
        repair_name, repair_params = ensure_operator_tuple(var_cfg["repair"], key="repair")
        repair_cfg = (ensure_supported_repair_name(encoding, repair_name), repair_params)

    return VariationPipeline(
        encoding=encoding,
        cross_method=cross_name,
        mut_method=mut_name,
        cross_params=c_kwargs,
        mut_params=m_kwargs,
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=repair_cfg,
        problem=problem,
        variation_kernels=variation_kernels,
    )


class AGEMOEA:
    """AGE-MOEA: Adaptive Geometry Estimation MOEA.

    Survival estimates the geometry of the first front and scores its points by
    diversity and proximity (Panichella, 2019). Later fronts, the truncated one
    included, are scored by proximity to the ideal point, and the binary mating
    tournament compares rank, then that survival score, as in the author's
    PlatEMO code. The rest follows pymoo, with PlatEMO's distance
    normalization: later fronts are measured in the first front's
    ideal-centred, normalized coordinates (PlatEMO and pymoo do not translate
    them), objectives are not rounded to 6 decimals before non-dominated
    sorting, the geometry exponent ``p`` is capped at 20, a hyperplane with a
    zero coefficient also falls back to the front maximum, and the default
    variation is SBX(0.9, 15) with polynomial mutation.

    Parameters
    ----------
    config : dict
        Algorithm configuration.
    kernel : KernelBackend, optional
        Backend for vectorized operations.

    Examples
    --------
    Batch mode:

    >>> algo = AGEMOEA(config, kernel)
    >>> result = algo.run(problem, ("max_evaluations", 10000), seed=42)

    Ask/tell interface:

    >>> algo = AGEMOEA(config, kernel)
    >>> algo.initialize(problem, ("max_evaluations", 10000), seed=42)
    >>> while not algo.should_terminate():
    ...     X = algo.ask()
    ...     F = evaluate(X)
    ...     algo.tell(F)
    >>> result = algo.result()
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: AGEMOEAState | None = None
        self._live_cb: LiveVisualization | None = None

    def _refresh_selection_metrics(self, st: AGEMOEAState) -> None:
        ranks, scores = population_selection_metrics(self.kernel, st.F, st.G, st.constraint_mode)
        st.selection_ranks = ranks
        st.selection_crowding = scores

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
        """Run AGE-MOEA optimization."""
        self.initialize(problem, termination, seed, eval_strategy, live_viz)
        backend = eval_strategy or SerialEvalBackend()

        assert self._st is not None
        stop_requested = False
        while not self.should_terminate():
            X_off = self.ask()
            eval_result = backend.evaluate(X_off, problem)
            stop_requested = self.tell(eval_result)
            if stop_requested:
                break

        return self.result()

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
        """Initialize algorithm state for ask/tell loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion, e.g., ``("max_evaluations", 10000)``.
        seed : int
            Random seed for reproducibility.
        eval_strategy : EvaluationBackend, optional
            Evaluation backend for the initial population.
        live_viz : LiveVisualization, optional
            Live visualization callback.
        """
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()
        live_cb = get_live_viz(live_viz)

        pop_size = int(self.cfg.get("pop_size", 100))
        constraint_mode = normalize_constraint_mode(self.cfg.get("constraint_mode", "feasibility"))
        n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
        term_key, term_val = termination
        if term_key == "max_evaluations":
            max_evals = int(term_val)
        elif term_key == "n_gen":
            max_evals = int(term_val) * pop_size
        else:
            raise ValueError("Unsupported termination criterion for AGE-MOEA.")
        validate_initial_budget(max_evals, pop_size, "AGE-MOEA")

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        xl, xu = resolve_bounds(problem, encoding)
        X = initialize_population(pop_size, problem.n_var, xl, xu, encoding, rng, problem, self.cfg.get("initializer"))
        F, G = extract_evaluation_arrays(
            backend.evaluate(X, problem),
            constraint_mode,
            n_constraints,
            algorithm="AGE-MOEA",
        )

        variation = _build_variation(self.cfg, encoding, xl, xu, problem, self.kernel.variation_kernels())
        ext_cfg = resolve_external_archive(self.cfg)
        archive_X, archive_F, archive_manager = setup_archive(
            self.kernel,
            X,
            F,
            problem.n_var,
            problem.n_obj,
            X.dtype,
            ext_cfg,
            G,
        )
        selection_ranks, selection_crowding = population_selection_metrics(self.kernel, F, G, constraint_mode)

        result_mode = str(self.cfg.get("result_mode", "non_dominated")).strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")

        self._live_cb = live_cb
        self._st = AGEMOEAState(
            X=X,
            F=F,
            G=G,
            rng=rng,
            pop_size=pop_size,
            n_eval=X.shape[0],
            generation=0,
            max_evals=max_evals,
            variation=variation,
            constraint_mode=constraint_mode,
            archive_size=ext_cfg.capacity if ext_cfg is not None else None,
            archive_X=archive_X,
            archive_F=archive_F,
            archive_manager=archive_manager,
            selection_ranks=selection_ranks,
            selection_crowding=selection_crowding,
            result_mode=result_mode,
        )
        live_cb.on_start(
            RunContext(
                problem=problem,
                algorithm=self,
                config=self.cfg,
                algorithm_name="agemoea",
                engine_name=str(self.kernel.name),
            )
        )

    def ask(self) -> np.ndarray[Any, Any]:
        """Generate offspring for external evaluation.

        Returns
        -------
        np.ndarray
            Offspring decision variables, shape ``(n_offspring, n_var)``.

        Raises
        ------
        RuntimeError
            If called before ``initialize()`` or previous offspring not consumed.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        st = self._st
        if st.selection_ranks is None or st.selection_crowding is None or st.selection_ranks.shape[0] != st.F.shape[0]:
            self._refresh_selection_metrics(st)
        assert st.selection_ranks is not None
        assert st.selection_crowding is not None
        assert st.variation is not None
        request_size = capped_offspring_size(st.n_eval, st.max_evals, st.pop_size, "AGE-MOEA")
        n_parents = int(np.ceil(request_size / st.variation.children_per_group) * st.variation.parents_per_group)
        parents_idx = self.kernel.tournament_selection(
            ranks=st.selection_ranks,
            crowding=st.selection_crowding,
            pressure=2,
            rng=st.rng,
            n_parents=n_parents,
        )

        X_off = st.variation.produce_offspring(st.X[parents_idx], st.rng)
        if X_off.shape[0] > request_size:
            X_off = X_off[:request_size]
        st.pending_offspring = X_off
        return np.asarray(X_off).copy()

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive evaluated offspring and update population.

        Parameters
        ----------
        eval_result : Any
            Objective values as ``np.ndarray``, or an object/dict with ``F`` and
            optional ``G`` constraint values.
        problem : ProblemProtocol | None
            Unused, kept for interface consistency.

        Returns
        -------
        bool
            Always ``False`` (AGE-MOEA has no early-stop criterion).

        Raises
        ------
        RuntimeError
            If called before ``ask()``.
        """
        if self._st is None or self._st.pending_offspring is None:
            raise RuntimeError("No pending offspring. Call ask() first.")

        st = self._st
        X_off = st.pending_offspring
        assert X_off is not None
        active_n_constraints = st.G.shape[1] if st.G is not None and st.constraint_mode != "none" else 0
        F_off, G_off = extract_evaluation_arrays(
            eval_result,
            st.constraint_mode,
            active_n_constraints,
            algorithm="AGE-MOEA",
            n_rows=X_off.shape[0],
            n_obj=st.F.shape[1],
        )

        X_combined = np.vstack([st.X, X_off])
        F_combined = np.vstack([st.F, F_off])
        G_combined = combine_constraints(st.G, G_off, st.constraint_mode, algorithm="AGE-MOEA")

        survivors, survival_scores = constraint_aware_age_survival(F_combined, G_combined, st.pop_size, self.kernel, st.constraint_mode)
        st.X = X_combined[survivors]
        st.F = F_combined[survivors]
        st.G = G_combined[survivors] if G_combined is not None else None
        st.n_eval += X_off.shape[0]
        st.selection_ranks, st.selection_crowding = selection_metrics(self.kernel, st.F, st.G, st.constraint_mode, survival_scores)
        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)

        st.pending_offspring = None
        st.generation += 1
        if self._live_cb is not None:
            self._live_cb.on_generation(st.generation, F=st.F, stats={"evals": st.n_eval})
            return live_should_stop(self._live_cb)
        return False

    def should_terminate(self) -> bool:
        """Check if termination criterion is met."""
        if self._st is None:
            return True
        return self._st.n_eval >= self._st.max_evals

    def result(self) -> dict[str, Any]:
        """Get optimization result.

        Returns
        -------
        dict
            Result dictionary with ``X``, ``F``, ``evaluations``, ``generation``,
            ``population``, and optionally ``archive``.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")
        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)
        return build_agemoea_result(self._st, kernel=self.kernel)

    @property
    def state(self) -> AGEMOEAState | None:
        """Access current algorithm state."""
        return self._st
