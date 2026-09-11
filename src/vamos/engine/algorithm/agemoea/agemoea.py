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

from vamos.engine.algorithm.components.hooks import get_live_viz, live_should_stop
from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.engine.algorithm.components.termination import capped_offspring_size, validate_initial_budget
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
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.observer import RunContext
from vamos.foundation.problem.types import ProblemProtocol

from .geometry import age_survival
from .state import AGEMOEAState, build_agemoea_result


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _build_variation(config: dict[str, Any], encoding: Any, xl: Any, xu: Any, problem: ProblemProtocol) -> VariationPipeline:
    explicit_overrides: dict[str, Any] = {}
    if "crossover" in config:
        explicit_overrides["crossover"] = config["crossover"]
    if "mutation" in config:
        explicit_overrides["mutation"] = config["mutation"]
    if "repair" in config and config["repair"] != "auto":
        explicit_overrides["repair"] = config["repair"]

    var_cfg = resolve_default_variation_config(encoding, explicit_overrides)
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
    )


class AGEMOEA:
    """AGE-MOEA: Adaptive Geometry Estimation MOEA.

    Implements adaptive geometry estimation for survival selection as in the
    original AGE-MOEA paper.

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
        ranks, crowding = self.kernel.nsga2_ranking(st.F)
        st.selection_ranks = np.asarray(ranks, dtype=int)
        st.selection_crowding = np.asarray(crowding, dtype=float)

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
            F_off = np.asarray(backend.evaluate(X_off, problem).F, dtype=float)
            stop_requested = self.tell(F_off)
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
        F = np.asarray(backend.evaluate(X, problem).F, dtype=float)

        variation = _build_variation(self.cfg, encoding, xl, xu, problem)
        ext_cfg = resolve_external_archive(self.cfg)
        archive_X, archive_F, archive_manager = setup_archive(
            self.kernel,
            X,
            F,
            problem.n_var,
            problem.n_obj,
            X.dtype,
            ext_cfg,
            None,
        )
        selection_ranks, selection_crowding = self.kernel.nsga2_ranking(F)

        result_mode = str(self.cfg.get("result_mode", "non_dominated")).strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")

        self._live_cb = live_cb
        self._st = AGEMOEAState(
            X=X,
            F=F,
            G=None,
            rng=rng,
            pop_size=pop_size,
            n_eval=X.shape[0],
            generation=0,
            max_evals=max_evals,
            variation=variation,
            archive_size=ext_cfg.capacity if ext_cfg is not None else None,
            archive_X=archive_X,
            archive_F=archive_F,
            archive_manager=archive_manager,
            selection_ranks=np.asarray(selection_ranks, dtype=int),
            selection_crowding=np.asarray(selection_crowding, dtype=float),
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
        return np.array(X_off, copy=True)

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive evaluated offspring and update population.

        Parameters
        ----------
        eval_result : Any
            Objective values as ``np.ndarray``, or an object with ``.F`` attribute,
            or a dict with ``"F"`` key.
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

        if hasattr(eval_result, "F"):
            F_off = np.asarray(eval_result.F, dtype=float)
        elif isinstance(eval_result, dict):
            F_off = np.asarray(eval_result["F"], dtype=float)
        else:
            F_off = np.asarray(eval_result, dtype=float)

        st.n_eval += X_off.shape[0]

        X_combined = np.vstack([st.X, X_off])
        F_combined = np.vstack([st.F, F_off])

        survivors = age_survival(F_combined, st.pop_size, self.kernel)
        st.X = X_combined[survivors]
        st.F = F_combined[survivors]
        self._refresh_selection_metrics(st)
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
