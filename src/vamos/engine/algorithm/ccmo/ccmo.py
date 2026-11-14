"""
CCMO: the coevolutionary framework for constrained multi-objective optimization.

Reference:
    Y. Tian, T. Zhang, J. Xiao, X. Zhang and Y. Jin, "A Coevolutionary Framework for Constrained
    Multiobjective Optimization Problems", IEEE Transactions on Evolutionary Computation
    25(1):102-116, 2021, doi:10.1109/TEVC.2020.3004012.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.engine.algorithm.components.evaluation import combine_constraints, extract_evaluation_arrays
from vamos.engine.algorithm.components.hooks import get_live_viz, live_should_stop
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import initialize_evaluated_population, resolve_bounds
from vamos.engine.algorithm.components.termination import capped_offspring_size, hv_target_reached, parse_termination
from vamos.engine.algorithm.config.base import normalize_constraint_mode
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.config.variation import ensure_operator_tuple, resolve_default_variation_config
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.engine.variation.helpers import ensure_supported_operator_names, ensure_supported_repair_name
from vamos.engine.variation.pipeline import VariationPipeline
from vamos.engine.variation.protocol import RepairConfigValue
from vamos.foundation.constraints.utils import compute_violation
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.kernel import default_kernel
from vamos.foundation.kernel.backend import KernelBackend, VariationKernels
from vamos.foundation.observer import RunContext
from vamos.foundation.problem.types import ProblemProtocol

from .selection import ccmo_environmental_selection, ccmo_fitness, ccmo_tournament
from .state import CCMOState, build_ccmo_result

_DE_DEFAULTS = {"cr": 1.0, "f": 0.5}


def _differential_evolution_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """``cr`` and ``f`` (or ``CR`` and ``F``) as the arguments of the ``de`` operator; 1.0 and 0.5 by default."""
    unknown = sorted(set(params) - {"cr", "CR", "f", "F", "prob"})
    if unknown:
        raise ValueError(f"CCMO's DE variation takes the parameters cr and f; got {', '.join(unknown)}.")
    cr = params.get("cr", params.get("CR", _DE_DEFAULTS["cr"]))
    f = params.get("f", params.get("F", _DE_DEFAULTS["f"]))
    return {"CR": float(cr), "F": float(f)}


def build_ccmo_variation(
    config: dict[str, Any],
    encoding: EncodingLike,
    xl: Any,
    xu: Any,
    problem: ProblemProtocol,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationPipeline, bool]:
    """The variation pipeline of a CCMO configuration, and whether it is the DE variation.

    The crossover ``de`` selects the DE variation (real encoding only), with ``cr`` and ``f``;
    any other crossover selects the GA variation, which mates parents two by two.
    """
    overrides: dict[str, Any] = {key: config[key] for key in ("crossover", "mutation") if key in config}
    if "repair" in config and config["repair"] != "auto":
        overrides["repair"] = config["repair"]
    # The configuration holds the resolved operators: they replace the encoding defaults whole.
    var_cfg = {**resolve_default_variation_config(str(normalize_encoding(encoding)), None), **overrides}
    c_name, c_params = ensure_operator_tuple(var_cfg.get("crossover", ("sbx", {})), key="crossover")
    m_name, m_params = ensure_operator_tuple(var_cfg.get("mutation", ("polynomial", {})), key="mutation")
    differential = c_name.strip().lower() == "de"
    if differential:
        if normalize_encoding(encoding) != "real":
            raise ValueError(f"CCMO's DE variation (crossover 'de') needs real variables; the problem's encoding is {encoding!r}.")
        c_params = _differential_evolution_parameters(dict(c_params))
    cross_name, mut_name = ensure_supported_operator_names(encoding, c_name, m_name)
    repair_cfg: RepairConfigValue = "auto"
    if "repair" in var_cfg:
        repair_name, repair_params = ensure_operator_tuple(var_cfg["repair"], key="repair")
        repair_cfg = (ensure_supported_repair_name(encoding, repair_name), repair_params)
    pipeline = VariationPipeline(
        encoding=encoding,
        cross_method=cross_name,
        cross_params=c_params,
        mut_method=mut_name,
        mut_params=m_params,
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=repair_cfg,
        problem=problem,
        variation_kernels=variation_kernels,
    )
    if not differential and pipeline.parents_per_group != 2:
        raise ValueError(
            f"CCMO's GA variation mates parents two by two, but crossover '{c_name}' needs {pipeline.parents_per_group} "
            "parents per group. Use a two-parent crossover, or crossover 'de' for the DE variation."
        )
    return pipeline, differential


class CCMO:
    """CCMO: two populations of size ``pop_size`` that share their offspring (Tian et al., 2021).

    Population 1 solves the problem and population 2 the problem without constraints. In each
    generation each population creates offspring from its own mating selection, and then each
    population selects its next generation from itself together with the offspring of both
    populations (in the order: itself, population 1's offspring, population 2's offspring). The
    result is population 1.

    The optimizer of each population is SPEA2 with a dominance relation in which the constraint
    violation comes first (see :mod:`vamos.engine.algorithm.ccmo.selection`), as in the
    authors' reference code and jMetal's port (pull request jMetal/jMetal#521), which differ from
    the paper: the paper embeds NSGA-II with SPEA2's truncation, and compares two infeasible
    solutions with the same violation as non-dominated. The variation is chosen by the crossover:

    - the GA variation (any two-parent crossover; SBX by default) selects ``pop_size`` parents by
      binary tournament, pairs the i-th of the first half with the i-th of the second half, and
      keeps the first child of each pair, then mutates it: ``pop_size // 2`` offspring per
      population;
    - the DE variation (crossover ``de``, real variables) selects ``2 * pop_size`` parents and
      creates, for the i-th member ``x_i`` of the population, ``x_i + F (r1_i - r2_i)`` with
      binomial crossover of rate ``CR``, then mutates it: ``pop_size`` offspring per population.

    The two initial populations take ``2 * pop_size`` evaluations, so the budget must cover them.
    Budgets are exact: when the last generation has fewer evaluations left than its offspring, it
    evaluates the first ones of ``[population 1's offspring, population 2's offspring]``, so
    population 2's offspring are cut first.

    Parameters
    ----------
    config : dict
        Algorithm configuration (see :class:`~vamos.engine.algorithm.config.CCMOConfig`).
    kernel : KernelBackend, optional
        Backend for vectorized operations.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: CCMOState | None = None
        self._live_cb: LiveVisualization | None = None

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
        """Run CCMO and return its result."""
        self.initialize(problem, termination, seed, eval_strategy, live_viz)
        backend = eval_strategy or SerialEvalBackend()

        assert self._st is not None
        stop_requested = live_should_stop(self._live_cb) if self._live_cb is not None else False
        while not stop_requested and not self.should_terminate():
            X_off = self.ask()
            stop_requested = self.tell(backend.evaluate(X_off, problem))

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
        """Create and evaluate both initial populations (``2 * pop_size`` evaluations)."""
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()
        live_cb = get_live_viz(live_viz)

        pop_size = int(self.cfg.get("pop_size", 100))
        if pop_size < 2:
            raise ValueError(f"CCMO needs pop_size >= 2, got {pop_size}.")
        constraint_mode = normalize_constraint_mode(self.cfg.get("constraint_mode", "feasibility"))
        n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
        max_evals, hv_config = parse_termination(termination, "CCMO", pop_size)
        if max_evals < 2 * pop_size:
            raise ValueError(
                "CCMO requires max_evaluations >= 2 * pop_size because its two initial populations consume "
                f"2 * pop_size evaluations (max_evaluations={max_evals}, pop_size={pop_size})."
            )
        result_mode = str(self.cfg.get("result_mode", "non_dominated") or "non_dominated").strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        xl, xu = resolve_bounds(problem, encoding)
        # Build (and validate) the variation before any evaluation.
        variation, differential = build_ccmo_variation(self.cfg, encoding, xl, xu, problem, self.kernel.variation_kernels())

        def evaluate(X: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
            return extract_evaluation_arrays(backend.evaluate(X, problem), constraint_mode, n_constraints, algorithm="CCMO")

        initializer = self.cfg.get("initializer")
        X, F, G, n_main = initialize_evaluated_population(
            pop_size,
            problem.n_var,
            xl,
            xu,
            encoding,
            rng,
            problem,
            initializer,
            evaluate=evaluate,
            max_evaluations=max_evals,
            algorithm="CCMO",
        )
        helper_X, helper_F, helper_G, n_helper = initialize_evaluated_population(
            pop_size,
            problem.n_var,
            xl,
            xu,
            encoding,
            rng,
            problem,
            initializer,
            evaluate=evaluate,
            max_evaluations=max_evals - n_main,
            algorithm="CCMO",
        )
        fitness, _ = ccmo_fitness(F, _violation(G, F.shape[0]))
        helper_fitness, _ = ccmo_fitness(helper_F, np.zeros(helper_F.shape[0]))

        ext_cfg = resolve_external_archive(self.cfg)
        archive_X, archive_F, archive_manager = setup_archive(self.kernel, X, F, problem.n_var, problem.n_obj, X.dtype, ext_cfg, G)

        self._live_cb = live_cb
        self._st = CCMOState(
            X=X,
            F=F,
            G=G,
            rng=rng,
            pop_size=pop_size,
            offspring_size=2 * (pop_size if differential else pop_size // 2),
            n_eval=n_main + n_helper,
            generation=0,
            max_evals=max_evals,
            constraint_mode=constraint_mode,
            archive_size=ext_cfg.capacity if ext_cfg is not None else None,
            archive_X=archive_X,
            archive_F=archive_F,
            archive_manager=archive_manager,
            result_mode=result_mode,
            hv_tracker=setup_hv_tracker(hv_config, self.kernel),
            fitness=fitness,
            helper_X=helper_X,
            helper_F=helper_F,
            helper_G=helper_G,
            helper_fitness=helper_fitness,
            variation=variation,
            differential_evolution=differential,
        )
        self._st.hv_reached = hv_target_reached(self._st)
        live_cb.on_start(
            RunContext(problem=problem, algorithm=self, config=self.cfg, algorithm_name="ccmo", engine_name=str(self.kernel.name))
        )
        # Generation 0 is the evaluated initial population 1, as in every algorithm.
        live_cb.on_generation(0, F=self._st.F, stats={"evals": self._st.n_eval})

    def ask(self) -> np.ndarray[Any, Any]:
        """Offspring of both populations, population 1's first, capped to the remaining budget.

        Raises
        ------
        RuntimeError
            If called before ``initialize()``, with offspring that ``tell()`` has not received,
            or when the budget is exhausted.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")
        st = self._st
        assert st.fitness is not None and st.helper_X is not None and st.helper_fitness is not None
        request = capped_offspring_size(st.n_eval, st.max_evals, st.offspring_size, "CCMO")
        main = self._offspring(st.X, st.fitness)
        helper = self._offspring(st.helper_X, st.helper_fitness)
        X_off = np.concatenate([main, helper])[:request]
        st.pending_offspring = X_off
        st.pending_from_main = min(main.shape[0], request)
        return np.asarray(X_off).copy()

    def _offspring(self, X: np.ndarray[Any, Any], fitness: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        st = self._st
        assert st is not None and st.variation is not None
        n, n_var = X.shape
        if st.differential_evolution:
            # The i-th child is x_i + F (r1_i - r2_i), with r1 and r2 from 2n tournaments.
            pool = ccmo_tournament(fitness, 2 * n, st.rng)
            groups = np.stack([X, X[pool[:n]], X[pool[n:]]], axis=1).reshape(3 * n, n_var)
            return st.variation.mutate(st.variation.recombine(groups, st.rng), st.rng)
        # n tournaments; the i-th parent of the first half mates with the i-th of the second half.
        half = n // 2
        pool = ccmo_tournament(fitness, n, st.rng)
        parents = np.stack([X[pool[:half]], X[pool[half : 2 * half]]], axis=1).reshape(2 * half, n_var)
        children = st.variation.recombine(parents, st.rng)
        first_children = np.ascontiguousarray(children[:: st.variation.children_per_group])
        return st.variation.mutate(first_children, st.rng)

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive the evaluated offspring; each population selects its next generation.

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
            ``True`` when the HV target of an ``("hv", ...)`` termination is
            reached or the live callback asks to stop.

        Raises
        ------
        RuntimeError
            If called before ``ask()``.
        """
        if self._st is None or self._st.pending_offspring is None:
            raise RuntimeError("No pending offspring. Call ask() first.")

        st = self._st
        X_off = st.pending_offspring
        assert X_off is not None and st.helper_X is not None and st.helper_F is not None
        active_n_constraints = st.G.shape[1] if st.G is not None and st.constraint_mode != "none" else 0
        F_off, G_off = extract_evaluation_arrays(
            eval_result,
            st.constraint_mode,
            active_n_constraints,
            algorithm="CCMO",
            n_rows=X_off.shape[0],
            n_obj=st.F.shape[1],
        )
        kernels = self.kernel.selection_kernels()

        # Population 1 selects from [itself, offspring] with the constraint violation.
        X_union = np.vstack([st.X, X_off])
        F_union = np.vstack([st.F, F_off])
        G_union = combine_constraints(st.G, G_off, st.constraint_mode, algorithm="CCMO")
        selected, st.fitness = ccmo_environmental_selection(F_union, _violation(G_union, F_union.shape[0]), st.pop_size, kernels)
        st.X, st.F = X_union[selected], F_union[selected]
        st.G = G_union[selected] if G_union is not None else None

        # Population 2 selects from [itself, offspring] without it.
        helper_X_union = np.vstack([st.helper_X, X_off])
        helper_F_union = np.vstack([st.helper_F, F_off])
        helper_G_union = combine_constraints(st.helper_G, G_off, st.constraint_mode, algorithm="CCMO")
        selected, st.helper_fitness = ccmo_environmental_selection(helper_F_union, np.zeros(helper_F_union.shape[0]), st.pop_size, kernels)
        st.helper_X, st.helper_F = helper_X_union[selected], helper_F_union[selected]
        st.helper_G = helper_G_union[selected] if helper_G_union is not None else None

        st.n_eval += X_off.shape[0]
        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)
        st.pending_offspring = None
        st.pending_from_main = 0
        st.generation += 1
        st.hv_reached = st.hv_reached or hv_target_reached(st)
        stop_requested = False
        if self._live_cb is not None:
            self._live_cb.on_generation(st.generation, F=st.F, stats={"evals": st.n_eval})
            stop_requested = live_should_stop(self._live_cb)
        return stop_requested or st.hv_reached

    def should_terminate(self) -> bool:
        """Whether the budget is spent or the HV target is reached."""
        if self._st is None:
            return True
        return self._st.n_eval >= self._st.max_evals or self._st.hv_reached

    def result(self) -> dict[str, Any]:
        """The result: population 1 (see :func:`~vamos.engine.algorithm.ccmo.state.build_ccmo_result`)."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")
        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)
        return build_ccmo_result(self._st, kernel=self.kernel)

    @property
    def state(self) -> CCMOState | None:
        """Access current algorithm state."""
        return self._st


def _violation(G: np.ndarray[Any, Any] | None, n: int) -> np.ndarray[Any, Any]:
    """Constraint violation of each row, zero without constraint values."""
    return compute_violation(G, n=n)


__all__ = ["CCMO", "build_ccmo_variation"]
