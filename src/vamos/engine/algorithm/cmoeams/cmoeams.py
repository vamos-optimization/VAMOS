"""
CMOEA-MS: the constrained multi-objective evolutionary algorithm with multiple stages.

Reference:
    Y. Tian, Y. Zhang, Y. Su, X. Zhang, K. C. Tan and Y. Jin, "Balancing Objective Optimization and
    Constraint Satisfaction in Constrained Evolutionary Multiobjective Optimization", IEEE
    Transactions on Cybernetics 52(9):9559-9572, 2022, doi:10.1109/TCYB.2020.3021138.
"""

from __future__ import annotations

import math
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
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.kernel import default_kernel
from vamos.foundation.kernel.backend import KernelBackend, VariationKernels
from vamos.foundation.observer import RunContext
from vamos.foundation.problem.types import ProblemProtocol

from .differential_evolution import exponential_crossover
from .fitness import is_stage_b, normalized_violation, stage_a_fitness, stage_b_fitness
from .selection import cmoeams_tournament, environmental_selection
from .state import CMOEAMSState, build_cmoeams_result

DE_VARIANTS = ("rand_1_bin", "rand_1_exp")
FEASIBLE_RATIO_THRESHOLD = 0.5
_DE_DEFAULTS = {"cr": 1.0, "f": 0.5}


def _differential_evolution_parameters(params: dict[str, Any]) -> tuple[float, float]:
    """CR and F from ``cr`` and ``f`` (or ``CR`` and ``F``); 1.0 and 0.5 by default. CR must be in [0, 1] and F positive and finite."""
    unknown = sorted(set(params) - {"cr", "CR", "f", "F"})
    if unknown:
        raise ValueError(f"CMOEA-MS's DE variation takes the parameters cr and f; got {', '.join(unknown)}.")
    cr = float(params.get("cr", params.get("CR", _DE_DEFAULTS["cr"])))
    f = float(params.get("f", params.get("F", _DE_DEFAULTS["f"])))
    if not 0.0 <= cr <= 1.0:
        raise ValueError(f"CMOEA-MS's crossover rate cr must be in [0, 1], got {cr}.")
    if not (math.isfinite(f) and f > 0.0):
        raise ValueError(f"CMOEA-MS's scale factor f must be positive and finite, got {f}.")
    return cr, f


def de_variant_name(value: object) -> str:
    """The DE variant, ``rand_1_bin`` (PlatEMO's, the default) or ``rand_1_exp``, in lower case."""
    name = "rand_1_bin" if value is None else str(value).strip().lower()
    if name not in DE_VARIANTS:
        raise ValueError(f"CMOEA-MS's de_variant must be one of {', '.join(DE_VARIANTS)}; got {value!r}.")
    return name


def build_cmoeams_variation(
    config: dict[str, Any],
    encoding: EncodingLike,
    xl: Any,
    xu: Any,
    problem: ProblemProtocol,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationPipeline, bool, float, float]:
    """The variation pipeline of a CMOEA-MS configuration, whether it is the DE variation, and its CR and F.

    The crossover ``de`` selects the DE variation (real encoding only), with ``cr`` and ``f``; any
    other crossover selects the GA variation, which needs two parents and two children per group.
    """
    overrides: dict[str, Any] = {key: config[key] for key in ("crossover", "mutation") if key in config}
    if "repair" in config and config["repair"] != "auto":
        overrides["repair"] = config["repair"]
    # The configuration holds the resolved operators: they replace the encoding defaults whole.
    var_cfg = {**resolve_default_variation_config(str(normalize_encoding(encoding)), None), **overrides}
    c_name, c_params = ensure_operator_tuple(var_cfg.get("crossover", ("sbx", {})), key="crossover")
    m_name, m_params = ensure_operator_tuple(var_cfg.get("mutation", ("polynomial", {})), key="mutation")
    differential = c_name.strip().lower() == "de"
    cr, f = _DE_DEFAULTS["cr"], _DE_DEFAULTS["f"]
    if differential:
        if normalize_encoding(encoding) != "real":
            raise ValueError(f"CMOEA-MS's DE variation (crossover 'de') needs real variables; the problem's encoding is {encoding!r}.")
        cr, f = _differential_evolution_parameters(dict(c_params))
        c_params = {"CR": cr, "F": f}
    elif de_variant_name(config.get("de_variant")) != "rand_1_bin":
        raise ValueError(f"CMOEA-MS's de_variant applies to the DE variation (crossover 'de'), not to crossover {c_name!r}.")
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
    if not differential and (pipeline.parents_per_group != 2 or pipeline.children_per_group != 2):
        raise ValueError(
            f"CMOEA-MS's GA variation needs a crossover with two parents and two children, but crossover '{c_name}' takes "
            f"{pipeline.parents_per_group} parents and gives {pipeline.children_per_group} children. Use a two-parent crossover, "
            "or crossover 'de' for the DE variation."
        )
    return pipeline, differential, cr, f


def _feasible_ratio_threshold(config: dict[str, Any]) -> float:
    """``feasible_ratio_threshold`` (the paper's ``lambda``), finite and in [0, 1]; 0.5 by default."""
    raw = config.get("feasible_ratio_threshold")
    value = FEASIBLE_RATIO_THRESHOLD if raw is None else float(raw)
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"CMOEA-MS's feasible_ratio_threshold must be in [0, 1], got {raw!r}.")
    return value


class CMOEAMS:
    """CMOEA-MS: a population of ``pop_size`` solutions whose fitness has two stages (Tian et al., 2022).

    Each generation selects parents by binary tournaments on the fitness that the population had
    in the union it was selected from (see :mod:`vamos.engine.algorithm.cmoeams.selection`),
    creates the offspring, and computes the fitness of the union ``Q`` of the population and the
    offspring, in that order, in one of two stages (see :mod:`vamos.engine.algorithm.cmoeams.fitness`):

    - **stage A** gives the objectives the same priority as the constraints: Pareto dominance on
      ``[SDE, CV]``, the shift-based density and the constraint violation;
    - **stage B** gives them a lower priority: a smaller violation dominates, and equal violations
      are compared by Pareto dominance on the objectives.

    ``Q`` is in stage B when its ratio of feasible solutions is greater than
    ``feasible_ratio_threshold`` (``lambda``, 0.5 by default) and at least 10% of the evaluations
    have been done, those of the offspring included. The fitness is SPEA2's, below 1 for the
    solutions that no other one dominates; the environmental selection keeps those, completed by
    the lowest fitness or truncated by cosine distance between the objective vectors, in the order
    of ``Q``. The fitness of the initial population is stage A's. The result is the final
    population, which can contain infeasible and dominated solutions. This follows jMetal's port
    (pull request jMetal/jMetal#526), which follows the authors' implementation in PlatEMO where it
    differs from the paper. The variation is chosen by the crossover:

    - the GA variation (any crossover with two parents and two children; SBX by default) selects
      ``pop_size`` parents, pairs the i-th of the first half with the i-th of the second half and
      keeps both children, the first ones at positions ``0, ..., pop_size // 2 - 1`` of the
      offspring and the second ones after them, then mutates them: ``2 * (pop_size // 2)``
      offspring. The crossover receives the second parent of each pair first: jMetal's SBX gives
      a variable that it does not cross to the other child, while VAMOS's leaves it with its own
      parent, so with this order each child has, at its position, the distribution of jMetal's;
    - the DE variation (crossover ``de``, real variables) selects two sets of ``pop_size`` parents
      by tournament and creates, for the i-th solution ``x_i``, the mutant ``x_i + F (x_r1 -
      x_r2)``, crossed with ``x_i`` by the binomial crossover of rate ``CR`` with one variable
      always from the mutant (``de_variant="rand_1_bin"``, VAMOS's ``de`` operator) or by the
      exponential crossover (``"rand_1_exp"``, see
      :mod:`vamos.engine.algorithm.cmoeams.differential_evolution`), then repairs (clamps by
      default) and mutates it: ``pop_size`` offspring.

    The initial population takes ``pop_size`` evaluations, so the budget must cover them. Budgets
    are exact: when fewer evaluations are left than offspring, the last generation creates the
    offspring of a full generation and evaluates the first ones, in the order of the offspring
    (the first children of the pairs, then the second ones; or the population's order), and ``Q``
    is the population and those. jMetal checks the budget before each generation, so it can
    exceed it by up to the offspring of a generation minus one. The 10% condition counts the
    evaluations done once the offspring are evaluated, as jMetal: every generation but the last
    has the evaluations of jMetal's, and in the last one both have done at least 10% of the
    budget, so every stage decision is jMetal's.

    Parameters
    ----------
    config : dict
        Algorithm configuration (see :class:`~vamos.engine.algorithm.config.CMOEAMSConfig`).
    kernel : KernelBackend, optional
        Backend for vectorized operations.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: CMOEAMSState | None = None
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
        """Run CMOEA-MS and return its result."""
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
        """Create and evaluate the initial population (``pop_size`` evaluations) and compute its stage-A fitness."""
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()
        live_cb = get_live_viz(live_viz)

        pop_size = int(self.cfg.get("pop_size", 100))
        if pop_size < 2:
            raise ValueError(f"CMOEA-MS needs pop_size >= 2, got {pop_size}.")
        constraint_mode = normalize_constraint_mode(self.cfg.get("constraint_mode", "feasibility"))
        n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
        max_evals, hv_config = parse_termination(termination, "CMOEA-MS", pop_size)
        if max_evals < pop_size:
            raise ValueError(
                "CMOEA-MS requires max_evaluations >= pop_size because its initial population consumes pop_size evaluations "
                f"(max_evaluations={max_evals}, pop_size={pop_size})."
            )
        result_mode = str(self.cfg.get("result_mode", "non_dominated") or "non_dominated").strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")
        threshold = _feasible_ratio_threshold(self.cfg)
        de_variant = de_variant_name(self.cfg.get("de_variant"))

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        xl, xu = resolve_bounds(problem, encoding)
        # Build (and validate) the variation before any evaluation.
        variation, differential, cr, f = build_cmoeams_variation(self.cfg, encoding, xl, xu, problem, self.kernel.variation_kernels())

        def evaluate(X: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
            return extract_evaluation_arrays(backend.evaluate(X, problem), constraint_mode, n_constraints, algorithm="CMOEA-MS")

        X, F, G, n_eval = initialize_evaluated_population(
            pop_size,
            problem.n_var,
            xl,
            xu,
            encoding,
            rng,
            problem,
            self.cfg.get("initializer"),
            evaluate=evaluate,
            max_evaluations=max_evals,
            algorithm="CMOEA-MS",
        )
        fitness = stage_a_fitness(F, normalized_violation(G, F.shape[0]))

        ext_cfg = resolve_external_archive(self.cfg)
        archive_X, archive_F, archive_manager = setup_archive(self.kernel, X, F, problem.n_var, problem.n_obj, X.dtype, ext_cfg, G)

        self._live_cb = live_cb
        self._st = CMOEAMSState(
            X=X,
            F=F,
            G=G,
            rng=rng,
            pop_size=pop_size,
            offspring_size=pop_size if differential else 2 * (pop_size // 2),
            n_eval=n_eval,
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
            feasible_ratio_threshold=threshold,
            variation=variation,
            differential_evolution=differential,
            de_variant=de_variant,
            de_cr=cr,
            de_f=f,
        )
        self._st.hv_reached = hv_target_reached(self._st)
        live_cb.on_start(
            RunContext(problem=problem, algorithm=self, config=self.cfg, algorithm_name="cmoeams", engine_name=str(self.kernel.name))
        )
        # Generation 0 is the evaluated initial population.
        live_cb.on_generation(0, F=self._st.F, stats={"evals": self._st.n_eval})

    def ask(self) -> np.ndarray[Any, Any]:
        """The offspring of the next generation, capped to the remaining budget: the first ones, in the order of the offspring.

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
        request = capped_offspring_size(st.n_eval, st.max_evals, st.offspring_size, "CMOEA-MS")
        X_off = self._differential_offspring() if st.differential_evolution else self._genetic_offspring()
        X_off = X_off[:request]
        st.pending_offspring = X_off
        return np.asarray(X_off).copy()

    def _genetic_offspring(self) -> np.ndarray[Any, Any]:
        """``pop_size`` tournaments; parent i mates with parent ``i + pop_size // 2``, and both children are kept."""
        st = self._st
        assert st is not None and st.variation is not None and st.fitness is not None
        n, n_var = st.X.shape
        half = n // 2
        pool = cmoeams_tournament(st.fitness, n, st.rng)
        # The i-th group is (i-th second parent, i-th first parent): VAMOS's SBX leaves a variable it does not cross
        # with its own parent, while jMetal's gives it to the other child, so each child here has jMetal's distribution.
        parents = np.stack([st.X[pool[half : 2 * half]], st.X[pool[:half]]], axis=1).reshape(2 * half, n_var)
        children = st.variation.recombine(parents, st.rng)
        offspring = np.concatenate([children[0::2], children[1::2]])
        return st.variation.mutate(np.ascontiguousarray(offspring), st.rng)

    def _differential_offspring(self) -> np.ndarray[Any, Any]:
        """The child of the i-th solution is ``x_i + F (x_r1 - x_r2)``, with r1 and r2 from two sets of ``pop_size`` tournaments."""
        st = self._st
        assert st is not None and st.variation is not None and st.fitness is not None
        X = st.X
        n, n_var = X.shape
        first = cmoeams_tournament(st.fitness, n, st.rng)
        second = cmoeams_tournament(st.fitness, n, st.rng)
        if st.de_variant == "rand_1_bin":
            # VAMOS's de operator: the mutant x + F (a - b) of each group [x, a, b], its binomial crossover with x and the repair.
            groups = np.stack([X, X[first], X[second]], axis=1).reshape(3 * n, n_var)
            trial = st.variation.recombine(groups, st.rng)
        else:
            mutant = X + st.de_f * (X[first] - X[second])
            start = st.rng.integers(0, n_var, size=n)
            trial = exponential_crossover(X, mutant, start, st.rng.random((n, n_var)), st.de_cr)
            repair = st.variation.repair_op
            if repair is not None:
                trial = np.asarray(repair(trial, st.variation.xl, st.variation.xu, st.rng), dtype=float)
        return st.variation.mutate(np.ascontiguousarray(trial), st.rng)

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive the evaluated offspring: the fitness of the union in its stage, then the environmental selection.

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
        assert X_off is not None
        active_n_constraints = st.G.shape[1] if st.G is not None and st.constraint_mode != "none" else 0
        F_off, G_off = extract_evaluation_arrays(
            eval_result,
            st.constraint_mode,
            active_n_constraints,
            algorithm="CMOEA-MS",
            n_rows=X_off.shape[0],
            n_obj=st.F.shape[1],
        )
        st.n_eval += X_off.shape[0]

        # The union of the population and the offspring, in that order.
        X_union = np.vstack([st.X, X_off])
        F_union = np.vstack([st.F, F_off])
        G_union = combine_constraints(st.G, G_off, st.constraint_mode, algorithm="CMOEA-MS")
        violation = normalized_violation(G_union, F_union.shape[0])
        st.stage_b = is_stage_b(violation, st.feasible_ratio_threshold, st.n_eval, st.max_evals)
        fitness = stage_b_fitness(F_union, violation) if st.stage_b else stage_a_fitness(F_union, violation)
        selected = environmental_selection(fitness, F_union, st.pop_size, self.kernel.selection_kernels())
        st.X, st.F = X_union[selected], F_union[selected]
        st.G = G_union[selected] if G_union is not None else None
        st.fitness = fitness[selected]
        st.stage_b_generations += int(st.stage_b)

        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)
        st.pending_offspring = None
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
        """The result: the final population (see :func:`~vamos.engine.algorithm.cmoeams.state.build_cmoeams_result`)."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")
        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)
        return build_cmoeams_result(self._st, kernel=self.kernel)

    @property
    def state(self) -> CMOEAMSState | None:
        """Access current algorithm state."""
        return self._st


__all__ = ["CMOEAMS", "DE_VARIANTS", "FEASIBLE_RATIO_THRESHOLD", "build_cmoeams_variation", "de_variant_name"]
