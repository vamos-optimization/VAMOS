"""
ToP: the two-phase framework for constrained multi-objective optimization, with NSGA-II in phase 2.

Reference:
    Z.-Z. Liu and Y. Wang, "Handling Constrained Multiobjective Optimization Problems With
    Constraints in Both the Decision and Objective Spaces", IEEE Transactions on Evolutionary
    Computation 23(5):870-884, 2019, doi:10.1109/TEVC.2019.2894743.
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

from .differential_evolution import binomial_crossover, current_to_rand_1, rand_1, rand_to_best_1
from .phases import (
    CURRENT_TO_RAND_PROBABILITY,
    DELTA_THRESHOLD,
    FEASIBLE_RATIO_THRESHOLD,
    INITIAL_DELTA,
    PHASE_ONE_EVALUATION_FRACTION,
    constraint_violation,
    feasible_ratio,
    index_of_lowest_mean_objective,
    mean_objectives,
    offspring_replaces_parent,
    phase_one_continues,
    updated_delta,
)
from .selection import constrained_front_numbers, crowding_distances, environmental_selection, top_tournament
from .state import ToPState, build_top_result

_DE_DEFAULTS = {"cr": 1.0, "f": 0.5}


def _differential_evolution_parameters(params: dict[str, Any]) -> tuple[float, float]:
    """CR and F from ``cr`` and ``f`` (or ``CR`` and ``F``, not both); 1.0 and 0.5 by default. CR must be in [0, 1] and F positive and finite."""
    unknown = sorted(set(params) - {"cr", "CR", "f", "F"})
    if unknown:
        raise ValueError(f"ToP's differential evolution takes the parameters cr and f; got {', '.join(unknown)}.")
    for name in ("cr", "f"):
        if name in params and name.upper() in params:
            raise ValueError(f"ToP's differential evolution takes {name} or {name.upper()}, not both.")
    cr = float(params.get("cr", params.get("CR", _DE_DEFAULTS["cr"])))
    f = float(params.get("f", params.get("F", _DE_DEFAULTS["f"])))
    if not 0.0 <= cr <= 1.0:
        raise ValueError(f"ToP's crossover rate cr must be in [0, 1], got {cr}.")
    if not (math.isfinite(f) and f > 0.0):
        raise ValueError(f"ToP's scale factor f must be positive and finite, got {f}.")
    return cr, f


def build_top_variation(
    config: dict[str, Any],
    encoding: EncodingLike,
    xl: Any,
    xu: Any,
    problem: ProblemProtocol,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationPipeline, float, float]:
    """The variation of a ToP configuration: the pipeline that repairs and mutates the trial vectors, CR and F.

    ToP's variation is differential evolution for real variables: the crossover must be ``de``,
    whose ``cr`` and ``f`` are ToP's CR and F. ToP's own operators create the trial vectors (see
    :mod:`vamos.engine.algorithm.top.differential_evolution`); the pipeline's crossover, VAMOS's
    ``de``, is never called, and its repair (``clip`` by default) and mutation (any real-coded
    mutation; polynomial by default) are applied to the trial vectors.
    """
    if normalize_encoding(encoding) != "real":
        raise ValueError(f"ToP's differential evolution needs real variables; the problem's encoding is {encoding!r}.")
    overrides: dict[str, Any] = {key: config[key] for key in ("crossover", "mutation") if key in config}
    if "repair" in config and config["repair"] != "auto":
        overrides["repair"] = config["repair"]
    var_cfg = {**resolve_default_variation_config("real", None), "crossover": ("de", dict(_DE_DEFAULTS)), **overrides}
    c_name, c_params = ensure_operator_tuple(var_cfg.get("crossover", ("de", {})), key="crossover")
    if c_name.strip().lower() != "de":
        raise ValueError(f"ToP's variation is differential evolution: its crossover must be 'de', got {c_name!r}.")
    cr, f = _differential_evolution_parameters(dict(c_params))
    m_name, m_params = ensure_operator_tuple(var_cfg.get("mutation", ("polynomial", {})), key="mutation")
    cross_name, mut_name = ensure_supported_operator_names(encoding, "de", m_name)
    repair_cfg: RepairConfigValue = "auto"
    if "repair" in var_cfg:
        repair_name, repair_params = ensure_operator_tuple(var_cfg["repair"], key="repair")
        repair_cfg = (ensure_supported_repair_name(encoding, repair_name), repair_params)
    pipeline = VariationPipeline(
        encoding=encoding,
        cross_method=cross_name,
        cross_params={"CR": cr, "F": f},
        mut_method=mut_name,
        mut_params=m_params,
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=repair_cfg,
        problem=problem,
        variation_kernels=variation_kernels,
    )
    return pipeline, cr, f


def _threshold(config: dict[str, Any], key: str, default: float, *, maximum: float = math.inf) -> float:
    """A switch threshold of the configuration, finite and in [0, ``maximum``]."""
    raw = config.get(key)
    value = default if raw is None else float(raw)
    if not (math.isfinite(value) and 0.0 <= value <= maximum):
        domain = f"in [0, {maximum:g}]" if math.isfinite(maximum) else "finite and >= 0"
        raise ValueError(f"ToP's {key} must be {domain}, got {raw!r}.")
    return value


class ToP:
    """ToP: a population of ``pop_size`` solutions that goes through two phases (Liu and Wang, 2019).

    **Phase 1** solves the problem as a constrained single-objective one, whose objective is the
    mean of the objectives (see :mod:`vamos.engine.algorithm.top.phases`). For each solution
    ``x_i``, with probability 0.5 a trial vector is created with DE/current-to-rand/1, ``x_i + F
    (x_r1 - x_i) + F (x_r2 - x_r3)``, and otherwise with DE/rand-to-best/1/bin, ``x_r1 + F (x_best
    - x_r1) + F (x_r2 - x_r3)``, where ``x_best`` is the first solution with the lowest mean
    objective and the indices are drawn with replacement; each trial vector replaces ``x_i`` if
    its violation is smaller, or if both are equal and its mean objective is smaller. Then ``Pf``,
    the ratio of feasible solutions, and ``delta`` are updated.

    **Phase 2** is NSGA-II with ToP's constrained dominance (see
    :mod:`vamos.engine.algorithm.top.selection`): each of the ``pop_size`` offspring is ``x_1 + F
    (x_2 - x_3)``, where ``x_1`` wins a binary tournament on the front number and the crowding
    distance and ``x_2`` and ``x_3`` are random, and the next population is the best ``pop_size``
    solutions of the population and the offspring.

    Every trial vector comes from a binomial crossover of rate CR without ``j_rand`` (see
    :mod:`vamos.engine.algorithm.top.differential_evolution`), is repaired (clamped to the bounds
    by default) and mutated (polynomial mutation with probability ``1/n`` and distribution index
    20 by default). A generation belongs to phase 1 while ``(delta >= 0.2 or Pf <= 1/3)`` and the
    evaluations done when it starts are not more than 90% of the maximum; as ``delta`` and ``Pf``
    only change in phase 1 and the evaluations only grow, once phase 2 starts it lasts until the
    end. The thresholds are ``delta_threshold``, ``feasible_ratio_threshold`` and
    ``phase_one_evaluation_fraction``. The result is the final population, which can contain
    infeasible solutions. This follows jMetal's port (pull request jMetal/jMetal#525), which
    follows PlatEMO's implementation where it differs from the paper.

    The initial population takes ``pop_size`` evaluations, so the budget must cover them. Budgets
    are exact: when fewer evaluations are left than ``pop_size``, the last generation, of either
    phase, creates the offspring of a full generation and evaluates only the first ones, in the
    order of the population. In phase 1 they replace the solutions at their positions, the other
    solutions being kept, and ``Pf`` and ``delta`` are updated; in phase 2 the next population is
    selected from the population and them. The phase of a generation is decided when it starts,
    from the evaluations done until then, so a partial generation does not change the switch:
    jMetal, which checks the budget after each generation and can exceed it by up to ``pop_size -
    1`` evaluations, takes the same decisions. ``phase_two_start`` in the result is the number of
    evaluations done when phase 2 started (``None`` if the run did not reach it).

    Parameters
    ----------
    config : dict
        Algorithm configuration (see :class:`~vamos.engine.algorithm.config.ToPConfig`).
    kernel : KernelBackend, optional
        Backend for vectorized operations.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: ToPState | None = None
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
        """Run ToP and return its result."""
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
        """Create and evaluate the initial population (``pop_size`` evaluations); ``delta`` starts at 1."""
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()
        live_cb = get_live_viz(live_viz)

        pop_size = int(self.cfg.get("pop_size", 100))
        if pop_size < 1:
            raise ValueError(f"ToP needs pop_size >= 1, got {pop_size}.")
        constraint_mode = normalize_constraint_mode(self.cfg.get("constraint_mode", "feasibility"))
        n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
        max_evals, hv_config = parse_termination(termination, "ToP", pop_size)
        if max_evals < pop_size:
            raise ValueError(
                "ToP requires max_evaluations >= pop_size because its initial population consumes pop_size evaluations "
                f"(max_evaluations={max_evals}, pop_size={pop_size})."
            )
        result_mode = str(self.cfg.get("result_mode", "non_dominated") or "non_dominated").strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")
        delta_threshold = _threshold(self.cfg, "delta_threshold", DELTA_THRESHOLD)
        feasible_ratio_threshold = _threshold(self.cfg, "feasible_ratio_threshold", FEASIBLE_RATIO_THRESHOLD, maximum=1.0)
        fraction = _threshold(self.cfg, "phase_one_evaluation_fraction", PHASE_ONE_EVALUATION_FRACTION, maximum=1.0)

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        if encoding != "real":
            raise ValueError(f"ToP's differential evolution needs real variables; the problem's encoding is {encoding!r}.")
        xl, xu = resolve_bounds(problem, encoding)
        # Build (and validate) the variation before any evaluation.
        variation, cr, f = build_top_variation(self.cfg, encoding, xl, xu, problem, self.kernel.variation_kernels())

        def evaluate(X: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
            return extract_evaluation_arrays(backend.evaluate(X, problem), constraint_mode, n_constraints, algorithm="ToP")

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
            algorithm="ToP",
        )

        ext_cfg = resolve_external_archive(self.cfg)
        archive_X, archive_F, archive_manager = setup_archive(self.kernel, X, F, problem.n_var, problem.n_obj, X.dtype, ext_cfg, G)

        self._live_cb = live_cb
        self._st = ToPState(
            X=X,
            F=F,
            G=G,
            rng=rng,
            pop_size=pop_size,
            offspring_size=pop_size,
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
            delta=INITIAL_DELTA,
            feasible_ratio=feasible_ratio(constraint_violation(G, F.shape[0])),
            cr=cr,
            f=f,
            delta_threshold=delta_threshold,
            feasible_ratio_threshold=feasible_ratio_threshold,
            phase_one_evaluation_fraction=fraction,
            variation=variation,
        )
        self._st.hv_reached = hv_target_reached(self._st)
        live_cb.on_start(
            RunContext(problem=problem, algorithm=self, config=self.cfg, algorithm_name="top", engine_name=str(self.kernel.name))
        )
        # Generation 0 is the evaluated initial population.
        live_cb.on_generation(0, F=self._st.F, stats={"evals": self._st.n_eval})

    def next_phase(self) -> int:
        """The phase, 1 or 2, of the next generation, from ``delta``, ``Pf`` and the evaluations done.

        Raises
        ------
        RuntimeError
            If called before ``initialize()``.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        st = self._st
        phase_one = phase_one_continues(
            st.delta,
            st.feasible_ratio,
            st.n_eval,
            st.max_evals,
            delta_threshold=st.delta_threshold,
            feasible_ratio_threshold=st.feasible_ratio_threshold,
            phase_one_evaluation_fraction=st.phase_one_evaluation_fraction,
        )
        return 1 if phase_one else 2

    def ask(self) -> np.ndarray[Any, Any]:
        """The offspring of the next generation, of phase 1 or 2, capped to the remaining budget: the first ones, in population order.

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
        request = capped_offspring_size(st.n_eval, st.max_evals, st.offspring_size, "ToP")
        st.pending_phase = self.next_phase()
        if st.pending_phase == 1:
            X_off = self._phase_one_offspring()
        else:
            if st.phase_two_start is None:
                st.phase_two_start = st.n_eval
            X_off = self._phase_two_offspring()
        X_off = X_off[:request]
        st.pending_offspring = X_off
        return np.asarray(X_off).copy()

    def _trial_vectors(self, base: np.ndarray[Any, Any], mutant: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """The binomial crossover of rate CR without ``j_rand``, then the repair (clamping by default) and the mutation."""
        st = self._st
        assert st is not None and st.variation is not None
        trial = binomial_crossover(base, mutant, st.rng.random(base.shape), st.cr)
        repair = st.variation.repair_op
        if repair is not None:
            trial = np.asarray(repair(trial, st.variation.xl, st.variation.xu, st.rng), dtype=float)
        return np.asarray(st.variation.mutate(trial, st.rng), dtype=float)

    def _phase_one_offspring(self) -> np.ndarray[Any, Any]:
        """One trial vector per solution: DE/current-to-rand/1 or DE/rand-to-best/1/bin with probability 0.5 each."""
        st = self._st
        assert st is not None
        X = np.asarray(st.X, dtype=float)
        n = X.shape[0]
        best = X[index_of_lowest_mean_objective(st.F)]
        current_to_rand = st.rng.random(n) < CURRENT_TO_RAND_PROBABILITY
        drawn = st.rng.integers(0, n, size=(n, 3))
        r1, r2, r3 = X[drawn[:, 0]], X[drawn[:, 1]], X[drawn[:, 2]]
        choice = current_to_rand[:, None]
        base = np.where(choice, X, r1)
        mutant = np.where(choice, current_to_rand_1(X, r1, r2, r3, st.f), rand_to_best_1(best, r1, r2, r3, st.f))
        return self._trial_vectors(base, mutant)

    def _phase_two_offspring(self) -> np.ndarray[Any, Any]:
        """One offspring per solution: DE/rand/1/bin with its first vector chosen by a binary tournament."""
        st = self._st
        assert st is not None
        X = np.asarray(st.X, dtype=float)
        n = X.shape[0]
        fronts = constrained_front_numbers(st.F, constraint_violation(st.G, n))
        distances = crowding_distances(st.F, fronts)
        parents = top_tournament(fronts, distances, n, st.rng)
        others = st.rng.integers(0, n, size=(n, 2))
        base = X[parents]
        return self._trial_vectors(base, rand_1(base, X[others[:, 0]], X[others[:, 1]], st.f))

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive the evaluated offspring: the one-to-one replacement of phase 1, or the selection of phase 2.

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
            algorithm="ToP",
            n_rows=X_off.shape[0],
            n_obj=st.F.shape[1],
        )
        G_union = combine_constraints(st.G, G_off, st.constraint_mode, algorithm="ToP")
        k = X_off.shape[0]

        if st.pending_phase == 1:
            # Each evaluated trial vector faces the solution at its position; the others are kept.
            violation = constraint_violation(st.G, st.F.shape[0])
            replaces = offspring_replaces_parent(
                constraint_violation(G_off, k), mean_objectives(F_off), violation[:k], mean_objectives(st.F[:k])
            )
            rows = np.flatnonzero(replaces)
            X, F = st.X.copy(), st.F.copy()
            X[rows], F[rows] = X_off[rows], F_off[rows]
            G = None
            if st.G is not None and G_off is not None:
                G = st.G.copy()
                G[rows] = G_off[rows]
            st.X, st.F, st.G = X, F, G
            violation = constraint_violation(st.G, st.F.shape[0])
            st.feasible_ratio = feasible_ratio(violation)
            st.delta = updated_delta(st.F, violation, st.delta)
        else:
            # The best pop_size of [population, offspring], in that order.
            X_union = np.vstack([st.X, X_off])
            F_union = np.vstack([st.F, F_off])
            selected = environmental_selection(F_union, constraint_violation(G_union, F_union.shape[0]), st.pop_size)
            st.X, st.F = X_union[selected], F_union[selected]
            st.G = G_union[selected] if G_union is not None else None

        st.n_eval += k
        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)
        st.pending_offspring = None
        st.pending_phase = None
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
        """The result: the final population (see :func:`~vamos.engine.algorithm.top.state.build_top_result`)."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")
        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)
        return build_top_result(self._st, kernel=self.kernel)

    @property
    def state(self) -> ToPState | None:
        """Access current algorithm state."""
        return self._st


__all__ = ["ToP", "build_top_variation"]
