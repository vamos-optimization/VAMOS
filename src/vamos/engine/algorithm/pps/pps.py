"""
PPS: the push and pull search framework for constrained multi-objective optimization, in MOEA/D.

Reference:
    Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb and E. Goodman, "Push and pull search
    for solving constrained multi-objective optimization problems", Swarm and Evolutionary
    Computation 44:665-679, 2019, doi:10.1016/j.swevo.2018.08.017.

This is PPS-MOEA/D, Algorithm 3 of the paper, ported from the paper. Where the paper leaves a
choice open, VAMOS takes the following one:

- ``r_k`` compares generation ``k`` with generation ``k - l``, so it is computed from
  ``k = l + 1``: the paper's ``k >= l`` would compare generation ``l`` with a generation 0 that
  its numbering, from 1, does not have;
- ``epsilon(0)`` is the largest violation of every solution evaluated before the switch, as
  Algorithms 3 and 5 compute it; the text describes it as that of the population;
- ``rf_k`` is, as the text says, "the ratio of feasible to infeasible solutions" of the
  population, not the fraction of feasible solutions of MOEA/D-IEpsilon by the same authors
  (see :func:`~vamos.engine.algorithm.pps.stages.feasible_ratio`);
- the replacement loop stops after ``replace_limit`` replacements or when ``S`` is used up (the
  pseudocode's ``or`` in line 25 is an ``and``);
- ``k`` in Eq. 8 is the generation counter of the run, not the generations since the switch;
- a search still in the push stage at ``k = Tc`` stays there, as the pseudocode switches only
  while ``k < Tc``;
- the union of ``NS`` and the population (line 37) counts each distinct solution once;
- the DE child is repaired before the mutation as well as after it, as MOEA/D-DE in VAMOS and
  jMetal, so the mutation works within the bounds;
- a zero weight counts as ``1e-4`` in Eq. 7, whose ``1 / lambda_j`` it would make infinite;
- a NaN constraint makes the violation infinite, and only finite violations count for
  ``epsilon(0)``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.evaluation import extract_evaluation_arrays
from vamos.engine.algorithm.components.hooks import get_live_viz, live_should_stop, send_live_generation
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import initialize_evaluated_population, resolve_bounds
from vamos.engine.algorithm.components.state import record_archive_rows
from vamos.engine.algorithm.components.termination import capped_offspring_size, parse_termination
from vamos.engine.algorithm.config.base import normalize_constraint_mode
from vamos.engine.algorithm.moead.neighborhood import compute_neighbors
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.kernel import default_kernel
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.observer import RunContext
from vamos.foundation.problem.types import ProblemProtocol

from .setup import build_pps_variation, pop_size_of, resolve_settings, weight_vectors
from .stages import (
    change_rate,
    feasible_ratio,
    feasible_selection,
    first_replacements,
    ideal_and_nadir,
    next_epsilon,
    overall_violation,
    pull_replaces,
    push_replaces,
    safe_weights,
    tchebycheff_values,
)
from .state import PPSState, build_pps_result, result_rows


class PPS:
    """PPS-MOEA/D: MOEA/D whose search goes from a push stage to a pull stage (Fan et al., 2019).

    The problem is decomposed into ``pop_size`` subproblems, one per weight vector (those of
    MOEA/D), each with the Tchebycheff function of Eq. 7, ``max_j |f_j - z*_j| / lambda_j``. A
    generation visits the subproblems in order. For subproblem ``i``, ``S`` is its
    ``neighbor_size`` nearest subproblems with probability ``delta`` and the whole population
    otherwise; the new solution is ``x_i + F (x_r2 - x_r3)`` with ``r2 != r3`` drawn from ``S``,
    crossed with ``x_i`` (CR), repaired, mutated and repaired again. It updates the ideal point
    ``z*`` and the largest violation seen, then replaces at most ``replace_limit`` solutions of
    ``S``, tried in random order:

    - in the **push stage**, which ignores the constraints, a solution whose Tchebycheff value is
      not smaller than the new one's (Algorithm 1);
    - in the **pull stage**, by an improved epsilon method: when both violations are within
      ``epsilon`` or equal, the Tchebycheff values decide; otherwise the less violated solution
      wins (Algorithm 2).

    The search starts in the push stage. Before each generation ``k`` (from 1), ``r_k`` is the
    largest relative change of the ideal and nadir points of the population over the last
    ``switch_window`` generations (``l``, Eqs. 4-6); while ``k < Tc``, ``r_k <=
    switch_threshold`` switches the search to the pull stage, with ``epsilon(0)`` the largest
    violation of every solution evaluated so far. In the pull stage and while ``k < Tc``, epsilon
    follows Eq. 8, also in the generation of the switch: ``(1 - tao) epsilon`` when the ratio of
    feasible to infeasible solutions of the population is below ``feasibility_ratio_threshold`` (``alpha``),
    and ``epsilon(0) (1 - k / Tc) ** cp`` otherwise. From ``k = Tc``, epsilon is 0. ``Tc`` is
    ``decay_fraction`` of the run's generations (the budget over the population): the paper's
    ``Tc = 800`` of 1000. After each generation, the output set ``NS`` keeps the feasible
    solutions of itself and the population, at most ``pop_size``, selected as NSGA-II does when
    there are more (Algorithm 8). The module lists the choices VAMOS makes where the paper leaves
    them open.

    The initial population takes ``pop_size`` evaluations, so the budget must cover them. The
    run is steady state: ``ask`` returns one new solution, for the next subproblem. Budgets are
    exact; a budget that ends within a generation merges the population into ``NS`` for the
    result. An ``("hv", ...)`` termination is checked at the end of each generation.

    Parameters
    ----------
    config : dict
        Algorithm configuration (see :class:`~vamos.engine.algorithm.config.PPSConfig`).
    kernel : KernelBackend, optional
        Backend for vectorized operations.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: PPSState | None = None
        self._live_cb: LiveVisualization | None = None

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> dict[str, Any]:
        """Run PPS and return its result."""
        self.initialize(problem, termination, seed, eval_strategy, live_viz)
        backend = eval_strategy or SerialEvalBackend()

        assert self._st is not None
        stop_requested = live_should_stop(self._live_cb) if self._live_cb is not None else False
        while not stop_requested and not self.should_terminate():
            X_off = self.ask()
            stop_requested = self.tell(backend.evaluate(X_off, problem))

        return self.result()

    def initialize(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> None:
        """Create and evaluate the initial population (``pop_size`` evaluations): lines 1-3 of Algorithm 3."""
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()
        live_cb = get_live_viz(live_viz)

        pop_size = pop_size_of(self.cfg)
        constraint_mode = normalize_constraint_mode(self.cfg.get("constraint_mode", "feasibility"))
        n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
        max_evals, hv_config = parse_termination(termination, "PPS", pop_size)
        if max_evals < pop_size:
            raise ValueError(
                "PPS requires max_evaluations >= pop_size because its initial population consumes pop_size evaluations "
                f"(max_evaluations={max_evals}, pop_size={pop_size})."
            )
        result_mode = str(self.cfg.get("result_mode", "non_dominated") or "non_dominated").strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")
        settings = resolve_settings(self.cfg, pop_size, max_evals)

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        if encoding != "real":
            raise ValueError(f"PPS's differential evolution needs real variables; the problem's encoding is {encoding!r}.")
        xl, xu = resolve_bounds(problem, encoding)
        # Build (and validate) the variation and the weights before any evaluation.
        variation = build_pps_variation(self.cfg, encoding, xl, xu, problem, self.kernel.variation_kernels())
        weights = weight_vectors(self.cfg, pop_size, problem.n_obj)

        def evaluate(X: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
            return extract_evaluation_arrays(backend.evaluate(X, problem), constraint_mode, n_constraints, algorithm="PPS")

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
            algorithm="PPS",
        )
        phi = overall_violation(G, F.shape[0])
        finite = phi[np.isfinite(phi)]

        ext_cfg = resolve_external_archive(self.cfg, seed=seed)
        archive_X, archive_F, archive_manager = setup_archive(self.kernel, X, F, problem.n_var, problem.n_obj, X.dtype, ext_cfg, G)

        self._live_cb = live_cb
        self._st = PPSState(
            X=np.array(X, dtype=float),
            F=np.array(F, dtype=float),
            G=None if G is None else np.array(G, dtype=float),
            rng=rng,
            pop_size=pop_size,
            offspring_size=1,
            n_eval=n_eval,
            generation=0,
            max_evals=max_evals,
            constraint_mode=constraint_mode,
            archive_size=ext_cfg.capacity if ext_cfg is not None else None,
            archive_X=archive_X,
            archive_F=archive_F,
            archive_manager=archive_manager,
            result_mode=result_mode,
            live_callback_mode=str(self.cfg.get("live_callback_mode", "population")),
            hv_tracker=setup_hv_tracker(hv_config, self.kernel),
            weights=weights,
            weights_safe=safe_weights(weights),
            neighbors=compute_neighbors(weights, settings.neighbor_size),
            phi=phi,
            # Line 1: z*_j = min_i f_j(x^i); line 3: maxViolation over the population, and the points of generation 1.
            ideal=ideal_and_nadir(F)[0],
            max_violation=float(finite.max()) if finite.size else -1.0,
            history=deque([ideal_and_nadir(F)], maxlen=settings.switch_window + 1),
            settings=settings,
            ns_X=np.zeros((0, problem.n_var), dtype=float),
            ns_F=np.zeros((0, F.shape[1]), dtype=float),
            ns_G=None if G is None else np.zeros((0, G.shape[1]), dtype=float),
            ns_phi=np.zeros(0, dtype=float),
            variation=variation,
        )
        self._st.hv_reached = self._hv_reached()
        live_cb.on_start(
            RunContext(problem=problem, algorithm=self, config=self.cfg, algorithm_name="pps", engine_name=str(self.kernel.name))
        )
        # Generation 0 is the evaluated initial population.
        send_live_generation(live_cb, 0, self._st)
        self._st.live_sent_evals = self._st.n_eval

    def ask(self) -> np.ndarray[Any, Any]:
        """The new solution of the next subproblem (lines 18-21 of Algorithm 3), one row.

        Raises
        ------
        RuntimeError
            If called before ``initialize()``, with a solution that ``tell()`` has not received,
            or when the budget is exhausted.
        """
        st = self._st
        if st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")
        capped_offspring_size(st.n_eval, st.max_evals, 1, "PPS")
        if not st.generation_started:
            self._start_generation(st)
        assert st.variation is not None and st.settings is not None
        i = st.cursor
        candidates = st.neighbors[i] if st.rng.random() < st.settings.delta else np.arange(st.pop_size)
        first = int(st.rng.integers(0, candidates.shape[0]))
        second = int(st.rng.integers(0, candidates.shape[0] - 1))
        second += int(second >= first)
        # VAMOS's de operator: the mutant x_i + F (x_r2 - x_r3) of the group [x_i, x_r2, x_r3], crossed with x_i, then repaired.
        trial = st.variation.recombine(np.stack([st.X[i], st.X[candidates[first]], st.X[candidates[second]]]), st.rng)
        child = np.asarray(st.variation.mutate(np.ascontiguousarray(trial), st.rng), dtype=float)
        st.pending_offspring = child
        st.pending_candidates = candidates
        return child.copy()

    @staticmethod
    def _start_generation(st: PPSState) -> None:
        """Lines 5-16 of Algorithm 3: ``r_k``, the switch to the pull stage, and the epsilon level of generation ``k``."""
        settings = st.settings
        assert settings is not None
        k = st.k
        if k > settings.switch_window:
            (ideal, nadir), (previous_ideal, previous_nadir) = st.history[-1], st.history[0]
            st.rate = change_rate(ideal, previous_ideal, nadir, previous_nadir)
        if k < settings.tc:
            if st.push_stage and st.rate <= settings.switch_threshold:
                st.push_stage = False
                st.switch_generation = k
                st.epsilon = st.initial_epsilon = st.max_violation
            if not st.push_stage:
                st.epsilon = next_epsilon(
                    st.epsilon,
                    st.initial_epsilon,
                    k,
                    feasible_ratio(st.phi),
                    tao=settings.tao,
                    threshold=settings.feasibility_ratio_threshold,
                    cp=settings.cp,
                    tc=settings.tc,
                )
        else:
            st.epsilon = 0.0
        st.generation_started = True

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive the evaluated solution: lines 22-34 of Algorithm 3, then lines 36-37 after the generation's last subproblem.

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
        st = self._st
        if st is None or st.pending_offspring is None or st.pending_candidates is None:
            raise RuntimeError("No pending offspring. Call ask() first.")
        settings = st.settings
        assert settings is not None
        child, candidates = st.pending_offspring, st.pending_candidates
        active_n_constraints = st.G.shape[1] if st.G is not None and st.constraint_mode != "none" else 0
        F_child, G_child = extract_evaluation_arrays(
            eval_result, st.constraint_mode, active_n_constraints, algorithm="PPS", n_rows=1, n_obj=st.F.shape[1]
        )
        st.n_eval += 1
        f = F_child[0]
        # Algorithm 7, which ignores NaN values, and Algorithm 5, over finite violations.
        st.ideal = np.fmin(st.ideal, f)
        child_phi = float(overall_violation(G_child, 1)[0])
        if math.isfinite(child_phi) and child_phi > st.max_violation:
            st.max_violation = child_phi

        # Lines 25-34: the solutions of S in random order, until replace_limit of them are replaced.
        order = st.rng.permutation(candidates)
        weights = st.weights_safe[order]
        child_values = tchebycheff_values(f[None, :], weights, st.ideal)
        values = tchebycheff_values(st.F[order], weights, st.ideal)
        if st.push_stage:
            replaces = push_replaces(child_values, values)
        else:
            replaces = pull_replaces(child_values, child_phi, values, st.phi[order], st.epsilon)
        replaced = order[first_replacements(replaces, settings.replace_limit)]
        st.X[replaced] = child[0]
        st.F[replaced] = f
        if st.G is not None and G_child is not None:
            st.G[replaced] = G_child[0]
        st.phi[replaced] = child_phi

        if st.archive_manager is not None:
            record_archive_rows(st, child, F_child, G_child)
        st.pending_offspring = None
        st.pending_candidates = None
        st.cursor += 1
        stop_requested = self._end_generation(st) if st.cursor == st.pop_size else False
        return stop_requested or st.hv_reached

    def _end_generation(self, st: PPSState) -> bool:
        """Lines 36-37 of Algorithm 3: the next ``k``, its ideal and nadir points, and ``NS``; whether the live callback asks to stop."""
        st.generation += 1
        st.cursor = 0
        st.generation_started = False
        st.history.append(ideal_and_nadir(st.F))
        X = np.vstack([st.ns_X, st.X])
        F = np.vstack([st.ns_F, st.F])
        phi = np.concatenate([st.ns_phi, st.phi])
        selected = feasible_selection(X, F, phi, st.pop_size, self.kernel)
        st.ns_X, st.ns_F, st.ns_phi = X[selected], F[selected], phi[selected]
        if st.ns_G is not None and st.G is not None:
            st.ns_G = np.vstack([st.ns_G, st.G])[selected]
        st.hv_reached = st.hv_reached or self._hv_reached()
        if self._live_cb is None:
            return False
        st.live_sent_evals = st.n_eval
        send_live_generation(self._live_cb, st.generation, st)
        return live_should_stop(self._live_cb)

    def _hv_reached(self) -> bool:
        """Whether the HV target of an ``("hv", ...)`` termination is reached by the rows the result reports."""
        st = self._st
        assert st is not None
        tracker = st.hv_tracker
        return bool(tracker is not None and tracker.enabled and tracker.reached(result_rows(st, self.kernel)[1]))

    def should_terminate(self) -> bool:
        """Whether the budget is spent or the HV target is reached."""
        if self._st is None:
            return True
        return self._st.n_eval >= self._st.max_evals or self._st.hv_reached

    def result(self) -> dict[str, Any]:
        """The result (see :func:`~vamos.engine.algorithm.pps.state.build_pps_result`)."""
        st = self._st
        if st is None:
            raise RuntimeError("Algorithm not initialized.")
        if self._live_cb is not None:
            if st.n_eval > st.live_sent_evals:
                # A budget that ends within a generation: its evaluations reach the callback as one more generation.
                st.live_sent_evals = st.n_eval
                send_live_generation(self._live_cb, st.generation + 1, st)
            self._live_cb.on_end(final_F=st.F)
        return build_pps_result(st, kernel=self.kernel)

    @property
    def state(self) -> PPSState | None:
        """Access current algorithm state."""
        return self._st


__all__ = ["PPS"]
