"""
C-TAEA: the two-archive evolutionary algorithm for constrained multi-objective optimization.

Reference:
    K. Li, R. Chen, G. Fu and X. Yao, "Two-Archive Evolutionary Algorithm for Constrained
    Multiobjective Optimization", IEEE Transactions on Evolutionary Computation 23(2):303-315,
    2019, doi:10.1109/TEVC.2018.2855411.
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

from .archives import convergence_archive_update, diversity_archive_update
from .mating import ctaea_tournament, mating_proportions
from .state import CTAEAState, build_ctaea_result
from .weights import ctaea_weight_vectors


def build_ctaea_variation(
    config: dict[str, Any],
    encoding: EncodingLike,
    xl: Any,
    xu: Any,
    problem: ProblemProtocol,
    variation_kernels: VariationKernels | None = None,
) -> VariationPipeline:
    """The variation pipeline of a C-TAEA configuration: a two-parent crossover, then the mutation of its first child.

    The configuration's operators replace the encoding defaults whole; a crossover that does not
    mate parents two by two is rejected.
    """
    overrides: dict[str, Any] = {key: config[key] for key in ("crossover", "mutation") if key in config}
    if "repair" in config and config["repair"] != "auto":
        overrides["repair"] = config["repair"]
    var_cfg = {**resolve_default_variation_config(str(normalize_encoding(encoding)), None), **overrides}
    c_name, c_params = ensure_operator_tuple(var_cfg.get("crossover", ("sbx", {})), key="crossover")
    m_name, m_params = ensure_operator_tuple(var_cfg.get("mutation", ("polynomial", {})), key="mutation")
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
    if pipeline.parents_per_group != 2:
        raise ValueError(
            f"C-TAEA mates parents two by two, but crossover '{c_name}' needs {pipeline.parents_per_group} parents per group. "
            "Use a two-parent crossover."
        )
    return pipeline


class CTAEA:
    """C-TAEA: a convergence archive and a diversity archive of N solutions, one per weight vector (Li et al., 2019).

    The weight vectors come from the requested population size ``pop_size`` (see
    :func:`~vamos.engine.algorithm.ctaea.weights.ctaea_weight_vectors`); their number N, for
    instance 91 for 100 and three objectives, is the size of both archives. The convergence
    archive (CA) is driven by feasibility and convergence, and the diversity archive (DA), which
    ignores the constraints, keeps solutions in the subregions where the CA has few (see
    :mod:`vamos.engine.algorithm.ctaea.archives`). The N random solutions of the initial
    population are the offspring of the first update of both archives.

    Each generation creates one offspring per weight vector. The first parent of every offspring
    comes from the CA when the proportion of non-dominated solutions of ``CA + DA`` that belong to
    the CA is larger than that of the DA, and from the DA otherwise; the second parent comes from
    the CA with a probability equal to the proportion of non-dominated solutions of the CA, and from
    the DA otherwise (see :mod:`vamos.engine.algorithm.ctaea.mating`). Each parent wins a binary
    tournament in its archive, and the offspring is the first child of the crossover, mutated. The
    CA and then the DA are updated with the offspring. The result is the CA. This follows jMetal's
    port (pull request jMetal/jMetal#524), which follows PlatEMO's implementation where it differs
    from the paper.

    The crossover receives the second parent first. jMetal's SBX gives a variable that it does
    not cross to the other child, so jMetal's first child takes those variables from the second
    parent; VAMOS's SBX, as Deb's code, leaves them with their own parent, so the order
    ``(second, first)`` gives the kept child jMetal's distribution. With the first parent first,
    the median IGD+ of 30 runs on C3-DTLZ4 is 0.064 instead of jMetal's 0.056.

    The initial population takes N evaluations, so the budget must cover them. Budgets are exact:
    when the last generation has fewer evaluations left than N, it creates the N offspring of a
    full generation and evaluates only the first ones, in the order of the weight vectors, which
    both archives are then updated with. jMetal checks the budget after each generation, so it
    can exceed it by up to N - 1 evaluations. ``n_gen`` counts generations of N evaluations, the
    initial population included.

    Parameters
    ----------
    config : dict
        Algorithm configuration (see :class:`~vamos.engine.algorithm.config.CTAEAConfig`).
    kernel : KernelBackend, optional
        Backend for vectorized operations.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: CTAEAState | None = None
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
        """Run C-TAEA and return its result."""
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
        """Create and evaluate the initial population (N evaluations), the offspring of the first update of both archives."""
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()
        live_cb = get_live_viz(live_viz)

        requested = int(self.cfg.get("pop_size", 100))
        if requested < 1:
            raise ValueError(f"C-TAEA needs pop_size >= 1, got {requested}.")
        n_obj = int(problem.n_obj)
        if n_obj < 2:
            raise ValueError(f"C-TAEA needs at least two objectives, got {n_obj}.")
        weights = ctaea_weight_vectors(requested, n_obj)
        size = int(weights.shape[0])
        constraint_mode = normalize_constraint_mode(self.cfg.get("constraint_mode", "feasibility"))
        n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
        max_evals, hv_config = parse_termination(termination, "C-TAEA", size)
        if max_evals < size:
            raise ValueError(
                f"C-TAEA requires max_evaluations >= {size}, the number of weight vectors that pop_size={requested} gives for "
                f"{n_obj} objectives, because its initial population consumes one evaluation per weight vector "
                f"(max_evaluations={max_evals})."
            )
        result_mode = str(self.cfg.get("result_mode", "non_dominated") or "non_dominated").strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        xl, xu = resolve_bounds(problem, encoding)
        # Build (and validate) the variation before any evaluation.
        variation = build_ctaea_variation(self.cfg, encoding, xl, xu, problem, self.kernel.variation_kernels())

        def evaluate(X: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
            return extract_evaluation_arrays(backend.evaluate(X, problem), constraint_mode, n_constraints, algorithm="C-TAEA")

        X, F, G, n_eval = initialize_evaluated_population(
            size,
            problem.n_var,
            xl,
            xu,
            encoding,
            rng,
            problem,
            self.cfg.get("initializer"),
            evaluate=evaluate,
            max_evaluations=max_evals,
            algorithm="C-TAEA",
        )
        violation = _violation(G, F.shape[0])
        kept = convergence_archive_update(F, violation, weights)
        diverse = diversity_archive_update(F[kept], F, weights)

        ext_cfg = resolve_external_archive(self.cfg)
        CA_X, CA_F, CA_G = X[kept], F[kept], None if G is None else G[kept]
        archive_X, archive_F, archive_manager = setup_archive(self.kernel, CA_X, CA_F, problem.n_var, problem.n_obj, X.dtype, ext_cfg, CA_G)

        self._live_cb = live_cb
        self._st = CTAEAState(
            X=CA_X,
            F=CA_F,
            G=CA_G,
            rng=rng,
            pop_size=requested,
            offspring_size=size,
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
            weights=weights,
            diversity_X=X[diverse],
            diversity_F=F[diverse],
            diversity_G=None if G is None else G[diverse],
            variation=variation,
        )
        self._st.hv_reached = hv_target_reached(self._st)
        live_cb.on_start(
            RunContext(problem=problem, algorithm=self, config=self.cfg, algorithm_name="ctaea", engine_name=str(self.kernel.name))
        )
        # Generation 0 is the CA built from the evaluated initial population.
        live_cb.on_generation(0, F=self._st.F, stats={"evals": self._st.n_eval})

    def ask(self) -> np.ndarray[Any, Any]:
        """One offspring per weight vector, capped to the remaining budget: the first ones, in the order of the weight vectors.

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
        request = capped_offspring_size(st.n_eval, st.max_evals, st.offspring_size, "C-TAEA")
        X_off = self._offspring()[:request]
        st.pending_offspring = X_off
        return np.asarray(X_off).copy()

    def _offspring(self) -> np.ndarray[Any, Any]:
        """The ``offspring_size`` offspring of a full generation (the restricted mating selection, then the variation)."""
        st = self._st
        assert st is not None and st.variation is not None
        assert st.diversity_X is not None and st.diversity_F is not None
        n = st.offspring_size
        proportions = mating_proportions(st.F, st.diversity_F)
        feasible = _violation(st.G, st.F.shape[0]) == 0.0
        diversity_feasible = _violation(st.diversity_G, st.diversity_F.shape[0]) == 0.0
        if proportions.first_parent_from_convergence_archive:
            first = st.X[ctaea_tournament(st.F, feasible, n, st.rng)]
        else:
            first = st.diversity_X[ctaea_tournament(st.diversity_F, diversity_feasible, n, st.rng)]
        from_convergence = st.rng.random(n) < proportions.non_dominated_share_of_convergence_archive
        in_convergence = st.X[ctaea_tournament(st.F, feasible, n, st.rng)]
        in_diversity = st.diversity_X[ctaea_tournament(st.diversity_F, diversity_feasible, n, st.rng)]
        second = np.where(from_convergence[:, None], in_convergence, in_diversity)
        # The i-th group is (i-th second parent, i-th first parent), and the first child of each is kept: VAMOS's
        # SBX leaves a variable it does not cross with its own parent, while jMetal's gives it to the other child,
        # so jMetal's first child inherits those variables from the second parent, as this one does.
        parents = np.stack([second, first], axis=1).reshape(2 * n, st.X.shape[1])
        children = st.variation.recombine(parents, st.rng)
        first_children = np.ascontiguousarray(children[:: st.variation.children_per_group])
        return st.variation.mutate(first_children, st.rng)

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive the evaluated offspring; update the CA, then the DA with the new CA.

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
        assert X_off is not None and st.weights is not None
        assert st.diversity_X is not None and st.diversity_F is not None
        active_n_constraints = st.G.shape[1] if st.G is not None and st.constraint_mode != "none" else 0
        F_off, G_off = extract_evaluation_arrays(
            eval_result,
            st.constraint_mode,
            active_n_constraints,
            algorithm="C-TAEA",
            n_rows=X_off.shape[0],
            n_obj=st.F.shape[1],
        )

        # The CA selects from [itself, offspring].
        X_union = np.vstack([st.X, X_off])
        F_union = np.vstack([st.F, F_off])
        G_union = combine_constraints(st.G, G_off, st.constraint_mode, algorithm="C-TAEA")
        kept = convergence_archive_update(F_union, _violation(G_union, F_union.shape[0]), st.weights)
        st.X, st.F = X_union[kept], F_union[kept]
        st.G = G_union[kept] if G_union is not None else None

        # The DA selects from [itself, offspring], to complement the new CA.
        diversity_X_union = np.vstack([st.diversity_X, X_off])
        diversity_F_union = np.vstack([st.diversity_F, F_off])
        diversity_G_union = combine_constraints(st.diversity_G, G_off, st.constraint_mode, algorithm="C-TAEA")
        diverse = diversity_archive_update(st.F, diversity_F_union, st.weights)
        st.diversity_X, st.diversity_F = diversity_X_union[diverse], diversity_F_union[diverse]
        st.diversity_G = diversity_G_union[diverse] if diversity_G_union is not None else None

        st.n_eval += X_off.shape[0]
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
        """The result: the CA (see :func:`~vamos.engine.algorithm.ctaea.state.build_ctaea_result`)."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")
        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)
        return build_ctaea_result(self._st, kernel=self.kernel)

    @property
    def state(self) -> CTAEAState | None:
        """Access current algorithm state."""
        return self._st


def _violation(G: np.ndarray[Any, Any] | None, n: int) -> np.ndarray[Any, Any]:
    """Constraint violation of each row, zero without constraint values."""
    return compute_violation(G, n=n)


__all__ = ["CTAEA", "build_ctaea_variation"]
