"""PPS state container and result."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.results import (
    apply_final_archive_subset,
    feasible_result_indices,
    get_external_archive_constraints,
    get_external_archive_contents,
    wants_population_result,
    with_constraints,
)
from vamos.engine.algorithm.components.state import AlgorithmState

from .stages import feasible_selection

if TYPE_CHECKING:
    from vamos.engine.variation.pipeline import VariationPipeline

    from .setup import PPSSettings


def _empty(n_cols: int) -> np.ndarray[Any, Any]:
    return np.zeros((0, n_cols), dtype=float)


@dataclass
class PPSState(AlgorithmState):
    """State of PPS-MOEA/D.

    ``X``, ``F``, ``G`` and ``phi`` (the overall violation, Eq. 3) are the population, one
    solution per subproblem, whose weight vectors are ``weights`` (``weights_safe`` without zero
    weights, for Eq. 7) and whose neighbourhoods are the rows of ``neighbors``. ``ideal`` is the
    ideal point ``z*`` of every solution evaluated (Algorithm 7), and ``max_violation`` the largest
    finite violation of every solution evaluated (Algorithm 5), -1 before the first.

    ``generation`` counts the completed generations, so the paper's counter is ``k = generation +
    1``; ``cursor`` is the subproblem of the next new solution in the current generation, and
    ``generation_started`` whether lines 5-16 of Algorithm 3 have set its stage and epsilon.
    ``history`` holds the ideal and nadir points of the population at the start of the last
    ``switch_window + 1`` generations (Algorithm 4). ``rate`` is ``r_k`` (Eq. 4), 1.0 until it can
    be computed; ``push_stage`` whether the search is in its push stage; ``switch_generation`` the
    ``k`` of the switch to the pull stage, ``None`` before it; ``epsilon`` the epsilon level of the
    current generation and ``initial_epsilon`` its value at the switch, ``epsilon(0)``.
    ``settings`` holds the run's parameters (see :class:`~vamos.engine.algorithm.pps.setup.PPSSettings`).
    ``ns_X``, ``ns_F``, ``ns_G`` and ``ns_phi`` are the output set ``NS``: feasible solutions,
    selected from it and the population at the end of each generation (Algorithm 8).
    """

    weights: np.ndarray[Any, Any] = field(default_factory=lambda: _empty(1), repr=False)
    weights_safe: np.ndarray[Any, Any] = field(default_factory=lambda: _empty(1), repr=False)
    neighbors: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64), repr=False)
    phi: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    ideal: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    max_violation: float = -1.0
    history: deque[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = field(default_factory=deque, repr=False)
    cursor: int = 0
    generation_started: bool = False
    rate: float = 1.0
    push_stage: bool = True
    switch_generation: int | None = None
    epsilon: float = 0.0
    initial_epsilon: float = 0.0
    settings: PPSSettings | None = None
    ns_X: np.ndarray[Any, Any] = field(default_factory=lambda: _empty(1), repr=False)
    ns_F: np.ndarray[Any, Any] = field(default_factory=lambda: _empty(1), repr=False)
    ns_G: np.ndarray[Any, Any] | None = field(default=None, repr=False)
    ns_phi: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    pending_candidates: np.ndarray[Any, Any] | None = field(default=None, repr=False)
    hv_reached: bool = False
    #: The evaluations that the live callback has been sent.
    live_sent_evals: int = 0
    variation: VariationPipeline | None = None

    @property
    def k(self) -> int:
        """The paper's generation counter, from 1."""
        return self.generation + 1


def feasible_set(state: PPSState, kernel: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """``NS`` once the current generation's population is merged into it: ``(X, F, G)``.

    At the end of a generation the population has been merged (line 37 of Algorithm 3), so this
    is ``NS`` itself; within a generation, which a budget can end, it is ``NDSelect(NS U P, N)``.
    """
    if state.cursor == 0:
        return state.ns_X, state.ns_F, state.ns_G
    X = np.vstack([state.ns_X, state.X])
    F = np.vstack([state.ns_F, state.F])
    G = np.vstack([state.ns_G, state.G]) if state.ns_G is not None and state.G is not None else None
    phi = np.concatenate([state.ns_phi, state.phi])
    selected = feasible_selection(X, F, phi, state.pop_size, kernel)
    return X[selected], F[selected], G[selected] if G is not None else None


def result_rows(state: PPSState, kernel: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """The rows the result reports, ``(X, F, G)``, as :func:`build_pps_result` describes them."""
    archive_contents = get_external_archive_contents(state)
    if archive_contents is not None and not wants_population_result(state):
        archive_G = get_external_archive_constraints(state)
        return archive_contents[0].copy(), archive_contents[1].copy(), archive_G.copy() if archive_G is not None else None
    if wants_population_result(state):
        return state.X.copy(), state.F.copy(), state.G.copy() if state.G is not None else None
    X, F, G = feasible_set(state, kernel)
    candidate_G = G
    if X.shape[0] == 0:
        X, F, G = state.X, state.F, state.G
        candidate_G = G
        if G is not None and state.constraint_mode != "none" and X.shape[0]:
            # The least violated by phi, where a NaN constraint is an infinite violation: by the raw G, one
            # NaN would make the least violation NaN and leave no candidate.
            least = np.flatnonzero(state.phi <= float(np.min(state.phi)) + 1e-12)
            X, F, G, candidate_G = X[least], F[least], G[least], None
    selected = feasible_result_indices(F, candidate_G, state.constraint_mode, kernel)
    return X[selected].copy(), F[selected].copy(), G[selected].copy() if G is not None else None


def build_pps_result(state: PPSState, kernel: Any = None) -> dict[str, Any]:
    """Result of PPS: the non-dominated solutions of ``NS``, the paper's output, unless ``result_mode`` is ``population``.

    ``NS`` holds feasible solutions only; when the run found none, the result is the non-dominated
    solutions of the least violated ones of the final population. With an external archive, the
    result is the archive (unless ``result_mode`` is ``population``); ``population`` gives the
    final population, which can contain infeasible and dominated solutions. ``feasible_set`` is
    the whole of ``NS``, which can contain dominated solutions when it holds ``pop_size`` or fewer
    (Algorithm 8). ``stage`` is ``"push"`` or ``"pull"``, the stage of the last generation;
    ``switch_generation`` the generation ``k`` of the switch to the pull stage, ``None`` without
    one; ``epsilon`` the epsilon level of the last generation.
    """
    result_X, result_F, result_G = result_rows(state, kernel)
    ns_X, ns_F, ns_G = feasible_set(state, kernel)
    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": state.n_eval,
        "generation": state.generation,
        "hv_reached": state.hv_reached,
        "stage": "push" if state.push_stage else "pull",
        "switch_generation": state.switch_generation,
        "epsilon": state.epsilon,
        "feasible_set": with_constraints({"X": ns_X.copy(), "F": ns_F.copy()}, None if ns_G is None else ns_G.copy()),
        "population": with_constraints({"X": state.X.copy(), "F": state.F.copy()}, None if state.G is None else state.G.copy()),
    }
    with_constraints(result, result_G)
    archive_contents = get_external_archive_contents(state)
    if archive_contents is not None:
        archive_X, archive_F = archive_contents
        result["archive"] = with_constraints({"X": archive_X, "F": archive_F}, get_external_archive_constraints(state))
    return apply_final_archive_subset(result, state)


__all__ = ["PPSState", "build_pps_result", "feasible_set", "result_rows"]
