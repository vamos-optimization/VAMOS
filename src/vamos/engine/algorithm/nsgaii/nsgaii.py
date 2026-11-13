"""
NSGA-II evolutionary algorithm core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.termination import HVTracker
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.eval.backends import EvaluationBackend
from vamos.foundation.kernel import default_kernel
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

from .ask_tell import ask_nsgaii, combine_ids, tell_nsgaii
from .run import advance_generation, finish_run, notify_generation, run_nsgaii, save_checkpoint
from .setup import initialize_run
from .state import NSGAIIState


@dataclass
class _AskTellSession:
    """What a run driven by ``initialize``/``ask``/``tell`` keeps between calls, as ``run`` keeps it in its loop."""

    problem: ProblemProtocol
    live_cb: LiveVisualization
    eval_strategy: EvaluationBackend
    stop_requested: bool = False
    hv_reached: bool = False
    ended: bool = False


class NSGAII:
    """
    Vectorized/SOA-style NSGA-II evolutionary core.
    Individuals are represented as array rows (X, F) without per-object instances.

    ``run`` drives a whole run. ``initialize``, ``ask``, ``tell``,
    ``should_terminate`` and ``result`` drive the same run step by step: with
    the same seed, the result equals ``run``'s.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: NSGAIIState | None = None
        self._session: _AskTellSession | None = None

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 50,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._session = None
        return run_nsgaii(
            self,
            problem,
            termination,
            seed,
            eval_strategy=eval_strategy,
            live_viz=live_viz,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            checkpoint=checkpoint,
        )

    def _save_checkpoint(self, checkpoint_dir: str, seed: int, generation: int, n_eval: int) -> None:
        save_checkpoint(self, checkpoint_dir, seed, generation, n_eval)

    def _notify_generation(
        self,
        live_cb: LiveVisualization,
        generation: int,
        F: np.ndarray[Any, Any],
        evals: int | None = None,
    ) -> bool:
        return notify_generation(self, live_cb, generation, F, evals=evals)

    def _initialize_run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_strategy: EvaluationBackend | None,
        live_viz: LiveVisualization | None,
        checkpoint: dict[str, Any] | None = None,
    ) -> tuple[LiveVisualization, EvaluationBackend, int, int, HVTracker]:
        return initialize_run(self, problem, termination, seed, eval_strategy, live_viz, checkpoint=checkpoint)

    def initialize(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        """Start a run driven by ``ask``/``tell``, with the arguments of ``run``.

        The initial population is evaluated and reported as generation 0. Then
        repeat ``ask``, evaluate the decision vectors, and ``tell`` the result
        until ``should_terminate``; ``result`` returns what ``run`` would.
        """
        live_cb, backend, _max_eval, n_eval, hv_tracker = self._initialize_run(
            problem, termination, seed, eval_strategy, live_viz, checkpoint=checkpoint
        )
        st = self._st
        assert st is not None, "State not initialized"
        stop_requested = notify_generation(self, live_cb, st.generation, st.F, problem=problem, evals=n_eval, eval_strategy=backend)
        self._session = _AskTellSession(
            problem=problem,
            live_cb=live_cb,
            eval_strategy=backend,
            stop_requested=stop_requested,
            hv_reached=bool(hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn())),
        )

    def ask(self) -> np.ndarray[Any, Any]:
        return ask_nsgaii(self)

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive the evaluated offspring of the last ``ask``.

        Returns ``True`` once the HV target of an ``("hv", ...)`` termination is reached.
        """
        session = self._session
        pending = self._st.pending_offspring if self._st is not None else None
        hv_reached = tell_nsgaii(self, eval_result)
        if session is None:
            return hv_reached
        n_offspring = 0 if pending is None else int(pending.shape[0])
        stop_requested, generation_hv_reached, _completed = advance_generation(
            self, session.live_cb, n_offspring, problem=session.problem, eval_strategy=session.eval_strategy
        )
        session.stop_requested = session.stop_requested or stop_requested
        session.hv_reached = session.hv_reached or hv_reached or generation_hv_reached
        return session.hv_reached

    def should_terminate(self) -> bool:
        """Whether the budget is spent, the HV target reached, or the live callback asked to stop."""
        st = self._st
        if st is None:
            return True
        session = self._session
        return st.n_eval >= st.max_evals or (session is not None and (session.hv_reached or session.stop_requested))

    def result(self) -> dict[str, Any]:
        """The result of a run started with ``initialize``, as ``run`` returns it."""
        session = self._session
        if self._st is None or session is None:
            raise RuntimeError("Call initialize() before result().")
        live_cb = None if session.ended else session.live_cb
        session.ended = True
        return finish_run(self, live_cb, session.hv_reached)

    def _combine_ids(self, st: NSGAIIState) -> np.ndarray[Any, Any] | None:
        return combine_ids(st)


__all__ = ["NSGAII"]
