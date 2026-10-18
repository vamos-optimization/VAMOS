# algorithm/moead/core.py
"""
MOEA/D evolutionary algorithm core.

This module contains the main MOEAD class with the evolutionary loop (run/ask/tell).
- Setup logic: setup.py
- Operator building: operators/policies/moead.py
- State and results: state.py
- Helper functions: helpers.py

References:
    Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
    Decomposition," IEEE Trans. Evolutionary Computation, vol. 11, no. 6, 2007.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    live_should_stop,
    notify_generation,
    track_offspring_genealogy,
)
from vamos.engine.algorithm.components.termination import capped_offspring_size
from vamos.engine.algorithm.components.utils import variation_operator_label
from vamos.engine.archive.factory import update_archive
from vamos.foundation.constraints.utils import compute_violation
from vamos.foundation.kernel import default_kernel

from .helpers import update_neighborhood, update_neighborhood_batch
from .initialization import initialize_moead_run
from .state import MOEADState, build_moead_result

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.engine.hooks.live_viz import LiveVisualization


class MOEAD:
    """Multi-Objective Evolutionary Algorithm based on Decomposition."""

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: MOEADState | None = None
        self._candidate_lengths: np.ndarray[Any, Any] | None = None
        self._candidate_orders: np.ndarray[Any, Any] | None = None
        self._parent_pairs: np.ndarray[Any, Any] | None = None
        self._row_index: np.ndarray[Any, Any] = np.empty(0, dtype=np.int64)

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run MOEA/D until the termination criterion is met."""
        live_cb, eval_strategy, max_eval, hv_tracker = self._initialize_run(
            problem,
            termination,
            seed,
            eval_strategy,
            live_viz,
            checkpoint=checkpoint,
        )
        st = self._st
        assert st is not None, "State not initialized"

        generation = st.generation
        live_cb.on_generation(generation, F=st.F, stats={"evals": st.n_eval})
        stop_requested = live_should_stop(live_cb)
        hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points())

        while st.n_eval < max_eval and not hv_reached and not stop_requested:
            st.generation = generation
            X_off = self.ask()
            eval_result = eval_strategy.evaluate(X_off, problem)
            hv_reached = self.tell(eval_result, problem)

            if hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

            generation += 1
            st.generation = generation
            stop_requested = notify_generation(live_cb, self.kernel, generation, st.F, stats={"evals": st.n_eval})

        result = build_moead_result(st, hv_reached, kernel=self.kernel)
        result["checkpoint"] = {
            "version": 1,
            "algorithm": "moead",
            "X": st.X,
            "F": st.F,
            "G": st.G,
            "generation": st.generation,
            "n_eval": st.n_eval,
            "rng_state": st.rng.bit_generator.state,
            "archive_X": st.archive_X,
            "archive_F": st.archive_F,
            "extra": {
                "subproblem_order": st.subproblem_order,
                "subproblem_cursor": st.subproblem_cursor,
                "ideal": st.ideal,
            },
        }
        finalize_genealogy(result, st, self.kernel)
        live_cb.on_end(final_F=st.F)
        return result

    def _initialize_run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, int, Any]:
        """Initialize algorithm state for a run."""
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_moead_run(
            self.cfg,
            self.kernel,
            problem,
            termination,
            seed,
            eval_strategy,
            live_viz,
            checkpoint=checkpoint,
        )
        cross_method = str(self.cfg.get("crossover", ("sbx", {}))[0]).lower()
        self._cross_is_de = cross_method in {"de", "differential", "differential_evolution"}
        self._op_label = variation_operator_label(self.cfg, "sbx+pm")
        return live_cb, eval_strategy, max_eval, hv_tracker

    def _ensure_parent_pair_buffer(self, batch_size: int) -> np.ndarray[Any, Any]:
        if self._parent_pairs is None or self._parent_pairs.shape[0] < batch_size:
            self._parent_pairs = np.empty((batch_size, 2), dtype=int)
        return self._parent_pairs[:batch_size]

    def _ensure_candidate_order_buffers(self, batch_size: int, pop_size: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if self._candidate_orders is None or self._candidate_orders.shape[0] < batch_size or self._candidate_orders.shape[1] != pop_size:
            self._candidate_orders = np.empty((batch_size, pop_size), dtype=int)
        if self._candidate_lengths is None or self._candidate_lengths.shape[0] < batch_size:
            self._candidate_lengths = np.empty(batch_size, dtype=np.int64)
        return self._candidate_orders[:batch_size], self._candidate_lengths[:batch_size]

    def _ensure_row_index(self, length: int) -> np.ndarray[Any, Any]:
        if self._row_index.shape[0] < length:
            self._row_index = np.arange(length, dtype=np.int64)
        return self._row_index[:length]

    def _sample_parent_pairs(
        self,
        st: MOEADState,
        active: np.ndarray[Any, Any],
        use_neighbors: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Sample one ordered parent pair per active subproblem."""
        batch_size = active.shape[0]
        parent_pairs = self._ensure_parent_pair_buffer(batch_size)
        if batch_size == 0:
            return parent_pairs
        if batch_size == 1:
            # One offspring per step: same generator calls and pairs as the batch path below.
            if bool(use_neighbors[0]):
                row = st.neighbors[active[0]]
                first = st.rng.integers(0, row.shape[0], size=1, dtype=np.int64)
                second = st.rng.integers(0, row.shape[0] - 1, size=1, dtype=np.int64)
                parent_pairs[0, 0] = row[first[0]]
                parent_pairs[0, 1] = row[second[0] + (second[0] >= first[0])]
            else:
                first = st.rng.integers(0, st.pop_size, size=1, dtype=np.int64)
                second = st.rng.integers(0, st.pop_size - 1, size=1, dtype=np.int64)
                parent_pairs[0, 0] = first[0]
                parent_pairs[0, 1] = second[0] + (second[0] >= first[0])
            return parent_pairs

        neighbor_mask = np.asarray(use_neighbors, dtype=bool)
        if bool(np.any(neighbor_mask)):
            active_neighbors = st.neighbors[active[neighbor_mask]]
            local_count = active_neighbors.shape[0]
            local_size = active_neighbors.shape[1]
            first_local = st.rng.integers(0, local_size, size=local_count, dtype=np.int64)
            second_local = st.rng.integers(0, local_size - 1, size=local_count, dtype=np.int64)
            second_local += second_local >= first_local
            rows = self._ensure_row_index(local_count)
            parent_pairs[neighbor_mask, 0] = active_neighbors[rows, first_local]
            parent_pairs[neighbor_mask, 1] = active_neighbors[rows, second_local]

        global_mask = ~neighbor_mask
        if bool(np.any(global_mask)):
            global_count = int(np.sum(global_mask))
            first_global = st.rng.integers(0, st.pop_size, size=global_count, dtype=np.int64)
            second_global = st.rng.integers(0, st.pop_size - 1, size=global_count, dtype=np.int64)
            second_global += second_global >= first_global
            parent_pairs[global_mask, 0] = first_global
            parent_pairs[global_mask, 1] = second_global

        return parent_pairs

    def _build_candidate_orders(
        self,
        st: MOEADState,
        active: np.ndarray[Any, Any],
        use_neighbors: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Precompute candidate replacement orders for a batch of offspring."""
        batch_size = active.shape[0]
        candidate_orders, candidate_lengths = self._ensure_candidate_order_buffers(batch_size, st.pop_size)
        if batch_size == 0:
            return candidate_orders, candidate_lengths

        neighbor_mask = np.asarray(use_neighbors, dtype=bool)
        if bool(np.any(neighbor_mask)):
            neighbor_rows = st.neighbors[active[neighbor_mask]]
            neighbor_size = neighbor_rows.shape[1]
            candidate_orders[neighbor_mask, :neighbor_size] = neighbor_rows
            candidate_lengths[neighbor_mask] = neighbor_size

        global_positions = np.flatnonzero(~neighbor_mask)
        for pos in global_positions:
            candidate_orders[pos, : st.pop_size] = st.rng.permutation(st.pop_size)
            candidate_lengths[pos] = st.pop_size

        return candidate_orders, candidate_lengths

    def ask(self) -> np.ndarray[Any, Any]:
        """Generate offspring solutions to be evaluated."""
        st = self._st
        if st is None:
            raise RuntimeError("ask() called before initialization.")
        if st.crossover_fn is None or st.mutation_fn is None:
            raise RuntimeError("MOEA/D variation operators are not initialized.")

        pop_size = st.pop_size
        max_eval = st.max_evals
        batch_size = capped_offspring_size(st.n_eval, max_eval, int(st.batch_size), "MOEA/D")
        if batch_size <= 0:
            raise ValueError("MOEA/D batch_size must be positive.")
        if batch_size > pop_size:
            batch_size = pop_size
        active = self._next_active_indices(st, batch_size)
        use_neighbors = st.rng.random(batch_size) < st.delta

        parent_pairs = self._sample_parent_pairs(st, active, use_neighbors)

        n_var = st.X.shape[1]
        if self._cross_is_de:
            parents = np.empty((batch_size, 3, n_var), dtype=st.X.dtype)
            parents[:, 0, :] = st.X[parent_pairs[:, 0]]
            parents[:, 1, :] = st.X[parent_pairs[:, 1]]
            parents[:, 2, :] = st.X[active]
            offspring = st.crossover_fn(parents, st.rng)
            children = offspring[:, 0, :].copy()
        else:
            parents = st.X[parent_pairs.reshape(-1)].reshape(batch_size, 2, n_var)
            offspring = st.crossover_fn(parents, st.rng)
            children = offspring[:, 0, :].copy()

        children = st.mutation_fn(children, st.rng)

        st.pending_offspring = children
        st.pending_active_indices = active
        st.pending_parent_pairs = parent_pairs
        st.pending_use_neighbors = use_neighbors

        if st.track_genealogy:
            parents_flat = np.column_stack([parent_pairs, active]).reshape(-1) if self._cross_is_de else parent_pairs.reshape(-1)
            track_offspring_genealogy(st, parents_flat, children.shape[0], self._op_label, "moead")

        return children

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive evaluated offspring and update the current MOEA/D state."""
        st = self._st
        if st is None:
            raise RuntimeError("tell() called before initialization.")

        children = st.pending_offspring
        active = st.pending_active_indices
        use_neighbors = st.pending_use_neighbors

        if children is None or active is None or use_neighbors is None:
            raise ValueError("tell() called without a pending ask().")

        F_child = eval_result.F
        G_child = eval_result.G if st.constraint_mode != "none" else None
        cv_child = compute_violation(G_child) if G_child is not None else None
        batch_size = children.shape[0]
        st.n_eval += batch_size

        st.pending_offspring = None
        st.pending_active_indices = None
        st.pending_parent_pairs = None
        st.pending_use_neighbors = None

        pop_size = st.pop_size

        st.ideal = np.minimum(st.ideal, F_child.min(axis=0))

        if batch_size == 1:
            child = children[0]
            child_f = F_child[0]
            child_g = G_child[0] if G_child is not None else None
            cv_penalty = float(cv_child[0]) if cv_child is not None else 0.0
            if use_neighbors[0]:
                candidate_order = st.neighbors[active[0]]
            else:
                candidate_order = st.rng.permutation(pop_size)
            update_neighborhood(
                st=st,
                idx=int(active[0]),
                child=child,
                child_f=child_f,
                child_g=child_g,
                cv_penalty=cv_penalty,
                candidate_order=candidate_order,
            )
        else:
            candidate_orders, candidate_lengths = self._build_candidate_orders(st, active, use_neighbors)
            update_neighborhood_batch(
                st=st,
                children=children,
                children_f=F_child,
                children_g=G_child,
                children_cv=cv_child,
                candidate_orders=candidate_orders,
                candidate_lengths=candidate_lengths,
            )

        update_archive(st, st.X, st.F, st.G)

        return st.hv_tracker is not None and st.hv_tracker.enabled and st.hv_tracker.reached(st.hv_points())

    @staticmethod
    def _next_active_indices(st: MOEADState, batch_size: int) -> np.ndarray[Any, Any]:
        """Return next subproblem indices using a rolling permutation."""
        if st.subproblem_order.size != st.pop_size:
            st.subproblem_order = st.rng.permutation(st.pop_size).astype(int, copy=False)
            st.subproblem_cursor = 0

        active = np.empty(batch_size, dtype=int)
        for i in range(batch_size):
            if st.subproblem_cursor >= st.pop_size:
                st.subproblem_order = st.rng.permutation(st.pop_size).astype(int, copy=False)
                st.subproblem_cursor = 0
            active[i] = int(st.subproblem_order[st.subproblem_cursor])
            st.subproblem_cursor += 1
        return active


__all__ = ["MOEAD"]
