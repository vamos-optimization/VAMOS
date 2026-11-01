"""
Immigration/injection support for NSGA-II.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.lifecycle import evaluate_batch, get_eval_strategy

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class ImmigrationStats:
    events: int = 0
    inserted: int = 0
    replaced_indices: list[int] = field(default_factory=list)
    replaced_pages: list[int] = field(default_factory=list)
    mating_participation: int = 0


@dataclass(frozen=True)
class ImmigrantCandidate:
    X: np.ndarray[Any, Any]
    F: np.ndarray[Any, Any] | None = None
    G: np.ndarray[Any, Any] | None = None
    tag: str | None = None


@dataclass
class _ActiveImmigrant:
    X: np.ndarray[Any, Any]
    F: np.ndarray[Any, Any] | None
    G: np.ndarray[Any, Any] | None
    tag: str | None


def _coerce_candidates(raw: Any) -> list[ImmigrantCandidate]:
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        arr = np.asarray(raw)
        if arr.ndim == 1:
            return [ImmigrantCandidate(X=arr.astype(float, copy=True))]
        if arr.ndim == 2:
            return [ImmigrantCandidate(X=row.astype(float, copy=True)) for row in arr]
        return []
    if isinstance(raw, Mapping):
        maybe_X = raw.get("X")
        maybe_F = raw.get("F")
        maybe_G = raw.get("G")
        maybe_tag = raw.get("tag")
        if isinstance(maybe_X, np.ndarray):
            X_arr = np.asarray(maybe_X)
            if X_arr.ndim == 1:
                return [
                    ImmigrantCandidate(
                        X=X_arr.astype(float, copy=True),
                        F=np.asarray(maybe_F, dtype=float).copy()
                        if isinstance(maybe_F, np.ndarray) and np.asarray(maybe_F).ndim == 1
                        else None,
                        G=np.asarray(maybe_G, dtype=float).copy()
                        if isinstance(maybe_G, np.ndarray) and np.asarray(maybe_G).ndim == 1
                        else None,
                        tag=str(maybe_tag) if maybe_tag is not None else None,
                    )
                ]
            if X_arr.ndim == 2:
                F_arr = np.asarray(maybe_F) if isinstance(maybe_F, np.ndarray) else None
                G_arr = np.asarray(maybe_G) if isinstance(maybe_G, np.ndarray) else None
                out: list[ImmigrantCandidate] = []
                for i in range(X_arr.shape[0]):
                    f_i = (
                        np.asarray(F_arr[i], dtype=float).copy()
                        if isinstance(F_arr, np.ndarray) and F_arr.ndim == 2 and i < F_arr.shape[0]
                        else None
                    )
                    g_i = (
                        np.asarray(G_arr[i], dtype=float).copy()
                        if isinstance(G_arr, np.ndarray) and G_arr.ndim == 2 and i < G_arr.shape[0]
                        else None
                    )
                    out.append(
                        ImmigrantCandidate(
                            X=np.asarray(X_arr[i], dtype=float).copy(),
                            F=f_i,
                            G=g_i,
                            tag=str(maybe_tag) if maybe_tag is not None else None,
                        )
                    )
                return out
        return []
    if isinstance(raw, Sequence):
        items_out: list[ImmigrantCandidate] = []
        for item in raw:
            items_out.extend(_coerce_candidates(item))
        return items_out
    return []


def _safe_call_provider(provider: Any, generation: int, state: Any) -> Any:
    if not callable(provider):
        return None
    try:
        return provider(generation, state)
    except TypeError:
        _logger().debug("Could not call provider with (generation, state) signature", exc_info=True)
    try:
        return provider(generation)
    except TypeError:
        _logger().debug("Could not call provider with (generation) signature", exc_info=True)
    return provider()


def _match_active_indices(X: np.ndarray[Any, Any], active: Sequence[_ActiveImmigrant]) -> np.ndarray[Any, Any]:
    if X.size == 0 or not active:
        return np.zeros(0, dtype=int)
    used = np.zeros(X.shape[0], dtype=bool)
    idx: list[int] = []
    for immigrant in active:
        eq = np.isclose(X, immigrant.X.reshape(1, -1), atol=1e-9, rtol=0.0)
        rows = np.flatnonzero(np.all(eq, axis=1) & (~used))
        if rows.size == 0:
            continue
        chosen = int(rows[0])
        used[chosen] = True
        idx.append(chosen)
    if not idx:
        return np.zeros(0, dtype=int)
    return np.asarray(idx, dtype=int)


class ImmigrationManager:
    """
    Generic injection manager.

    Config keys:
      provider: callable(generation, state) -> immigrant payload
      static_candidates: payload accepted by _coerce_candidates
      enabled: bool
      start_generation: int
      every: int
      max_in_pop: int
      max_insert_per_event: int
      breeding_mode: "normal" | "exclude" | "freeze_reinsert"
      replacement_policy: "worst_rank_crowding" | "worst_crowding" | "random" | callable
      skip_if: callable(generation, state) -> bool
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.enabled = bool(cfg.get("enabled", True))
        self.provider = cfg.get("provider")
        self.static_candidates = _coerce_candidates(cfg.get("static_candidates"))
        self.start_generation = int(cfg.get("start_generation", 0))
        self.every = int(max(1, int(cfg.get("every", 1))))
        self.max_in_pop = int(max(0, int(cfg.get("max_in_pop", 1))))
        self.max_insert_per_event = int(max(0, int(cfg.get("max_insert_per_event", 1))))
        self.breeding_mode = str(cfg.get("breeding_mode", "exclude")).lower()
        self.replacement_policy = cfg.get("replacement_policy", "worst_rank_crowding")
        self.skip_if = cfg.get("skip_if")
        self._static_cursor = 0
        self._active: list[_ActiveImmigrant] = []
        self._stats: dict[int, ImmigrationStats] = {}
        self._active_indices = np.zeros(0, dtype=int)
        if self.breeding_mode not in {"normal", "exclude", "freeze_reinsert"}:
            raise ValueError("immigration.breeding_mode must be one of: normal, exclude, freeze_reinsert")

    def _stats_for(self, generation: int) -> ImmigrationStats:
        g = int(max(0, generation))
        if g not in self._stats:
            self._stats[g] = ImmigrationStats()
        return self._stats[g]

    def stats_for_generation(self, generation: int) -> ImmigrationStats:
        st = self._stats.get(int(generation))
        if st is None:
            return ImmigrationStats()
        return ImmigrationStats(
            events=int(st.events),
            inserted=int(st.inserted),
            replaced_indices=list(st.replaced_indices),
            replaced_pages=list(st.replaced_pages),
            mating_participation=int(st.mating_participation),
        )

    def totals(self) -> ImmigrationStats:
        totals = ImmigrationStats()
        for st in self._stats.values():
            totals.events += int(st.events)
            totals.inserted += int(st.inserted)
            totals.mating_participation += int(st.mating_participation)
            if st.replaced_indices:
                totals.replaced_indices.extend(int(i) for i in st.replaced_indices)
            if st.replaced_pages:
                totals.replaced_pages.extend(int(p) for p in st.replaced_pages)
        return totals

    def active_indices(self) -> np.ndarray[Any, Any]:
        return self._active_indices.copy()

    def excluded_indices(self) -> np.ndarray[Any, Any]:
        if self.breeding_mode in {"exclude", "freeze_reinsert"}:
            return self._active_indices.copy()
        return np.zeros(0, dtype=int)

    def _schedule_active(self, generation: int) -> bool:
        g = int(generation)
        if g < self.start_generation:
            return False
        return ((g - self.start_generation) % self.every) == 0

    def _fetch_candidates(self, generation: int, state: Any) -> list[ImmigrantCandidate]:
        dynamic_raw = _safe_call_provider(self.provider, generation, state)
        dynamic_candidates = _coerce_candidates(dynamic_raw)
        if dynamic_candidates:
            return dynamic_candidates
        if not self.static_candidates:
            return []
        out: list[ImmigrantCandidate] = []
        n_take = min(self.max_insert_per_event, len(self.static_candidates))
        for _ in range(n_take):
            cand = self.static_candidates[self._static_cursor % len(self.static_candidates)]
            self._static_cursor = (self._static_cursor + 1) % len(self.static_candidates)
            out.append(cand)
        return out

    def _evaluate_missing_FG(
        self,
        *,
        problem: Any,
        candidates: list[ImmigrantCandidate],
        state: Any,
        eval_strategy: EvaluationBackend,
    ) -> list[ImmigrantCandidate]:
        """Evaluate the candidates without objectives through the run's backend, within its budget.

        The evaluations count in ``state.n_eval``. Candidates beyond the remaining budget keep
        ``F=None`` and are not inserted.
        """
        if not candidates:
            return []
        needs_eval = [i for i, c in enumerate(candidates) if c.F is None]
        if not needs_eval:
            return candidates
        needs_eval = needs_eval[: max(0, int(state.max_evals) - int(state.n_eval))]
        if not needs_eval:
            return candidates
        X_batch = np.vstack([candidates[i].X.reshape(1, -1) for i in needs_eval])
        constraint_mode = str(getattr(state, "constraint_mode", "feasibility"))
        F_raw, G = evaluate_batch(problem, eval_strategy, X_batch, constraint_mode)
        state.n_eval += len(needs_eval)
        F = np.asarray(F_raw)
        if F.ndim != 2 or F.shape[0] != len(needs_eval):
            return candidates
        result = list(candidates)
        for j, idx in enumerate(needs_eval):
            g_i = np.asarray(G[j], dtype=float).copy() if isinstance(G, np.ndarray) and G.ndim == 2 else None
            result[idx] = ImmigrantCandidate(
                X=np.asarray(candidates[idx].X, dtype=float).copy(),
                F=np.asarray(F[j], dtype=float).copy(),
                G=g_i,
                tag=candidates[idx].tag,
            )
        return result

    def _coerce_replace_index(self, idx: int | None, size: int) -> int | None:
        if idx is None:
            return None
        i = int(idx)
        if i < 0 or i >= size:
            return None
        return i

    def _choose_replace_idx(
        self,
        *,
        state: Any,
        kernel: Any,
        protected: np.ndarray[Any, Any],
    ) -> int | None:
        X = getattr(state, "X", None)
        F = getattr(state, "F", None)
        if not isinstance(X, np.ndarray) or not isinstance(F, np.ndarray):
            return None
        n = X.shape[0]
        if n == 0:
            return None
        protected_set = {int(i) for i in protected.tolist()}
        candidates = np.array(
            [i for i in range(n) if i not in protected_set],
            dtype=int,
        )
        if candidates.size == 0:
            candidates = np.arange(n, dtype=int)

        policy = self.replacement_policy
        if callable(policy):
            idx = policy(state, candidates)
            return self._coerce_replace_index(cast(int | None, idx), n)

        policy_name = str(policy).lower()
        ranks, crowding = kernel.nsga2_ranking(F)
        if policy_name == "random":
            rng = cast(np.random.Generator, state.rng)
            return int(rng.choice(candidates))
        if policy_name == "worst_crowding":
            return int(candidates[np.argmin(crowding[candidates])])

        worst_rank = int(np.max(ranks[candidates]))
        worst = candidates[ranks[candidates] == worst_rank]
        if worst.size == 0:
            worst = candidates
        return int(worst[np.argmin(crowding[worst])])

    def _inject_records(
        self,
        *,
        generation: int,
        state: Any,
        kernel: Any,
        records: Sequence[ImmigrantCandidate],
        track_active: bool = True,
    ) -> int:
        if not records:
            return 0
        X = getattr(state, "X", None)
        F = getattr(state, "F", None)
        G = getattr(state, "G", None)
        if not isinstance(X, np.ndarray) or not isinstance(F, np.ndarray):
            return 0
        inserted = 0
        st = self._stats_for(generation)
        protected = self._active_indices.copy()

        for cand in records:
            if track_active and self.max_in_pop > 0 and len(self._active) >= self.max_in_pop:
                break
            if cand.F is None:
                continue
            idx = self._choose_replace_idx(state=state, kernel=kernel, protected=protected)
            if idx is None:
                continue
            replaced_page = int(np.rint(F[idx, 0])) if F.ndim == 2 and F.shape[1] > 0 else 0
            X[idx] = np.asarray(cand.X, dtype=float)
            F[idx] = np.asarray(cand.F, dtype=float)
            if isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == X.shape[0]:
                if cand.G is None:
                    G[idx] = 0.0
                else:
                    G[idx] = np.asarray(cand.G, dtype=float)
            if track_active:
                self._active.append(
                    _ActiveImmigrant(
                        X=np.asarray(cand.X, dtype=float).copy(),
                        F=np.asarray(cand.F, dtype=float).copy() if cand.F is not None else None,
                        G=np.asarray(cand.G, dtype=float).copy() if cand.G is not None else None,
                        tag=cand.tag,
                    )
                )
            inserted += 1
            st.replaced_indices.append(int(idx))
            st.replaced_pages.append(int(replaced_page))
        if inserted > 0:
            st.events += 1
            st.inserted += int(inserted)
        return inserted

    def record_parent_indices(self, generation: int, parent_idx: np.ndarray[Any, Any]) -> None:
        if parent_idx.size == 0 or self._active_indices.size == 0:
            return
        part = int(np.count_nonzero(np.isin(parent_idx, self._active_indices)))
        if part <= 0:
            return
        self._stats_for(generation).mating_participation += int(part)

    def apply_generation(
        self,
        *,
        generation: int,
        state: Any,
        problem: Any,
        kernel: Any,
        eval_strategy: EvaluationBackend | None = None,
    ) -> bool:
        if not self.enabled or self.max_in_pop <= 0:
            self._active_indices = np.zeros(0, dtype=int)
            state.non_breeding_indices = np.zeros(0, dtype=int)
            return False

        X = getattr(state, "X", None)
        if not isinstance(X, np.ndarray) or X.size == 0:
            self._active_indices = np.zeros(0, dtype=int)
            state.non_breeding_indices = np.zeros(0, dtype=int)
            return False

        changed = False
        self._active_indices = _match_active_indices(X, self._active)

        if self.breeding_mode == "freeze_reinsert" and int(generation) >= self.start_generation:
            missing = max(0, self.max_in_pop - int(self._active_indices.size))
            if missing > 0 and self._active:
                pool: list[ImmigrantCandidate] = []
                for rec in self._active:
                    pool.append(
                        ImmigrantCandidate(
                            X=rec.X.copy(),
                            F=rec.F.copy() if rec.F is not None else None,
                            G=rec.G.copy() if rec.G is not None else None,
                            tag=rec.tag,
                        )
                    )
                inserted = self._inject_records(
                    generation=generation,
                    state=state,
                    kernel=kernel,
                    records=pool[:missing],
                    track_active=False,
                )
                changed = changed or inserted > 0

        do_inject = self._schedule_active(generation)
        if callable(self.skip_if):
            try:
                do_inject = do_inject and (not bool(self.skip_if(generation, state)))
            except Exception:
                _logger().debug("skip_if callback raised; proceeding with injection", exc_info=True)

        if do_inject:
            candidates = self._fetch_candidates(generation, state)
            slots = max(0, self.max_in_pop - len(self._active))
            # Only the candidates that can enter are evaluated.
            n_take = min(slots, self.max_insert_per_event, len(candidates))
            if n_take > 0:
                candidates = self._evaluate_missing_FG(
                    problem=problem,
                    candidates=candidates[:n_take],
                    state=state,
                    eval_strategy=get_eval_strategy(eval_strategy),
                )
                inserted = self._inject_records(
                    generation=generation,
                    state=state,
                    kernel=kernel,
                    records=candidates,
                )
                changed = changed or inserted > 0

        if len(self._active) > self.max_in_pop:
            self._active = self._active[: self.max_in_pop]

        X_new = getattr(state, "X", None)
        self._active_indices = _match_active_indices(X_new, self._active) if isinstance(X_new, np.ndarray) else np.zeros(0, dtype=int)
        if self.breeding_mode in {"exclude", "freeze_reinsert"}:
            state.non_breeding_indices = self._active_indices.copy()
        else:
            state.non_breeding_indices = np.zeros(0, dtype=int)
        return bool(changed)
