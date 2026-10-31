from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast, overload

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.exceptions import NoSolutionsError, ResultSelectionError
from vamos.foundation.quality_indicators.pareto import pareto_filter

from .ranking import BestMethod, RankingMethod, RankingSource, normalize_best_method, selection_candidates, warn_if_knee_alias
from .ranking import top_k as rank_top_k
from .ranking import top_k_report as build_top_k_report

if TYPE_CHECKING:
    from vamos.experiment.artifacts.models import RunManifest


class BestResult(TypedDict):
    X: NDArray[np.generic] | None
    F: NDArray[np.float64]
    index: int
    front_index: int


class TopKResult(TypedDict):
    X: NDArray[np.generic] | None
    F: NDArray[np.float64]
    indices: NDArray[np.int_]
    scores: NDArray[np.float64]
    source: RankingSource
    method: RankingMethod


class OptimizationResult:
    """
    Container returned by optimize() with Pareto front data and selection helpers.

    Use `vamos.ux.api` for summaries, plotting, and export helpers.
    """

    F: NDArray[np.float64] | None
    X: NDArray[np.generic] | None
    data: dict[str, Any]
    meta: dict[str, Any]

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        meta: Mapping[str, Any] | None = None,
        manifest: RunManifest | None = None,
    ):
        self.F = payload.get("F")
        self.X = payload.get("X")
        self.data = dict(payload)
        self.meta = dict(meta or {})
        self._manifest = manifest

    def __len__(self) -> int:
        return len(self.F) if self.F is not None else 0

    def __repr__(self) -> str:
        n_sol = len(self)
        n_obj = self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0
        return f"OptimizationResult({n_sol} solutions, {n_obj} objectives)"

    @property
    def n_objectives(self) -> int:
        return self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0

    @property
    def manifest(self) -> RunManifest | None:
        """Immutable source manifest when this result was loaded from a run."""
        return self._manifest

    @overload
    def front(self, *, return_indices: Literal[False] = False, feasible_only: bool = True) -> NDArray[np.float64] | None: ...

    @overload
    def front(self, *, return_indices: Literal[True], feasible_only: bool = True) -> tuple[NDArray[np.float64], NDArray[np.int_]]: ...

    def front(
        self,
        *,
        return_indices: bool = False,
        feasible_only: bool = True,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.int_]] | None:
        """Non-dominated rows of the top-level ``F``, feasible solutions first.

        When the result carries constraint values ``G`` for its rows, only the
        feasible rows are Pareto-filtered or, when none is feasible, the least
        violated ones: the rule the algorithms use for their non-dominated
        results. ``feasible_only=False`` filters every row by ``F`` alone, as
        VAMOS 1.1.0 and earlier did. The indices refer to the top-level
        ``X``/``F`` arrays.
        """
        if self.F is None:
            return pareto_filter(None, return_indices=True) if return_indices else None
        F = np.asarray(self.F, dtype=float)
        candidates = selection_candidates(F, self.data.get("G"), feasible_only=feasible_only)
        front_F, local_idx = pareto_filter(F[candidates], return_indices=True)
        front_idx = np.asarray(candidates[np.asarray(local_idx, dtype=int)], dtype=int)
        if return_indices:
            return front_F, front_idx
        return front_F

    def best(self, method: BestMethod = "balanced_sum", *, feasible_only: bool = True) -> BestResult:
        """One solution of :meth:`front`, chosen by ``method``.

        ``balanced_sum`` (default) minimizes the sum of objectives normalized over
        the front; ``balanced`` minimizes their maximum; ``min_f1`` and ``min_f2``
        minimize one objective. ``knee`` is a deprecated alias of ``balanced_sum``,
        not a knee point; use :func:`vamos.ux.api.knee_point_scores` for a
        geometric knee. ``feasible_only`` is passed to :meth:`front`.
        """
        if self.F is None or len(self.F) == 0:
            raise NoSolutionsError("No solutions available.")

        resolved_method = normalize_best_method(method)
        warn_if_knee_alias(resolved_method)
        front_F, front_idx = self.front(return_indices=True, feasible_only=feasible_only)
        if len(front_F) == 0:
            raise NoSolutionsError("No solutions available.")

        if resolved_method in {"balanced_sum", "knee"}:
            F_norm = (front_F - front_F.min(axis=0)) / (np.ptp(front_F, axis=0) + 1e-12)
            front_pos = int(np.argmin(F_norm.sum(axis=1)))
        elif resolved_method == "min_f1":
            front_pos = int(np.argmin(front_F[:, 0]))
        elif resolved_method == "min_f2":
            if front_F.shape[1] < 2:
                raise ResultSelectionError(f"'min_f2' requires at least 2 objectives, but this result has {front_F.shape[1]}.")
            front_pos = int(np.argmin(front_F[:, 1]))
        elif resolved_method == "balanced":
            F_norm = (front_F - front_F.min(axis=0)) / (np.ptp(front_F, axis=0) + 1e-12)
            front_pos = int(np.argmin(F_norm.max(axis=1)))
        else:
            raise AssertionError(f"Unhandled best() method '{resolved_method}'.")

        idx = int(front_idx[front_pos])
        return {
            "X": self.X[idx] if self.X is not None else None,
            "F": self.F[idx],
            "index": idx,
            "front_index": front_pos,
        }

    def top_k(
        self,
        k: int = 100,
        *,
        source: RankingSource = "auto",
        method: RankingMethod = "balanced_sum",
        nondominated_only: bool = True,
        weights: NDArray[np.float64] | None = None,
        feasible_only: bool = True,
    ) -> TopKResult:
        """Up to ``k`` solutions of ``source``, ranked by ``method``.

        ``source="auto"`` (default) ranks the external archive when the result has
        one and the top-level result otherwise; ``"result"``, ``"archive"`` and
        ``"population"`` select one explicitly. When the source carries
        constraint values ``G`` for its rows, only its feasible rows are ranked
        or, when none is feasible, the least violated ones; ``feasible_only=False``
        ranks every row. ``method="knee"`` is a deprecated alias of
        ``"balanced_sum"``.
        """
        warn_if_knee_alias(method)
        return cast(
            TopKResult,
            rank_top_k(
                self,
                k=k,
                source=source,
                method=method,
                nondominated_only=nondominated_only,
                weights=weights,
                feasible_only=feasible_only,
            ),
        )

    def top_k_report(
        self,
        k: int = 100,
        *,
        source: RankingSource = "auto",
        method: RankingMethod = "balanced_sum",
        nondominated_only: bool = True,
        weights: NDArray[np.float64] | None = None,
        feasible_only: bool = True,
    ) -> list[dict[str, Any]]:
        """The :meth:`top_k` selection as one row per solution, with its objectives and variables."""
        warn_if_knee_alias(method)
        return build_top_k_report(
            result=self,
            k=k,
            source=source,
            method=method,
            nondominated_only=nondominated_only,
            weights=weights,
            feasible_only=feasible_only,
        )

    def explain_defaults(self) -> dict[str, object]:
        explained: dict[str, object] = {}
        resolved = self.meta.get("run_artifact_resolved_spec")
        sources = self.meta.get("default_sources")
        if resolved is not None:
            explained["resolved_spec"] = resolved
        if sources is not None:
            explained["default_sources"] = sources
        return explained


__all__ = ["BestResult", "OptimizationResult", "TopKResult"]
