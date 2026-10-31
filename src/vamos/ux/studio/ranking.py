"""Panel-free MCDM rankings of loaded fronts for the Studio Results page."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

from vamos.ux.studio.algorithm_catalog import algorithm_label
from vamos.ux.studio.data import FrontRecord
from vamos.ux.studio.dm import build_decision_view, rank_by_score

MCDM_METHODS: dict[str, str] = {
    "weighted_sum": "Weighted sum (equal weights)",
    "tchebycheff": "Tchebycheff (equal weights)",
    "knee": "Knee point",
    "topsis": "TOPSIS (equal weights)",
}


@dataclass(frozen=True)
class Ranking:
    """The top solutions of each front of one problem, and the fronts that could not be ranked."""

    rows: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def problems_in(fronts: Sequence[FrontRecord]) -> list[str]:
    return list(dict.fromkeys(front.problem_name for front in fronts))


def fronts_for(fronts: Sequence[FrontRecord], problem: str | None) -> list[FrontRecord]:
    return [front for front in fronts if problem is None or front.problem_name == problem]


def rank_fronts(fronts: Sequence[FrontRecord], *, method: str, top_k: int, problem: str | None = None) -> Ranking:
    """Rank the solutions of every front of ``problem`` by ``method``.

    Objectives are normalized per front before scoring, so ranks compare the
    solutions of one front, not solutions of different algorithms. Warnings
    raised while scoring a front, such as a front without a knee, are reported
    in ``Ranking.warnings``.
    """
    ranking = Ranking()
    for front in fronts_for(fronts, problem):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            view = build_decision_view(front, methods=[method])
        label = algorithm_label(front.algorithm_name)
        for item in caught:
            if issubclass(item.category, UserWarning):
                ranking.warnings.append(f"{label}: {item.message}")
            else:
                warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
        try:
            order = rank_by_score(view, method)
        except ValueError:
            ranking.warnings.append(
                f"{MCDM_METHODS.get(method, method)} cannot rank the {front.points_F.shape[0]} solutions of {algorithm_label(front.algorithm_name)}."
            )
            continue
        scores = view.mcdm_scores[method]
        for rank, index in enumerate(order[: max(int(top_k), 1)], start=1):
            row: dict[str, object] = {
                "Problem": front.problem_name,
                "Algorithm": algorithm_label(front.algorithm_name),
                "Rank": rank,
                "Score": float(scores[index]),
            }
            for objective, value in enumerate(front.points_F[index], start=1):
                row[f"f{objective}"] = float(value)
            ranking.rows.append(row)
    return ranking


def front_summary_rows(fronts: Sequence[FrontRecord]) -> list[dict[str, object]]:
    return [
        {
            "Problem": front.problem_name,
            "Algorithm": algorithm_label(front.algorithm_name),
            "Seeds": len(front.extra.get("seeds", []) or []),
            "Solutions": int(front.points_F.shape[0]),
            "Objectives": int(front.points_F.shape[1]),
        }
        for front in fronts
    ]


__all__ = ["MCDM_METHODS", "Ranking", "front_summary_rows", "fronts_for", "problems_in", "rank_fronts"]
