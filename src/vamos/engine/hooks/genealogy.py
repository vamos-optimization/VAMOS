from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

IndividualID = int


@dataclass
class GenealogyRecord:
    individual_id: IndividualID
    generation: int
    parents: list[IndividualID]
    operator_name: str | None
    algorithm_name: str | None
    fitness: NDArray[np.float64] | None = None
    is_final_front: bool = False


class GenealogyTracker(Protocol):
    records: dict[IndividualID, GenealogyRecord]
    next_id: int

    def new_individual(
        self,
        generation: int,
        parents: list[IndividualID],
        operator_name: str | None,
        algorithm_name: str | None,
        fitness: NDArray[np.float64] | None = None,
    ) -> IndividualID: ...

    def mark_final_front(self, ids: list[IndividualID]) -> None: ...


@dataclass
class DefaultGenealogyTracker:
    records: dict[IndividualID, GenealogyRecord] = field(default_factory=dict)
    next_id: int = 0

    def new_individual(
        self,
        generation: int,
        parents: list[IndividualID],
        operator_name: str | None,
        algorithm_name: str | None,
        fitness: NDArray[np.float64] | None = None,
    ) -> IndividualID:
        idx = self.next_id
        self.next_id += 1
        self.records[idx] = GenealogyRecord(
            individual_id=idx,
            generation=generation,
            parents=list(parents),
            operator_name=operator_name,
            algorithm_name=algorithm_name,
            fitness=fitness.copy() if fitness is not None else None,
        )
        return idx

    def mark_final_front(self, ids: list[IndividualID]) -> None:
        for i in ids:
            if i in self.records:
                self.records[i].is_final_front = True


def get_lineage(tracker: GenealogyTracker, individual_id: IndividualID) -> list[GenealogyRecord]:
    lineage: list[GenealogyRecord] = []
    stack = [individual_id]
    visited = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        rec = tracker.records.get(current)
        if rec is None:
            continue
        lineage.append(rec)
        stack.extend(rec.parents)
    return lineage


def _resolve_final_ids(tracker: GenealogyTracker, final_ids: list[IndividualID] | None) -> list[IndividualID]:
    if final_ids is not None:
        return list(final_ids)
    return [rid for rid, rec in tracker.records.items() if rec.is_final_front]


def _final_ancestor_ids(tracker: GenealogyTracker, final_ids: list[IndividualID] | None) -> set[IndividualID]:
    ancestors: set[IndividualID] = set()
    for fid in _resolve_final_ids(tracker, final_ids):
        for rec in get_lineage(tracker, fid):
            ancestors.add(rec.individual_id)
    return ancestors


def operator_success_stats(
    tracker: GenealogyTracker,
    final_ids: list[IndividualID] | None = None,
) -> list[dict[str, object]]:
    final_ancestors = _final_ancestor_ids(tracker, final_ids)
    totals: dict[str, int] = {}
    finals: dict[str, int] = {}
    for rec in tracker.records.values():
        if rec.operator_name is None:
            continue
        op = rec.operator_name
        totals[op] = totals.get(op, 0) + 1
        if rec.individual_id in final_ancestors:
            finals[op] = finals.get(op, 0) + 1
    rows: list[dict[str, object]] = []
    for op, cnt in totals.items():
        good = finals.get(op, 0)
        rows.append(
            {
                "operator": op,
                "total_uses": cnt,
                "uses_in_final_lineages": good,
                "ratio": (good / cnt) if cnt else 0.0,
            }
        )
    return rows


def generation_contributions(
    tracker: GenealogyTracker,
    final_ids: list[IndividualID] | None = None,
) -> list[dict[str, object]]:
    final_ancestors = _final_ancestor_ids(tracker, final_ids)
    gen_totals: dict[int, int] = {}
    gen_final: dict[int, int] = {}
    for rec in tracker.records.values():
        gen_totals[rec.generation] = gen_totals.get(rec.generation, 0) + 1
        if rec.individual_id in final_ancestors:
            gen_final[rec.generation] = gen_final.get(rec.generation, 0) + 1
    rows: list[dict[str, object]] = []
    for gen in sorted(gen_totals):
        tot = gen_totals[gen]
        fin = gen_final.get(gen, 0)
        rows.append(
            {
                "generation": gen,
                "total": tot,
                "final_lineage": fin,
                "ratio": (fin / tot) if tot else 0.0,
            }
        )
    return rows


__all__ = [
    "IndividualID",
    "GenealogyRecord",
    "GenealogyTracker",
    "DefaultGenealogyTracker",
    "get_lineage",
    "operator_success_stats",
    "generation_contributions",
]
