"""Derived scalar analysis over canonical StudySummary run references."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.experiment.study.report_models import StudySummaryRow
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.kernel.numpy_backend import _fast_non_dominated_sort
from vamos.foundation.quality_indicators.hypervolume import hypervolume
from vamos.foundation.quality_indicators.moocore_indicators import HVIndicator, QualityIndicator, get_indicator, has_moocore
from vamos.run_artifacts import load_result
from vamos.study_artifacts import StudySummary

# Indicators measured against the problem's reference front. They are distances
# where lower is better, so a run without feasible solutions gets +inf.
_REFERENCE_FRONT_INDICATORS = {"igd", "igd+", "igd_plus", "epsilon_additive", "epsilon_mult", "avg_hausdorff"}


@dataclass(frozen=True, slots=True)
class SummarySource:
    """A canonical summary and its root plus presentation-only annotations."""

    root: Path
    summary: StudySummary
    annotations: Mapping[str, object]


@dataclass(slots=True)
class _DerivedRow:
    record: dict[str, Any]
    problem_id: str | None
    # Feasible objective vectors of the selected run; None when the task has no run.
    front: NDArray[np.float64] | None


def derive_summary_rows(
    sources: Sequence[SummarySource],
    *,
    indicators: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Add caller metrics to summary rows without reinterpreting study state.

    Indicator references are built per problem: rows that share a problem
    identity and objective count, across all sources, are measured against one
    HV reference point and one non-dominated reference front derived only from
    their own fronts.

    Only feasible solutions (every stored constraint ``G <= 0``) enter the
    indicators and the references. A run without feasible solutions gets HV 0
    and +inf for the reference-front indicators. ``n_feasible`` counts the
    feasible solutions of each run; ``feasibility_known`` is False when the run
    stores no constraint values, and then every solution counts as feasible.
    """
    rows = [_load_row(source, row) for source in sources for row in source.summary.rows]
    for group in _problem_groups(rows):
        _attach_problem_metrics(group, indicators)
    return tuple(row.record for row in rows)


def _problem_groups(rows: Sequence[_DerivedRow]) -> tuple[list[_DerivedRow], ...]:
    groups: dict[tuple[str | None, int], list[_DerivedRow]] = {}
    for row in rows:
        if row.front is not None:
            groups.setdefault((row.problem_id, row.front.shape[-1]), []).append(row)
    return tuple(groups.values())


def _attach_problem_metrics(rows: list[_DerivedRow], indicators: Sequence[str]) -> None:
    fronts = [row.front for row in rows if row.front is not None and row.front.size]
    reference = np.asarray(compute_hv_reference(fronts), dtype=float) if fronts else None
    reference_text = None if reference is None else " ".join(f"{value:.6f}" for value in reference)
    for row in rows:
        if row.front is not None:
            # A run without feasible solutions covers no volume.
            row.record["hv"] = float(hypervolume(row.front, reference)) if reference is not None and row.front.size else 0.0
            row.record["hv_source"] = "problem"
            row.record["hv_reference"] = reference_text
    _attach_indicators(rows, fronts, reference, indicators)


def _load_row(source: SummarySource, row: StudySummaryRow) -> _DerivedRow:
    record = row.as_dict()
    metrics = record.pop("metrics")
    record.update(source.annotations)
    record["problem"] = _component_name(row.problem_id)
    record["algorithm"] = _component_name(row.algorithm_id)
    record["engine"] = _component_name(row.backend_id)
    record["time_ms"] = row.runtime_ms
    record["evals"] = row.evaluations
    if isinstance(metrics, Mapping):
        record.update(metrics)
    if isinstance(row.evaluations, int) and isinstance(row.runtime_ms, (int, float)) and row.runtime_ms > 0:
        record["evals_per_sec"] = float(row.evaluations * 1000.0 / row.runtime_ms)
    loaded = _load_front(source.root, row)
    if loaded is None:
        return _DerivedRow(record, row.problem_id, None)
    front, feasibility_known = loaded
    record["n_feasible"] = int(front.shape[0])
    record["feasibility_known"] = feasibility_known
    return _DerivedRow(record, row.problem_id, front)


def _load_front(root: Path, row: StudySummaryRow) -> tuple[NDArray[np.float64], bool] | None:
    """Return the run's feasible objective vectors and whether its G was stored."""
    if row.selected_run_id is None or row.run_manifest_path is None:
        return None
    result = load_result(root / Path(row.run_manifest_path).parent)
    front = np.asarray(result.F, dtype=float)
    constraints = result.data.get("G")
    if constraints is None:
        return front, False
    violations = np.asarray(constraints, dtype=float).reshape(front.shape[0], -1)
    return front[np.all(violations <= 0.0, axis=1)], True


def _attach_indicators(
    rows: list[_DerivedRow],
    fronts: list[NDArray[np.float64]],
    reference: NDArray[np.float64] | None,
    indicators: Sequence[str],
) -> None:
    requested = tuple(name for name in indicators if name.lower() not in {"hv", "hypervolume"})
    if not requested or not has_moocore():
        return
    reference_front = _nondominated_union(fronts) if fronts else None
    for row in rows:
        if row.front is None:
            continue
        for name in requested:
            row.record[f"indicator_{name}"] = _row_indicator(name, row.front, reference_front, reference)


def _row_indicator(
    name: str,
    front: NDArray[np.float64],
    reference_front: NDArray[np.float64] | None,
    reference: NDArray[np.float64] | None,
) -> float | NDArray[np.float64] | None:
    if reference_front is None or reference is None or not front.size:
        # Without feasible solutions the distance to the reference front is +inf, the worst value.
        return math.inf if name in _REFERENCE_FRONT_INDICATORS else None
    try:
        return _indicator_value(name, front, reference_front, reference)
    except Exception as exc:
        logging.getLogger(__name__).warning("Derived study indicator '%s' failed: %s", name, exc)
        return None


def _indicator_value(
    name: str,
    front: NDArray[np.float64],
    reference_front: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> float | NDArray[np.float64] | None:
    if name in {"hv", "hypervolume"}:
        return HVIndicator(reference_point=reference).compute(front).value
    if name in _REFERENCE_FRONT_INDICATORS:
        indicator: QualityIndicator = get_indicator(name, reference_front=reference_front)
    else:
        indicator = get_indicator(name)
    return indicator.compute(front).value


def _nondominated_union(fronts: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    combined = np.vstack(fronts)
    front_indices, _ = _fast_non_dominated_sort(combined)
    if not front_indices:
        return np.asarray(combined, dtype=float)
    first = front_indices[0]
    return np.asarray(combined[first] if first else combined, dtype=float)


def _component_name(component_id: str | None) -> str | None:
    if component_id is None or ":" not in component_id:
        return component_id
    return component_id.split(":", 1)[1].split("@", 1)[0]


__all__ = ["SummarySource", "derive_summary_rows"]
