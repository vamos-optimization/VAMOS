"""
Notebook-friendly helpers for inspecting tuning results and fronts.
Uses only NumPy/Pandas/Matplotlib; Pandas and Matplotlib are imported when a helper needs them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from vamos.study_artifacts import StudySummary
from vamos.ux.analysis.core_objective_reduction import reduce_objectives

if TYPE_CHECKING:
    import pandas as pd


class TuningResultLike(Protocol):
    def __getitem__(self, index: int) -> object: ...


def tuning_result_to_dataframe(tuning_result: TuningResultLike, param_names: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Convert a TuningResult-like object to a DataFrame.

    Expected keys/attributes: unit_vectors (X), objectives (F), assignments/configs (optional).
    """
    import pandas as pd

    X_attr = getattr(tuning_result, "unit_vectors", None)
    if X_attr is None:
        X_attr = getattr(tuning_result, "X_nd", None)
    X = X_attr if X_attr is not None else tuning_result[0]

    F_attr = getattr(tuning_result, "objectives", None)
    if F_attr is None:
        F_attr = getattr(tuning_result, "F_nd", None)
    F = F_attr if F_attr is not None else tuning_result[1]

    assignments_raw = getattr(tuning_result, "assignments", None)
    if assignments_raw is None:
        assignments_raw = getattr(tuning_result, "configs", None)
    assignments: Sequence[object]
    if isinstance(assignments_raw, Sequence) and not isinstance(assignments_raw, (str, bytes)):
        assignments = assignments_raw
    else:
        assignments = []
    X = np.asarray(X)
    F = np.asarray(F)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(X.shape[1])]
    rows = []
    for i in range(X.shape[0]):
        row = {param_names[j]: X[i, j] for j in range(X.shape[1])}
        for k, val in enumerate(F[i]):
            row[f"obj_{k}"] = val
        if assignments:
            row["assignment"] = assignments[i]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_tuning_scatter(df: pd.DataFrame, x_param: str, y_param: str, color_by: str = "obj_0") -> object:
    """Scatter plot of hyperparameters colored by an objective/metric."""
    import matplotlib.pyplot as plt

    plt.figure()
    sc = plt.scatter(df[x_param], df[y_param], c=df[color_by], cmap="viridis", edgecolor="k", alpha=0.8)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    cbar = plt.colorbar(sc)
    cbar.set_label(color_by)
    plt.tight_layout()
    return plt.gca()


def plot_objective_tradeoff(df: pd.DataFrame, obj_x: str = "obj_0", obj_y: str = "obj_1") -> object:
    """Plot objective trade-off scatter."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(df[obj_x], df[obj_y], color="tab:blue", edgecolor="k", alpha=0.8)
    plt.xlabel(obj_x)
    plt.ylabel(obj_y)
    plt.tight_layout()
    return plt.gca()


def plot_reduced_front(
    F: NDArray[Any],
    labels: Sequence[str] | None = None,
    target_dim: int = 2,
    method: str = "angle",
) -> object:
    """
    Reduce a front to 2D/3D and plot it.

    The axes are labelled with the kept objectives: their entries of ``labels`` or, without
    ``labels``, ``f1``, ``f2``, ... numbered from 1 by their position in ``F``.
    """
    import matplotlib.pyplot as plt

    F_red, selected = reduce_objectives(F, method=method, target_dim=target_dim)
    indices = [int(i) for i in selected]
    labels = [labels[i] for i in indices] if labels else [f"f{i + 1}" for i in indices]
    if F_red.shape[1] == 2:
        plt.figure()
        plt.scatter(F_red[:, 0], F_red[:, 1], color="tab:orange", edgecolor="k", alpha=0.8)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.tight_layout()
        return plt.gca()
    if F_red.shape[1] >= 3:
        fig = plt.figure()
        ax3d = cast(Any, fig.add_subplot(111, projection="3d"))
        ax3d.scatter(F_red[:, 0], F_red[:, 1], F_red[:, 2], color="tab:orange", alpha=0.8)
        ax3d.set_xlabel(labels[0])
        ax3d.set_ylabel(labels[1])
        ax3d.set_zlabel(labels[2])
        fig.tight_layout()
        return ax3d
    raise ValueError("Reduced front must have at least 2 objectives.")


def study_summary_to_dataframe(summary: StudySummary) -> pd.DataFrame:
    """Convert one canonical summary to a traceable analysis DataFrame."""
    import pandas as pd

    rows: list[dict[str, object]] = []
    for source in summary.rows:
        record = source.as_dict()
        metrics = record.pop("metrics")
        record["problem"] = _component_name(source.problem_id)
        record["algorithm"] = _component_name(source.algorithm_id)
        record["engine"] = _component_name(source.backend_id)
        record["time_ms"] = source.runtime_ms
        record["evals"] = source.evaluations
        if isinstance(metrics, Mapping):
            record.update(metrics)
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _component_name(component_id: str | None) -> str | None:
    if component_id is None or ":" not in component_id:
        return component_id
    return component_id.split(":", 1)[1].split("@", 1)[0]


def summarize_by_algorithm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean/std summary grouped by problem and algorithm.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    grouped = df.groupby(["problem", "algorithm"])[numeric_cols].agg(["mean", "std"])
    grouped.columns = ["_".join(col).rstrip("_") for col in grouped.columns]
    return grouped.reset_index()
