from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .config import ExternalArchiveConfig

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.archive import (
        CrowdingDistanceArchive,
        HypervolumeArchive,
        MaxMinArchive,
        ReferenceDirectionsArchive,
        SPEA2Archive,
        UnboundedArchive,
    )
    from vamos.engine.algorithm.components.state import AlgorithmState
    from vamos.foundation.kernel.backend import KernelBackend

    ArchiveManager: TypeAlias = (
        CrowdingDistanceArchive | HypervolumeArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | UnboundedArchive
    )
    ResultArchiveManager: TypeAlias = (
        CrowdingDistanceArchive | HypervolumeArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive
    )
else:
    ArchiveManager: TypeAlias = Any
    ResultArchiveManager: TypeAlias = Any


def resolve_external_archive(cfg: dict[str, Any]) -> ExternalArchiveConfig | None:
    """Extract :class:`ExternalArchiveConfig` from a serialized config dict."""
    raw = cfg.get("external_archive")
    if raw is None:
        return None
    if isinstance(raw, ExternalArchiveConfig):
        return raw
    return ExternalArchiveConfig(**raw)


def setup_archive(
    kernel: KernelBackend,
    X: np.ndarray[Any, Any],
    F: np.ndarray[Any, Any],
    n_var: int,
    n_obj: int,
    dtype: np.dtype[Any],
    ext_cfg: ExternalArchiveConfig | None,
    G: np.ndarray[Any, Any] | None = None,
) -> tuple[np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None, ArchiveManager | None]:
    """Initialize the shared external archive manager when configured."""
    if ext_cfg is None:
        return None, None, None

    from vamos.engine.algorithm.components.archive import (
        CrowdingDistanceArchive,
        HypervolumeArchive,
        MaxMinArchive,
        ReferenceDirectionsArchive,
        SPEA2Archive,
        UnboundedArchive,
    )

    capacity = ext_cfg.capacity
    pruning = ext_cfg.pruning
    tol = ext_cfg.objective_tolerance
    dedup_mode = ext_cfg.deduplicate_in
    decision_tol = ext_cfg.decision_tolerance
    truncate_size = ext_cfg.truncate_size
    n_con = G.shape[1] if G is not None else None

    manager: ArchiveManager
    if capacity is None:
        manager = UnboundedArchive(
            n_var=n_var,
            n_obj=n_obj,
            dtype=dtype,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
            n_con=n_con,
        )
    elif pruning in {"hv", "mc_hv"}:
        manager = HypervolumeArchive(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
            n_con=n_con,
            ref_point=ext_cfg.hv_ref_point,
        )
    elif pruning == "knn":
        manager = SPEA2Archive(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
            n_con=n_con,
            constraint_mode="feasibility",
        )
    elif pruning == "maxmin":
        manager = MaxMinArchive(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
            n_con=n_con,
        )
    elif pruning == "ref_dirs":
        manager = ReferenceDirectionsArchive(
            capacity,
            n_var,
            n_obj,
            dtype,
            rng_seed=ext_cfg.rng_seed,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
            n_con=n_con,
        )
    else:
        manager = CrowdingDistanceArchive(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
            n_con=n_con,
            selection_kernels=kernel.selection_kernels(),
        )

    archive_X, archive_F = manager.update(X, F, G)
    return archive_X, archive_F, manager


def setup_result_archive(
    ext_cfg: ExternalArchiveConfig | None,
    n_var: int,
    n_obj: int,
    dtype: np.dtype[Any],
) -> ResultArchiveManager | None:
    """Create a bounded result archive when the external archive is bounded."""
    if ext_cfg is None or ext_cfg.capacity is None:
        return None

    from vamos.engine.algorithm.components.archive import (
        CrowdingDistanceArchive,
        HypervolumeArchive,
        MaxMinArchive,
        ReferenceDirectionsArchive,
        SPEA2Archive,
    )

    truncate_size = ext_cfg.truncate_size
    tol = ext_cfg.objective_tolerance
    dedup_mode = ext_cfg.deduplicate_in
    decision_tol = ext_cfg.decision_tolerance

    if ext_cfg.pruning in {"hv", "mc_hv"}:
        return HypervolumeArchive(
            ext_cfg.capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
            ref_point=ext_cfg.hv_ref_point,
        )
    if ext_cfg.pruning == "knn":
        return SPEA2Archive(
            ext_cfg.capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
            constraint_mode="feasibility",
        )
    if ext_cfg.pruning == "maxmin":
        return MaxMinArchive(
            ext_cfg.capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
        )
    if ext_cfg.pruning == "ref_dirs":
        return ReferenceDirectionsArchive(
            ext_cfg.capacity,
            n_var,
            n_obj,
            dtype,
            rng_seed=ext_cfg.rng_seed,
            truncate_size=truncate_size,
            objective_tolerance=tol,
            deduplicate_in=dedup_mode,
            decision_tolerance=decision_tol,
        )
    return CrowdingDistanceArchive(
        ext_cfg.capacity,
        n_var,
        n_obj,
        dtype,
        truncate_size=truncate_size,
        objective_tolerance=tol,
        deduplicate_in=dedup_mode,
        decision_tolerance=decision_tol,
    )


def update_archive(
    state: AlgorithmState,
    X_new: np.ndarray[Any, Any] | None = None,
    F_new: np.ndarray[Any, Any] | None = None,
    G_new: np.ndarray[Any, Any] | None = None,
) -> None:
    """Update archive contents from the provided arrays or current algorithm state."""
    if state.archive_manager is None:
        return

    X = X_new if X_new is not None else state.X
    F = F_new if F_new is not None else state.F
    G = G_new if G_new is not None else state.G

    state.archive_X, state.archive_F = state.archive_manager.update(X, F, G)


__all__ = [
    "ArchiveManager",
    "ResultArchiveManager",
    "resolve_external_archive",
    "setup_archive",
    "setup_result_archive",
    "update_archive",
]
