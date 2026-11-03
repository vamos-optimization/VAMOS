# algorithm/nsgaii/setup.py
"""
Setup and initialization helpers for NSGA-II.

This module contains functions for parsing configuration, initializing populations,
archives, and genealogy tracking. These are extracted from the main NSGAII class
to keep the core algorithm focused on the evolutionary loop.

Operator pool building lives in operators/policies/nsgaii.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    MaxMinArchive,
    ReferenceDirectionsArchive,
    SPEA2Archive,
    UnboundedArchive,
)
from vamos.engine.algorithm.components.lifecycle import setup_initial_population
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.hooks.genealogy import DefaultGenealogyTracker, GenealogyTracker
from vamos.foundation.eval.backends import EvaluationBackend
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

# Constants
DEFAULT_TOURNAMENT_PRESSURE = 2


def parse_termination(termination: tuple[str, Any]) -> tuple[int, dict[str, Any] | None]:
    """Parse termination criterion and return (max_eval, hv_config).

    Parameters
    ----------
    termination : tuple[str, Any]
        Termination criterion as (type, value). Supported types:
        - "max_evaluations": value is the max number of evaluations
        - "hv": value is a dict with hypervolume config

    Returns
    -------
    tuple[int, dict[str, Any] | None]
        (max_evaluations, hv_config or None)

    Raises
    ------
    ValueError
        If termination type is unsupported or HV config is invalid.
    """
    term_type, term_val = termination
    hv_config = None
    if term_type == "max_evaluations":
        max_eval = int(term_val)
    elif term_type == "hv":
        hv_config = dict(term_val)
        max_eval = int(hv_config.get("max_evaluations", 0))
        if max_eval <= 0:
            raise ValueError("HV-based termination requires a positive max_evaluations value.")
    else:
        raise ValueError("Unsupported termination criterion for NSGA-II.")
    return max_eval, hv_config


def setup_population(
    problem: ProblemProtocol,
    eval_strategy: EvaluationBackend,
    rng: np.random.Generator,
    pop_size: int,
    constraint_mode: str,
    initializer_cfg: dict[str, Any] | None = None,
    max_evaluations: int | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None, int]:
    """Initialize and evaluate the starting population.

    Parameters
    ----------
    problem : ProblemProtocol
        The optimization problem.
    eval_strategy : EvaluationBackend
        Backend for evaluating solutions.
    rng : np.random.Generator
        Random number generator.
    pop_size : int
        Population size.
    constraint_mode : str
        Constraint handling mode ('none', 'feasibility', etc.).
    initializer_cfg : dict[str, Any] | None
        Optional initializer configuration.
    max_evaluations : int | None
        Evaluation budget of the run; the opposition-based initializer needs ``2 * pop_size``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None, int]
        (X, F, G, n_eval) - decision variables, objectives, constraints, and every
        evaluation the initialization took.
    """
    return setup_initial_population(
        problem, eval_strategy, rng, pop_size, constraint_mode, initializer_cfg, max_evaluations=max_evaluations, algorithm="NSGA-II"
    )


def setup_archive(
    kernel: KernelBackend,
    X: np.ndarray[Any, Any],
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    n_var: int,
    n_obj: int,
    dtype: np.dtype[Any],
    ext_cfg: ExternalArchiveConfig | None,
) -> tuple[
    np.ndarray[Any, Any] | None,
    np.ndarray[Any, Any] | None,
    CrowdingDistanceArchive | HypervolumeArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | UnboundedArchive | None,
]:
    """Initialize archive if configured.

    Parameters
    ----------
    kernel : KernelBackend
        The kernel backend.
    X : np.ndarray
        Initial population decision variables.
    F : np.ndarray
        Initial population objectives.
    G : np.ndarray | None
        Initial population constraints.
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives.
    dtype : np.dtype
        Data type for arrays.
    ext_cfg : ExternalArchiveConfig | None
        External archive configuration, or ``None`` to disable.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None, archive_manager | None]
        (archive_X, archive_F, archive_manager)
    """
    if ext_cfg is None:
        return None, None, None

    capacity = ext_cfg.capacity
    pruning = ext_cfg.pruning

    tol = ext_cfg.objective_tolerance
    dedup_mode = ext_cfg.deduplicate_in
    decision_tol = ext_cfg.decision_tolerance
    truncate_size = ext_cfg.truncate_size
    n_con = G.shape[1] if G is not None else None

    manager: CrowdingDistanceArchive | HypervolumeArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | UnboundedArchive
    if capacity is None:
        # Unbounded archive
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
        )

    archive_X, archive_F = manager.update(X, F, G)
    return archive_X, archive_F, manager


def setup_genealogy(
    pop_size: int,
    F: np.ndarray[Any, Any],
    track_genealogy: bool,
) -> tuple[GenealogyTracker | None, np.ndarray[Any, Any] | None]:
    """Initialize genealogy tracking if enabled.

    Parameters
    ----------
    pop_size : int
        Population size.
    F : np.ndarray
        Initial fitness values.
    track_genealogy : bool
        Whether to enable genealogy tracking.

    Returns
    -------
    tuple[GenealogyTracker | None, np.ndarray | None]
        (genealogy_tracker, individual_ids)
    """
    if not track_genealogy:
        return None, None

    genealogy_tracker = DefaultGenealogyTracker()
    ids = np.arange(pop_size, dtype=int)
    for i in range(pop_size):
        genealogy_tracker.new_individual(
            generation=0,
            parents=[],
            operator_name=None,
            algorithm_name="nsgaii",
            fitness=F[i] if F is not None and i < F.shape[0] else None,
        )
    return genealogy_tracker, ids


def setup_selection(
    sel_method: str,
    sel_params: dict[str, Any],
) -> tuple[str, int]:
    """Parse selection config and return (method, tournament_size).

    Parameters
    ----------
    sel_method : str
        Selection method name ('tournament' or 'random').
    sel_params : dict[str, Any]
        Selection parameters.

    Returns
    -------
    tuple[str, int]
        (method, tournament_size)

    Raises
    ------
    ValueError
        If selection method is not supported.
    """
    allowed = ("tournament", "random", "boltzmann", "ranking", "sus")
    if sel_method not in allowed:
        raise ValueError(f"Unsupported selection method '{sel_method}'. Must be one of {allowed}.")
    if sel_method == "tournament":
        unexpected = sorted(set(sel_params) - {"size"})
        if unexpected:
            raise ValueError(f"Unsupported tournament selection options: {', '.join(unexpected)}")
    tournament_size = (
        int(sel_params.get("size", DEFAULT_TOURNAMENT_PRESSURE)) if sel_method == "tournament" else DEFAULT_TOURNAMENT_PRESSURE
    )
    return sel_method, tournament_size


def setup_result_archive(
    ext_cfg: ExternalArchiveConfig | None,
    n_var: int,
    n_obj: int,
    dtype: np.dtype[Any],
) -> HypervolumeArchive | CrowdingDistanceArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | None:
    """Create result archive if configured.

    Parameters
    ----------
    ext_cfg : ExternalArchiveConfig | None
        External archive configuration. The ``pruning`` field determines
        the archive class (``"hv"`` -> HypervolumeArchive,
        ``"knn"`` -> SPEA2Archive, ``"maxmin"`` -> MaxMinArchive,
        ``"ref_dirs"`` -> ReferenceDirectionsArchive, otherwise CrowdingDistanceArchive).
        ``None`` or unbounded (``capacity is None``) disables the result archive.
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives.
    dtype : np.dtype
        Data type for arrays.

    Returns
    -------
    HypervolumeArchive | CrowdingDistanceArchive | MaxMinArchive | ReferenceDirectionsArchive | None
        The result archive, or None if not configured.
    """
    if ext_cfg is None or ext_cfg.capacity is None:
        return None

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


def resolve_external_archive(cfg: dict[str, Any]) -> ExternalArchiveConfig | None:
    """Extract :class:`ExternalArchiveConfig` from a serialised config dict.

    Returns ``None`` when no external archive is configured.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration.

    Returns
    -------
    ExternalArchiveConfig | None
        External archive configuration, or ``None``.
    """
    raw = cfg.get("external_archive")
    if raw is None:
        return None
    if isinstance(raw, ExternalArchiveConfig):
        return raw
    return ExternalArchiveConfig(**raw)


__all__ = [
    "parse_termination",
    "setup_population",
    "setup_archive",
    "setup_genealogy",
    "setup_selection",
    "setup_result_archive",
    "resolve_external_archive",
    "DEFAULT_TOURNAMENT_PRESSURE",
]
