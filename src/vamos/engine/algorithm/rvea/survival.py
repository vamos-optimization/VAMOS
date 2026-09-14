"""Survival helpers for RVEA, including feasibility-first constraint handling."""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.constraints.utils import compute_violation, is_feasible


def apd_survival(
    F: np.ndarray[Any, Any],
    V: np.ndarray[Any, Any],
    gamma: np.ndarray[Any, Any],
    ideal: np.ndarray[Any, Any],
    n_survive: int,
    n_gen: int,
    n_max_gen: int,
    alpha: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Select survivors using RVEA's angle-penalized distance criterion."""
    if F.size == 0:
        return np.empty(0, dtype=int), ideal, None

    n_obj = F.shape[1]
    ideal = np.minimum(F.min(axis=0), ideal)

    F_shift = F - ideal
    dist_to_ideal = np.linalg.norm(F_shift, axis=1)
    dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64
    F_prime = F_shift / dist_to_ideal[:, None]

    cos_theta = np.clip(F_prime @ V.T, -1.0, 1.0)
    acute_angle = np.arccos(cos_theta)
    niches = acute_angle.argmin(axis=1)

    M = float(n_obj) if n_obj > 2 else 1.0
    progress = (n_gen / n_max_gen) ** alpha
    theta = acute_angle[np.arange(F.shape[0]), niches]
    penalty = M * progress * (theta / gamma[niches])
    apd = dist_to_ideal * (1.0 + penalty)

    order = np.lexsort((np.arange(F.shape[0]), apd, niches))
    sorted_niches = niches[order]
    first_in_niche = np.empty(sorted_niches.shape[0], dtype=bool)
    first_in_niche[0] = True
    first_in_niche[1:] = sorted_niches[1:] != sorted_niches[:-1]
    survivors = order[first_in_niche]
    target = min(int(n_survive), F.shape[0])
    if survivors.size < target:
        selected = np.zeros(F.shape[0], dtype=bool)
        selected[survivors] = True
        fill = order[~selected[order]][: target - survivors.size]
        survivors = np.concatenate([survivors, fill])
    elif survivors.size > target:
        survivors = survivors[:target]

    nadir = F[survivors].max(axis=0) if survivors.size else None
    return survivors, ideal, nadir


def constraint_aware_apd_survival(
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    V: np.ndarray[Any, Any],
    gamma: np.ndarray[Any, Any],
    ideal: np.ndarray[Any, Any],
    n_survive: int,
    n_gen: int,
    n_max_gen: int,
    alpha: float,
    constraint_mode: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Apply feasibility-first ordering and APD diversity among feasible points."""
    if G is None or constraint_mode == "none":
        return apd_survival(F, V, gamma, ideal, n_survive, n_gen, n_max_gen, alpha)

    feasible = is_feasible(G, n=G.shape[0])
    feasible_idx = np.flatnonzero(feasible)
    target = min(int(n_survive), F.shape[0])
    if feasible_idx.size >= target:
        local, updated_ideal, nadir = apd_survival(
            F[feasible_idx],
            V,
            gamma,
            ideal,
            target,
            n_gen,
            n_max_gen,
            alpha,
        )
        return np.asarray(feasible_idx[local], dtype=int), updated_ideal, nadir

    violation = compute_violation(G, n=G.shape[0])
    infeasible_idx = np.flatnonzero(~feasible)
    needed = target - feasible_idx.size
    order = np.argsort(violation[infeasible_idx], kind="stable")
    fill = infeasible_idx[order[:needed]]
    survivors = np.concatenate([feasible_idx, fill]).astype(int, copy=False)

    # Infeasible objective values must never seed RVEA's geometry. Until the
    # first feasible point appears the ideal stays untouched and adaptation is
    # disabled because no feasible nadir exists.
    if feasible_idx.size == 0:
        return survivors, ideal, None
    updated_ideal = np.minimum(F[feasible_idx].min(axis=0), ideal)
    nadir = F[feasible_idx].max(axis=0)
    return survivors, updated_ideal, nadir


__all__ = ["apd_survival", "constraint_aware_apd_survival"]
