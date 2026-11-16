from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._mixed_spec import (
    CUSTOM_CROSSOVER_KEYS,
    extract_index_array,
    has_customized_segment_settings,
    resolve_choice,
    resolve_perm_crossover,
    resolve_probability,
    validate_mixed_spec,
    validate_segment_bounds,
)
from .integer import integer_sbx_crossover


def _complementary(
    p1: NDArray[np.generic], p2: NDArray[np.generic], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The arithmetic crossover's children ``lam * p1 + (1 - lam) * p2`` and ``(1 - lam) * p1 + lam * p2``."""
    lam = rng.random()
    first, second = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    return lam * first + (1.0 - lam) * second, (1.0 - lam) * first + lam * second


def _mixed_crossover_default(
    X_parents: NDArray[np.generic],
    prob: float,
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> NDArray[np.generic]:
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    n_original = Np
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    pairs = X_parents.reshape(Np // 2, 2, D).copy()
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        offspring = pairs.reshape(Np, D)
        return offspring[:n_original] if n_original % 2 != 0 else offspring

    validate_mixed_spec(spec, D)
    perm_idx = extract_index_array(spec, "perm_idx")
    real_idx = extract_index_array(spec, "real_idx")
    int_idx = extract_index_array(spec, "int_idx")
    cat_idx = extract_index_array(spec, "cat_idx")
    perm_crossover = resolve_perm_crossover(spec) if perm_idx.size else None

    for row in np.flatnonzero(rng.random(pairs.shape[0]) <= prob):
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = p1.copy()
        child2 = p2.copy()
        if perm_idx.size:
            parents_perm = np.stack([p1[perm_idx], p2[perm_idx]], axis=0).astype(np.int32, copy=True)
            assert perm_crossover is not None
            perm_children = perm_crossover(parents_perm, 1.0, rng)
            child1[perm_idx] = perm_children[0]
            child2[perm_idx] = perm_children[1]
        if real_idx.size:
            child1[real_idx], child2[real_idx] = _complementary(p1[real_idx], p2[real_idx], rng)
        if int_idx.size or cat_idx.size:
            swap_positions = np.concatenate([int_idx, cat_idx])
            if swap_positions.size:
                swap_cols = swap_positions[rng.random(swap_positions.size) < 0.5]
                child1[swap_cols] = p2[swap_cols]
                child2[swap_cols] = p1[swap_cols]
        pairs[row, 0], pairs[row, 1] = child1, child2

    offspring = pairs.reshape(Np, D)
    return offspring[:n_original] if n_original % 2 != 0 else offspring


def mixed_crossover(X_parents: NDArray[np.generic], prob: float, spec: dict[str, Any], rng: np.random.Generator) -> NDArray[np.generic]:
    """
    Mixed crossover: permutation crossover for perm_idx, arithmetic crossover for real,
    uniform swap for int/cat.

    The arithmetic crossover draws ``lam`` uniformly from [0, 1) for each mated pair and
    gives the complementary children ``lam * p1 + (1 - lam) * p2`` and
    ``(1 - lam) * p1 + lam * p2``; ``real_crossover="mean"`` gives both children the
    parents' mean instead. ``int_crossover="arithmetic"`` rounds the same children.
    """
    if not has_customized_segment_settings(spec, CUSTOM_CROSSOVER_KEYS):
        return _mixed_crossover_default(X_parents, prob, spec, rng)

    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    n_original = Np
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    pairs = X_parents.reshape(Np // 2, 2, D).copy()
    validate_mixed_spec(spec, D)

    perm_idx = extract_index_array(spec, "perm_idx")
    real_idx = extract_index_array(spec, "real_idx")
    int_idx = extract_index_array(spec, "int_idx")
    cat_idx = extract_index_array(spec, "cat_idx")
    int_lower = np.asarray(spec.get("int_lower") if spec.get("int_lower") is not None else [], dtype=int)
    int_upper = np.asarray(spec.get("int_upper") if spec.get("int_upper") is not None else [], dtype=int)
    validate_segment_bounds(
        index_name="int_idx", idx=int_idx, lower_name="int_lower", lower=int_lower, upper_name="int_upper", upper=int_upper
    )

    perm_prob = resolve_probability(spec, "perm_crossover_prob", prob)
    real_prob = resolve_probability(spec, "real_crossover_prob", prob)
    int_prob = resolve_probability(spec, "int_crossover_prob", prob)
    cat_prob = resolve_probability(spec, "cat_crossover_prob", prob)
    real_method = resolve_choice(spec, "real_crossover", "arithmetic", allowed={"arithmetic", "mean"})
    int_method = resolve_choice(spec, "int_crossover", "uniform", allowed={"uniform", "arithmetic", "sbx", "integer_sbx"})
    cat_method = resolve_choice(spec, "cat_crossover", "uniform", allowed={"uniform"})
    int_eta = float(spec.get("int_crossover_eta", 20.0))
    perm_crossover = resolve_perm_crossover(spec) if perm_idx.size else None

    if perm_idx.size and perm_prob > 0.0:
        for row in np.flatnonzero(rng.random(pairs.shape[0]) <= perm_prob):
            parents_perm = np.stack([pairs[row, 0, perm_idx], pairs[row, 1, perm_idx]], axis=0).astype(np.int32, copy=True)
            assert perm_crossover is not None
            perm_children = perm_crossover(parents_perm, 1.0, rng)
            pairs[row, 0, perm_idx] = perm_children[0]
            pairs[row, 1, perm_idx] = perm_children[1]

    if real_idx.size and real_prob > 0.0:
        for row in np.flatnonzero(rng.random(pairs.shape[0]) <= real_prob):
            if real_method not in {"arithmetic", "mean"}:
                raise ValueError(f"Unsupported real_crossover '{real_method}'.")
            p1_real, p2_real = pairs[row, 0, real_idx], pairs[row, 1, real_idx]
            if real_method == "mean":
                mean_vals = 0.5 * (p1_real + p2_real)
                pairs[row, 0, real_idx] = mean_vals
                pairs[row, 1, real_idx] = mean_vals
            else:
                pairs[row, 0, real_idx], pairs[row, 1, real_idx] = _complementary(p1_real, p2_real, rng)

    if int_idx.size and int_prob > 0.0:
        for row in np.flatnonzero(rng.random(pairs.shape[0]) <= int_prob):
            p1 = pairs[row, 0, int_idx]
            p2 = pairs[row, 1, int_idx]
            if int_method == "uniform":
                swap_mask = rng.random(int_idx.size) < 0.5
                if np.any(swap_mask):
                    tmp = p1[swap_mask].copy()
                    p1[swap_mask] = p2[swap_mask]
                    p2[swap_mask] = tmp
            elif int_method == "arithmetic":
                first, second = _complementary(p1, p2, rng)
                p1[:] = np.rint(first)
                p2[:] = np.rint(second)
            elif int_method in {"sbx", "integer_sbx"}:
                children = integer_sbx_crossover(
                    np.stack([p1, p2], axis=0).astype(float, copy=True),
                    prob=1.0,
                    eta=int_eta,
                    lower=int_lower,
                    upper=int_upper,
                    rng=rng,
                )
                p1[:] = children[0]
                p2[:] = children[1]
            else:
                raise ValueError(f"Unsupported int_crossover '{int_method}'.")
            pairs[row, 0, int_idx] = p1
            pairs[row, 1, int_idx] = p2

    if cat_idx.size and cat_prob > 0.0:
        if cat_method != "uniform":
            raise ValueError(f"Unsupported cat_crossover '{cat_method}'.")
        for row in np.flatnonzero(rng.random(pairs.shape[0]) <= cat_prob):
            swap_mask = rng.random(cat_idx.size) < 0.5
            if not np.any(swap_mask):
                continue
            cols = cat_idx[swap_mask]
            tmp = pairs[row, 0, cols].copy()
            pairs[row, 0, cols] = pairs[row, 1, cols]
            pairs[row, 1, cols] = tmp

    offspring = pairs.reshape(Np, D)
    return offspring[:n_original] if n_original % 2 != 0 else offspring


__all__ = ["mixed_crossover"]
