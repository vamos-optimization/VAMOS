from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._permutation_common import (
    RNG,
    PermPop,
    PermVec,
    ensure_distinct_indices,
    get_swap_rows_jit,
    two_cut_points,
    validate_permutation_population,
)


def swap_mutation(X: PermPop, prob: float, rng: RNG) -> None:
    validate_permutation_population(X, label="X")
    N, D = X.shape
    if N == 0 or D < 2:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    rows = np.flatnonzero(rng.random(N) <= prob)
    if rows.size == 0:
        return
    idx_pairs = rng.integers(0, D, size=(rows.size, 2))
    ensure_distinct_indices(idx_pairs, D, rng)
    first = idx_pairs[:, 0]
    second = idx_pairs[:, 1]
    jit_fn = get_swap_rows_jit()
    if jit_fn is not None:
        jit_fn(X, rows.astype(np.int64), first.astype(np.int64), second.astype(np.int64))
        return
    tmp = X[rows, first].copy()
    X[rows, first] = X[rows, second]
    X[rows, second] = tmp


def insert_mutation(X: PermPop, prob: float, rng: RNG) -> None:
    _apply_row_mutation(X, prob, rng, _insert_row_mutation)


def scramble_mutation(X: PermPop, prob: float, rng: RNG, max_segment_length: int = 0) -> None:
    _apply_row_mutation(X, prob, rng, lambda row, local_rng: _scramble_row_mutation(row, local_rng, max_segment_length=max_segment_length))


def inversion_mutation(X: PermPop, prob: float, rng: RNG) -> None:
    _apply_row_mutation(X, prob, rng, _inversion_row_mutation)


def displacement_mutation(X: PermPop, prob: float, rng: RNG) -> None:
    _apply_row_mutation(X, prob, rng, _displacement_row_mutation)


def two_opt_mutation(X: PermPop, prob: float, rng: RNG) -> None:
    """A 2-opt move: reverse the segment between two random positions.

    On a tour, removing two edges and reconnecting the path reverses the segment
    between them, so this is the same operator as :func:`inversion_mutation`, with
    the same random draws; the name is kept for routing problems.
    """
    _apply_row_mutation(X, prob, rng, _two_opt_row_mutation)


def _apply_row_mutation(X: PermPop, prob: float, rng: RNG, mut_fn: Callable[[PermVec, RNG], None]) -> None:
    validate_permutation_population(X, label="X")
    N, D = X.shape
    if N == 0 or D < 2:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    for idx in np.flatnonzero(rng.random(N) <= prob):
        mut_fn(X[idx], rng)


def _insert_row_mutation(row: PermVec, rng: RNG) -> None:
    """Move the gene at a random position ``i`` to another random position ``j``, to the left or right."""
    n = row.size
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n - 1))
    if j >= i:
        j += 1
    gene = row[i]
    if i < j:
        row[i:j] = row[i + 1 : j + 1]
        row[j] = gene
    else:
        row[j + 1 : i + 1] = row[j:i]
        row[j] = gene


def _scramble_row_mutation(row: PermVec, rng: RNG, max_segment_length: int = 0) -> None:
    """Shuffle the genes between two distinct random positions, both included.

    The segment has at least two genes (at most ``max_segment_length`` when it is
    positive), so the shuffle can change the row.
    """
    n = row.size
    if n < 2:
        return
    lo, hi = two_cut_points(n, rng)
    if max_segment_length > 0 and hi - lo + 1 > max_segment_length:
        hi = lo + max_segment_length - 1
    segment = row[lo : hi + 1].copy()
    rng.shuffle(segment)
    row[lo : hi + 1] = segment


def _inversion_row_mutation(row: PermVec, rng: RNG) -> None:
    lo, hi = two_cut_points(row.size, rng)
    row[lo : hi + 1] = row[lo : hi + 1][::-1]


def _displacement_row_mutation(row: PermVec, rng: RNG) -> None:
    lo, hi = two_cut_points(row.size, rng)
    segment = row[lo:hi].copy()
    remaining = np.concatenate([row[:lo], row[hi:]])
    insert_pos = rng.integers(0, remaining.size + 1)
    row[:] = np.concatenate([remaining[:insert_pos], segment, remaining[insert_pos:]])


def _two_opt_row_mutation(row: PermVec, rng: RNG) -> None:
    lo, hi = two_cut_points(row.size, rng)
    if hi > lo:
        row[lo : hi + 1] = row[lo : hi + 1][::-1]


__all__ = [
    "displacement_mutation",
    "insert_mutation",
    "inversion_mutation",
    "scramble_mutation",
    "swap_mutation",
    "two_opt_mutation",
]
