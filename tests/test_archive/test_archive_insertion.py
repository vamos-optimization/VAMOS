"""Updates of a few rows are inserted into the archive; they must give exactly what the full update gives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.components import archive_core, archive_insertion
from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    MaxMinArchive,
    ReferenceDirectionsArchive,
    SPEA2Archive,
    UnboundedArchive,
)
from vamos.foundation.kernel.numpy_backend import NumPyKernel

N_VAR = 3


def _archive(kind: str, n_obj: int, n_con: int | None) -> Any:
    common: dict[str, Any] = {"n_var": N_VAR, "n_obj": n_obj, "dtype": float, "n_con": n_con}
    if kind == "unbounded":
        return UnboundedArchive(**common)
    if kind == "unbounded-tolerance":
        return UnboundedArchive(**common, objective_tolerance=0.5)
    if kind == "unbounded-decision":
        return UnboundedArchive(**common, deduplicate_in="decision")
    if kind == "crowding":
        return CrowdingDistanceArchive(12, **common)
    if kind == "crowding-kernels":
        return CrowdingDistanceArchive(12, **common, truncate_size=9, selection_kernels=NumPyKernel().selection_kernels())
    if kind == "hv":
        return HypervolumeArchive(12, **common)
    if kind == "knn":
        return SPEA2Archive(12, **common)
    if kind == "maxmin":
        return MaxMinArchive(12, **common, truncate_size=8)
    if kind == "ref_dirs":
        return ReferenceDirectionsArchive(12, **common, rng_seed=4)
    raise ValueError(kind)


def _batch(rng: np.random.Generator, n_obj: int, n_con: int | None, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Rows near a linear front on a coarse grid, so that ties, duplicates and signed zeros are common."""
    size = int(rng.choice([1, 1, 1, 2, 3, 5, 16, 17, 40]))
    first = rng.integers(0, 25, size=size)
    if n_obj == 2:
        F = np.column_stack([first, 24 - first]).astype(float)
    else:
        second = (rng.random(size) * (25 - first)).astype(int)
        F = np.column_stack([first, second, 24 - first - second]).astype(float)
    F += rng.integers(0, 3, size=F.shape) * (rng.random(F.shape) < 0.5)
    F += np.where(rng.random(F.shape) < 0.2, rng.random(F.shape).round(1), 0.0)
    F = np.where(F == 0.0, np.where(rng.random(F.shape) < 0.5, -0.0, 0.0), F)
    if step >= 180 and rng.random() < 0.1:  # non-finite rows end the clean contents: keep them for the end
        # moocore 0.3.2 crashes on -inf with three objectives, whatever the archive does.
        F[int(rng.integers(size)), int(rng.integers(n_obj))] = rng.choice([np.inf, np.nan] if n_obj == 3 else [np.inf, -np.inf, np.nan])
    if rng.random() < 0.2 and size > 1:
        F[1:] = F[0]  # duplicates, with the same or other decision vectors
    X = rng.integers(0, 3, size=(size, N_VAR)).astype(float)
    if n_con is None:
        return X, F, None
    # Infeasible rows only at first, then a mix.
    G = rng.normal(size=(size, n_con)) + (2.0 if step < 15 else 0.0)
    if step >= 180 and rng.random() < 0.1:
        G[0, 0] = np.nan
    return X, F, G


def _state(archive: Any) -> tuple[Any, ...]:
    X, F = archive.contents()
    G = archive.constraint_values()
    rng = archive._rng.bit_generator.state if isinstance(archive, ReferenceDirectionsArchive) else None
    return X.tobytes(), F.tobytes(), None if G is None else G.tobytes(), archive.last_capacity_pruned, archive.covers_last_input, rng


def _rows(archive: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    X, F = archive.contents()
    return X, F, archive.constraint_values()


def _check_flags(archive: Any, before: tuple[np.ndarray, np.ndarray, np.ndarray | None], F: np.ndarray, G: np.ndarray | None) -> None:
    """What ``unchanged_by_last_input`` and ``covers_last_input`` claim, checked by brute force."""
    after = _rows(archive)
    same = all(
        (left is None and right is None) or (left is not None and right is not None and left.tobytes() == right.tobytes())
        for left, right in zip(before, after)
    )
    random_truncation = isinstance(archive, ReferenceDirectionsArchive) and archive.last_capacity_pruned > 0
    assert archive.unchanged_by_last_input == (same and not random_truncation)
    if not archive.covers_last_input:
        return
    _, archived_F, archived_G = after
    feasible_archive = archived_G is None or bool(np.all(archived_G <= 0.0))
    assert archived_F.shape[0] > 0 and feasible_archive and np.isfinite(archived_F).all()
    rows_G = None if G is None else np.concatenate((G, before[2])) if before[2] is not None else G
    rows_F = np.concatenate((F, before[1]))
    for index, row in enumerate(rows_F):
        if rows_G is not None and index < rows_G.shape[0] and not np.all(rows_G[index] <= 0.0):
            continue  # infeasible, while the archive holds feasible rows
        assert np.isfinite(row).all() and np.all(archived_F <= row, axis=1).any()


KINDS = [
    "unbounded",
    "unbounded-tolerance",
    "unbounded-decision",
    "crowding",
    "crowding-kernels",
    "hv",
    "knn",
    "maxmin",
    "ref_dirs",
]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the non-finite rows at the end
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n_con", [None, 2])
@pytest.mark.parametrize("moocore", [True, False])
def test_insertion_matches_the_full_update(kind: str, n_con: int | None, moocore: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if not moocore:
        monkeypatch.setattr(archive_core, "_moocore", None)
    elif archive_core._moocore is None:
        pytest.skip("moocore is not installed")
    n_obj = 2 if kind == "hv" and not moocore else 3 if kind == "ref_dirs" else 2
    rng = np.random.default_rng(len(kind) * 7 + (n_con or 0) + moocore)
    archive, reference = _archive(kind, n_obj, n_con), _archive(kind, n_obj, n_con)
    reference._insertable = lambda *args: False
    inserted = pruned_by_insertion = pruned = 0
    original_insert = archive._insert

    def counting_insert(*args: Any) -> bool:
        nonlocal inserted, pruned_by_insertion
        done = bool(original_insert(*args))
        inserted += done
        pruned_by_insertion += done and archive.last_capacity_pruned > 0
        return done

    archive._insert = counting_insert
    for step in range(220):
        X, F, G = _batch(rng, n_obj, n_con, step)
        before = _rows(archive)
        outcomes = []
        for target in (archive, reference):
            try:
                X_out, F_out = target.update(X, F, G)
                outcomes.append((X_out.tobytes(), F_out.tobytes(), _state(target)))
            except ValueError as error:  # pareto_filter rejects NaN objectives: both must raise alike
                outcomes.append((type(error), str(error)))
        assert outcomes[0] == outcomes[1], f"step {step}"
        if len(outcomes[0]) == 3:
            _check_flags(archive, before, F, G)
        pruned += reference.last_capacity_pruned > 0
    if kind in ("unbounded-tolerance", "unbounded-decision"):
        assert inserted == 0
    else:
        assert inserted > 50
    if archive.capacity is not None:
        assert pruned > 10 and pruned_by_insertion > 0


def test_insertion_appends_new_rows_after_the_rows_it_keeps() -> None:
    archive = UnboundedArchive(n_var=1, n_obj=2, dtype=float)
    archive.update(np.array([[0.0], [1.0], [2.0]]), np.array([[0.0, 3.0], [1.0, 2.0], [3.0, 0.0]]))
    assert archive.covers_last_input

    X, F = archive.update(np.array([[5.0]]), np.array([[0.5, 1.5]]))

    assert F.tolist() == [[0.0, 3.0], [3.0, 0.0], [0.5, 1.5]]
    assert X.ravel().tolist() == [0.0, 2.0, 5.0]
    assert archive.covers_last_input


def test_capacity_pruning_and_decision_deduplication_do_not_cover_the_input() -> None:
    bounded = CrowdingDistanceArchive(2, n_var=1, n_obj=2, dtype=float)
    bounded.update(np.zeros((3, 1)), np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]))
    assert bounded.last_capacity_pruned == 1 and not bounded.covers_last_input

    decision = UnboundedArchive(n_var=1, n_obj=2, dtype=float, deduplicate_in="decision")
    decision.update(np.array([[0.0], [1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert not decision.covers_last_input


def test_constrained_archive_covers_its_input_only_once_it_holds_a_feasible_row() -> None:
    archive = UnboundedArchive(n_var=1, n_obj=2, dtype=float, n_con=1)
    archive.update(np.zeros((2, 1)), np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[1.0], [2.0]]))
    assert not archive.covers_last_input

    archive.update(np.ones((1, 1)), np.array([[5.0, 5.0]]), np.array([[-1.0]]))
    assert archive.covers_last_input
    assert archive.contents()[1].tolist() == [[5.0, 5.0]]


@pytest.mark.parametrize("n_dropped", [1, 2, 5, 16, 17, 60])
def test_drop_rows_keeps_the_other_rows_in_order(n_dropped: int) -> None:
    rng = np.random.default_rng(n_dropped)
    size = 80
    X = rng.random((size + 5, 3))
    F = rng.random((size + 5, 2))
    dropped = np.sort(rng.choice(size, n_dropped, replace=False))
    expected_X, expected_F = np.delete(X[:size], dropped, axis=0), np.delete(F[:size], dropped, axis=0)

    left = archive_insertion.drop_rows((X, F), size, dropped)

    assert left == size - n_dropped
    assert X[:left].tobytes() == expected_X.tobytes() and F[:left].tobytes() == expected_F.tobytes()
