"""Study planning checks of the reference directions of NSGA-III and the reference vectors of RVEA.

Planning never reads an external source of vectors: a path is rejected, and a lattice must fit the
population (NSGA-III needs at least one individual per direction, RVEA exactly one per vector).
"""

from __future__ import annotations

from collections.abc import Mapping
from math import comb
from typing import Any

from vamos.engine.reference_directions import two_layer_shared

from .errors import UnresolvedStudyTaskError


def validate_nsgaiii_reference_directions(config: Mapping[str, Any], *, population: int, n_obj: int, index: int) -> None:
    raw = config.get("reference_directions")
    if not isinstance(raw, Mapping):
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{index}].algorithm_config.reference_directions",
            expected="object with path=null and positive integer divisions",
            actual=type(raw).__name__,
            action="Use generated built-in reference directions; planning never reads an external source path.",
        )
    path = raw.get("path")
    divisions = raw.get("divisions")
    if path is not None:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNSAFE_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{index}].algorithm_config.reference_directions.path",
            expected="null (built-in generated reference directions)",
            actual="external path",
            action="Remove the reference-direction path; planning never follows external scientific inputs.",
        )
    if isinstance(divisions, bool) or not isinstance(divisions, int) or divisions < 1:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{index}].algorithm_config.reference_directions.divisions",
            expected="positive integer",
            actual=divisions,
            action="Choose a positive reference-direction division count.",
        )
    inner = _inner_divisions(raw, "reference_directions", index, divisions=divisions, n_obj=n_obj)
    expected = comb(divisions + n_obj - 1, n_obj - 1) + (0 if inner is None else comb(inner + n_obj - 1, n_obj - 1))
    lattice = f"divisions={divisions}" if inner is None else f"divisions={divisions} and inner_divisions={inner}"
    if population < expected:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="REFERENCE_DIRECTION_POPULATION_MISMATCH",
            field=f"$.matrix[{index}].pop_size",
            expected=f">= {expected} for n_obj={n_obj} and {lattice}",
            actual=population,
            action="Use at least the reference-direction count as pop_size or choose fewer divisions.",
        )


def _inner_divisions(raw: Mapping[str, Any], field: str, index: int, *, divisions: int, n_obj: int) -> int | None:
    """VAMOS 1.5.0: the inside layer of a two-layer lattice, ``None`` without one.

    An inside layer that repeats directions of the boundary layer is rejected, as the run would reject it.
    """
    inner = raw.get("inner_divisions")
    if inner is not None and (isinstance(inner, bool) or not isinstance(inner, int) or inner < 1):
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{index}].algorithm_config.{field}.inner_divisions",
            expected="positive integer or null",
            actual=inner,
            action="Choose a positive division count for the inside layer, or leave it out.",
        )
    shared = 0 if inner is None or n_obj < 2 else two_layer_shared(n_obj, divisions, inner)
    if shared:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{index}].algorithm_config.{field}.inner_divisions",
            expected=f"an inside layer that shares no direction with the boundary layer of divisions={divisions}",
            actual=f"inner_divisions={inner} repeats {shared} direction(s) for n_obj={n_obj}",
            action="Choose other divisions or inner_divisions.",
        )
    return inner


def validate_rvea_reference_directions(config: Mapping[str, Any], *, population: int, n_obj: int, index: int) -> None:
    vectors = config.get("reference_vectors")
    if vectors is not None and not isinstance(vectors, Mapping):
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{index}].algorithm_config.reference_vectors",
            expected="object with path=null and positive integer divisions, or null",
            actual=type(vectors).__name__,
            action="Use generated built-in reference vectors; planning never reads an external source path.",
        )
    vectors = vectors or {}
    if vectors.get("path") is not None:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNSAFE_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{index}].algorithm_config.reference_vectors.path",
            expected="null (built-in generated reference vectors)",
            actual="external path",
            action="Remove the reference-vector path; planning never follows external scientific inputs.",
        )
    # VAMOS 1.5.0: reference_vectors.divisions takes the place of n_partitions.
    use_divisions = vectors.get("divisions") is not None
    partitions = vectors.get("divisions") if use_divisions else config.get("n_partitions")
    if isinstance(partitions, bool) or not isinstance(partitions, int) or partitions < 1:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{index}].algorithm_config." + ("reference_vectors.divisions" if use_divisions else "n_partitions"),
            expected="positive integer",
            actual=partitions,
            action="Choose a positive RVEA reference-vector partition count.",
        )
    inner = _inner_divisions(vectors, "reference_vectors", index, divisions=partitions, n_obj=n_obj)
    expected = comb(partitions + n_obj - 1, n_obj - 1) + (0 if inner is None else comb(inner + n_obj - 1, n_obj - 1))
    lattice = f"n_partitions={partitions}" if inner is None else f"divisions={partitions} and inner_divisions={inner}"
    if population != expected:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="REFERENCE_DIRECTION_POPULATION_MISMATCH",
            field=f"$.matrix[{index}].pop_size",
            expected=f"{expected} for n_obj={n_obj} and {lattice}",
            actual=population,
            action="Use the reference-vector count as pop_size or choose compatible n_partitions.",
        )


__all__ = ["validate_nsgaiii_reference_directions", "validate_rvea_reference_directions"]
