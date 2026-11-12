"""Assignment -> config builders for tuning."""

from __future__ import annotations

from typing import Any, cast

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol

from ._bridge_assignment_builders import build_moead_permutation_config, config_builders
from ._bridge_assignment_shared import sanitize_assignment


def config_from_assignment(algorithm_name: str, assignment: dict[str, Any]) -> AlgorithmConfigProtocol:
    """Build a concrete algorithm config dataclass from a sampled assignment.

    The tuning spaces do not know the problem, so the caller adds its number of objectives as
    ``assignment["n_obj"]`` for the algorithms that depend on it. RVEA derives its population
    from ``n_partitions`` and ``n_obj`` (two objectives without it). MOEA/D with ``n_obj`` runs the
    largest population, at most the sampled ``pop_size``, that has weight vectors: with three or
    more objectives a complete Das-Dennis lattice or a packaged jMetal file (see
    :func:`~vamos.engine.algorithm.components.weight_vectors.largest_weighted_population`);
    without it, exactly ``pop_size``.
    """
    algo = algorithm_name.lower()
    assignment = sanitize_assignment(assignment)

    if algo == "moead_permutation":
        return build_moead_permutation_config(assignment)

    for base_name, (names, builder) in config_builders().items():
        if algo == f"{base_name}_mixed":
            return cast(AlgorithmConfigProtocol, builder(assignment, mixed=True))
        if algo in names:
            return cast(AlgorithmConfigProtocol, builder(assignment))

    raise ValueError(f"Unsupported algorithm for config construction: {algorithm_name}")


__all__ = ["config_from_assignment"]
