"""Assignment -> config builders for tuning."""

from __future__ import annotations

from typing import Any, cast

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol

from ._bridge_assignment_builders import build_moead_permutation_config, config_builders
from ._bridge_assignment_shared import sanitize_assignment
from .parameter_names import bridge_assignment


def config_from_assignment(algorithm_name: str, assignment: dict[str, Any]) -> AlgorithmConfigProtocol:
    """Build a concrete algorithm config dataclass from a sampled assignment.

    The tuning spaces do not know the problem, so the caller adds its number of objectives as
    ``assignment["n_obj"]`` for the algorithms that depend on it. RVEA runs the largest complete
    Das-Dennis lattice with at most ``pop_size`` directions for ``n_obj`` (two objectives without
    it), or the lattice of ``n_partitions`` when the assignment has one. MOEA/D with ``n_obj`` runs the
    largest population, at most the sampled ``pop_size``, that has weight vectors: with three or
    more objectives a complete Das-Dennis lattice or a packaged jMetal file (see
    :func:`~vamos.engine.algorithm.components.weight_vectors.largest_weighted_population`);
    without it, exactly ``pop_size``.

    The assignment may name its parameters as the spaces do, after Evolver's names
    (``sbx_distribution_index``), or with the earlier names (``crossover_eta``); see
    :mod:`vamos.engine.tuning.racing.parameter_names`.
    """
    algo = algorithm_name.lower()
    assignment = sanitize_assignment(bridge_assignment(assignment))

    if algo == "moead_permutation":
        return build_moead_permutation_config(assignment)

    for base_name, (names, builder) in config_builders().items():
        if algo == f"{base_name}_mixed":
            return cast(AlgorithmConfigProtocol, builder(assignment, mixed=True))
        if algo in names:
            return cast(AlgorithmConfigProtocol, builder(assignment))

    raise ValueError(f"Unsupported algorithm for config construction: {algorithm_name}")


__all__ = ["config_from_assignment"]
