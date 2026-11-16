"""Checks of ``optimize`` arguments that would otherwise be ignored or applied only in part."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from vamos.engine.algorithm.catalog import ENABLED_ALGORITHMS
from vamos.engine.algorithm.config import GenericAlgorithmConfig
from vamos.engine.algorithm.config.defaults import config_algorithm
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.variants import canonical_algorithm_name
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.archive.factory import resolve_archive_seed
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.problem.registry import get_problem_specs

_DIMENSIONS = ("n_var", "n_obj")


def check_problem_arguments(
    problem: object,
    *,
    n_var: int | None,
    n_obj: int | None,
    problem_kwargs: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """Reject dimension arguments that a problem instance would ignore and kwargs a name cannot take.

    Returns ``problem_kwargs`` with a registered problem's options as the run artifact records them, so that the
    problem is built with the recorded values.
    """
    if isinstance(problem, str):
        spec = get_problem_specs().get(problem)
        options = {key: value for key, value in (problem_kwargs or {}).items() if key not in _DIMENSIONS}
        if spec is not None and options:
            # The registry entry's factory declares the problem's other options; an unknown name fails later.
            dimensions = {key: value for key, value in (problem_kwargs or {}).items() if key in _DIMENSIONS}
            return {**dimensions, **spec.check_options(options)}
        return problem_kwargs
    if isinstance(problem, type):
        return problem_kwargs
    given = [name for name, value in (("n_var", n_var), ("n_obj", n_obj), ("problem_kwargs", problem_kwargs)) if value is not None]
    if given:
        raise ConfigurationError(
            f"{', '.join(given)} cannot change a problem instance, which already has its dimensions "
            f"(n_var={getattr(problem, 'n_var', '?')}, n_obj={getattr(problem, 'n_obj', '?')}).",
            suggestion="Build the instance with the dimensions you want, or pass a registered problem name or a problem class.",
        )
    return problem_kwargs


def check_budget(max_evaluations: int | None, termination: tuple[str, object] | None) -> None:
    """Reject a ``max_evaluations`` that ``termination`` would override."""
    if termination is None or max_evaluations is None:
        return
    kind, value = termination
    if kind == "max_evaluations" and not isinstance(value, bool) and value == max_evaluations:
        return
    raise ConfigurationError(
        f"max_evaluations={max_evaluations} conflicts with termination={termination!r}; termination decides when the run stops.",
        suggestion="Pass only termination, or only max_evaluations.",
    )


def check_algorithm_config(algorithm: str, algorithm_config: object) -> None:
    """Reject a configuration of another built-in algorithm, and a generic one for a built-in."""
    canonical = canonical_algorithm_name(algorithm)
    if isinstance(algorithm_config, GenericAlgorithmConfig):
        if canonical in ENABLED_ALGORITHMS:
            raise ConfigurationError(
                f"GenericAlgorithmConfig is for plugin algorithms; {algorithm} is built in, and a generic mapping can set "
                "keys its typed configuration does not record, so the run could not be reproduced.",
                suggestion=f"Use the configuration class of {algorithm}, for example from vamos.algorithms.",
            )
        return
    owner = config_algorithm(algorithm_config)
    if owner is not None and owner != canonical:
        raise ConfigurationError(
            f"{type(algorithm_config).__name__} configures {owner}, not {algorithm}.",
            suggestion=f"Pass algorithm={owner!r}, or a configuration of {algorithm}.",
        )


def with_archive_seed(algorithm_config: AlgorithmConfigProtocol, seed: int) -> AlgorithmConfigProtocol:
    """``algorithm_config`` with the effective ``rng_seed`` of its external archive for the run ``seed``.

    ``ref_dirs`` pruning without an ``rng_seed`` draws from the stream derived from the run
    seed; resolving it here records the seed in the run's resolved configuration, which
    replay then reuses.
    """
    archive = getattr(algorithm_config, "external_archive", None)
    if (
        not isinstance(archive, ExternalArchiveConfig)
        or not dataclasses.is_dataclass(algorithm_config)
        or isinstance(algorithm_config, type)
    ):
        return algorithm_config
    resolved = resolve_archive_seed(archive, seed)
    if resolved is archive:
        return algorithm_config
    return dataclasses.replace(algorithm_config, external_archive=resolved)


__all__ = ["check_algorithm_config", "check_budget", "check_problem_arguments", "with_archive_seed"]
