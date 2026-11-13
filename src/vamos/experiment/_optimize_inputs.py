"""Checks of ``optimize`` arguments that would otherwise be ignored or applied only in part."""

from __future__ import annotations

from collections.abc import Mapping

from vamos.engine.algorithm.catalog import ENABLED_ALGORITHMS
from vamos.engine.algorithm.config import GenericAlgorithmConfig
from vamos.engine.algorithm.config.defaults import config_algorithm
from vamos.engine.algorithm.variants import canonical_algorithm_name
from vamos.foundation.exceptions import ConfigurationError

# The registry selects a named problem by its dimensions only.
_NAMED_PROBLEM_KWARGS = ("n_var", "n_obj")


def check_problem_arguments(
    problem: object,
    *,
    n_var: int | None,
    n_obj: int | None,
    problem_kwargs: Mapping[str, object] | None,
) -> None:
    """Reject dimension arguments that a problem instance would ignore and kwargs a name cannot take."""
    if isinstance(problem, str):
        unsupported = sorted(str(key) for key in (problem_kwargs or {}) if key not in _NAMED_PROBLEM_KWARGS)
        if unsupported:
            raise ConfigurationError(
                f"problem_kwargs for a registered problem accepts only n_var and n_obj; got {', '.join(unsupported)}.",
                suggestion="Build the problem instance with these arguments and pass it as problem.",
            )
        return
    if isinstance(problem, type):
        return
    given = [name for name, value in (("n_var", n_var), ("n_obj", n_obj), ("problem_kwargs", problem_kwargs)) if value is not None]
    if given:
        raise ConfigurationError(
            f"{', '.join(given)} cannot change a problem instance, which already has its dimensions "
            f"(n_var={getattr(problem, 'n_var', '?')}, n_obj={getattr(problem, 'n_obj', '?')}).",
            suggestion="Build the instance with the dimensions you want, or pass a registered problem name or a problem class.",
        )


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


__all__ = ["check_algorithm_config", "check_budget", "check_problem_arguments"]
