"""Rule G7: the default configuration of a space runs as its algorithm's default configuration.

Equivalent configurations can be written differently, such as ``("pm", {"prob": 1/n})`` and a
``mutation_prob_factor`` of 1.0, so the check runs both with the same seed and compares the
results instead of the configurations. It runs the algorithm twice, so loading a space does not
check it; :func:`check_default_configuration` does, and the test suite checks every built-in space.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from vamos.engine.algorithm.config.defaults import build_default_algorithm_config
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import resolve_builtin_algorithm
from vamos.foundation.eval.backends import resolve_eval_strategy
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol

from ..racing.bridge_assignments import config_from_assignment
from .model import ParameterSpace, ParameterSpaceError


def default_assignment(space: ParameterSpace, *, pop_size: int, n_obj: int) -> dict[str, Any]:
    """The default values of ``space`` with the run settings that the tuning bridges read: ``pop_size`` and ``n_obj``."""
    return {**space.default_values(), "pop_size": int(pop_size), "n_obj": int(n_obj)}


def check_default_configuration(
    space: ParameterSpace,
    problem: ProblemProtocol,
    *,
    seed: int = 1,
    max_evaluations: int = 600,
    pop_size: int | None = None,
) -> None:
    """Raise a G7 :class:`ParameterSpaceError` unless the space's defaults run as the algorithm's default on ``problem``.

    ``pop_size`` is the run's population, which the space does not tune; ``None`` is the size of the
    algorithm's default configuration (100, or a reference lattice for ``n_obj``). Both
    configurations run ``max_evaluations`` evaluations with ``seed``, and the check compares their
    final variables and objectives.
    """
    n_obj = int(problem.n_obj)
    reference = build_default_algorithm_config(
        space.algorithm, pop_size=pop_size, n_var=int(problem.n_var), n_obj=n_obj, encoding=space.encoding
    )
    if reference is None:  # pragma: no cover - rule G6 admits built-in algorithms only
        raise ParameterSpaceError("G7", "algorithm", f"{space.algorithm!r} has no default configuration")
    size = int(cast(int, reference.to_dict()["pop_size"]))
    tuned = config_from_assignment(space.variant, default_assignment(space, pop_size=size, n_obj=n_obj))
    expected = _run(space.algorithm, reference, problem, seed, max_evaluations)
    actual = _run(space.algorithm, tuned, problem, seed, max_evaluations)
    for key in ("X", "F"):
        if not np.array_equal(np.asarray(expected[key]), np.asarray(actual[key])):
            differences = _differences(reference.to_dict(), tuned.to_dict())
            raise ParameterSpaceError(
                "G7",
                "parameters",
                f"the default configuration of {space.name!r} does not run as {type(reference).__name__}.default(): "
                f"seed {seed} gives other final {key}. Configuration fields that differ: {differences}",
            )


def _run(algorithm: str, config: AlgorithmConfigProtocol, problem: ProblemProtocol, seed: int, budget: int) -> Mapping[str, Any]:
    instance = resolve_builtin_algorithm(algorithm)(config.to_dict(), resolve_kernel("numpy"))
    backend = resolve_eval_strategy("serial")
    try:
        return instance.run(problem=problem, termination=("max_evaluations", int(budget)), seed=int(seed), eval_strategy=backend)
    finally:
        backend.close()


def _differences(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    keys = sorted(set(expected) | set(actual))
    return {key: (expected.get(key), actual.get(key)) for key in keys if repr(expected.get(key)) != repr(actual.get(key))}


__all__ = ["check_default_configuration", "default_assignment"]
