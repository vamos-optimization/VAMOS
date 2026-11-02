"""Build explicit requested and resolved v1 run specifications."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any

from vamos.foundation.problem.builder import FunctionalProblem
from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection

from .jsonio import normalize_json
from .manifest import RESOLVED_SPEC_VERSION

_OPERATOR_ALIASES = {
    "bitflip": "bitflip",
    "clip": "clip",
    "de": "differential_evolution",
    "differential_evolution": "differential_evolution",
    "mixed": "mixed",
    "neighborhood": "neighborhood",
    "order": "order",
    "ox": "order",
    "pm": "polynomial",
    "polynomial": "polynomial",
    "random": "random",
    "sbx": "sbx",
    "swap": "swap",
    "tournament": "tournament",
    "uniform": "uniform",
}
_BUILT_IN_ALGORITHMS = {
    "agemoea",
    "ibea",
    "moead",
    "nsgaii",
    "nsgaiii",
    "rvea",
    "smpso",
    "smsemoa",
    "spea2",
}


@dataclass(frozen=True, slots=True)
class RunSpecInputs:
    """Resolved execution facts retained at the Python API boundary."""

    problem_built_in: bool
    problem_label: str
    problem_kwargs: Mapping[str, object] | None
    n_var_requested: int | None
    n_obj_requested: int | None
    n_var: int
    n_obj: int
    encoding: str
    algorithm_requested: str
    algorithm: str
    algorithm_config: Mapping[str, object]
    algorithm_config_explicit: bool
    max_evaluations_requested: int | None
    termination: tuple[str, object]
    pop_size_requested: int | None
    resolved_pop_size: object
    engine_requested: str | None
    engine: str
    eval_strategy: object
    seed_requested: int | None
    seed: int
    default_sources: Mapping[str, str]
    #: Module and qualified name of the user code behind a problem that is not a registry problem.
    problem_code: tuple[str, str] | None = None


def is_registry_problem(requested: object, instance: object, label: str) -> bool:
    """Whether a run's problem is the registry problem its label names, which replay rebuilds by that name.

    A problem requested by registry name qualifies. An instance (or a class) qualifies only when its label is a
    registry key and its class is exactly the one the registry builds for that key; user code, such as a
    ``make_problem`` function or a subclass, never does, whatever its name.
    """
    if isinstance(requested, str):
        return True
    key = _slug(label)
    if key not in get_problem_specs() or isinstance(instance, FunctionalProblem):
        return False
    try:
        reference = make_problem_selection(
            key,
            n_var=getattr(instance, "n_var", None),
            n_obj=getattr(instance, "n_obj", None),
        ).instantiate()
    except (KeyError, TypeError, ValueError):
        return False
    return type(reference) is type(instance)


def custom_problem_code(instance: object) -> tuple[str, str]:
    """Module and qualified name of the user code behind a problem: a ``make_problem`` function, else the class."""
    code: object = instance._fn if isinstance(instance, FunctionalProblem) else type(instance)
    module = getattr(code, "__module__", None)
    qualified_name = getattr(code, "__qualname__", None) or getattr(code, "__name__", None)
    return (
        module if isinstance(module, str) else "unknown",
        qualified_name if isinstance(qualified_name, str) else type(code).__name__,
    )


def build_run_specs(inputs: RunSpecInputs) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return detached canonical JSON for intent and effective execution state."""
    requested = _requested_spec(inputs)
    resolved = _resolved_spec(inputs)
    normalized_requested = normalize_json(_configuration_value(requested), field="$.requested_spec")
    normalized_resolved = normalize_json(_configuration_value(resolved), field="$.resolved_spec")
    if not isinstance(normalized_requested, dict) or not isinstance(normalized_resolved, dict):
        raise AssertionError("run spec builders returned a non-object")
    return normalized_requested, normalized_resolved


def _requested_spec(inputs: RunSpecInputs) -> dict[str, Any]:
    problem_config: dict[str, Any] = {}
    if inputs.n_var_requested is not None:
        problem_config["n_var"] = inputs.n_var_requested
    if inputs.n_obj_requested is not None:
        problem_config["n_obj"] = inputs.n_obj_requested
    if inputs.problem_kwargs:
        problem_config.update(inputs.problem_kwargs)
    requested: dict[str, Any] = {
        "version": "1",
        "problems": {inputs.problem_label: problem_config},
        "defaults": {"seed": inputs.seed_requested},
    }
    defaults = requested["defaults"]
    if not isinstance(defaults, dict):
        raise AssertionError("requested defaults is not an object")
    if inputs.algorithm_requested != "auto":
        defaults["algorithm"] = inputs.algorithm_requested
    if inputs.engine_requested is not None:
        defaults["engine"] = inputs.engine_requested
    if inputs.max_evaluations_requested is not None:
        defaults["max_evaluations"] = inputs.max_evaluations_requested
    if inputs.pop_size_requested is not None:
        defaults["population_size"] = inputs.pop_size_requested
    if inputs.algorithm_config_explicit:
        requested["algorithms"] = {inputs.algorithm: dict(inputs.algorithm_config)}
    return requested


def _resolved_spec(inputs: RunSpecInputs) -> dict[str, Any]:
    algorithm_config = dict(inputs.algorithm_config)
    operators = _operators(algorithm_config, encoding=inputs.encoding, n_var=inputs.n_var, algorithm=inputs.algorithm)
    for key in ("crossover", "mutation", "selection", "repair", "initializer"):
        algorithm_config.pop(key, None)
    defaults_applied = [
        {
            "field": f"/{field}",
            "source": source,
            "reason": "value resolved by the VAMOS Python API",
        }
        for field, source in sorted(inputs.default_sources.items())
        if source in {"auto", "default"}
    ]
    pop_size = _optional_int(inputs.resolved_pop_size)
    offspring_size = _optional_int(algorithm_config.get("offspring_size")) or pop_size
    archive_size = _optional_int(algorithm_config.get("archive_size"))
    problem_config = dict(inputs.problem_kwargs or {})
    problem_config.update(
        {
            "n_var": inputs.n_var,
            "n_obj": inputs.n_obj,
            "encoding": inputs.encoding,
            "constraint_convention": "g_lte_0",
        }
    )
    result: dict[str, Any] = {
        "spec_version": RESOLVED_SPEC_VERSION,
        "problem": _problem_component(inputs, problem_config),
        "algorithm": _component(
            "algorithm",
            f"vamos.algorithm:{_slug(inputs.algorithm)}@1",
            algorithm_config,
            built_in=inputs.algorithm in _BUILT_IN_ALGORITHMS,
        ),
        "operators": operators,
        "backend": {
            "kernel": _kernel_component(inputs.engine),
            "evaluation": _evaluation_component(inputs.eval_strategy),
        },
        "termination": _termination_component(inputs.termination),
        "seed": inputs.seed,
        "population": {
            "initial_size": pop_size,
            "offspring_size": offspring_size,
            "archive_size": archive_size,
        },
        "defaults_applied": defaults_applied,
        "determinism": {
            "declared": _is_deterministic(inputs.engine, inputs.eval_strategy)
            and inputs.problem_built_in
            and not _contains_executable(inputs.algorithm_config),
            "rng": "numpy.random.Generator/PCG64",
        },
    }
    return result


def _operators(config: Mapping[str, object], *, encoding: str, n_var: int, algorithm: str) -> dict[str, Any]:
    operators: dict[str, Any] = {}
    for role in ("crossover", "mutation", "selection"):
        raw = config.get(role)
        if raw is None and role == "selection":
            raw = ("neighborhood", {"delta": config.get("delta", 0.9)}) if algorithm == "moead" else ("tournament", {"size": 2})
        if raw is not None:
            operators[role] = _operator_component(role, raw, n_var=n_var)
    initializer = config.get("initializer")
    if initializer is None:
        operators["initializer"] = _operator_component("initializer", ("random", {}), n_var=n_var)
    elif isinstance(initializer, Mapping):
        init_config = dict(initializer)
        name = str(init_config.pop("type", "random"))
        operators["initializer"] = _operator_component("initializer", (name, init_config), n_var=n_var)
    else:
        operators["initializer"] = _operator_component("initializer", initializer, n_var=n_var)
    repair = config.get("repair", "auto")
    if repair == "auto":
        if encoding == "real":
            operators["repair"] = _operator_component("repair", ("clip", {}), n_var=n_var)
        else:
            operators["repair"] = _inactive_operator("repair", f"auto repair is inactive for {encoding} encoding")
    elif repair is None:
        operators["repair"] = _inactive_operator("repair", "repair was explicitly disabled")
    else:
        operators["repair"] = _operator_component("repair", repair, n_var=n_var)
    return operators


def _operator_component(role: str, raw: object, *, n_var: int) -> dict[str, Any]:
    name: str
    config: dict[str, Any]
    if isinstance(raw, str):
        name, config = raw, {}
    elif isinstance(raw, (tuple, list)) and len(raw) == 2 and isinstance(raw[0], str) and isinstance(raw[1], Mapping):
        name, config = raw[0], dict(raw[1])
    else:
        name, config = f"unavailable-{role}", {"recorded_value": raw}
    if config.get("prob") == "1/n":
        config["prob"] = 1.0 / max(1, n_var)
    if role == "selection" and name == "tournament" and "size" not in config:
        config["size"] = 2
    stable_name = _OPERATOR_ALIASES.get(name, _slug(name))
    return _component("operator", f"vamos.operator:{stable_name}@1", config, built_in=name in _OPERATOR_ALIASES)


def _inactive_operator(role: str, reason: str) -> dict[str, Any]:
    descriptor = _component("operator", f"vamos.operator:none-{role}@1", {}, built_in=True)
    descriptor["resolution"] = {"contract_version": 1, "active": False, "reason": reason}
    return descriptor


def _kernel_component(engine: str) -> dict[str, Any]:
    package = {"numpy": "numpy", "numba": "numba", "moocore": "moocore"}.get(engine, engine)
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        version = None
    descriptor = _component(
        "kernel_backend",
        f"vamos.kernel:{_slug(engine)}@1",
        {},
        built_in=engine in {"numpy", "numba", "moocore"},
    )
    descriptor["resolution"] = {"name": engine, "version": version, "capabilities": []}
    return descriptor


def _evaluation_component(value: object) -> dict[str, Any]:
    config: dict[str, Any]
    if value is None:
        name, config, built_in = "serial", {"ordered": True, "workers": 1}, True
    elif isinstance(value, str):
        name, config, built_in = value.lower(), {}, value.lower() in {"serial", "multiprocessing", "dask"}
    else:
        name = value.__class__.__name__
        config = {"configuration": "unavailable"}
        built_in = value.__class__.__module__.startswith("vamos.")
    return _component("evaluation_backend", f"vamos.evaluation:{_slug(name)}@1", config, built_in=built_in)


def _termination_component(termination: tuple[str, object]) -> dict[str, Any]:
    name, raw_value = termination
    config: dict[str, Any]
    if name == "max_evaluations":
        config = {"max_evaluations": raw_value, "hard_max_evaluations": raw_value}
    elif isinstance(raw_value, Mapping):
        config = dict(raw_value)
    else:
        config = {"value": raw_value}
    return _component("termination", f"vamos.termination:{_slug(name)}@1", config, built_in=True)


def _problem_component(inputs: RunSpecInputs, config: Mapping[str, Any]) -> dict[str, Any]:
    if inputs.problem_built_in:
        return _component("problem", f"vamos.problem:{_slug(inputs.problem_label)}@1", config, built_in=True)
    # User code: identified by where it lives, never by its free name, which may match a registry key.
    module, qualified_name = inputs.problem_code or ("unknown", "unknown")
    return {
        "kind": "problem",
        "component_id": f"custom.problem:{_slug(f'{module}.{qualified_name}')}",
        "provider": {"type": "custom_python", "distribution": None},
        "config": dict(config),
        "resolution": {
            "contract_version": 1,
            "module": module,
            "qualified_name": qualified_name,
            "label": inputs.problem_label,
        },
    }


def _component(kind: str, component_id: str, config: Mapping[str, Any], *, built_in: bool) -> dict[str, Any]:
    return {
        "kind": kind,
        "component_id": component_id,
        "provider": {
            "type": "built_in" if built_in else "unavailable",
            "distribution": "vamos-optimization" if built_in else None,
        },
        "config": dict(config),
        "resolution": {"contract_version": 1},
    }


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_deterministic(engine: str, evaluation: object) -> bool:
    return engine in {"numpy", "numba", "moocore"} and (evaluation is None or evaluation == "serial")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-") or "unavailable"


def _configuration_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else {"unavailable_reason": "non_finite_configuration_number"}
    if isinstance(value, numbers.Real):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else {"unavailable_reason": "non_finite_configuration_number"}
    if isinstance(value, Mapping):
        return {str(key): _configuration_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_configuration_value(item) for item in value]
    if callable(value):
        return {
            "unavailable_reason": "executable_configuration_not_serialized",
            "module": getattr(value, "__module__", None),
            "qualified_name": getattr(value, "__qualname__", None),
        }
    return {
        "unavailable_reason": "configuration_value_not_json_serializable",
        "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
    }


def _contains_executable(value: object) -> bool:
    if callable(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_executable(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_executable(item) for item in value)
    return False


__all__ = ["RunSpecInputs", "build_run_specs", "custom_problem_code", "is_registry_problem"]
