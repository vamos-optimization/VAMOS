"""Exact reconstruction of executable built-in inputs from a resolved run spec."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol, EngineName
from vamos.experiment.types import TerminationSpec
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.problem.registry import make_problem_selection

from .component_support import resolved_component_reconstructability
from .errors import ComponentNotReconstructableError, ResolvedSpecMismatchError, UnsupportedReplayProviderError
from .jsonio import canonical_json_bytes, sha256_bytes
from .models import deep_freeze, deep_thaw
from .specs import RunSpecInputs, build_run_specs

_CONFIG_TYPES = {
    "agemoea": AGEMOEAConfig,
    "ibea": IBEAConfig,
    "moead": MOEADConfig,
    "nsgaii": NSGAIIConfig,
    "nsgaiii": NSGAIIIConfig,
    "rvea": RVEAConfig,
    "smpso": SMPSOConfig,
    "smsemoa": SMSEMOAConfig,
    "spea2": SPEA2Config,
}
_OPERATOR_INPUT_NAMES = {
    "differential_evolution": "de",
    "order": "ox",
    "polynomial": "pm",
}
_OPERATOR_ROLES = ("crossover", "initializer", "mutation", "repair", "selection")


@dataclass(frozen=True, slots=True)
class ReconstructedRun:
    """Canonical built-in execution inputs reconstructed from persisted science."""

    resolved_spec: Mapping[str, Any]
    problem: str
    n_var: int
    n_obj: int
    encoding: str
    algorithm: str
    algorithm_config: AlgorithmConfigProtocol
    termination: TerminationSpec
    engine: EngineName
    eval_strategy: str | None
    seed: int


def reconstruct_resolved_run(resolved_spec: Mapping[str, Any], *, root: Path) -> ReconstructedRun:
    """Reconstruct and prove executable built-in inputs from a resolved spec."""
    _ensure_resolved_components(resolved_spec, root)
    resolved = cast(dict[str, Any], deep_thaw(deep_freeze(resolved_spec)))
    problem, n_var, n_obj, encoding = _problem_fields(resolved, root)
    algorithm = _component_name(_mapping(resolved.get("algorithm")).get("component_id"), "vamos.algorithm")
    if algorithm is None or algorithm not in _CONFIG_TYPES:
        raise _component_error(root, "$.resolved_spec.algorithm.component_id", algorithm)
    config = _reconstruct_algorithm_config(algorithm, resolved, root)
    termination = _termination(resolved, root)
    engine = _engine(resolved, root)
    eval_strategy = _evaluation_strategy(resolved)
    seed = _integer(resolved.get("seed"), root, "$.resolved_spec.seed")
    _assert_problem_registered(problem, n_var, n_obj, encoding, root)
    _assert_semantic_equality(resolved, problem, n_var, n_obj, encoding, algorithm, config, termination, engine, seed, root)
    return ReconstructedRun(
        resolved_spec=cast(Mapping[str, Any], deep_freeze(resolved)),
        problem=problem,
        n_var=n_var,
        n_obj=n_obj,
        encoding=encoding,
        algorithm=algorithm,
        algorithm_config=config,
        termination=termination,
        engine=cast(EngineName, engine),
        eval_strategy=eval_strategy,
        seed=seed,
    )


def instantiate_reconstructed_problem(reconstructed: ReconstructedRun, *, root: Path) -> Any:
    """Instantiate only the registered problem proven by reconstruction."""
    try:
        selection = make_problem_selection(
            reconstructed.problem,
            n_var=reconstructed.n_var,
            n_obj=reconstructed.n_obj,
        )
        problem = selection.instantiate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ComponentNotReconstructableError(
            operation="reconstruct resolved run",
            field="$.resolved_spec.problem",
            path=root,
            reason="cannot instantiate the registered built-in problem",
            expected=reconstructed.problem,
            actual=f"{type(exc).__name__}: {exc}",
            action="Use the exact VAMOS implementation that created the persisted spec.",
        ) from exc
    dimensions = (getattr(problem, "n_var", None), getattr(problem, "n_obj", None))
    expected = (reconstructed.n_var, reconstructed.n_obj)
    if dimensions != expected:
        raise ResolvedSpecMismatchError(
            operation="reconstruct resolved run",
            field="$.resolved_spec.problem.config",
            path=root,
            reason="instantiates with dimensions different from the persisted spec",
            expected={"n_var": expected[0], "n_obj": expected[1]},
            actual={"n_var": dimensions[0], "n_obj": dimensions[1]},
            action="Use the exact VAMOS implementation that created the persisted spec.",
        )
    return problem


def _ensure_resolved_components(resolved: Mapping[str, Any], root: Path) -> None:
    status, reasons = resolved_component_reconstructability(resolved)
    if status == "reconstructable":
        return
    reason = reasons[0] if reasons else None
    error_type = (
        UnsupportedReplayProviderError
        if reason is not None and reason.code in {"custom_provider", "unsupported_provider"}
        else ComponentNotReconstructableError
    )
    raise error_type(
        operation="reconstruct resolved run",
        field=reason.field if reason is not None else "$.resolved_spec",
        path=root,
        reason="contains a component outside built-in execution support",
        expected="reconstructable built-in components",
        actual=status,
        action=reason.action if reason is not None else "Regenerate with supported built-ins.",
    )


def _reconstruct_algorithm_config(algorithm: str, resolved: Mapping[str, Any], root: Path) -> AlgorithmConfigProtocol:
    algorithm_descriptor = _mapping(resolved.get("algorithm"))
    raw_config = algorithm_descriptor.get("config")
    if not isinstance(raw_config, Mapping):
        raise _component_error(root, "$.resolved_spec.algorithm.config", raw_config)
    config = cast(dict[str, object], deep_thaw(deep_freeze(raw_config)))
    operators = resolved.get("operators")
    if not isinstance(operators, Mapping):
        raise _component_error(root, "$.resolved_spec.operators", operators)
    for role, descriptor in operators.items():
        if role not in _OPERATOR_ROLES or not isinstance(descriptor, Mapping):
            raise _component_error(root, f"$.resolved_spec.operators.{role}", descriptor)
        _restore_operator(config, str(role), descriptor, root)
    config_type = _CONFIG_TYPES[algorithm]
    try:
        return cast(AlgorithmConfigProtocol, cast(Any, config_type).from_dict(config))
    except (TypeError, ValueError) as exc:
        raise ComponentNotReconstructableError(
            operation="reconstruct resolved run",
            field="$.resolved_spec.algorithm.config",
            path=root,
            reason="cannot reconstruct the typed built-in algorithm configuration",
            expected=config_type.__name__,
            actual=f"{type(exc).__name__}: {exc}",
            action="Regenerate the run with a complete current resolved configuration.",
        ) from exc


def _restore_operator(config: dict[str, object], role: str, descriptor: Mapping[str, Any], root: Path) -> None:
    resolution = descriptor.get("resolution")
    if isinstance(resolution, Mapping) and resolution.get("active") is False:
        if role == "repair":
            config[role] = "auto"
        return
    stable_name = _component_name(descriptor.get("component_id"), "vamos.operator")
    parameters = descriptor.get("config")
    if stable_name is None or not isinstance(parameters, Mapping):
        raise _component_error(root, f"$.resolved_spec.operators.{role}", descriptor)
    name = _OPERATOR_INPUT_NAMES.get(stable_name, stable_name)
    values = cast(dict[str, Any], deep_thaw(deep_freeze(parameters)))
    config[role] = {"type": name, **values} if role == "initializer" else (name, values)


def _assert_semantic_equality(
    source: Mapping[str, Any],
    problem: str,
    n_var: int,
    n_obj: int,
    encoding: str,
    algorithm: str,
    config: AlgorithmConfigProtocol,
    termination: TerminationSpec,
    engine: str,
    seed: int,
    root: Path,
) -> None:
    _, reconstructed = build_run_specs(
        RunSpecInputs(
            problem_built_in=True,
            problem_label=problem,
            problem_kwargs=None,
            n_var_requested=n_var,
            n_obj_requested=n_obj,
            n_var=n_var,
            n_obj=n_obj,
            encoding=encoding,
            algorithm_requested=algorithm,
            algorithm=algorithm,
            algorithm_config=dict(config.to_dict()),
            algorithm_config_explicit=True,
            max_evaluations_requested=None,
            termination=termination,
            pop_size_requested=None,
            resolved_pop_size=config.to_dict().get("pop_size"),
            engine_requested=engine,
            engine=engine,
            eval_strategy=_evaluation_strategy(source),
            seed_requested=seed,
            seed=seed,
            default_sources=_default_sources(source),
        )
    )
    expected_bytes = canonical_json_bytes(source)
    actual_bytes = canonical_json_bytes(reconstructed)
    if expected_bytes == actual_bytes:
        return
    raise ResolvedSpecMismatchError(
        operation="reconstruct resolved run",
        field="$.resolved_spec",
        path=root,
        reason="does not equal the explicitly reconstructed effective specification",
        expected=sha256_bytes(expected_bytes),
        actual=sha256_bytes(actual_bytes),
        action="Regenerate the source with the current complete canonical spec; current defaults were not substituted.",
    )


def _problem_fields(resolved: Mapping[str, Any], root: Path) -> tuple[str, int, int, str]:
    descriptor = _mapping(resolved.get("problem"))
    problem = _component_name(descriptor.get("component_id"), "vamos.problem")
    config = _mapping(descriptor.get("config"))
    if problem is None:
        raise _component_error(root, "$.resolved_spec.problem.component_id", descriptor.get("component_id"))
    n_var = _integer(config.get("n_var"), root, "$.resolved_spec.problem.config.n_var")
    n_obj = _integer(config.get("n_obj"), root, "$.resolved_spec.problem.config.n_obj")
    encoding = config.get("encoding")
    if not isinstance(encoding, str):
        raise _component_error(root, "$.resolved_spec.problem.config.encoding", encoding)
    extras = set(config) - {"constraint_convention", "encoding", "n_obj", "n_var"}
    if extras:
        raise _component_error(root, "$.resolved_spec.problem.config", {"unsupported_fields": sorted(extras)})
    return problem, n_var, n_obj, encoding


def _assert_problem_registered(problem: str, n_var: int, n_obj: int, encoding: str, root: Path) -> None:
    try:
        selection = make_problem_selection(problem, n_var=n_var, n_obj=n_obj)
    except (KeyError, TypeError, ValueError) as exc:
        raise _component_error(root, "$.resolved_spec.problem", problem) from exc
    registered_encoding = normalize_encoding(selection.spec.encoding)
    if registered_encoding != encoding:
        raise ResolvedSpecMismatchError(
            operation="reconstruct resolved run",
            field="$.resolved_spec.problem.config.encoding",
            path=root,
            reason="differs from the current built-in problem registration",
            expected=encoding,
            actual=registered_encoding,
            action="Use the exact VAMOS implementation that created the source run.",
        )


def _termination(resolved: Mapping[str, Any], root: Path) -> TerminationSpec:
    descriptor = _mapping(resolved.get("termination"))
    name = _component_name(descriptor.get("component_id"), "vamos.termination")
    config = _mapping(descriptor.get("config"))
    if name == "max_evaluations":
        value = _integer(config.get("max_evaluations"), root, "$.resolved_spec.termination.config.max_evaluations")
        if config.get("hard_max_evaluations") != value or set(config) != {"max_evaluations", "hard_max_evaluations"}:
            raise _component_error(root, "$.resolved_spec.termination.config", config)
        return (name, value)
    if name == "hv":
        return (name, cast(dict[str, object], deep_thaw(deep_freeze(config))))
    raise _component_error(root, "$.resolved_spec.termination.component_id", name)


def _engine(resolved: Mapping[str, Any], root: Path) -> str:
    backend = _mapping(resolved.get("backend"))
    kernel = _mapping(backend.get("kernel"))
    engine = _component_name(kernel.get("component_id"), "vamos.kernel")
    resolution = _mapping(kernel.get("resolution"))
    if engine not in {"numpy", "numba", "moocore"} or resolution.get("name") != engine:
        raise _component_error(root, "$.resolved_spec.backend.kernel", kernel)
    evaluation = _mapping(backend.get("evaluation"))
    if not is_reconstructable_evaluation_backend(evaluation):
        raise _component_error(root, "$.resolved_spec.backend.evaluation", evaluation)
    return engine


def is_reconstructable_evaluation_backend(evaluation: Mapping[str, Any]) -> bool:
    """Return whether exact reconstruction, and so study execution, supports ``evaluation``.

    Only ordered serial evaluation is reconstructed; study planning applies the same
    predicate so that a published plan never contains a task ``Study.run()`` rejects.
    """
    evaluation_name = _component_name(evaluation.get("component_id"), "vamos.evaluation")
    return evaluation_name == "serial" and evaluation.get("config") in ({}, {"ordered": True, "workers": 1})


def _evaluation_strategy(resolved: Mapping[str, Any]) -> str | None:
    backend = _mapping(resolved.get("backend"))
    evaluation = _mapping(backend.get("evaluation"))
    return "serial" if evaluation.get("config") == {} else None


def _default_sources(resolved: Mapping[str, Any]) -> dict[str, str]:
    items = resolved.get("defaults_applied")
    if not isinstance(items, list):
        return {}
    return {
        str(item["field"]).lstrip("/"): str(item["source"])
        for item in items
        if isinstance(item, Mapping) and "field" in item and "source" in item
    }


def _component_name(value: object, namespace: str) -> str | None:
    prefix = f"{namespace}:"
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith("@1"):
        return None
    return value[len(prefix) : -2]


def _integer(value: object, root: Path, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _component_error(root, field, value)
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _component_error(root: Path, field: str, actual: object) -> ComponentNotReconstructableError:
    return ComponentNotReconstructableError(
        operation="reconstruct resolved run",
        field=field,
        path=root,
        reason="is not a complete supported built-in execution component",
        expected="stable schema-1 built-in descriptor",
        actual=actual,
        action="Regenerate the run with a current registered built-in component.",
    )


__all__ = [
    "ReconstructedRun",
    "instantiate_reconstructed_problem",
    "is_reconstructable_evaluation_backend",
    "reconstruct_resolved_run",
]
