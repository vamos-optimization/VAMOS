"""
Schema validation helpers for experiment YAML/JSON specs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields
from difflib import get_close_matches
from typing import TypedDict, cast

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
from vamos.engine.hooks.hv_convergence import HVConvergenceConfig
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import available_problem_names

SpecBlock = dict[str, object]
ProblemOverrides = dict[str, SpecBlock | None]


class ExperimentSpec(TypedDict, total=False):
    version: str
    defaults: SpecBlock
    problems: ProblemOverrides
    stopping: SpecBlock
    archive: SpecBlock


EXPERIMENT_SPEC_VERSION = "1"

_ALGORITHM_BLOCKS: dict[str, type] = {
    "nsgaii": NSGAIIConfig,
    "moead": MOEADConfig,
    "smsemoa": SMSEMOAConfig,
    "nsgaiii": NSGAIIIConfig,
    "spea2": SPEA2Config,
    "ibea": IBEAConfig,
    "smpso": SMPSOConfig,
    "agemoea": AGEMOEAConfig,
    "rvea": RVEAConfig,
}

_OPERATOR_SPEC_KEYS = ("crossover", "mutation", "selection", "repair", "aggregation")
# Algorithm fields a CLI config sets elsewhere, or that only the Python API can provide.
_CLI_SET_ELSEWHERE = {
    "pop_size": "set population_size in defaults or the problem block",
    "pop_size_auto": "set population_size in defaults or the problem block",
    "offspring_size": "set offspring_population_size in defaults or the problem block",
    "external_archive": "use the archive block",
    "track_genealogy": "set track_genealogy in defaults or the problem block",
    "generation_callback": "it is available only in the Python API",
    "generation_callback_copy": "it is available only in the Python API",
    "parent_selection_filter": "it is available only in the Python API",
}
_HOOK_ARCHIVE_KEYS = {
    "enabled",
    "capacity",
    "truncate_size",
    "pruning",
    "hv_ref_point",
    "rng_seed",
    "objective_tolerance",
    "deduplicate_in",
    "decision_tolerance",
}


def allowed_override_keys(extra_keys: Iterable[str] | None = None) -> set[str]:
    """
    Build the allowed override keys for defaults/problem blocks.

    Pass CLI-derived keys (argument spec keys) via `extra_keys`.
    """
    keys = set(extra_keys or [])
    keys.update(_dataclass_field_names(ExperimentConfig))
    keys.update(_ALGORITHM_BLOCKS.keys())
    keys.update({"stopping", "archive"})
    return keys


def validate_experiment_spec(spec: object, *, allowed_overrides: Iterable[str]) -> ExperimentSpec:
    """
    Validate an experiment spec and return a normalized mapping.
    """
    spec_dict = _as_str_dict(spec, path="Experiment spec")

    unknown = _unknown_keys(spec_dict, {"version", "defaults", "problems", "stopping", "archive"})
    if unknown:
        raise ValueError(f"Unknown top-level keys in experiment spec: {', '.join(unknown)}")

    version = spec_dict.get("version")
    if version is None:
        raise ValueError(f"Experiment spec must declare 'version: {EXPERIMENT_SPEC_VERSION}'.")
    version_str = str(version).strip()
    if version_str != EXPERIMENT_SPEC_VERSION:
        raise ValueError(f"Unsupported experiment spec version. Expected version={EXPERIMENT_SPEC_VERSION!r}, got {version_str!r}.")

    override_keys = allowed_override_keys(allowed_overrides)
    defaults = _as_str_dict(spec_dict.get("defaults"), path="Experiment spec 'defaults'")
    _validate_overrides_block(defaults, path="defaults", allowed_keys=override_keys)

    problems = _as_str_dict(spec_dict.get("problems"), path="Experiment spec 'problems'")
    known_problems = sorted(available_problem_names())
    for key, value in problems.items():
        normalized_key = key.lower()
        if normalized_key not in known_problems:
            suggestions = get_close_matches(normalized_key, known_problems, n=3, cutoff=0.6)
            suggestion_text = ""
            if suggestions:
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?"
            raise ValueError(f"Unknown problem key in experiment spec: {key!r}.{suggestion_text}")
        if value is None:
            continue
        overrides = _as_str_dict(value, path=f"Experiment spec 'problems.{key}'")
        _validate_overrides_block(overrides, path=f"problems.{key}", allowed_keys=override_keys)

    if "stopping" in spec_dict:
        _validate_stopping_block(spec_dict.get("stopping"), path="stopping")
    if "archive" in spec_dict:
        _validate_archive_block(spec_dict.get("archive"), path="archive")

    return cast(ExperimentSpec, spec_dict)


def _validate_overrides_block(block: SpecBlock, *, path: str, allowed_keys: set[str]) -> None:
    unknown = _unknown_keys(block, allowed_keys)
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")

    for algo_key, cfg_cls in _ALGORITHM_BLOCKS.items():
        if algo_key not in block:
            continue
        algo_block = block.get(algo_key)
        if algo_block is None:
            continue
        algo_dict = _as_str_dict(algo_block, path=f"'{path}.{algo_key}'")
        _validate_algorithm_block(algo_dict, config_cls=cfg_cls, path=f"{path}.{algo_key}")

    if "stopping" in block:
        _validate_stopping_block(block.get("stopping"), path=f"{path}.stopping")
    if "archive" in block:
        _validate_archive_block(block.get("archive"), path=f"{path}.archive")


def _validate_algorithm_block(block: SpecBlock, *, config_cls: type, path: str) -> None:
    allowed = set(_dataclass_field_names(config_cls))
    unknown = _unknown_keys(block, allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")
    for key, hint in _CLI_SET_ELSEWHERE.items():
        if key in block:
            raise ValueError(f"'{path}.{key}' cannot be set in an algorithm block: {hint}.")

    for op_key in _OPERATOR_SPEC_KEYS:
        if op_key not in block:
            continue
        spec = block.get(op_key)
        if spec is None:
            continue
        _validate_operator_spec(spec, path=f"{path}.{op_key}")


def _validate_operator_spec(spec: object, *, path: str) -> None:
    if isinstance(spec, str):
        if not spec.strip():
            raise ValueError(f"'{path}' cannot be an empty operator name.")
        return
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise ValueError(f"'{path}' tuple/list form must have exactly 2 elements: [method, params].")
        method, params = spec
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"'{path}' operator method must be a non-empty string.")
        if not isinstance(params, Mapping):
            raise ValueError(f"'{path}' operator params must be a mapping.")
        return
    if isinstance(spec, Mapping):
        spec_map = _as_str_dict(spec, path=f"'{path}'")
        method = spec_map.get("method") or spec_map.get("name")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"'{path}' mapping form requires a non-empty 'method' (or 'name') key.")
        return
    raise ValueError(f"'{path}' must be a string, a 2-tuple/list, or a mapping with 'method'.")


def _validate_stopping_block(block: object, *, path: str) -> None:
    if block is None:
        return
    block_dict = _as_str_dict(block, path=f"'{path}'")
    unknown = _unknown_keys(block_dict, {"hv_convergence"})
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")
    hv = block_dict.get("hv_convergence")
    if hv is None:
        return
    hv_dict = _as_str_dict(hv, path=f"'{path}.hv_convergence'")
    allowed = {"enabled", "ref_point", *_dataclass_field_names(HVConvergenceConfig)}
    unknown_hv = _unknown_keys(hv_dict, allowed)
    if unknown_hv:
        raise ValueError(f"Unknown keys in '{path}.hv_convergence': {', '.join(unknown_hv)}")

    ref = hv_dict.get("ref_point")
    if ref is not None and not (
        (isinstance(ref, str) and ref.lower() == "auto") or (isinstance(ref, (list, tuple)) and all(_is_number(x) for x in ref))
    ):
        raise ValueError(f"'{path}.hv_convergence.ref_point' must be 'auto' or a list of numbers.")


def _validate_archive_block(block: object, *, path: str) -> None:
    if block is None:
        return
    block_dict = _as_str_dict(block, path=f"'{path}'")
    unknown = _unknown_keys(block_dict, {"external"})
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")
    external = block_dict.get("external")
    if external is not None:
        external_dict = _as_str_dict(external, path=f"'{path}.external'")
        unknown_external = _unknown_keys(external_dict, _HOOK_ARCHIVE_KEYS)
        if unknown_external:
            raise ValueError(f"Unknown keys in '{path}.external': {', '.join(unknown_external)}")


def _dataclass_field_names(cls: object) -> list[str]:
    return [field.name for field in fields(cls)]  # type: ignore[arg-type]


def _unknown_keys(d: Mapping[str, object], allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    return sorted(key for key in d.keys() if key not in allowed_set)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_str_dict(value: object, *, path: str) -> SpecBlock:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping (YAML/JSON object).")
    out: SpecBlock = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{path} keys must be non-empty strings.")
        out[key] = item
    return out


__all__ = [
    "EXPERIMENT_SPEC_VERSION",
    "ExperimentSpec",
    "ProblemOverrides",
    "SpecBlock",
    "allowed_override_keys",
    "validate_experiment_spec",
]
