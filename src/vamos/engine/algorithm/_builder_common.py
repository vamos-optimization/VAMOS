"""Internal helpers for algorithm builder assembly."""

from __future__ import annotations

from typing import Any, cast

from vamos.engine.algorithm.config import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import resolve_algorithm
from vamos.engine.config.variation import ensure_operator_tuple, normalize_operator_tuple, resolve_default_variation_config
from vamos.engine.variation.helpers import ensure_supported_repair_name
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol


def resolve_problem_encoding(problem: ProblemProtocol) -> EncodingLike:
    return normalize_encoding(getattr(problem, "encoding", "real"))


def resolve_variation_config(
    *,
    encoding: EncodingLike,
    overrides: dict[str, Any] | None,
    extra_exclude: set[str] | None = None,
) -> dict[str, Any]:
    normalized_overrides = dict(overrides or {})
    for key in ("crossover", "mutation", "selection", "repair", "aggregation"):
        value = normalized_overrides.get(key)
        if isinstance(value, (tuple, list)):
            # The list form ([name, {params}]) that ablations use is the same operator as {method: name, **params}.
            operator = normalize_operator_tuple(value)
            if operator is None:
                raise TypeError(f"Variation '{key}' must be a mapping with a 'method' field or a [name, params] pair.")
            name, params = operator
            normalized_overrides[key] = {"method": name, **params}
    var_cfg = resolve_default_variation_config(encoding, normalized_overrides)
    excluded = extra_exclude or set()
    for key, value in normalized_overrides.items():
        if key not in var_cfg and key not in excluded:
            var_cfg[key] = value
    return var_cfg


def apply_operator_config(
    builder: Any,
    *,
    method_name: str,
    operator_cfg: Any,
    key: str,
    encoding: EncodingLike | None = None,
) -> None:
    operator_name, operator_kwargs = ensure_operator_tuple(operator_cfg, key=key)
    if key == "repair":
        if encoding is None:
            raise ValueError("Repair operator wiring requires an encoding.")
        operator_name = ensure_supported_repair_name(encoding, operator_name)
    getattr(builder, method_name)(operator_name, **operator_kwargs)


def apply_optional_operator_config(
    builder: Any,
    *,
    method_name: str,
    operator_cfg: Any | None,
    key: str,
    encoding: EncodingLike | None = None,
) -> None:
    if operator_cfg is None:
        return
    apply_operator_config(
        builder,
        method_name=method_name,
        operator_cfg=operator_cfg,
        key=key,
        encoding=encoding,
    )


def finalize_algorithm(
    *,
    algorithm_name: str,
    builder: Any,
    kernel: KernelBackend,
) -> tuple[Any, AlgorithmConfigProtocol]:
    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm(algorithm_name)
    return algo_ctor(cfg_data.to_dict(), kernel), cfg_data


__all__ = [
    "apply_operator_config",
    "apply_optional_operator_config",
    "finalize_algorithm",
    "resolve_problem_encoding",
    "resolve_variation_config",
]
