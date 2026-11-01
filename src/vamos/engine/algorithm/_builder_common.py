"""Internal helpers for algorithm builder assembly."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

from vamos.engine.algorithm.config import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import resolve_algorithm
from vamos.engine.config.variation import ensure_operator_tuple, resolve_default_variation_config
from vamos.engine.variation.helpers import ensure_supported_repair_name
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol


def resolve_problem_encoding(problem: ProblemProtocol) -> EncodingLike:
    return normalize_encoding(getattr(problem, "encoding", "real"))


def resolve_variation_config(
    *,
    encoding: EncodingLike,
    overrides: dict[str, Any] | None,
    extra_exclude: set[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The variation an algorithm runs with: its ``defaults`` (or the encoding's) merged with ``overrides``.

    An operator override that names the default operator keeps the default parameters it does not
    set; see :func:`resolve_default_variation_config`.
    """
    normalized_overrides = dict(overrides or {})
    for key in ("crossover", "mutation", "selection", "repair", "aggregation"):
        value = normalized_overrides.get(key)
        if isinstance(value, (tuple, list)):
            # The list form ([name, {params}]) that ablations use is the same operator as {method: name, **params}.
            name, params = ensure_operator_tuple(value, key=key)
            normalized_overrides[key] = {"method": name, **params}
    var_cfg = resolve_default_variation_config(encoding, normalized_overrides, defaults=defaults)
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


# Operators, which the builders apply themselves through ``apply_operator_config``.
_OPERATOR_KEYS = ("crossover", "mutation", "selection", "repair", "aggregation")


def apply_block_fields(
    config: AlgorithmConfigProtocol, block: Mapping[str, Any] | None, *, handled: tuple[str, ...] = ()
) -> AlgorithmConfigProtocol:
    """``config`` with the other fields of its dataclass that an algorithm block sets.

    The builders apply the operators and the ``handled`` fields themselves. The result goes
    through ``from_dict``, which validates every field.
    """
    names = {field.name for field in dataclasses.fields(cast(Any, config))}
    fields = {
        key: value
        for key, value in (block or {}).items()
        if key in names and key not in _OPERATOR_KEYS and key not in handled and value is not None
    }
    if not fields:
        return config
    try:
        return cast(AlgorithmConfigProtocol, cast(Any, type(config)).from_dict({**config.to_dict(), **fields}))
    except ConfigurationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid algorithm field(s) {', '.join(sorted(fields))}: {exc}") from exc


def finalize_algorithm(
    *,
    algorithm_name: str,
    builder: Any,
    kernel: KernelBackend,
    block: Mapping[str, Any] | None = None,
    handled: tuple[str, ...] = (),
) -> tuple[Any, AlgorithmConfigProtocol]:
    cfg_data = apply_block_fields(cast(AlgorithmConfigProtocol, builder.build()), block, handled=handled)
    algo_ctor = resolve_algorithm(algorithm_name)
    return algo_ctor(cfg_data.to_dict(), kernel), cfg_data


__all__ = [
    "apply_block_fields",
    "apply_operator_config",
    "apply_optional_operator_config",
    "finalize_algorithm",
    "resolve_problem_encoding",
    "resolve_variation_config",
]
