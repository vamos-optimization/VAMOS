from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, TypedDict, cast

from vamos.engine.archive import ExternalArchiveConfig, PrunePolicy
from vamos.engine.archive.config import check_external_archive_field
from vamos.engine.hooks.hv_convergence import HVConvergenceConfig, check_hv_convergence_field, coerce_hv_convergence_field

_ARCHIVE_KEYS = {
    "capacity",
    "truncate_size",
    "pruning",
    "hv_ref_point",
    "rng_seed",
    "objective_tolerance",
    "deduplicate_in",
    "decision_tolerance",
}
# The hook archive keeps objective vectors only: there is no decision vector to deduplicate on.
HOOK_ARCHIVE_DEDUPLICATE_IN: Final = "objective"
_DEFAULT_HOOK_ARCHIVE_CAPACITY = 200
# The convergence monitor measures a two-objective hypervolume.
HOOK_REFERENCE_POINT_LENGTH: Final = 2


class StoppingArchiveConfig(TypedDict):
    stopping_enabled: bool
    stop_cfg: HVConvergenceConfig
    archive_enabled: bool
    archive_cfg: ExternalArchiveConfig
    hv_ref_point: list[float] | None


def build_hv_stop_cfg(d: Mapping[str, object] | None) -> HVConvergenceConfig:
    """Build the convergence settings of a ``stopping.hv_convergence`` block (without ``enabled`` and ``ref_point``)."""
    data = dict(HVConvergenceConfig().__dict__)
    for key, raw in (d or {}).items():
        if key not in data:
            raise TypeError(f"Unknown stopping.hv_convergence field: {key}.")
        value = coerce_hv_convergence_field(key, raw)
        check_hv_convergence_field(key, value)
        data[key] = value
    return HVConvergenceConfig(**data)


def check_hook_reference_point(value: object) -> list[float] | None:
    """Return the reference point of ``stopping.hv_convergence.ref_point``, or ``None`` for ``auto``.

    It is ``"auto"``, ``null`` or a list of two finite numbers, because the
    convergence monitor measures a two-objective hypervolume.
    """
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return None
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ValueError(f"ref_point must be 'auto' or a list of numbers, got {value!r}.")
    point = [float(item) for item in value]
    if len(point) != HOOK_REFERENCE_POINT_LENGTH:
        raise ValueError(
            f"ref_point must have {HOOK_REFERENCE_POINT_LENGTH} values, one per objective of the two-objective "
            f"HV convergence monitor; got {len(point)}."
        )
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"ref_point must hold finite numbers, got {point!r}.")
    return point


def check_hook_archive_field(name: str, value: object) -> None:
    """Raise when ``value`` is not valid for the field ``name`` of an ``archive.external`` block."""
    if name == "enabled":
        if not isinstance(value, bool):
            raise TypeError(f"enabled must be true or false, got {value!r}.")
        return
    if name not in _ARCHIVE_KEYS:
        raise TypeError(f"Unknown archive.external field: {name}.")
    check_external_archive_field(name, value)
    if name == "deduplicate_in" and value != HOOK_ARCHIVE_DEDUPLICATE_IN:
        raise ValueError(
            f"deduplicate_in must be {HOOK_ARCHIVE_DEDUPLICATE_IN!r} for the hook archive, which stores objective "
            f"vectors only; got {value!r}."
        )


def build_archive_cfg(d: Mapping[str, object] | None) -> ExternalArchiveConfig:
    """Build the hook archive of an ``archive.external`` block (without ``enabled``).

    ``capacity`` defaults to 200; ``capacity: null`` keeps every non-dominated vector.
    """
    if d is None:
        return ExternalArchiveConfig(capacity=_DEFAULT_HOOK_ARCHIVE_CAPACITY)
    data = dict(d)
    unknown = sorted(set(data) - _ARCHIVE_KEYS)
    if unknown:
        joined = ", ".join(unknown)
        raise TypeError(f"Unknown archive.external fields: {joined}.")
    for key, value in data.items():
        check_hook_archive_field(key, value)

    return ExternalArchiveConfig(
        capacity=cast("int | None", data.get("capacity", _DEFAULT_HOOK_ARCHIVE_CAPACITY)),
        truncate_size=cast("int | None", data.get("truncate_size")),
        pruning=cast(PrunePolicy, data.get("pruning", "crowding")),
        hv_ref_point=cast("list[float] | None", data.get("hv_ref_point")),
        rng_seed=cast(int, data.get("rng_seed", 0)),
        objective_tolerance=cast(float, data.get("objective_tolerance", 0.0)),
        deduplicate_in=HOOK_ARCHIVE_DEDUPLICATE_IN,
        decision_tolerance=cast(float, data.get("decision_tolerance", 1e-32)),
    )


def merge_spec_blocks(base: Mapping[str, object], override: object) -> dict[str, object]:
    """Merge ``override`` into ``base`` key by key, recursing into mappings present in both.

    A value that is not a mapping, ``null`` included, replaces the inherited one.
    """
    merged = dict(base)
    if not isinstance(override, Mapping):
        return merged
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_spec_blocks(current, value)
        else:
            merged[key] = value
    return merged


def _problem_entry(problems: Mapping[str, object], problem_key: str) -> object:
    """The ``problems`` entry of ``problem_key``; problem keys are case-insensitive, like the registry."""
    wanted = problem_key.lower()
    for key, value in problems.items():
        if isinstance(key, str) and key.lower() == wanted:
            return value
    return None


def _extract_block(spec: Mapping[str, object], key: str, problem_key: str | None) -> dict[str, object]:
    """The ``key`` block in effect for one problem: ``defaults``, then the top level, then ``problems.<key>``."""
    block: dict[str, object] = {}
    defaults = spec.get("defaults")
    if isinstance(defaults, Mapping):
        block = merge_spec_blocks(block, defaults.get(key))
    block = merge_spec_blocks(block, spec.get(key))
    if problem_key:
        problems = spec.get("problems")
        if isinstance(problems, Mapping):
            p_cfg = _problem_entry(problems, problem_key)
            if isinstance(p_cfg, Mapping):
                block = merge_spec_blocks(block, p_cfg.get(key))
    return block


def parse_stopping_archive(spec: Mapping[str, object] | None, problem_key: str | None = None) -> StoppingArchiveConfig:
    """
    Reads an experiment spec dict and returns:
      stopping_enabled, stop_cfg, archive_enabled, archive_cfg, hv_ref_point

    Each block merges key by key: ``defaults``, then the top-level block, then
    ``problems.<problem_key>``. Invalid values raise ``TypeError`` or ``ValueError``.
    """
    if not isinstance(spec, Mapping):
        spec = {}

    stopping = _extract_block(spec, "stopping", problem_key)
    archive = _extract_block(spec, "archive", problem_key)

    hv_raw = stopping.get("hv_convergence")
    hv_block = hv_raw if isinstance(hv_raw, Mapping) else {}
    stop_enabled = hv_block.get("enabled", False)
    if not isinstance(stop_enabled, bool):
        raise TypeError(f"stopping.hv_convergence.enabled must be true or false, got {stop_enabled!r}.")
    hv_ref_point = check_hook_reference_point(hv_block.get("ref_point"))
    stop_cfg = build_hv_stop_cfg({k: v for k, v in hv_block.items() if k not in ("enabled", "ref_point")})

    arch_raw = archive.get("external")
    arch_block = arch_raw if isinstance(arch_raw, Mapping) else {}
    arch_enabled = arch_block.get("enabled", False)
    if not isinstance(arch_enabled, bool):
        raise TypeError(f"archive.external.enabled must be true or false, got {arch_enabled!r}.")
    arch_cfg = build_archive_cfg({k: v for k, v in arch_block.items() if k != "enabled"})

    return {
        "stopping_enabled": stop_enabled,
        "stop_cfg": stop_cfg,
        "archive_enabled": arch_enabled,
        "archive_cfg": arch_cfg,
        "hv_ref_point": hv_ref_point,
    }


__all__ = [
    "StoppingArchiveConfig",
    "build_archive_cfg",
    "check_hook_archive_field",
    "check_hook_reference_point",
    "merge_spec_blocks",
    "parse_stopping_archive",
]
