from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast

from vamos.engine.archive import DeduplicateIn, ExternalArchiveConfig, PrunePolicy
from vamos.engine.hooks.hv_convergence import HVConvergenceConfig

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


class StoppingArchiveConfig(TypedDict):
    stopping_enabled: bool
    stop_cfg: HVConvergenceConfig
    archive_enabled: bool
    archive_cfg: ExternalArchiveConfig
    hv_ref_point: list[float] | None


def build_hv_stop_cfg(d: Mapping[str, object] | None) -> HVConvergenceConfig:
    base = HVConvergenceConfig()
    data = dict(base.__dict__)
    data.update({k: v for k, v in (d or {}).items() if k in data})
    return HVConvergenceConfig(**data)


def build_archive_cfg(d: Mapping[str, object] | None) -> ExternalArchiveConfig:
    if d is None:
        return ExternalArchiveConfig(capacity=200)
    data = dict(d)
    unknown = sorted(set(data) - _ARCHIVE_KEYS)
    if unknown:
        joined = ", ".join(unknown)
        raise TypeError(f"Unknown archive.external fields: {joined}.")

    capacity = data.get("capacity", 200)
    if not isinstance(capacity, int):
        raise TypeError("archive.capacity must be an integer.")

    truncate_size = data.get("truncate_size")
    if truncate_size is not None and not isinstance(truncate_size, int):
        raise TypeError("archive.truncate_size must be an integer or null.")

    rng_seed = data.get("rng_seed", 0)
    if not isinstance(rng_seed, int):
        raise TypeError("archive.rng_seed must be an integer.")

    hv_ref_point_raw = data.get("hv_ref_point")
    hv_ref_point: list[float] | None
    if hv_ref_point_raw is None:
        hv_ref_point = None
    elif isinstance(hv_ref_point_raw, list):
        hv_ref_point = [float(value) for value in hv_ref_point_raw]
    else:
        raise TypeError("archive.hv_ref_point must be a list of floats or null.")

    pruning = data.get("pruning", "crowding")
    if not isinstance(pruning, str):
        raise TypeError("archive.pruning must be a string.")

    objective_tolerance = data.get("objective_tolerance", 0.0)
    if not isinstance(objective_tolerance, (int, float)):
        raise TypeError("archive.objective_tolerance must be numeric.")

    decision_tolerance = data.get("decision_tolerance", 1e-32)
    if not isinstance(decision_tolerance, (int, float)):
        raise TypeError("archive.decision_tolerance must be numeric.")

    deduplicate_in = data.get("deduplicate_in", "objective")
    if not isinstance(deduplicate_in, str):
        raise TypeError("archive.deduplicate_in must be a string.")

    return ExternalArchiveConfig(
        capacity=capacity,
        truncate_size=truncate_size,
        pruning=cast(PrunePolicy, pruning),
        hv_ref_point=hv_ref_point,
        rng_seed=rng_seed,
        objective_tolerance=float(objective_tolerance),
        deduplicate_in=cast(DeduplicateIn, deduplicate_in),
        decision_tolerance=float(decision_tolerance),
    )


def _extract_block(spec: Mapping[str, object], key: str, problem_key: str | None) -> dict[str, object]:
    block: dict[str, object] = {}
    value = spec.get(key)
    if isinstance(value, Mapping):
        block = dict(value)
    defaults = spec.get("defaults")
    if not block and isinstance(defaults, Mapping):
        defaults_block = defaults.get(key)
        if isinstance(defaults_block, Mapping):
            block = dict(defaults_block)
    if problem_key:
        problems = spec.get("problems")
        if isinstance(problems, Mapping):
            p_cfg = problems.get(problem_key)
            if isinstance(p_cfg, Mapping):
                override_block = p_cfg.get(key)
                if isinstance(override_block, Mapping):
                    override = dict(override_block)
                    merged = dict(block)
                    merged.update(override)
                    block = merged
    return block


def parse_stopping_archive(spec: Mapping[str, object] | None, problem_key: str | None = None) -> StoppingArchiveConfig:
    """
    Reads an experiment spec dict and returns:
      stopping_enabled, stop_cfg, archive_enabled, archive_cfg, hv_ref_point
    """
    if not isinstance(spec, Mapping):
        spec = {}

    stopping = _extract_block(spec, "stopping", problem_key)
    archive = _extract_block(spec, "archive", problem_key)

    hv_raw = stopping.get("hv_convergence") if isinstance(stopping, Mapping) else None
    hv_block = hv_raw if isinstance(hv_raw, Mapping) else {}
    stop_enabled = bool(hv_block.get("enabled", False))
    hv_ref_point = hv_block.get("ref_point")
    if isinstance(hv_ref_point, str) and hv_ref_point.lower() == "auto":
        hv_ref_point = None
    stop_cfg = build_hv_stop_cfg({k: v for k, v in hv_block.items() if k not in ("enabled", "ref_point")})

    arch_raw = archive.get("external") if isinstance(archive, Mapping) else None
    arch_block = arch_raw if isinstance(arch_raw, Mapping) else {}
    arch_enabled = bool(arch_block.get("enabled", False))
    arch_cfg = build_archive_cfg({k: v for k, v in arch_block.items() if k != "enabled"})

    return {
        "stopping_enabled": stop_enabled,
        "stop_cfg": stop_cfg,
        "archive_enabled": arch_enabled,
        "archive_cfg": arch_cfg,
        "hv_ref_point": hv_ref_point,
    }


__all__ = ["StoppingArchiveConfig", "build_archive_cfg", "parse_stopping_archive"]
