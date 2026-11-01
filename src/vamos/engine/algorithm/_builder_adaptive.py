"""Internal builders for adaptive and swarm-style algorithms."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

from vamos.engine.algorithm._builder_common import (
    apply_operator_config,
    apply_optional_operator_config,
    finalize_algorithm,
    resolve_problem_encoding,
    resolve_variation_config,
)
from vamos.engine.algorithm.config import AGEMOEAConfig, AlgorithmConfigProtocol, SMPSOConfig
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.config.variation import merge_variation_overrides
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

if TYPE_CHECKING:
    from vamos.engine.algorithm.agemoea import AGEMOEA
    from vamos.engine.algorithm.smpso import SMPSO


def _as_float(value: object) -> float:
    return float(cast(float | int | str, value))


def _apply_external_archive(builder: Any, external_archive: ExternalArchiveConfig | None) -> None:
    if external_archive is not None:
        builder.external_archive(**asdict(external_archive))


def build_smpso_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    external_archive: ExternalArchiveConfig | None,
    smpso_variation: dict[str, Any] | None,
) -> tuple[SMPSO, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    mut_cfg = merge_variation_overrides(
        {
            "mutation": ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
        },
        smpso_variation,
    )
    builder = SMPSOConfig.builder()
    builder.pop_size(pop_size)
    ext_capacity = external_archive.capacity if external_archive is not None and external_archive.capacity is not None else None
    builder.archive_size(pop_size if ext_capacity is None else ext_capacity)
    _apply_external_archive(builder, external_archive)
    apply_operator_config(
        builder,
        method_name="mutation",
        operator_cfg=mut_cfg.get("mutation", ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0})),
        key="mutation",
    )
    if "inertia" in mut_cfg:
        builder.inertia(_as_float(mut_cfg["inertia"]))
    if "c1" in mut_cfg:
        builder.c1(_as_float(mut_cfg["c1"]))
    if "c2" in mut_cfg:
        builder.c2(_as_float(mut_cfg["c2"]))
    if "vmax_fraction" in mut_cfg:
        builder.vmax_fraction(_as_float(mut_cfg["vmax_fraction"]))
    apply_optional_operator_config(
        builder,
        method_name="repair",
        operator_cfg=mut_cfg.get("repair"),
        key="repair",
        encoding=encoding,
    )
    return cast(
        "tuple[SMPSO, AlgorithmConfigProtocol]",
        finalize_algorithm(
            algorithm_name="smpso",
            builder=builder,
            kernel=kernel,
            block=smpso_variation,
            handled=("inertia", "c1", "c2", "vmax_fraction"),
        ),
    )


def build_agemoea_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    agemoea_variation: dict[str, Any] | None,
) -> tuple[AGEMOEA, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    agemoea_overrides = agemoea_variation or {}
    var_cfg = resolve_variation_config(encoding=encoding, overrides=agemoea_overrides)
    if encoding == "real":
        if "crossover" not in agemoea_overrides:
            var_cfg["crossover"] = ("sbx", {"prob": 0.9, "eta": 15.0})
        if "mutation" not in agemoea_overrides:
            var_cfg["mutation"] = ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0})

    builder = AGEMOEAConfig.builder()
    builder.pop_size(pop_size)
    apply_operator_config(builder, method_name="crossover", operator_cfg=var_cfg["crossover"], key="crossover")
    apply_operator_config(builder, method_name="mutation", operator_cfg=var_cfg["mutation"], key="mutation")
    apply_optional_operator_config(
        builder,
        method_name="repair",
        operator_cfg=var_cfg.get("repair"),
        key="repair",
        encoding=encoding,
    )
    return cast(
        "tuple[AGEMOEA, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="agemoea", builder=builder, kernel=kernel, block=agemoea_overrides),
    )


__all__ = [
    "build_agemoea_algorithm",
    "build_smpso_algorithm",
]
