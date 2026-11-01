"""Internal builders for archive-based and population MOEAs."""

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
from vamos.engine.algorithm.config import (
    AlgorithmConfigProtocol,
    IBEAConfig,
    NSGAIIConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

if TYPE_CHECKING:
    from vamos.engine.algorithm.ibea import IBEA
    from vamos.engine.algorithm.nsgaii import NSGAII
    from vamos.engine.algorithm.smsemoa import SMSEMOA
    from vamos.engine.algorithm.spea2 import SPEA2


def _as_int(value: object) -> int:
    return int(cast(int | float | str, value))


def _as_float(value: object) -> float:
    return float(cast(float | int | str, value))


def _apply_external_archive(builder: Any, external_archive: ExternalArchiveConfig | None) -> None:
    if external_archive is not None:
        builder.external_archive(**asdict(external_archive))


def build_nsgaii_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    offspring_size: int,
    selection_pressure: int,
    external_archive: ExternalArchiveConfig | None,
    nsgaii_variation: dict[str, Any] | None,
    track_genealogy: bool,
) -> tuple[NSGAII, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    var_cfg = resolve_variation_config(encoding=encoding, overrides=nsgaii_variation)

    builder = NSGAIIConfig.builder()
    builder.pop_size(pop_size)
    builder.offspring_size(offspring_size)
    builder.result_mode("population")
    apply_optional_operator_config(builder, method_name="crossover", operator_cfg=var_cfg.get("crossover"), key="crossover")
    apply_optional_operator_config(builder, method_name="mutation", operator_cfg=var_cfg.get("mutation"), key="mutation")
    if "selection" in var_cfg:
        apply_operator_config(builder, method_name="selection", operator_cfg=var_cfg["selection"], key="selection")
    else:
        builder.selection("tournament", size=selection_pressure)
    apply_optional_operator_config(
        builder,
        method_name="repair",
        operator_cfg=var_cfg.get("repair"),
        key="repair",
        encoding=encoding,
    )
    _apply_external_archive(builder, external_archive)
    if track_genealogy:
        builder.track_genealogy(True)
    return cast(
        "tuple[NSGAII, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="nsgaii", builder=builder, kernel=kernel, block=nsgaii_variation),
    )


def build_smsemoa_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    smsemoa_variation: dict[str, Any] | None,
) -> tuple[SMSEMOA, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    var_cfg = resolve_variation_config(encoding=encoding, overrides=smsemoa_variation)

    builder = SMSEMOAConfig.builder()
    builder.pop_size(pop_size)
    apply_operator_config(builder, method_name="crossover", operator_cfg=var_cfg["crossover"], key="crossover")
    apply_operator_config(builder, method_name="mutation", operator_cfg=var_cfg["mutation"], key="mutation")
    apply_operator_config(
        builder,
        method_name="selection",
        operator_cfg=var_cfg.get("selection", ("random", {})),
        key="selection",
    )
    apply_optional_operator_config(
        builder,
        method_name="repair",
        operator_cfg=var_cfg.get("repair"),
        key="repair",
        encoding=encoding,
    )
    return cast(
        "tuple[SMSEMOA, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="smsemoa", builder=builder, kernel=kernel, block=smsemoa_variation),
    )


def build_spea2_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    selection_pressure: int,
    external_archive: ExternalArchiveConfig | None,
    spea2_variation: dict[str, Any] | None,
) -> tuple[SPEA2, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    var_cfg = resolve_variation_config(encoding=encoding, overrides=spea2_variation)

    builder = SPEA2Config.builder()
    builder.pop_size(pop_size)
    archive_override = var_cfg.get("archive_size")
    ext_capacity = external_archive.capacity if external_archive is not None and external_archive.capacity is not None else None
    builder.archive_size(_as_int(archive_override) if archive_override is not None else (ext_capacity or pop_size))
    _apply_external_archive(builder, external_archive)
    if "k_neighbors" in var_cfg and var_cfg["k_neighbors"] is not None:
        builder.k_neighbors(_as_int(var_cfg["k_neighbors"]))
    apply_operator_config(
        builder,
        method_name="crossover",
        operator_cfg=var_cfg.get("crossover", ("sbx", {"prob": 1.0, "eta": 20.0})),
        key="crossover",
    )
    apply_operator_config(
        builder,
        method_name="mutation",
        operator_cfg=var_cfg.get("mutation", ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0})),
        key="mutation",
    )
    apply_operator_config(
        builder,
        method_name="selection",
        operator_cfg=var_cfg.get("selection", ("tournament", {"size": selection_pressure})),
        key="selection",
    )
    apply_optional_operator_config(
        builder,
        method_name="repair",
        operator_cfg=var_cfg.get("repair"),
        key="repair",
        encoding=encoding,
    )
    return cast(
        "tuple[SPEA2, AlgorithmConfigProtocol]",
        finalize_algorithm(
            algorithm_name="spea2", builder=builder, kernel=kernel, block=spea2_variation, handled=("archive_size", "k_neighbors")
        ),
    )


def build_ibea_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    selection_pressure: int,
    ibea_variation: dict[str, Any] | None,
) -> tuple[IBEA, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    var_cfg = resolve_variation_config(encoding=encoding, overrides=ibea_variation)

    builder = IBEAConfig.builder()
    builder.pop_size(pop_size)
    apply_operator_config(
        builder,
        method_name="crossover",
        operator_cfg=var_cfg.get("crossover", ("sbx", {"prob": 1.0, "eta": 20.0})),
        key="crossover",
    )
    apply_operator_config(
        builder,
        method_name="mutation",
        operator_cfg=var_cfg.get("mutation", ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0})),
        key="mutation",
    )
    apply_operator_config(
        builder,
        method_name="selection",
        operator_cfg=var_cfg.get("selection", ("tournament", {"size": selection_pressure})),
        key="selection",
    )
    indicator = var_cfg.get("indicator")
    if indicator is None:
        builder.indicator("eps")
    elif isinstance(indicator, str):
        builder.indicator(indicator)
    else:
        raise ValueError("IBEA indicator must be a string.")
    builder.kappa(_as_float(var_cfg.get("kappa", 0.05)))
    apply_optional_operator_config(
        builder,
        method_name="repair",
        operator_cfg=var_cfg.get("repair"),
        key="repair",
        encoding=encoding,
    )
    return cast(
        "tuple[IBEA, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="ibea", builder=builder, kernel=kernel, block=ibea_variation, handled=("indicator", "kappa")),
    )


__all__ = [
    "build_ibea_algorithm",
    "build_nsgaii_algorithm",
    "build_smsemoa_algorithm",
    "build_spea2_algorithm",
]
