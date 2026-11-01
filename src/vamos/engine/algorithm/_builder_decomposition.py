"""Internal builders for decomposition and reference-direction algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vamos.engine.algorithm._builder_common import (
    apply_operator_config,
    apply_optional_operator_config,
    finalize_algorithm,
    resolve_problem_encoding,
    resolve_variation_config,
)
from vamos.engine.algorithm.config import AlgorithmConfigProtocol, MOEADConfig, NSGAIIIConfig, RVEAConfig
from vamos.engine.algorithm.config.rvea import rvea_partitions_for_population
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol
from vamos.resources import weight_path

if TYPE_CHECKING:
    from vamos.engine.algorithm.moead import MOEAD
    from vamos.engine.algorithm.nsgaiii import NSGAIII
    from vamos.engine.algorithm.rvea import RVEA


# MOEA/D fields its builder below reads from the algorithm block.
_MOEAD_BUILDER_FIELDS = ("neighbor_size", "delta", "replace_limit", "batch_size", "weight_vectors", "use_numba_variation")


def build_moead_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    moead_variation: dict[str, Any] | None,
) -> tuple[MOEAD, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    moead_overrides = moead_variation or {}
    real_defaults = {
        "crossover": ("de", {"cr": 1.0, "f": 0.5}),
        "mutation": ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
    }
    var_cfg = resolve_variation_config(encoding=encoding, overrides=moead_overrides, defaults=real_defaults if encoding == "real" else None)
    var_cfg.setdefault("aggregation", ("pbi", {"theta": 5.0}))

    builder = MOEADConfig.builder()
    builder.pop_size(pop_size)
    # Set through the builder, which checks the neighbourhood against the population.
    builder.neighbor_size(moead_overrides.get("neighbor_size", min(20, pop_size)))
    builder.delta(moead_overrides.get("delta", 0.9))
    builder.replace_limit(moead_overrides.get("replace_limit", 2))
    if "batch_size" in moead_overrides:
        builder.batch_size(moead_overrides["batch_size"])
    weight_cfg = moead_overrides.get("weight_vectors")
    if weight_cfg is None:
        builder.weight_vectors(path=str(weight_path("W3D_91.dat").parent))
    elif isinstance(weight_cfg, dict):
        builder.weight_vectors(**weight_cfg)
    else:
        builder.weight_vectors(path=str(weight_cfg))

    apply_operator_config(builder, method_name="crossover", operator_cfg=var_cfg["crossover"], key="crossover")
    apply_operator_config(builder, method_name="mutation", operator_cfg=var_cfg["mutation"], key="mutation")
    apply_optional_operator_config(builder, method_name="aggregation", operator_cfg=var_cfg.get("aggregation"), key="aggregation")
    if "use_numba_variation" in var_cfg:
        builder.use_numba_variation(bool(var_cfg["use_numba_variation"]))
    apply_optional_operator_config(
        builder,
        method_name="repair",
        operator_cfg=var_cfg.get("repair"),
        key="repair",
        encoding=encoding,
    )
    return cast(
        "tuple[MOEAD, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="moead", builder=builder, kernel=kernel, block=moead_overrides, handled=_MOEAD_BUILDER_FIELDS),
    )


def build_nsgaiii_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    nsgaiii_variation: dict[str, Any] | None,
    selection_pressure: int,
) -> tuple[NSGAIII, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    var_cfg = resolve_variation_config(encoding=encoding, overrides=nsgaiii_variation)

    builder = NSGAIIIConfig.builder()
    builder.pop_size(pop_size)
    apply_operator_config(builder, method_name="crossover", operator_cfg=var_cfg["crossover"], key="crossover")
    apply_operator_config(builder, method_name="mutation", operator_cfg=var_cfg["mutation"], key="mutation")
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
        "tuple[NSGAIII, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="nsgaiii", builder=builder, kernel=kernel, block=nsgaiii_variation),
    )


def build_rvea_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    rvea_variation: dict[str, Any] | None,
) -> tuple[RVEA, AlgorithmConfigProtocol]:
    encoding = resolve_problem_encoding(problem)
    rvea_overrides = rvea_variation or {}
    real_defaults = {
        "crossover": ("sbx", {"prob": 0.9, "eta": 30.0}),
        "mutation": ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
    }
    var_cfg = resolve_variation_config(encoding=encoding, overrides=rvea_overrides, defaults=real_defaults if encoding == "real" else None)

    builder = RVEAConfig.builder()
    builder.pop_size(pop_size)
    partitions = rvea_overrides.get("n_partitions")
    # RVEA needs one reference vector per individual: derive the lattice from the population unless it is given.
    builder.n_partitions(int(partitions) if partitions is not None else rvea_partitions_for_population(pop_size, problem.n_obj))
    builder.alpha(float(rvea_overrides.get("alpha", 2.0)))
    builder.adapt_freq(rvea_overrides["adapt_freq"] if "adapt_freq" in rvea_overrides else 0.1)
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
        "tuple[RVEA, AlgorithmConfigProtocol]",
        finalize_algorithm(
            algorithm_name="rvea", builder=builder, kernel=kernel, block=rvea_overrides, handled=("n_partitions", "alpha", "adapt_freq")
        ),
    )


__all__ = [
    "build_moead_algorithm",
    "build_nsgaiii_algorithm",
    "build_rvea_algorithm",
]
