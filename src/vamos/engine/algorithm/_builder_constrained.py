"""Internal builders for the algorithms designed for constrained problems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vamos.engine.algorithm._builder_common import (
    apply_operator_config,
    apply_optional_operator_config,
    finalize_algorithm,
    resolve_problem_encoding,
    resolve_variation_config,
)
from vamos.engine.algorithm.config import AlgorithmConfigProtocol, CCMOConfig, CTAEAConfig, ToPConfig
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

if TYPE_CHECKING:
    from vamos.engine.algorithm.ccmo import CCMO
    from vamos.engine.algorithm.ctaea import CTAEA
    from vamos.engine.algorithm.top import ToP


def build_ccmo_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    ccmo_variation: dict[str, Any] | None,
) -> tuple[CCMO, AlgorithmConfigProtocol]:
    """CCMO with the paper's variation for real variables, SBX(1.0, 20) and polynomial mutation (1/n, 20), unless overridden."""
    encoding = resolve_problem_encoding(problem)
    ccmo_overrides = ccmo_variation or {}
    real_defaults = {
        "crossover": ("sbx", {"prob": 1.0, "eta": 20.0}),
        "mutation": ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
    }
    var_cfg = resolve_variation_config(encoding=encoding, overrides=ccmo_overrides, defaults=real_defaults if encoding == "real" else None)

    builder = CCMOConfig.builder()
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
        "tuple[CCMO, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="ccmo", builder=builder, kernel=kernel, block=ccmo_overrides),
    )


def build_ctaea_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    ctaea_variation: dict[str, Any] | None,
) -> tuple[CTAEA, AlgorithmConfigProtocol]:
    """C-TAEA with the paper's variation for real variables, SBX(1.0, 20) and polynomial mutation (1/n, 20), unless overridden.

    ``pop_size`` is the requested number of weight vectors.
    """
    encoding = resolve_problem_encoding(problem)
    ctaea_overrides = ctaea_variation or {}
    real_defaults = {
        "crossover": ("sbx", {"prob": 1.0, "eta": 20.0}),
        "mutation": ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
    }
    var_cfg = resolve_variation_config(encoding=encoding, overrides=ctaea_overrides, defaults=real_defaults if encoding == "real" else None)

    builder = CTAEAConfig.builder()
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
        "tuple[CTAEA, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="ctaea", builder=builder, kernel=kernel, block=ctaea_overrides),
    )


def build_top_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    top_variation: dict[str, Any] | None,
) -> tuple[ToP, AlgorithmConfigProtocol]:
    """ToP with PlatEMO's variation, DE with CR = 1.0 and F = 0.5 and polynomial mutation (1/n, 20), unless overridden.

    ToP's differential evolution needs real variables; another encoding is rejected.
    """
    encoding = resolve_problem_encoding(problem)
    if encoding != "real":
        raise ValueError(f"ToP's differential evolution needs real variables; the problem's encoding is {encoding!r}.")
    top_overrides = top_variation or {}
    defaults = {
        "crossover": ("de", {"cr": 1.0, "f": 0.5}),
        "mutation": ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
    }
    var_cfg = resolve_variation_config(encoding="real", overrides=top_overrides, defaults=defaults)

    builder = ToPConfig.builder()
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
        "tuple[ToP, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="top", builder=builder, kernel=kernel, block=top_overrides),
    )


__all__ = ["build_ccmo_algorithm", "build_ctaea_algorithm", "build_top_algorithm"]
