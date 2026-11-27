"""The settings and the variation of a PPS run, from its configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors, vector_method
from vamos.engine.config.variation import ensure_operator_tuple, resolve_default_variation_config
from vamos.engine.variation.helpers import ensure_supported_operator_names, ensure_supported_repair_name
from vamos.engine.variation.pipeline import VariationPipeline
from vamos.engine.variation.protocol import RepairConfigValue
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.kernel.backend import VariationKernels
from vamos.foundation.problem.types import ProblemProtocol

from .stages import control_generation

#: The paper's population size N (section 4.1), the default of :class:`~vamos.engine.algorithm.config.PPSConfig`.
PAPER_POP_SIZE = 300

#: The paper's settings (section 4.1), the defaults of :class:`~vamos.engine.algorithm.config.PPSConfig`:
#: T, delta, n_r, the epsilon of Eq. 4, l, alpha, tau, cp, and Tc = 800 of 1000 generations.
PAPER_SETTINGS: dict[str, float] = {
    "neighbor_size": 30,
    "delta": 0.9,
    "replace_limit": 2,
    "switch_threshold": 1e-3,
    "switch_window": 20,
    "feasibility_ratio_threshold": 0.95,
    "tao": 0.1,
    "cp": 2.0,
    "decay_fraction": 0.8,
}
_DE_DEFAULTS = {"cr": 1.0, "f": 0.5}


@dataclass(frozen=True)
class PPSSettings:
    """The parameters of a PPS run, validated; ``tc`` is the control generation ``Tc`` of its budget."""

    neighbor_size: int
    delta: float
    replace_limit: int
    switch_threshold: float
    switch_window: int
    feasibility_ratio_threshold: float
    tao: float
    cp: float
    tc: int


def _number(config: Mapping[str, Any], key: str, *, low: float, high: float, low_closed: bool, high_closed: bool) -> float:
    """A parameter of the configuration, the paper's value by default, finite and within its interval."""
    raw = config.get(key)
    value = float(PAPER_SETTINGS[key] if raw is None else raw)
    above = value >= low if low_closed else value > low
    below = value <= high if high_closed else value < high
    if not (math.isfinite(value) and above and below):
        interval = f"{'[' if low_closed else '('}{low:g}, {high:g}{']' if high_closed else ')'}"
        raise ValueError(f"PPS's {key} must be in {interval}, got {raw!r}.")
    return value


def _integer(raw: object, key: str, *, low: int, high: int | None = None) -> int:
    """An integer parameter in ``[low, high]``."""
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)) or raw < low or (high is not None and raw > high):
        bound = f"in [{low}, {high}]" if high is not None else f">= {low}"
        raise ValueError(f"PPS's {key} must be an integer {bound}, got {raw!r}.")
    return int(raw)


def pop_size_of(config: Mapping[str, Any]) -> int:
    """The configuration's ``pop_size``, at least 2; the paper's 300 when it has none."""
    raw = config.get("pop_size")
    return PAPER_POP_SIZE if raw is None else _integer(raw, "pop_size", low=2)


def resolve_settings(config: Mapping[str, Any], pop_size: int, max_evaluations: int) -> PPSSettings:
    """The settings of a run: the configuration's, the paper's by default.

    ``neighbor_size`` defaults to the paper's 30, or the population when it is smaller, and must
    lie in ``[2, pop_size]``, as DE draws two distinct solutions from it.
    """
    raw_neighbors = config.get("neighbor_size")
    neighbor_size = (
        min(int(PAPER_SETTINGS["neighbor_size"]), pop_size)
        if raw_neighbors is None
        else _integer(raw_neighbors, "neighbor_size", low=2, high=pop_size)
    )
    window = config.get("switch_window")
    limit = config.get("replace_limit")
    decay_fraction = _number(config, "decay_fraction", low=0.0, high=1.0, low_closed=False, high_closed=True)
    return PPSSettings(
        neighbor_size=neighbor_size,
        delta=_number(config, "delta", low=0.0, high=1.0, low_closed=True, high_closed=True),
        replace_limit=_integer(int(PAPER_SETTINGS["replace_limit"]) if limit is None else limit, "replace_limit", low=1),
        switch_threshold=_number(config, "switch_threshold", low=0.0, high=math.inf, low_closed=True, high_closed=False),
        switch_window=_integer(int(PAPER_SETTINGS["switch_window"]) if window is None else window, "switch_window", low=1),
        feasibility_ratio_threshold=_number(
            config, "feasibility_ratio_threshold", low=0.0, high=math.inf, low_closed=True, high_closed=False
        ),
        tao=_number(config, "tao", low=0.0, high=1.0, low_closed=False, high_closed=False),
        cp=_number(config, "cp", low=0.0, high=math.inf, low_closed=False, high_closed=False),
        tc=control_generation(decay_fraction, max_evaluations, pop_size),
    )


def differential_evolution_parameters(params: Mapping[str, Any]) -> tuple[float, float]:
    """CR and F from ``cr`` and ``f`` (or ``CR`` and ``F``, not both); 1.0 and 0.5 by default. CR must be in [0, 1] and F positive and finite."""
    unknown = sorted(set(params) - {"cr", "CR", "f", "F"})
    if unknown:
        raise ValueError(f"PPS's differential evolution takes the parameters cr and f; got {', '.join(unknown)}.")
    for name in ("cr", "f"):
        if name in params and name.upper() in params:
            raise ValueError(f"PPS's differential evolution takes {name} or {name.upper()}, not both.")
    cr = float(params.get("cr", params.get("CR", _DE_DEFAULTS["cr"])))
    f = float(params.get("f", params.get("F", _DE_DEFAULTS["f"])))
    if not 0.0 <= cr <= 1.0:
        raise ValueError(f"PPS's crossover rate cr must be in [0, 1], got {cr}.")
    if not (math.isfinite(f) and f > 0.0):
        raise ValueError(f"PPS's scale factor f must be positive and finite, got {f}.")
    return cr, f


def build_pps_variation(
    config: Mapping[str, Any],
    encoding: EncodingLike,
    xl: Any,
    xu: Any,
    problem: ProblemProtocol,
    variation_kernels: VariationKernels | None = None,
) -> VariationPipeline:
    """The variation of a PPS configuration: DE/rand/1/bin from the subproblem's solution, then a mutation.

    PPS's variation is differential evolution for real variables: the crossover must be ``de``,
    whose ``cr`` and ``f`` are CR and F (1.0 and 0.5). VAMOS's ``de`` operator creates the mutant
    ``x_i + F (x_r2 - x_r3)`` and its binomial crossover with ``x_i``, then the repair (``clip``
    by default) applies, then the mutation (any real-coded one; polynomial by default) and the
    repair again.
    """
    if normalize_encoding(encoding) != "real":
        raise ValueError(f"PPS's differential evolution needs real variables; the problem's encoding is {encoding!r}.")
    overrides: dict[str, Any] = {key: config[key] for key in ("crossover", "mutation") if key in config}
    if "repair" in config and config["repair"] != "auto":
        overrides["repair"] = config["repair"]
    var_cfg = {**resolve_default_variation_config("real", None), "crossover": ("de", dict(_DE_DEFAULTS)), **overrides}
    c_name, c_params = ensure_operator_tuple(var_cfg.get("crossover", ("de", {})), key="crossover")
    if c_name.strip().lower() != "de":
        raise ValueError(f"PPS's variation is differential evolution: its crossover must be 'de', got {c_name!r}.")
    cr, f = differential_evolution_parameters(dict(c_params))
    m_name, m_params = ensure_operator_tuple(var_cfg.get("mutation", ("polynomial", {})), key="mutation")
    cross_name, mut_name = ensure_supported_operator_names(encoding, "de", m_name)
    repair_cfg: RepairConfigValue = "auto"
    if "repair" in var_cfg:
        repair_name, repair_params = ensure_operator_tuple(var_cfg["repair"], key="repair")
        repair_cfg = (ensure_supported_repair_name(encoding, repair_name), repair_params)
    return VariationPipeline(
        encoding=encoding,
        cross_method=cross_name,
        cross_params={"CR": cr, "F": f},
        mut_method=mut_name,
        mut_params=m_params,
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=repair_cfg,
        problem=problem,
        variation_kernels=variation_kernels,
    )


def weight_vectors(config: Mapping[str, Any], pop_size: int, n_obj: int) -> np.ndarray[Any, Any]:
    """One weight vector per subproblem, as MOEA/D's (see :class:`~vamos.engine.algorithm.config.MOEADConfig`)."""
    source = config.get("weight_vectors") or {}
    weights = load_or_generate_weight_vectors(
        pop_size,
        n_obj,
        path=source.get("path"),
        divisions=source.get("divisions"),
        mode="jmetalpy",
        inner_divisions=source.get("inner_divisions"),
        sha256=source.get("sha256"),
        **vector_method(source),
    )
    if weights.shape[0] != pop_size:
        raise ValueError(f"PPS needs one weight vector per subproblem: {weights.shape[0]} vectors for pop_size={pop_size}.")
    return weights


__all__ = [
    "PAPER_POP_SIZE",
    "PAPER_SETTINGS",
    "PPSSettings",
    "build_pps_variation",
    "differential_evolution_parameters",
    "pop_size_of",
    "resolve_settings",
    "weight_vectors",
]
