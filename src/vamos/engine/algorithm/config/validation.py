"""Domain checks shared by the typed algorithm configurations."""

from __future__ import annotations

import difflib
import math
from numbers import Integral, Real
from typing import Any, Literal, cast

from vamos.foundation.exceptions import ConfigurationError

_IBEA_EPSILON_NAMES = ("eps", "epsilon", "additive_epsilon")
_RESULT_MODES = ("non_dominated", "population")
OPERATOR_PAIR_FIELDS = ("crossover", "mutation", "selection", "aggregation", "repair")
# Operator parameters that are probabilities and distribution indices.
_PROBABILITY_PARAMS = ("prob", "cr", "CR")
_DISTRIBUTION_INDEX_PARAMS = ("eta",)
_PROB_FACTOR_PARAMS = ("prob_factor", "mutation_prob_factor")


def did_you_mean(value: object, options: tuple[str, ...]) -> str:
    matches = difflib.get_close_matches(str(value), options, n=1, cutoff=0.6)
    return f" Did you mean '{matches[0]}'?" if matches else ""


def normalize_ibea_indicator(value: object) -> Literal["epsilon", "hypervolume"]:
    """Indicator IBEA runs for a configured name, or raise ``ConfigurationError``.

    ``"eps"``, ``"epsilon"`` and ``"additive_epsilon"`` select the additive epsilon
    indicator and ``"hypervolume"`` the hypervolume one, in any letter case.
    """
    name = value.lower() if isinstance(value, str) else None
    if name == "hypervolume":
        return "hypervolume"
    if name in _IBEA_EPSILON_NAMES:
        return "epsilon"
    options = (*_IBEA_EPSILON_NAMES, "hypervolume")
    raise ConfigurationError(
        f"IBEA indicator must be one of {', '.join(options)} (any letter case); got {value!r}.{did_you_mean(name, options)}"
    )


def normalize_ibea_kappa(value: object) -> float:
    """IBEA's fitness scaling factor as a float, or raise ``ConfigurationError`` unless it is finite and positive."""
    _check_finite("kappa", value, minimum=0.0, strict=True)
    return float(cast(float, value))


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _check_positive_int(field: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ConfigurationError(f"{field} must be a positive integer; got {value!r}.")


def _check_finite(field: str, value: object, *, minimum: float, maximum: float = math.inf, strict: bool = False) -> None:
    number = float(cast(float, value)) if _is_real(value) else math.nan
    below = number <= minimum if strict else number < minimum
    if not math.isfinite(number) or below or number > maximum:
        bound = f"> {minimum:g}" if strict else f">= {minimum:g}"
        domain = f"in [{minimum:g}, {maximum:g}]" if math.isfinite(maximum) else f"finite and {bound}"
        raise ConfigurationError(f"{field} must be {domain}; got {value!r}.")


def _check_operator_params(field: str, method: str, params: dict[str, Any]) -> None:
    from vamos.engine.variation.helpers import resolve_prob_expression

    for key in _PROBABILITY_PARAMS:
        if key in params:
            try:
                resolve_prob_expression(params[key], 1, 0.0)
            except ValueError as exc:
                raise ConfigurationError(f"{field} '{method}' parameter {key}: {exc}") from None
    for key in _DISTRIBUTION_INDEX_PARAMS:
        if key in params:
            _check_finite(f"{field} '{method}' parameter {key}", params[key], minimum=0.0)
    for key in _PROB_FACTOR_PARAMS:
        if params.get(key) is not None:
            _check_finite(f"{field} '{method}' parameter {key}", params[key], minimum=0.0, strict=True)
    if field == "selection" and method.strip().lower() == "tournament" and "size" in params:
        _check_positive_int("selection 'tournament' parameter size", params["size"])


def _check_aggregation(method: str) -> None:
    from vamos.engine.algorithm.moead.aggregation import aggregation_methods

    options = aggregation_methods()
    if method.lower() not in options:
        raise ConfigurationError(f"aggregation must be one of {', '.join(options)}; got {method!r}.{did_you_mean(method.lower(), options)}")


# Domain of each scalar field, wherever a config declares it.
_POSITIVE_INT_FIELDS = ("pop_size", "batch_size", "neighbor_size", "replace_limit", "archive_size", "n_partitions")
_OPTIONAL_POSITIVE_INT_FIELDS = ("k_neighbors", "offspring_size", "replacement_size")
_FINITE_FIELDS: dict[str, dict[str, Any]] = {
    "delta": {"minimum": 0.0, "maximum": 1.0},
    "kappa": {"minimum": 0.0, "strict": True},
    "alpha": {"minimum": 0.0},
    "inertia": {"minimum": 0.0},
    "c1": {"minimum": 0.0},
    "c2": {"minimum": 0.0},
    "vmax_fraction": {"minimum": 0.0, "strict": True},
}
_OPTIONAL_FINITE_FIELDS: dict[str, dict[str, Any]] = {
    "mutation_prob_factor": {"minimum": 0.0, "strict": True},
    "adapt_freq": {"minimum": 0.0, "maximum": 1.0},
}


def validate_config_fields(config: Any, fields: dict[str, Any]) -> None:
    """Raise ``ConfigurationError`` for the first field of ``config`` outside its domain."""
    for name in _POSITIVE_INT_FIELDS:
        if name in fields:
            _check_positive_int(name, getattr(config, name))
    for name in _OPTIONAL_POSITIVE_INT_FIELDS:
        if name in fields and getattr(config, name) is not None:
            _check_positive_int(name, getattr(config, name))
    for name, bounds in _FINITE_FIELDS.items():
        if name in fields:
            _check_finite(name, getattr(config, name), **bounds)
    for name, bounds in _OPTIONAL_FINITE_FIELDS.items():
        if name in fields and getattr(config, name) is not None:
            _check_finite(name, getattr(config, name), **bounds)
    if "result_mode" in fields and config.result_mode is not None and config.result_mode not in _RESULT_MODES:
        raise ConfigurationError(f"result_mode must be 'non_dominated' or 'population'; got {config.result_mode!r}.")
    if "indicator" in fields:
        normalize_ibea_indicator(config.indicator)
    for name in OPERATOR_PAIR_FIELDS:
        pair = getattr(config, name, None) if name in fields else None
        if isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str) and isinstance(pair[1], dict):
            if name == "aggregation":
                _check_aggregation(pair[0])
            else:
                _check_operator_params(name, pair[0], pair[1])


def operator_pair(value: object) -> object:
    """``(name, params)`` for a ``[name, params]`` list, as JSON returns operator pairs; other values unchanged."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], dict):
        return (value[0], value[1])
    return value
