from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from vamos.engine.algorithm.factory import build_algorithm
from vamos.engine.config.variation import (
    merge_variation_overrides,
    normalize_operator_tuple,
    normalize_variation_config,
    resolve_default_variation_config,
)
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection


def test_normalize_operator_tuple_accepts_str_tuple_dict():
    assert normalize_operator_tuple("sbx") == ("sbx", {})
    assert normalize_operator_tuple(("polynomial", {"prob": 0.1})) == ("polynomial", {"prob": 0.1})
    assert normalize_operator_tuple({"method": "uniform", "prob": 0.5}) == ("uniform", {"prob": 0.5})
    assert normalize_operator_tuple({"name": "repair", "alpha": 0.2}) == ("repair", {"alpha": 0.2})


def test_normalize_variation_config_handles_ops_and_extras():
    raw = {
        "crossover": {"method": "sbx", "prob": 0.9},
        "mutation": {"name": "polynomial", "eta": 20.0},
        "repair": "auto",
        "kappa": 0.05,
    }
    cfg = normalize_variation_config(raw)
    assert cfg == {
        "crossover": ("sbx", {"prob": 0.9}),
        "mutation": ("polynomial", {"eta": 20.0}),
        "repair": "auto",
        "kappa": 0.05,
    }


def test_merge_variation_overrides_prefers_non_null_override():
    base = {"crossover": ("sbx", {"prob": 0.9}), "kappa": 0.05}
    override = {"kappa": 0.1, "repair": ("clip", {})}
    merged = merge_variation_overrides(base, override)
    assert merged["crossover"] == base["crossover"]
    assert merged["kappa"] == 0.1
    assert merged["repair"] == ("clip", {})


def test_resolve_default_variation_config_defaults_by_encoding():
    real_cfg = resolve_default_variation_config("real", None)
    assert real_cfg["crossover"][0] == "sbx"
    assert real_cfg["mutation"][0] == "polynomial"
    assert real_cfg["repair"] == ("clip", {})

    real_cfg_auto = resolve_default_variation_config("real", {"repair": "auto"})
    assert real_cfg_auto["repair"] == ("clip", {})

    binary_cfg = resolve_default_variation_config("binary", None)
    assert binary_cfg["crossover"][0] == "hux"
    assert binary_cfg["mutation"][0] == "bitflip"
    assert "repair" not in binary_cfg

    perm_cfg = resolve_default_variation_config("permutation", {"repair": ("clip", {})})
    assert perm_cfg["crossover"][0] == "ox"
    assert perm_cfg["repair"] == ("clip", {})


_SBX_ALGORITHMS = ["nsgaii", "spea2", "smsemoa", "ibea", "nsgaiii", "agemoea", "rvea"]


def _build(algorithm: str, variation: dict[str, object] | None = None) -> tuple[Any, Any, Any]:
    problem = make_problem_selection("zdt1", n_var=6).instantiate()
    config = ExperimentConfig(population_size=8, offspring_population_size=8, max_evaluations=16, seed=1)
    variations = None if variation is None else {algorithm: variation}
    built, cfg = build_algorithm(algorithm, "numpy", problem, config, variations=variations)
    return problem, built, cfg


@pytest.mark.parametrize("algorithm", _SBX_ALGORITHMS)
def test_partial_sbx_override_keeps_default_probability(algorithm: str) -> None:
    _, _, default_cfg = _build(algorithm)
    default_name, default_params = default_cfg.to_dict()["crossover"]
    assert default_name == "sbx"
    assert default_params["prob"] == 0.9

    for override in ({"method": "sbx", "eta": 5}, {"method": "sbx", "eta": 5, "prob": None}):
        _, _, cfg = _build(algorithm, {"crossover": override})
        assert cfg.to_dict()["crossover"] == ("sbx", {**default_params, "eta": 5})
    for override in ("sbx", {"method": "sbx"}):
        _, _, cfg = _build(algorithm, {"crossover": override})
        assert cfg.to_dict()["crossover"] == ("sbx", default_params)


def test_partial_sbx_override_reaches_the_live_operator() -> None:
    def live_crossover(variation: dict[str, object] | None) -> Any:
        problem, algorithm, _ = _build("nsgaii", variation)
        algorithm.run(problem, ("max_evaluations", 16), seed=1)
        return algorithm._st.variation.crossover_op.operator

    default = live_crossover(None)
    partial = live_crossover({"crossover": {"method": "sbx", "eta": 5}})
    assert partial.prob == default.prob == 0.9
    assert partial.eta == 5.0


def test_partial_moead_override_keeps_the_other_de_parameters() -> None:
    _, _, cfg = _build("moead", {"crossover": {"method": "de", "f": 0.7}})
    assert cfg.to_dict()["crossover"] == ("de", {"cr": 1.0, "f": 0.7})
    _, _, cfg = _build("moead", {"crossover": {"method": "sbx", "eta": 5}})
    assert cfg.to_dict()["crossover"] == ("sbx", {"eta": 5})


@pytest.mark.parametrize(
    "spec",
    [("sbx", 0.9), ("sbx",), ("sbx", {}, 1), (1, {}), ("", {}), {"prob": 0.2}, {"method": "", "eta": 5}, 0.9],
    ids=repr,
)
def test_normalize_operator_tuple_rejects_scalar_params(spec: object) -> None:
    with pytest.raises(ValueError):
        normalize_operator_tuple(spec)


def test_normalize_operator_tuple_accepts_any_mapping_and_drops_none_parameters() -> None:
    assert normalize_operator_tuple(MappingProxyType({"method": "sbx", "eta": 5})) == ("sbx", {"eta": 5})
    assert normalize_operator_tuple(("sbx", MappingProxyType({"eta": 5, "prob": None}))) == ("sbx", {"eta": 5})
    assert normalize_operator_tuple(None) is None


def test_normalize_variation_config_rejects_operators_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="crossover"):
        normalize_variation_config({"crossover": ["sbx"]})
    with pytest.raises(ValueError, match="mutation"):
        normalize_variation_config({"mutation": {"prob": 0.2}})


def test_resolve_default_variation_config_merges_parameters_of_the_default_operator() -> None:
    cfg = resolve_default_variation_config("real", {"crossover": {"method": "sbx", "eta": 5}, "mutation": "pm"})
    assert cfg["crossover"] == ("sbx", {"prob": 0.9, "eta": 5})
    assert cfg["mutation"] == ("pm", {"prob": "1/n", "eta": 20.0})
    cfg = resolve_default_variation_config("real", {"crossover": {"method": "blx_alpha", "alpha": 0.3}})
    assert cfg["crossover"] == ("blx_alpha", {"alpha": 0.3})
