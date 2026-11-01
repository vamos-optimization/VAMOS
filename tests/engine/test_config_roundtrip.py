"""Tests for config from_dict/to_dict roundtrip for all 9 algorithms."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.foundation.exceptions import ConfigurationError

ALL_CONFIGS = [
    NSGAIIConfig,
    MOEADConfig,
    SPEA2Config,
    SMSEMOAConfig,
    NSGAIIIConfig,
    IBEAConfig,
    SMPSOConfig,
    AGEMOEAConfig,
    RVEAConfig,
]


@pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
class TestConfigRoundtrip:
    def test_default_roundtrip(self, config_cls):
        """from_dict(cfg.to_dict()) should reconstruct an equal config."""
        cfg = config_cls.default()
        d = cfg.to_dict()
        cfg2 = config_cls.from_dict(d)
        assert cfg == cfg2

    def test_roundtrip_preserves_type(self, config_cls):
        """Reconstructed config should be the same class."""
        cfg = config_cls.default()
        cfg2 = config_cls.from_dict(cfg.to_dict())
        assert type(cfg2) is config_cls

    def test_to_dict_returns_plain_dict(self, config_cls):
        """to_dict() should return a plain dict, not a dataclass."""
        cfg = config_cls.default()
        d = cfg.to_dict()
        assert isinstance(d, dict)


@pytest.mark.parametrize("algorithm", ["nsgaii", "moead", "smsemoa", "nsgaiii", "spea2", "ibea", "smpso", "agemoea", "rvea"])
@pytest.mark.parametrize("n_obj", [2, 3])
def test_none_pop_size_is_uniform(algorithm, n_obj):
    from vamos.engine.algorithm.config.defaults import build_default_algorithm_config

    config = build_default_algorithm_config(algorithm, pop_size=None, n_var=10, n_obj=n_obj, encoding="real")

    assert isinstance(config.pop_size, int) and config.pop_size > 0


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_default_accepts_none_pop_size_and_builders_reject_invalid_sizes(config_cls):
    from vamos.foundation.exceptions import ConfigurationError

    config = config_cls.default(pop_size=None)
    assert isinstance(config.pop_size, int) and config.pop_size > 0
    for invalid in (0, -3, 2.5, None, True):
        with pytest.raises(ConfigurationError, match="pop_size"):
            config_cls.builder().pop_size(invalid)


@pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
def test_config_json_roundtrip_is_equal(config_cls):
    """JSON turns operator tuples into lists; from_dict restores an equal config."""
    cfg = config_cls.default()
    assert config_cls.from_dict(json.loads(cfg.to_json())) == cfg


def test_from_dict_rejects_unknown_keys():
    payload = {**NSGAIIConfig.default().to_dict(), "pop_sise": 7}
    with pytest.raises(ConfigurationError, match=r"pop_sise.*pop_size"):
        NSGAIIConfig.from_dict(payload)


def _replaced(config_cls, **changes):
    return lambda: dataclasses.replace(config_cls.default(), **changes)


_OUT_OF_DOMAIN = {
    "sbx_prob_1.7": (_replaced(NSGAIIConfig, crossover=("sbx", {"prob": 1.7, "eta": 20.0})), "prob"),
    "sbx_eta_negative": (_replaced(NSGAIIConfig, crossover=("sbx", {"prob": 0.9, "eta": -5.0})), "eta"),
    "sbx_eta_nan": (_replaced(NSGAIIConfig, crossover=("sbx", {"prob": 0.9, "eta": math.nan})), "eta"),
    "pm_prob_negative": (_replaced(NSGAIIConfig, mutation=("pm", {"prob": -0.2, "eta": 20.0})), "prob"),
    "pm_prob_text": (_replaced(NSGAIIConfig, mutation=("pm", {"prob": "abc", "eta": 20.0})), "prob"),
    "de_cr_above_one": (_replaced(MOEADConfig, crossover=("de", {"cr": 1.5, "f": 0.5})), "cr"),
    "tournament_size_0": (_replaced(NSGAIIConfig, selection=("tournament", {"size": 0})), "size"),
    "result_mode_xyz": (_replaced(NSGAIIConfig, result_mode="xyz"), "result_mode"),
    "live_callback_mode_all": (_replaced(NSGAIIConfig, live_callback_mode="all"), "live_callback_mode"),
    "mutation_prob_factor_0": (_replaced(NSGAIIConfig, mutation_prob_factor=0.0), "mutation_prob_factor"),
    "kappa_0": (
        lambda: IBEAConfig.builder().pop_size(40).crossover("sbx").mutation("pm").selection("tournament").indicator("eps").kappa(0).build(),
        "kappa",
    ),
    "kappa_negative": (_replaced(IBEAConfig, kappa=-1.0), "kappa"),
    "kappa_inf": (_replaced(IBEAConfig, kappa=math.inf), "kappa"),
    "delta_1.5": (_replaced(MOEADConfig, delta=1.5), "delta"),
    "replace_limit_0": (_replaced(MOEADConfig, replace_limit=0), "replace_limit"),
    "batch_size_0": (_replaced(MOEADConfig, batch_size=0), "batch_size"),
    "aggregation_foo": (_replaced(MOEADConfig, aggregation=("foo", {})), "aggregation"),
    "neighbor_size_500": (
        lambda: (
            MOEADConfig.builder()
            .pop_size(100)
            .neighbor_size(500)
            .delta(0.9)
            .replace_limit(2)
            .crossover("de", cr=1.0, f=0.5)
            .mutation("pm", prob="1/n", eta=20.0)
            .aggregation("pbi", theta=5.0)
            .build()
        ),
        "neighbor_size",
    ),
    "smpso_archive_size_0": (_replaced(SMPSOConfig, archive_size=0), "archive_size"),
    "smpso_inertia_negative": (_replaced(SMPSOConfig, inertia=-3.0), "inertia"),
    "spea2_archive_size_negative": (_replaced(SPEA2Config, archive_size=-1), "archive_size"),
    "spea2_k_neighbors_0": (_replaced(SPEA2Config, k_neighbors=0), "k_neighbors"),
    "rvea_alpha_nan": (_replaced(RVEAConfig, alpha=math.nan), "alpha"),
}


@pytest.mark.parametrize("case", list(_OUT_OF_DOMAIN))
def test_builders_reject_out_of_domain_parameters(case):
    build, field = _OUT_OF_DOMAIN[case]
    with pytest.raises(ConfigurationError, match=field):
        build()


def test_valid_probability_expressions_are_accepted():
    for prob in ("1/n", "1/N", " 2 / n ", 0.0, 1.0, "0.5"):
        cfg = dataclasses.replace(NSGAIIConfig.default(), mutation=("pm", {"prob": prob, "eta": 20.0}))
        assert cfg.mutation[1]["prob"] == prob


@pytest.mark.parametrize("indicator", ["hv", "HV", "foo", " hypervolume"])
def test_ibea_rejects_unknown_indicator(indicator):
    builder = IBEAConfig.builder().pop_size(40).crossover("sbx").mutation("pm").selection("tournament").kappa(0.05)
    with pytest.raises(ConfigurationError, match=r"indicator.*hypervolume"):
        builder.indicator(indicator).build()


@pytest.mark.parametrize("indicator", ["eps", "EPS", "epsilon", "additive_epsilon", "hypervolume", "Hypervolume"])
def test_ibea_accepts_its_indicator_names(indicator):
    assert dataclasses.replace(IBEAConfig.default(), indicator=indicator).indicator == indicator


def test_ibea_run_rejects_unknown_indicator_before_evaluating():
    from vamos.engine.algorithm.ibea import IBEA
    from vamos.foundation.kernel.numpy_backend import NumPyKernel
    from vamos.foundation.problem.zdt1 import ZDT1Problem

    class CountingZDT1(ZDT1Problem):
        calls = 0

        def evaluate(self, X, out):
            CountingZDT1.calls += 1
            super().evaluate(X, out)

    cfg = {**IBEAConfig.default(pop_size=10).to_dict(), "indicator": "hv"}
    with pytest.raises(ConfigurationError, match="indicator"):
        IBEA(cfg, kernel=NumPyKernel()).run(CountingZDT1(n_var=6), termination=("max_evaluations", 40), seed=1)
    assert CountingZDT1.calls == 0


def test_moead_default_neighbourhood_fits_small_populations():
    assert MOEADConfig.default(pop_size=8).neighbor_size == 8
    assert MOEADConfig.default(pop_size=100).neighbor_size == 20


def test_moead_tuning_bridge_keeps_the_neighbourhood_within_the_population():
    from vamos.engine.tuning.racing._bridge_assignment_builders import build_moead_config

    assignment = {
        "pop_size": 20,
        "neighbor_size": 40,
        "delta": 0.9,
        "replace_limit": 2,
        "crossover": "de",
        "mutation": "pm",
        "mutation_prob": 0.1,
        "aggregation": "tchebycheff",
    }
    assert build_moead_config(assignment).neighbor_size == 20
