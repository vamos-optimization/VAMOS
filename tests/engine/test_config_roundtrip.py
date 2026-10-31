"""Tests for config from_dict/to_dict roundtrip for all 9 algorithms."""

from __future__ import annotations

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
