"""Tests for config shortcut methods (default)."""

from __future__ import annotations

import warnings
from math import comb

import pytest

from vamos.algorithms import IBEAConfig, MOEADConfig, NSGAIIConfig, NSGAIIIConfig, SMSEMOAConfig, SPEA2Config
from vamos.engine.operators.impl.registry import get_operator_registry


class TestNSGAIIConfigShortcuts:
    """Test NSGAIIConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = NSGAIIConfig.default()

        assert cfg.pop_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.mutation[0] == "pm"
        assert cfg.selection[0] == "tournament"
        assert cfg.repair == "auto"

    def test_default_with_custom_pop_size(self):
        """default() should accept custom pop_size."""
        cfg = NSGAIIConfig.default(pop_size=50)
        assert cfg.pop_size == 50

    def test_default_with_n_var(self):
        """default() should set mutation prob based on n_var."""
        cfg = NSGAIIConfig.default(n_var=30)
        # mutation prob should be ~1/30 = 0.033
        assert abs(cfg.mutation[1]["prob"] - 1 / 30) < 0.001

    def test_default_with_permutation_encoding(self):
        """default() should pick permutation-compatible operators when requested."""
        cfg = NSGAIIConfig.default(encoding="permutation")

        assert cfg.crossover[0] == "ox"
        assert cfg.mutation[0] == "swap"
        assert cfg.selection[0] == "tournament"
        assert cfg.repair == "auto"

    def test_builder_defaults_pop_size_and_selection(self):
        """Builder should fill defaults for pop_size and selection if omitted."""
        cfg = NSGAIIConfig.builder().crossover("sbx", prob=1.0, eta=20.0).mutation("polynomial", prob=0.1, eta=20.0).build()

        assert cfg.pop_size == 100
        assert cfg.selection[0] == "tournament"

    def test_builder_rejects_tuple_operator_syntax(self):
        with pytest.raises(TypeError, match="keyword arguments"):
            NSGAIIConfig.builder().crossover(("sbx", {"prob": 1.0}))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="keyword arguments"):
            NSGAIIConfig.builder().mutation(("pm", {"prob": "1/n"}))  # type: ignore[arg-type]

    def test_tournament_selection_accepts_size_key(self):
        """Tournament selection should use the new 'size' key."""
        cfg = (
            NSGAIIConfig.builder()
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("polynomial", prob=0.1, eta=20.0)
            .selection("tournament", size=3)
            .build()
        )
        assert cfg.selection[1]["size"] == 3
        assert "pressure" not in cfg.selection[1]

    def test_tournament_selection_rejects_unknown_options(self):
        with pytest.raises(ValueError, match="Unsupported tournament selection options"):
            NSGAIIConfig.builder().selection("tournament", unexpected_option=2)

    def test_available_operators_lists_pm_operator_id(self):
        operators = NSGAIIConfig.available_operators("mutation")
        assert "pm" in operators["mutation"]

    def test_available_operators_are_derived_from_runtime_registries(self):
        operators = NSGAIIConfig.available_operators("crossover")
        assert "sbx" in operators["crossover"]

    def test_builder_accepts_custom_registered_operator_without_warning(self):
        registry = get_operator_registry()
        registry.register("_pytest_custom_operator", object, override=True)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cfg = NSGAIIConfig.builder().crossover("_pytest_custom_operator").mutation("_pytest_custom_operator").build()

        assert cfg.crossover[0] == "_pytest_custom_operator"
        assert cfg.mutation[0] == "_pytest_custom_operator"
        assert [str(item.message) for item in captured] == []

    def test_builder_validates_operator_names_eagerly(self):
        with pytest.raises(ValueError, match="Unknown mutation"):
            (NSGAIIConfig.builder().crossover("sbx", prob=1.0, eta=20.0).mutation("polynomia", prob=0.1, eta=20.0).build())

    def test_repair_auto_round_trips(self):
        cfg = NSGAIIConfig.default(pop_size=25, n_var=5)
        restored = NSGAIIConfig.from_dict(cfg.to_dict())
        assert restored.repair == "auto"


class TestMOEADConfigShortcuts:
    """Test MOEADConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = MOEADConfig.default()

        assert cfg.pop_size == 91
        assert cfg.neighbor_size == 20
        assert cfg.delta == 0.9
        assert cfg.replace_limit == 2
        assert cfg.aggregation[0] == "pbi"
        assert cfg.repair == "auto"

    def test_default_mutation_operator_id_is_valid(self):
        cfg = MOEADConfig.default()
        assert cfg.mutation[0] == "pm"


class TestIBEAConfigShortcuts:
    def test_default_mutation_operator_id_is_valid(self):
        cfg = IBEAConfig.default()
        assert cfg.mutation[0] == "pm"


class TestSPEA2ConfigShortcuts:
    """Test SPEA2Config.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = SPEA2Config.default()

        assert cfg.pop_size == 100
        assert cfg.archive_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.selection[0] == "tournament"
        assert cfg.repair == "auto"


class TestSMSEMOAConfigShortcuts:
    """Test SMSEMOAConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = SMSEMOAConfig.default()

        assert cfg.pop_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.reference_point["adaptive"] is True
        assert cfg.repair == "auto"


class TestNSGAIIIConfigShortcuts:
    """Test NSGAIIIConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = NSGAIIIConfig.default()

        assert cfg.pop_size == comb(12 + 3 - 1, 3 - 1)
        assert cfg.crossover[0] == "sbx"
        assert cfg.reference_directions["divisions"] == 12
        assert cfg.repair == "auto"
