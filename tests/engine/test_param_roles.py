"""Tests for parameter role annotations across all 9 algorithm config spaces."""

from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.tuning import (
    build_agemoea_config_space,
    build_ibea_config_space,
    build_moead_config_space,
    build_nsgaii_config_space,
    build_nsgaiii_config_space,
    build_rvea_config_space,
    build_smpso_config_space,
    build_smsemoa_config_space,
    build_spea2_config_space,
)
from vamos.engine.tuning.racing.param_space import PARAM_ROLES

ALL_BUILDERS = [
    build_nsgaii_config_space,
    build_moead_config_space,
    build_nsgaiii_config_space,
    build_smsemoa_config_space,
    build_spea2_config_space,
    build_ibea_config_space,
    build_agemoea_config_space,
    build_rvea_config_space,
    build_smpso_config_space,
]


@pytest.mark.parametrize("builder", ALL_BUILDERS, ids=lambda b: b.__name__)
def test_all_params_have_valid_roles(builder):
    cs = builder()
    space = cs.to_param_space()
    for name, param in space.params.items():
        assert hasattr(param, "role"), f"Param {name} missing role attribute"
        assert param.role in PARAM_ROLES, f"Param {name} has invalid role '{param.role}'"


@pytest.mark.parametrize("builder", ALL_BUILDERS, ids=lambda b: b.__name__)
def test_group_by_role_covers_all_params(builder):
    cs = builder()
    groups = cs.group_by_role()
    grouped_names = set()
    for role, params in groups.items():
        assert role in PARAM_ROLES, f"Invalid role '{role}' in groups"
        for p in params:
            grouped_names.add(p.name)
    all_params = set(cs.to_param_space().params.keys())
    assert grouped_names == all_params, f"Ungrouped: {all_params - grouped_names}"


@pytest.mark.parametrize("builder", ALL_BUILDERS, ids=lambda b: b.__name__)
def test_sample_produces_valid_config(builder):
    cs = builder()
    rng = np.random.default_rng(42)
    config = cs.to_param_space().sample(rng)
    assert isinstance(config, dict)
    assert len(config) > 0


def test_nsgaii_has_all_five_roles():
    groups = build_nsgaii_config_space().group_by_role()
    assert set(groups.keys()) == PARAM_ROLES


def test_moead_aggregation_is_structural():
    cs = build_moead_config_space()
    groups = cs.group_by_role()
    structural_names = {p.name for p in groups["structural"]}
    assert "aggregation_function" in structural_names


def test_moead_neighbor_size_is_population():
    cs = build_moead_config_space()
    groups = cs.group_by_role()
    pop_names = {p.name for p in groups["population"]}
    assert "neighborhood_size" in pop_names


def test_rvea_adapt_freq_is_adaptive():
    cs = build_rvea_config_space()
    groups = cs.group_by_role()
    adaptive_names = {p.name for p in groups["adaptive"]}
    assert "adapt_freq" in adaptive_names


def test_ibea_indicator_is_structural():
    cs = build_ibea_config_space()
    groups = cs.group_by_role()
    structural_names = {p.name for p in groups["structural"]}
    assert "indicator" in structural_names


def test_ibea_kappa_is_operator_rate():
    cs = build_ibea_config_space()
    groups = cs.group_by_role()
    rate_names = {p.name for p in groups["operator_rate"]}
    assert "kappa" in rate_names


def test_spea2_archive_size_is_population():
    cs = build_spea2_config_space()
    groups = cs.group_by_role()
    pop_names = {p.name for p in groups["population"]}
    assert "archive_size" in pop_names


def test_smpso_swarm_params_are_operator_rate():
    cs = build_smpso_config_space()
    groups = cs.group_by_role()
    rate_names = {p.name for p in groups["operator_rate"]}
    for name in ("inertia", "c1", "c2", "vmax_fraction"):
        assert name in rate_names, f"{name} should be operator_rate"
