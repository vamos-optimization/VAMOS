"""RVEA's survival in pymoo's reading (the default) and in the authors' PlatEMO code, as jMetal (``variant``)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import vamos
from vamos.algorithms import RVEAConfig
from vamos.engine.algorithm.rvea.rvea import _calc_gamma, _calc_V, _generate_reference_vectors
from vamos.engine.algorithm.rvea.survival import angle_penalized_distances, apd_survival
from vamos.exceptions import ConfigurationError


def _penalty(variant: str, F: np.ndarray, t: int, t_max: int, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = _calc_V(_generate_reference_vectors(2, 9))
    gamma = _calc_gamma(V)
    niches, penalty, _apd = angle_penalized_distances(F, F.min(axis=0), V, gamma, t, t_max, alpha, variant)
    return niches, penalty, gamma


def _angle(F: np.ndarray, V: np.ndarray, niches: np.ndarray) -> np.ndarray:
    shifted = F - F.min(axis=0)
    unit = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
    return np.arccos(np.clip(np.sum(unit * V[niches], axis=1), -1.0, 1.0))


@pytest.mark.parametrize(("variant", "M"), [("platemo", 2.0), ("pymoo", 1.0)])
def test_with_two_objectives_the_penalty_is_m_times_progress_times_the_relative_angle(variant: str, M: float) -> None:
    rng = np.random.default_rng(4)
    F = rng.random((30, 2)) + 0.1
    t, t_max, alpha = 7, 20, 2.0
    V = _calc_V(_generate_reference_vectors(2, 9))

    niches, penalty, gamma = _penalty(variant, F, t, t_max, alpha)

    theta = _angle(F, V, niches)
    np.testing.assert_allclose(penalty, M * (t / t_max) ** alpha * theta / gamma[niches], rtol=1e-12, atol=1e-15)


def test_with_three_objectives_both_variants_use_m_objectives() -> None:
    rng = np.random.default_rng(5)
    F = rng.random((40, 3)) + 0.1
    V = _calc_V(_generate_reference_vectors(3, 4))
    gamma = _calc_gamma(V)

    _, pymoo, _ = angle_penalized_distances(F, F.min(axis=0), V, gamma, 3, 10, 2.0, "pymoo")
    _, platemo, _ = angle_penalized_distances(F, F.min(axis=0), V, gamma, 3, 10, 2.0, "platemo")

    np.testing.assert_array_equal(pymoo, platemo)


def test_platemo_translates_by_the_candidates_minimum_and_pymoo_by_the_accumulated_ideal() -> None:
    rng = np.random.default_rng(6)
    F = rng.random((20, 3)) + 1.0
    V = _calc_V(_generate_reference_vectors(3, 3))
    gamma = _calc_gamma(V)
    ideal = np.zeros(3)  # accumulated from an earlier generation, below every candidate

    pymoo, pymoo_ideal, _ = apd_survival(F, V, gamma, ideal, 10, 2, 10, 2.0, "pymoo")
    platemo, platemo_ideal, _ = apd_survival(F, V, gamma, ideal, 10, 2, 10, 2.0, "platemo")

    np.testing.assert_array_equal(pymoo_ideal, platemo_ideal)  # both keep the accumulated ideal
    niches, _, apd = angle_penalized_distances(F, F.min(axis=0), V, gamma, 2, 10, 2.0, "platemo")
    expected = sorted(int(np.flatnonzero(niches == niche)[np.argmin(apd[niches == niche])]) for niche in np.unique(niches))
    assert sorted(platemo.tolist()) == expected
    assert sorted(pymoo.tolist()) != sorted(platemo.tolist())


def _config(variant: str | None, pop_size: int = 100) -> RVEAConfig:
    config = RVEAConfig.default(pop_size=pop_size, n_obj=2)
    return config if variant is None else dataclasses.replace(config, variant=variant)


def test_the_default_is_pymoos_reading_and_keeps_its_results() -> None:
    assert RVEAConfig.default(n_obj=2).variant == "pymoo"
    assert "variant" not in RVEAConfig.default(n_obj=2).to_dict()

    default = vamos.optimize("zdt1", algorithm="rvea", algorithm_config=_config(None), max_evaluations=2000, seed=2, engine="numpy")
    pymoo = vamos.optimize("zdt1", algorithm="rvea", algorithm_config=_config("pymoo"), max_evaluations=2000, seed=2, engine="numpy")
    platemo = vamos.optimize("zdt1", algorithm="rvea", algorithm_config=_config("platemo"), max_evaluations=2000, seed=2, engine="numpy")

    np.testing.assert_array_equal(default.F, pymoo.F)
    assert platemo.data["evaluations"] == 2000
    assert default.F.shape != platemo.F.shape or not np.array_equal(default.F, platemo.F)


def test_the_variant_is_validated_serialized_and_replayed(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="variant"):
        _config("jmetal")
    builder = (
        RVEAConfig.builder()
        .pop_size(91)
        .n_partitions(12)
        .alpha(2.0)
        .adapt_freq(0.1)
        .crossover("sbx", prob=0.9, eta=30.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .variant("platemo")
    )
    config = builder.build()
    assert config.to_dict()["variant"] == "platemo"
    assert RVEAConfig.from_dict(config.to_dict()) == config

    result = vamos.optimize("dtlz2", algorithm="rvea", algorithm_config=config, max_evaluations=364, seed=1)
    source = vamos.save_result(result, tmp_path / "source")
    assert vamos.verify_run(source.root).effective_replayability == "exact"
    assert vamos.reproduce(source.root, output=tmp_path / "replay").exact is True
