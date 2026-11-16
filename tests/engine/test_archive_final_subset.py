"""The final subset of an external archive: ``ExternalArchiveConfig.final_size`` and ``final_selection``.

An unbounded archive returns every non-dominated solution found. jMetal's ``BestSolutionsArchive`` and
Evolver return a subset as large as the population instead. With ``final_selection``, the result reports
such a subset of the archive when the run ends, while ``result.data["archive"]`` keeps the whole archive.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.components.subset_selection import (
    select_top_k_crowding,
    select_top_k_farthest,
    select_top_k_hv_greedy,
    select_top_k_kmeans,
)
from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    CCMOConfig,
    CMOEAMSConfig,
    CTAEAConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
    ToPConfig,
)
from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.rng import derived_seed

SELECTORS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "crowding": select_top_k_crowding,
    "farthest": select_top_k_farthest,
    "hv_greedy": select_top_k_hv_greedy,
    # Without an rng_seed, kmeans draws from the archive's stream derived from the run seed of _run.
    "kmeans": lambda F, k: select_top_k_kmeans(F, k, rng=derived_seed(4, "external_archive")),
}


def _nsgaii(**archive: Any) -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(20)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob=1.0 / 6.0, eta=20.0)
        .selection("tournament", size=2)
        .external_archive(capacity=None, **archive)
        .build()
    )


def _archive(result: Any, key: str = "F") -> np.ndarray:
    """The whole external archive of a result."""
    data = result.data
    if isinstance(data.get("external_archive"), dict):  # SPEA2, whose own archive is "archive"
        return np.asarray(data["external_archive"][key])
    if isinstance(data.get("archive"), dict):
        return np.asarray(data["archive"][key])
    return np.asarray(data[f"archive_{key}"])  # IBEA and NSGA-III


def _run(config: Any, *, algorithm: str = "nsgaii", evaluations: int = 600, **kwargs: Any) -> Any:
    return vamos.optimize("zdt1", algorithm=algorithm, algorithm_config=config, max_evaluations=evaluations, n_var=6, seed=4, **kwargs)


def test_by_default_the_archive_has_no_final_subset() -> None:
    config = ExternalArchiveConfig()

    assert (config.final_size, config.final_selection) == (None, "none")


def test_a_configuration_without_a_final_subset_serializes_as_before() -> None:
    payload = _nsgaii().to_dict()["external_archive"]

    assert isinstance(payload, dict)
    assert "final_size" not in payload and "final_selection" not in payload


def test_a_final_subset_round_trips() -> None:
    config = _nsgaii(final_size=10, final_selection="kmeans", rng_seed=3)

    assert NSGAIIConfig.from_dict(config.to_dict()) == config
    assert config.external_archive is not None
    assert (config.external_archive.final_size, config.external_archive.final_selection) == (10, "kmeans")


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"final_selection": "random"}, ValueError),
        ({"final_selection": 3}, ValueError),
        ({"final_size": 0, "final_selection": "crowding"}, ValueError),
        ({"final_size": 2.5, "final_selection": "crowding"}, TypeError),
        ({"final_size": True, "final_selection": "crowding"}, TypeError),
    ],
)
def test_invalid_final_subsets_are_rejected(arguments: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error, match="final_"):
        ExternalArchiveConfig(**arguments)


def test_a_final_size_without_a_selection_warns_that_it_has_no_effect() -> None:
    with pytest.warns(UserWarning, match="final_size"):
        ExternalArchiveConfig(final_size=10)


def test_rng_seed_is_used_by_the_kmeans_final_selection() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExternalArchiveConfig(final_size=10, final_selection="kmeans", rng_seed=5)


@pytest.mark.parametrize("selection", sorted(SELECTORS))
def test_the_result_is_the_selected_subset_of_the_whole_archive(selection: str) -> None:
    result = _run(_nsgaii(final_size=10, final_selection=selection))
    archive_F = np.asarray(result.data["archive"]["F"])
    archive_X = np.asarray(result.data["archive"]["X"])

    assert archive_F.shape[0] > 10
    indices = SELECTORS[selection](archive_F, 10)
    np.testing.assert_array_equal(result.F, archive_F[indices])
    np.testing.assert_array_equal(result.X, archive_X[indices])
    subset = result.data["archive_subset"]
    assert (subset["selection"], subset["size"]) == (selection, 10)
    np.testing.assert_array_equal(subset["indices"], indices)


def test_without_a_final_size_the_subset_is_as_large_as_the_population() -> None:
    result = _run(_nsgaii(final_selection="crowding"))

    assert result.data["archive"]["F"].shape[0] > 20
    assert result.F.shape[0] == 20
    assert result.data["archive_subset"]["size"] == 20


def test_an_archive_smaller_than_the_final_size_is_reported_whole() -> None:
    result = _run(_nsgaii(final_size=500, final_selection="crowding"), evaluations=100)

    np.testing.assert_array_equal(result.F, result.data["archive"]["F"])
    np.testing.assert_array_equal(result.data["archive_subset"]["indices"], np.arange(result.F.shape[0]))


def test_without_a_final_selection_the_result_is_the_whole_archive() -> None:
    result = _run(_nsgaii())

    np.testing.assert_array_equal(result.F, result.data["archive"]["F"])
    assert "archive_subset" not in result.data


def test_a_population_result_ignores_the_final_subset() -> None:
    config = dataclasses.replace(_nsgaii(final_size=10, final_selection="crowding"), result_mode="population")
    result = _run(config)

    assert result.F.shape[0] == 20
    assert "archive_subset" not in result.data


def _with_archive(config: Any) -> Any:
    archive = ExternalArchiveConfig(capacity=None, final_size=5, final_selection="crowding")
    return dataclasses.replace(config, external_archive=archive, result_mode="non_dominated")


ALGORITHMS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("nsgaii", lambda: NSGAIIConfig.default(pop_size=8, n_var=6)),
    ("moead", lambda: MOEADConfig.default(pop_size=8, n_var=6, n_obj=2)),
    ("smsemoa", lambda: SMSEMOAConfig.default(pop_size=8, n_var=6)),
    ("nsgaiii", lambda: NSGAIIIConfig.default(pop_size=8, n_var=6, n_obj=2)),
    ("spea2", lambda: SPEA2Config.default(pop_size=8, n_var=6)),
    ("ibea", lambda: IBEAConfig.default(pop_size=8, n_var=6)),
    ("smpso", lambda: SMPSOConfig.default(pop_size=8, n_var=6)),
    ("agemoea", lambda: AGEMOEAConfig.default(pop_size=8, n_var=6)),
    ("rvea", lambda: RVEAConfig.default(pop_size=8, n_var=6, n_obj=2)),
    ("ccmo", lambda: CCMOConfig.default(pop_size=8, n_var=6)),
    ("ctaea", lambda: CTAEAConfig.default(pop_size=8, n_var=6)),
    ("top", lambda: ToPConfig.default(pop_size=8, n_var=6)),
    ("cmoeams", lambda: CMOEAMSConfig.default(pop_size=8, n_var=6)),
)


@pytest.mark.parametrize(("algorithm", "config_factory"), ALGORITHMS, ids=[name for name, _ in ALGORITHMS])
def test_every_algorithm_with_an_external_archive_reports_its_final_subset(algorithm: str, config_factory: Callable[[], Any]) -> None:
    result = _run(_with_archive(config_factory()), algorithm=algorithm, evaluations=400)
    archive_F = _archive(result)

    indices = select_top_k_crowding(archive_F, 5)
    np.testing.assert_array_equal(result.F, archive_F[indices])
    assert result.data["archive_subset"]["selection"] == "crowding"


def test_the_artifact_keeps_the_whole_archive_and_records_the_selection(tmp_path: Path) -> None:
    result = _run(_nsgaii(final_size=10, final_selection="farthest"))
    stored = vamos.save_result(result, tmp_path / "run")
    loaded = vamos.load_result(stored.root)

    np.testing.assert_array_equal(loaded.F, result.F)
    np.testing.assert_array_equal(loaded.data["archive"]["F"], result.data["archive"]["F"])
    archive = stored.manifest.resolved_spec["algorithm"]["config"]["external_archive"]
    assert (archive["final_size"], archive["final_selection"]) == (10, "farthest")
    assert vamos.reproduce(stored.root, output=tmp_path / "replay").exact


def test_the_kmeans_final_selection_records_the_seed_derived_from_the_run_seed(tmp_path: Path) -> None:
    result = _run(_nsgaii(final_size=10, final_selection="kmeans"))
    stored = vamos.save_result(result, tmp_path / "run")

    archive = stored.manifest.resolved_spec["algorithm"]["config"]["external_archive"]
    assert archive["rng_seed"] == derived_seed(4, "external_archive")
    assert vamos.reproduce(stored.root, output=tmp_path / "replay").exact


def test_the_hook_archive_has_no_final_subset() -> None:
    from vamos.engine.hooks.config_parse import build_archive_cfg

    with pytest.raises(TypeError, match="final_size"):
        build_archive_cfg({"capacity": 10, "final_size": 5})
