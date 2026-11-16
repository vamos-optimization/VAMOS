"""``live_callback_mode`` in every algorithm that takes an external archive."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config.defaults import build_algorithm_config
from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.quality_indicators.pareto import pareto_filter

ARCHIVE_ALGORITHMS = (
    "agemoea",
    "ccmo",
    "cmoeams",
    "ctaea",
    "ibea",
    "moead",
    "nsgaii",
    "nsgaiii",
    "rvea",
    "smpso",
    "smsemoa",
    "spea2",
    "top",
)
POP_SIZE = 20


class _Recording:
    def __init__(self) -> None:
        self.calls: list[tuple[int, np.ndarray, np.ndarray, dict[str, Any]]] = []

    def on_start(self, ctx: object = None) -> None:
        return None

    def on_generation(self, generation: int, F: Any = None, X: Any = None, stats: Any = None) -> None:
        archive = (stats or {}).get("archive")
        copied = dict(stats or {})
        if archive is not None:
            copied["archive"] = {"F": np.array(archive["F"]), "X": np.array(archive["X"])}
        self.calls.append((generation, np.array(F), np.array(X), copied))

    def on_end(self, final_F: object = None, final_stats: object = None) -> None:
        return None


def _config(name: str, mode: str, *, archive: bool = True) -> Any:
    config = build_algorithm_config(name, pop_size=POP_SIZE, n_var=30, n_obj=2, encoding="real", result_mode="non_dominated")
    changes: dict[str, Any] = {"live_callback_mode": mode}
    if archive:
        changes["external_archive"] = ExternalArchiveConfig(capacity=None)
    return dataclasses.replace(config, **changes)


def _run(name: str, config: Any, budget: int = 10 * POP_SIZE) -> tuple[vamos.OptimizationResult, _Recording]:
    callback = _Recording()
    result = vamos.optimize(
        "zdt1", algorithm=name, algorithm_config=config, max_evaluations=budget, seed=1, engine="numpy", live_viz=callback
    )
    return result, callback


def _rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.lexsort(values.T[::-1])]


@pytest.mark.parametrize("name", ARCHIVE_ALGORITHMS)
def test_every_archive_algorithm_takes_live_callback_mode(name: str) -> None:
    config = build_algorithm_config(name, pop_size=POP_SIZE, n_var=30, n_obj=2, encoding="real", result_mode="population")
    expected = "nd_only" if name == "nsgaii" else "population"
    assert config.to_dict()["live_callback_mode"] == expected
    builder = type(config).builder()
    assert builder.live_callback_mode("population_archive") is builder
    with pytest.raises(ConfigurationError, match="live_callback_mode"):
        dataclasses.replace(config, live_callback_mode="all")


@pytest.mark.parametrize("name", ARCHIVE_ALGORITHMS)
def test_population_archive_sends_the_archive_and_ends_with_the_result(name: str) -> None:
    result, callback = _run(name, _config(name, "population_archive"))

    assert len(callback.calls) >= 2
    for _, F, X, stats in callback.calls:
        # The population, as in the "population" mode, and the archive beside it.
        assert X.shape == (F.shape[0], 30)
        assert {"evals", "archive"} <= set(stats)
        assert stats["archive"]["X"].shape == (stats["archive"]["F"].shape[0], 30)
    last = callback.calls[-1][3]["archive"]
    # What a live view shows last is what the run returns.
    np.testing.assert_array_equal(_rows(last["F"]), _rows(result.F))
    np.testing.assert_array_equal(_rows(last["X"]), _rows(result.X))


@pytest.mark.parametrize("name", ARCHIVE_ALGORITHMS)
def test_population_archive_ends_with_the_result_when_the_budget_ends_inside_a_generation(name: str) -> None:
    # 195 evaluations: the last generation is cut short. Generational algorithms report it as a shorter generation;
    # MOEA/D and SMS-EMOA, which report one generation per pop_size evaluations, send the last 15 in a final one.
    result, callback = _run(name, _config(name, "population_archive"), budget=10 * POP_SIZE - 5)

    generations = [call[0] for call in callback.calls]
    assert generations == sorted(set(generations)), generations
    last_generation, _, _, stats = callback.calls[-1]
    assert stats["evals"] == 10 * POP_SIZE - 5
    np.testing.assert_array_equal(_rows(stats["archive"]["F"]), _rows(result.F))
    np.testing.assert_array_equal(_rows(stats["archive"]["X"]), _rows(result.X))


@pytest.mark.parametrize("name", ["moead", "smsemoa", "spea2"])
def test_population_archive_without_an_archive_sends_the_population_only(name: str) -> None:
    _, callback = _run(name, _config(name, "population_archive", archive=False))
    assert all("archive" not in stats for *_, stats in callback.calls)


@pytest.mark.parametrize("name", ["moead", "smsemoa", "ibea", "spea2"])
def test_nd_only_sends_the_first_front(name: str) -> None:
    _, population = _run(name, _config(name, "population", archive=False))
    _, front = _run(name, _config(name, "nd_only", archive=False))

    assert [call[0] for call in front.calls] == [call[0] for call in population.calls]
    for (_, F_all, X_all, _), (_, F, X, _) in zip(population.calls, front.calls, strict=True):
        _, indices = pareto_filter(F_all, return_indices=True)
        np.testing.assert_array_equal(F, F_all[np.sort(indices)])
        np.testing.assert_array_equal(X, X_all[np.sort(indices)])


@pytest.mark.parametrize("name", ["nsgaii", "nsgaiii", "moead", "smsemoa", "spea2", "ibea", "smpso"])
def test_population_archive_ends_with_the_result_when_the_hv_target_stops_the_run(name: str) -> None:
    # The batch that reaches the hypervolume target ends the run; its archive still reaches the callback last.
    callback = _Recording()
    termination = ("hv", {"target_value": 0.3, "reference_point": [1.1, 1.1], "max_evaluations": 6000})
    result = vamos.optimize(
        "zdt1",
        algorithm=name,
        algorithm_config=_config(name, "population_archive"),
        termination=termination,
        seed=1,
        engine="numpy",
        live_viz=callback,
    )

    assert result.data["hv_reached"] is True, name
    last = callback.calls[-1][3]["archive"]
    np.testing.assert_array_equal(_rows(last["F"]), _rows(result.F))
    np.testing.assert_array_equal(_rows(last["X"]), _rows(result.X))
