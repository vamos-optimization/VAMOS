from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import vamos
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


def _nsgaiii() -> NSGAIIIConfig:
    return (
        NSGAIIIConfig.builder()
        .pop_size(6)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob=1.0 / 6.0, eta=20.0)
        .selection("tournament", size=2)
        .reference_directions(divisions=2)
        .result_mode("population")
        .build()
    )


def _rvea() -> RVEAConfig:
    return (
        RVEAConfig.builder()
        .pop_size(6)
        .n_partitions(2)
        .alpha(2.0)
        .adapt_freq(0.1)
        .crossover("sbx", prob=1.0, eta=30.0)
        .mutation("pm", prob=1.0 / 6.0, eta=20.0)
        .result_mode("population")
        .build()
    )


_ALGORITHMS: tuple[tuple[str, str, Callable[[], Any], dict[str, int]], ...] = (
    ("nsgaii", "zdt1", lambda: NSGAIIConfig.default(pop_size=8, n_var=6), {"n_var": 6}),
    ("nsgaiii", "dtlz2", _nsgaiii, {"n_var": 6, "n_obj": 3}),
    ("moead", "zdt1", lambda: MOEADConfig.default(pop_size=6, n_var=6, n_obj=2), {"n_var": 6}),
    ("spea2", "zdt1", lambda: SPEA2Config.default(pop_size=8, n_var=6), {"n_var": 6}),
    ("ibea", "zdt1", lambda: IBEAConfig.default(pop_size=8, n_var=6), {"n_var": 6}),
    ("smsemoa", "zdt1", lambda: SMSEMOAConfig.default(pop_size=8, n_var=6), {"n_var": 6}),
    ("smpso", "zdt1", lambda: SMPSOConfig.default(pop_size=8, n_var=6), {"n_var": 6}),
    ("agemoea", "zdt1", lambda: AGEMOEAConfig.default(pop_size=8, n_var=6), {"n_var": 6}),
    ("rvea", "dtlz2", _rvea, {"n_var": 6, "n_obj": 3}),
)


@pytest.mark.parametrize(("algorithm", "problem", "config_factory", "problem_dimensions"), _ALGORITHMS)
@pytest.mark.parametrize("engine", ["numpy", "auto"])
def test_all_registered_built_in_algorithms_replay_exactly(
    algorithm: str,
    problem: str,
    config_factory: Callable[[], Any],
    problem_dimensions: dict[str, int],
    tmp_path: Path,
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if engine == "auto":
        pytest.importorskip("numba")
    config = config_factory()
    pop_size = int(config.to_dict()["pop_size"])
    result = vamos.optimize(
        problem,
        algorithm=algorithm,
        algorithm_config=config,
        termination=("max_evaluations", max(12, pop_size * 2)),
        engine=engine,
        seed=3,
        **problem_dimensions,
    )
    source = vamos.save_result(result, tmp_path / f"{algorithm}-source")

    if engine == "auto":
        assert source.manifest.requested_spec["defaults"]["engine"] == "auto"
        assert source.manifest.resolved_spec["backend"]["kernel"]["resolution"]["name"] == "numba"
        # Replay must use stored resolution even if today's auto policy differs.
        monkeypatch.setattr("vamos.experiment.runtime.catalog._has_numba", lambda: False)

    replay = vamos.reproduce(source.root, output=tmp_path / f"{algorithm}-replay")

    assert replay.exact
    assert replay.task_id == source.manifest.task_id
    assert {"F", "X"}.issubset({item.role for item in replay.comparisons if item.exact})


@pytest.mark.parametrize("problem", ["zdt1", "re21", "int_alloc", "bin_feat", "tsp6", "mixed_design"])
def test_representative_builtin_problem_encodings_replay(problem: str, tmp_path: Path) -> None:
    result = vamos.optimize(problem, algorithm="nsgaii", pop_size=6, max_evaluations=12, engine="numpy", seed=4)
    source = vamos.save_result(result, tmp_path / f"{problem}-source")

    replay = vamos.reproduce(source.root, output=tmp_path / f"{problem}-replay")

    assert replay.exact


@pytest.mark.parametrize("engine", ["numba", "moocore"])
def test_optional_backend_exact_replay_without_substitution(engine: str, tmp_path: Path) -> None:
    pytest.importorskip(engine)
    result = vamos.optimize("zdt1", algorithm="nsgaii", pop_size=8, max_evaluations=16, engine=engine, seed=5, n_var=6)
    source = vamos.save_result(result, tmp_path / f"{engine}-source")

    replay = vamos.reproduce(source.root, output=tmp_path / f"{engine}-replay")
    replay_run = vamos.load_run(replay.output_root)

    kernel = replay_run.manifest.resolved_spec["backend"]["kernel"]
    assert replay.exact
    assert kernel["resolution"]["name"] == engine
