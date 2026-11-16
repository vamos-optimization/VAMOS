"""Fields of the stable algorithm configurations either change the run or warn that they have no effect."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

import vamos
from vamos.algorithms import (
    AGEMOEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMSEMOAConfig,
    SPEA2Config,
)

N_VAR = 10


def _defaults() -> dict[str, object]:
    # Two-objective populations of 20 that every algorithm runs, RVEA's and NSGA-III's lattices included.
    return {
        "nsgaii": NSGAIIConfig.default(pop_size=20, n_var=N_VAR),
        "moead": MOEADConfig.default(pop_size=20, n_var=N_VAR, n_obj=2),
        "nsgaiii": NSGAIIIConfig.default(pop_size=20, n_var=N_VAR, n_obj=2),
        "smsemoa": SMSEMOAConfig.default(pop_size=20, n_var=N_VAR),
        "spea2": SPEA2Config.default(pop_size=20, n_var=N_VAR),
        "agemoea": AGEMOEAConfig.default(pop_size=20, n_var=N_VAR),
        "rvea": RVEAConfig.default(pop_size=20, n_var=N_VAR, n_obj=2),
    }


def _final_front(algorithm: str, config: object) -> np.ndarray:
    result = vamos.optimize("zdt1", algorithm=algorithm, algorithm_config=config, max_evaluations=400, seed=3, engine="numpy", n_var=N_VAR)
    return np.asarray(result.F)


@pytest.mark.parametrize("algorithm", ["nsgaii", "moead", "nsgaiii", "smsemoa", "spea2", "agemoea", "rvea"])
def test_mutation_prob_factor_changes_the_run(algorithm: str) -> None:
    # prob = factor / n_var, NSGA-II's and IBEA's rule: 3.0 mutates three genes per child instead of one.
    config = _defaults()[algorithm]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaled = dataclasses.replace(config, mutation_prob_factor=3.0)
    baseline = _final_front(algorithm, config)
    changed = _final_front(algorithm, scaled)
    assert baseline.shape != changed.shape or not np.array_equal(baseline, changed)


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (
            lambda: AGEMOEAConfig.builder().pop_size(20).crossover("sbx", prob=0.9).mutation("pm", prob=0.1).track_genealogy(True).build(),
            "track_genealogy",
        ),
        (lambda: dataclasses.replace(RVEAConfig.default(pop_size=20, n_obj=2), track_genealogy=True), "track_genealogy"),
        (
            lambda: (
                SPEA2Config.builder()
                .pop_size(20)
                .archive_size(20)
                .crossover("sbx", prob=0.9)
                .mutation("pm", prob=0.1)
                .selection("random")
                .build()
            ),
            "selection",
        ),
        (lambda: dataclasses.replace(SPEA2Config.default(pop_size=20), selection=("tournament", {"size": 5})), "selection"),
        (lambda: SPEA2Config.from_dict({**SPEA2Config.default(pop_size=20).to_dict(), "selection": ["random", {}]}), "selection"),
        (
            lambda: (
                NSGAIIIConfig.builder()
                .pop_size(20)
                .crossover("sbx", prob=0.9)
                .mutation("pm", prob=0.1)
                .selection("tournament")
                .pop_size_auto(True)
                .build()
            ),
            "pop_size_auto",
        ),
        (
            lambda: (
                NSGAIIIConfig.builder()
                .pop_size(20)
                .crossover("sbx", prob=0.9)
                .mutation("pm", prob=0.1)
                .selection("tournament")
                .pop_size_auto(False)
                .build()
            ),
            "pop_size_auto",
        ),
        (
            lambda: (
                NSGAIIIConfig.builder()
                .pop_size(20)
                .crossover("sbx", prob=0.9)
                .mutation("pm", prob=0.1)
                .selection("tournament")
                .enforce_ref_dirs(True)
                .build()
            ),
            "enforce_ref_dirs",
        ),
        (
            lambda: (
                NSGAIIIConfig.builder()
                .pop_size(20)
                .crossover("sbx", prob=0.9)
                .mutation("pm", prob=0.1)
                .selection("tournament")
                .enforce_ref_dirs(False)
                .build()
            ),
            "enforce_ref_dirs",
        ),
        (lambda: dataclasses.replace(NSGAIIIConfig.default(pop_size=20, n_obj=2), enforce_ref_dirs=False), "enforce_ref_dirs"),
        (
            lambda: NSGAIIIConfig.from_dict({**NSGAIIIConfig.default(pop_size=20, n_obj=2).to_dict(), "enforce_ref_dirs": False}),
            "enforce_ref_dirs",
        ),
    ],
)
def test_fields_without_effect_warn_once_when_the_user_sets_them(build, field: str) -> None:
    with pytest.warns(FutureWarning, match=rf"`{field}` has no effect") as record:
        build()
    assert len([warning for warning in record if issubclass(warning.category, FutureWarning)]) == 1


def test_defaults_and_their_round_trips_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for config in _defaults().values():
            type(config).from_dict(config.to_dict())
        # SPEA2's own binary tournament, as the command line sets it.
        SPEA2Config.builder().pop_size(20).archive_size(20).crossover("sbx", prob=0.9).mutation("pm", prob=0.1).selection(
            "tournament", size=2
        ).build()
        NSGAIIIConfig.default(pop_size=100)


def test_nsgaiii_default_population_keeps_its_size_without_warning() -> None:
    # The largest lattice that fits in 100 individuals has 91 directions, and NSGA-III runs N >= H.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        config = NSGAIIIConfig.default(pop_size=100)
        result = vamos.optimize("dtlz2", algorithm="nsgaiii", pop_size=100, max_evaluations=200, seed=1, engine="numpy")
    assert config.pop_size == 100
    assert np.asarray(result.data["population"]["F"]).shape[0] == 100
