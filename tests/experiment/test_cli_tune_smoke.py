"""`vamos tune --smoke`: populations within the smoke budget, failure warning and output folder (review finding VR-296)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.engine.tuning import AlgorithmConfigSpace, EvalContext, Instance, Int, ParamSpace
from vamos.experiment.cli import _tune_runtime
from vamos.experiment.cli import tune as tune_cli
from vamos.experiment.cli._tune_smoke import smoke_param_space

# The quick verification command of docs/guide/cli.md and docs/topics/tuning.md.
GUIDE_SMOKE = ["--instances", "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1", "--algorithm", "nsgaii", "--backend", "random", "--smoke"]


def _failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("[tune] evaluation failed")]


def _artifact(root: Path, name: str) -> Any:
    (path,) = root.rglob(name)
    return json.loads(path.read_text(encoding="utf-8"))


def _param_space(algorithm: str) -> ParamSpace:
    space = _tune_runtime.BUILDERS[algorithm]()
    return space.to_param_space() if isinstance(space, AlgorithmConfigSpace) else space


def _smoke_profile(argv: list[str]) -> dict[str, Any]:
    return tune_cli._apply_smoke_mode(tune_cli._parse_args([*argv, "--smoke"]))


def test_tune_smoke_samples_feasible_population(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert tune_cli.main([*GUIDE_SMOKE, "--output-dir", str(tmp_path)]) == 0

    assert _failures(caplog) == []
    budget = _artifact(tmp_path, "tuning_summary.json")["smoke_profile"]["budget"]
    sampled = [trial["config"]["pop_size"] for trial in _artifact(tmp_path, "tuning_history.json")]
    assert sampled and all(1 <= pop_size <= budget // 2 for pop_size in sampled), sampled


@pytest.mark.parametrize("n_obj", [2, 3])
@pytest.mark.parametrize("algorithm", sorted(_tune_runtime.BUILDERS))
def test_smoke_space_keeps_every_population_within_half_the_budget(algorithm: str, n_obj: int) -> None:
    profile = _smoke_profile(["--algorithm", algorithm, "--n-obj", str(n_obj)])
    space, fixed = smoke_param_space(
        _param_space(algorithm), max_population=profile["pop_size"], n_obj=n_obj, algorithm=algorithm, budget=profile["budget"]
    )
    params = space.params
    assert fixed == {}

    population = params["pop_size"] if "pop_size" in params else params["n_partitions"]
    assert isinstance(population, Int)
    # SMAC3, BOHB and irace reject an integer range without width.
    assert population.low < population.high
    if "pop_size" in params:
        largest = population.high
    else:
        # RVEA's population is its number of reference directions.
        largest = math.comb(population.high + n_obj - 1, n_obj - 1)
    # The opposition-based initializer evaluates a population and its opposite.
    assert 1 <= largest and 2 * largest <= profile["budget"], (largest, profile)


def test_smoke_population_leaves_room_for_the_opposition_based_initializer(caplog: pytest.LogCaptureFixture) -> None:
    profile = _smoke_profile(["--problem", "zdt1"])
    space, _ = smoke_param_space(
        _param_space("nsgaii"), max_population=profile["pop_size"], n_obj=2, algorithm="nsgaii", budget=profile["budget"]
    )
    config = space.sample(np.random.default_rng(0))
    config.update(initializer="obl", pop_size=space.params["pop_size"].high)
    logger = logging.getLogger("test_cli_tune_smoke")
    evaluate = _tune_runtime.make_evaluator("zdt1", 30, 2, "nsgaii", profile["pop_size"], None, False, 0.0, -1.0, logger=lambda: logger)

    with caplog.at_level(logging.WARNING):
        score = evaluate(config, EvalContext(instance=Instance(name="zdt1", n_var=30, kwargs={}), seed=1, budget=profile["budget"]))

    assert _failures(caplog) == [] and score >= 0.0


@pytest.mark.parametrize(
    "argv",
    [
        ["--problem", "dtlz2", "--n-obj", "3", "--algorithm", "rvea"],
        ["--problem", "dtlz2", "--n-obj", "3", "--algorithm", "nsgaiii"],
        ["--problem", "zdt1", "--algorithm", "smpso"],
    ],
)
def test_tune_smoke_runs_every_sampled_configuration(argv: list[str], tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert tune_cli.main([*argv, "--backend", "random", "--smoke", "--output-dir", str(tmp_path)]) == 0

    assert _failures(caplog) == []


def test_tune_warns_when_most_evaluations_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    calls: list[int] = []
    actual = _tune_runtime.optimize

    def mostly_failing_optimize(*args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) % 4:
            raise RuntimeError("synthetic failure")
        return actual(*args, **kwargs)

    monkeypatch.setattr(_tune_runtime, "optimize", mostly_failing_optimize)
    with caplog.at_level(logging.WARNING):
        assert tune_cli.main(["--problem", "zdt1", "--backend", "random", "--smoke", "--output-dir", str(tmp_path)]) == 0

    # 4 configurations x 1 instance x 2 seeds, of which every fourth runs.
    assert len(_failures(caplog)) == 6
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("6 of 8 tuning evaluations failed" in message for message in warnings), warnings


def test_tune_without_failures_does_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert tune_cli.main(["--problem", "zdt1", "--backend", "random", "--smoke", "--output-dir", str(tmp_path)]) == 0

    assert not any("tuning evaluations failed" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("instances", "label"),
    [("dtlz2", "dtlz2"), ("dtlz2,wfg1", "dtlz2-wfg1"), ("zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1", "zdt1-zdt2-zdt3-and-3-more")],
)
def test_tune_output_folder_names_the_instances(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, instances: str, label: str) -> None:
    def fake_make_evaluator(**_: object):
        def evaluate(config, ctx):
            return 1.0

        return evaluate

    monkeypatch.setattr(tune_cli, "make_evaluator", fake_make_evaluator)
    assert tune_cli.main(["--instances", instances, "--backend", "random", "--smoke", "--output-dir", str(tmp_path)]) == 0

    (folder,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert folder.name.startswith(f"{label}_nsgaii_random_seed1_"), folder.name


# Review follow-up (PR #230): reference-direction algorithms need at least one direction per objective.
@pytest.mark.parametrize("algorithm", ["moead", "nsgaiii", "rvea"])
def test_smoke_space_keeps_reference_lattices_feasible(
    algorithm: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from vamos.experiment.cli import _tune_scoring

    # The runs are what this test checks; a 16-objective hypervolume without moocore takes minutes.
    monkeypatch.setattr(_tune_scoring, "_score_population_result", lambda *_args: 1.0)
    argv = ["--problem", "dtlz2", "--n-obj", "16", "--algorithm", algorithm, "--backend", "random", "--smoke"]
    with caplog.at_level(logging.WARNING):
        assert tune_cli.main([*argv, "--output-dir", str(tmp_path)]) == 0

    assert _failures(caplog) == []
    sampled = _artifact(tmp_path, "tuning_history.json")
    if algorithm == "rvea":
        # One partition gives the 16 axis directions; two give 136, more than the smoke budget. The
        # fixed count leaves the search space, and the smoke profile records it.
        assert all("n_partitions" not in trial["config"] for trial in sampled), sampled
        assert _artifact(tmp_path, "tuning_summary.json")["smoke_profile"]["fixed_parameters"] == {"n_partitions": 1}
    else:
        assert all(trial["config"]["pop_size"] >= 16 for trial in sampled), sampled


@pytest.mark.parametrize(("algorithm", "label"), [("nsgaiii", "NSGA-III"), ("moead", "MOEA/D"), ("rvea", "RVEA")])
def test_smoke_rejects_lattices_larger_than_its_budget(algorithm: str, label: str, tmp_path: Path) -> None:
    argv = ["--problem", "dtlz2", "--n-obj", "31", "--n-var", "50", "--algorithm", algorithm, "--backend", "random", "--smoke"]
    with pytest.raises(SystemExit, match=rf"{label} with 31 objectives needs a population of at least 31.*smoke budget of 30"):
        tune_cli.main([*argv, "--output-dir", str(tmp_path)])
    assert not list(tmp_path.rglob("*.json")), "nothing may be written before the check"


# Review follow-up (PR #230): a parameter that the smoke caps fix to one value must not reach the model
# backends as an integer range without width, which ConfigSpace (SMAC3, BOHB) and irace reject.
@pytest.mark.parametrize(
    ("algorithm", "n_obj"),
    # With 30 objectives the smallest lattice is exactly the 30-evaluation smoke budget.
    [("rvea", 6), ("rvea", 16), ("moead", 16), ("nsgaiii", 16), ("moead", 30), ("nsgaiii", 30), ("rvea", 30)],
)
def test_smoke_space_translates_for_smac_bohb_and_irace(
    algorithm: str, n_obj: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("ConfigSpace")
    import re

    from vamos.engine.tuning._irace_space import parameter_file
    from vamos.engine.tuning._model_backend_utils import build_configspace
    from vamos.experiment.cli import _tune_scoring
    from vamos.experiment.cli._tune_scoring import evaluation_ledger

    # The runs are what this test checks; a many-objective hypervolume without moocore takes minutes.
    monkeypatch.setattr(_tune_scoring, "_score_population_result", lambda *_args: 1.0)
    seen: dict[str, object] = {}

    def fake_run_backend(args, task, eval_fn, resolved_jobs):  # noqa: ANN001
        _ = args, resolved_jobs
        config = task.param_space.sample(np.random.default_rng(0))
        eval_fn(config, EvalContext(instance=task.instances[0], seed=1, budget=task.budget_per_run))
        seen.update(space=task.param_space, ledger=evaluation_ledger(eval_fn).counts())
        return config, []

    monkeypatch.setattr(tune_cli, "_run_backend", fake_run_backend)
    argv = ["--problem", "dtlz2", "--n-obj", str(n_obj), "--n-var", "40", "--algorithm", algorithm, "--backend", "random", "--smoke"]
    assert tune_cli.main([*argv, "--output-dir", str(tmp_path)]) == 0

    space = seen["space"]
    build_configspace(space, seed=1)
    assert not re.search(r"\((-?[\d.e+-]+), \1\)", parameter_file(space)), parameter_file(space)
    assert seen["ledger"] == (1, 0), "the sampled configuration must run"
