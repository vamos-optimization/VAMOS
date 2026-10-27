"""irace backend of ModelBasedTuner.

The end-to-end tests need R with irace >= 4.4.4: ``Rscript`` on PATH, or its path in
``VAMOS_IRACE_RSCRIPT``. Translation, validation and the evaluation server are tested without R.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from functools import cache
from pathlib import Path

import pytest

from vamos.engine.tuning import (
    Boolean,
    Categorical,
    Condition,
    EvalContext,
    Instance,
    Int,
    ModelBasedTuner,
    ParamSpace,
    Real,
    TuningTask,
)
from vamos.engine.tuning import _irace_target_runner as target_runner
from vamos.engine.tuning._irace_space import configurations_file, decode_switches, parameter_file
from vamos.engine.tuning._model_backend_irace import MIN_IRACE_VERSION, _Evaluations, _serve


def _space() -> ParamSpace:
    return ParamSpace(
        params={
            "x": Real("x", 0.0, 1.0),
            "kind": Categorical("kind", ["a", "b"]),
            "y": Int("y", 1, 5),
            "rate": Real("rate", 0.01, 1.0, log=True),
        },
        conditions=[Condition("y", "cfg['kind'] == 'b'"), Condition("rate", "cfg['kind'] == 'a' or cfg['x'] > 0.5")],
    )


def _task(space: ParamSpace | None = None) -> TuningTask:
    return TuningTask(
        name="irace_toy",
        param_space=space or _space(),
        instances=[Instance("i1", 3), Instance("i2", 3)],
        seeds=[11, 12],
        budget_per_run=50,
        maximize=True,
    )


def _score(config: dict, ctx: EvalContext) -> float:
    return 1.0 - abs(config["x"] - 0.3) + (0.2 if config.get("y") == 4 else 0.0) + 0.01 * ctx.seed


def test_parameter_file_translates_domains_and_conditions() -> None:
    space = ParamSpace(
        params={
            "use": Boolean("use"),
            "op": Categorical("op", ["sbx", "pm", 0.25]),
            "n": Int("n", 2, 64, log=True),
            "eta": Real("eta", 5.0, 40.0),
            "w": Real("w", 0.001, 1.0, log=True),
        },
        conditions=[
            Condition("op", "cfg['use']"),
            Condition("eta", "cfg['op'] == 'sbx'"),
            Condition("eta", "cfg['n'] >= 8"),
            Condition("w", "not cfg['use'] or cfg['op'] != 0.25"),
        ],
    )
    assert parameter_file(space).splitlines() == [
        'use "--use=" c ("FALSE", "TRUE")',
        'op "--op=" c ("sbx", "pm", "0.25") | use == "TRUE"',
        'n "--n=" i,log (2, 64)',
        'eta "--eta=" r (5.0, 40.0) | (op == "sbx") & (n >= 8)',
        'w "--w=" r,log (0.001, 1.0) | !is.na(use) & !is.na(op) & (!(use == "TRUE") | op != "0.25")',
    ]


@pytest.mark.parametrize(
    ("params", "expr", "message"),
    [
        ({"op": Categorical("op", ["a", "b"])}, "cfg['op'] in 'ab'", "Supported comparisons"),
        ({"op": Categorical("op", ["a", "b"])}, "cfg['op'] < 'b'", "Ordering comparisons"),
        ({"op": Categorical("op", ["a", "b"])}, "cfg['op'] == 'c'", "not a choice"),
        ({"op": Categorical("op", ["a b", "c"])}, "cfg['op'] == 'c'", "distinct irace tokens"),
        ({"n": Int("n", 1, 3)}, "cfg['n'] == 'x'", "numeric literal"),
    ],
)
def test_untranslatable_conditions_fail_before_irace(params: dict, expr: str, message: str) -> None:
    space = ParamSpace(params={**params, "child": Real("child", 0.0, 1.0)}, conditions=[Condition("child", expr)])
    with pytest.raises(ValueError, match=message):
        parameter_file(space)


def test_names_and_unknown_parents_fail_before_irace() -> None:
    with pytest.raises(ValueError, match="parameter names"):
        parameter_file(ParamSpace(params={"_x": Real("_x", 0.0, 1.0)}))
    with pytest.raises(ValueError, match="unknown parent"):
        parameter_file(ParamSpace(params={"x": Real("x", 0.0, 1.0)}, conditions=[Condition("x", "cfg['z'] == 1")]))


def test_decode_switches_returns_the_active_assignment() -> None:
    space = _space()
    assert decode_switches(space, ["--x=0.25", "--kind=b", "--y=4"]) == {"x": 0.25, "kind": "b", "y": 4}
    with pytest.raises(ValueError, match="inactive"):
        decode_switches(space, ["--x=0.25", "--kind=b", "--y=4", "--rate=0.1"])
    with pytest.raises(ValueError, match="missing"):
        decode_switches(space, ["--x=0.25", "--kind=b"])
    with pytest.raises(ValueError, match="Unexpected irace argument"):
        decode_switches(space, ["--x=0.25", "--kind=b", "--y=4", "--z=1"])


def test_configurations_file_marks_inactive_parameters_na() -> None:
    text = configurations_file(_space(), [{"x": 0.75, "kind": "b", "y": 2, "rate": 0.5}, {"x": 0.1, "kind": "b", "y": 3}])
    assert text.splitlines() == ["x kind y rate", "0.75 b 2 0.5", "0.1 b 3 NA"]
    with pytest.raises(ValueError, match="inactive or unknown"):
        configurations_file(_space(), [{"x": 0.1, "kind": "b", "y": 3, "rate": 0.5}])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 10.0}, "wall-clock"),
        ({"budget_levels": [10, 50]}, "one fidelity level"),
        ({"irace_options": {"parallel": 4}}, "manages them"),
        ({"irace_instance_seeds": "fixed"}, "irace_instance_seeds"),
        ({"irace_max_experiments": 0}, "positive"),
    ],
)
def test_invalid_irace_settings_fail_before_r(kwargs: dict, message: str) -> None:
    tuner = ModelBasedTuner(task=_task(), max_trials=5, backend="irace", **kwargs)
    with pytest.raises(ValueError, match=message):
        tuner.run(_score)


def test_missing_rscript_raises_clearly(tmp_path: Path) -> None:
    tuner = ModelBasedTuner(task=_task(), max_trials=5, backend="irace", irace_rscript=str(tmp_path / "no-rscript"))
    with pytest.raises(RuntimeError, match="requires Rscript"):
        tuner.run(_score)


@pytest.mark.parametrize(("rscript", "expected"), [(os.path.join("R", "bin", "Rscript"), "resolved"), ("Rscript", "Rscript")])
def test_rscript_paths_resolve_against_the_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rscript: str, expected: str
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command[0], cwd))
        return subprocess.CompletedProcess(command, 1, "", "no irace")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("vamos.engine.tuning._model_backend_irace.subprocess.run", fake_run)
    tuner = ModelBasedTuner(task=_task(), max_trials=5, backend="irace", irace_rscript=rscript)
    with pytest.raises(RuntimeError, match="R package `irace`"):
        tuner.run(_score)
    (executable, cwd), *_ = calls
    assert cwd != tmp_path
    assert executable == (str(tmp_path / "R" / "bin" / "Rscript") if expected == "resolved" else expected)


def test_evaluation_server_and_target_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    tuner = ModelBasedTuner(task=_task(), max_trials=1, backend="irace")
    pair = (tuner.task.instances[1], 12)
    evaluations = _Evaluations(tuner, _score, 50, {"vamos_pair_3": pair})
    server = _serve(evaluations, "secret")
    monkeypatch.setenv("VAMOS_IRACE_PORT", str(server.server_address[1]))
    try:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            monkeypatch.setenv("VAMOS_IRACE_TOKEN", "secret")
            ok = target_runner.main(["7", "4", "999", "vamos_pair_3", "--x=0.25", "--kind=b", "--y=4"])
            bad = target_runner.main(["8", "4", "999", "vamos_pair_3", "--x=0.25", "--kind=b"])
            monkeypatch.setenv("VAMOS_IRACE_TOKEN", "wrong")
            rejected = target_runner.main(["9", "4", "999", "vamos_pair_3", "--x=0.25", "--kind=b", "--y=4"])
    finally:
        server.shutdown()
        server.server_close()
    assert (ok, bad, rejected) == (0, 1, 1)
    assert float(out.getvalue()) == pytest.approx(-(1.0 - 0.05 + 0.2 + 0.12))
    assert "missing" in err.getvalue() and "wrong token" in err.getvalue()
    assert evaluations.records == [
        dict(config_id=7, config={"x": 0.25, "kind": "b", "y": 4}, instance="i2", seed=12, score=pytest.approx(1.27))
    ]
    assert isinstance(evaluations.error, ValueError)


@cache
def _rscript() -> str | None:
    rscript = os.environ.get("VAMOS_IRACE_RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        return None
    probe = subprocess.run(
        [rscript, "-e", 'cat(as.character(utils::packageVersion("irace")))'], capture_output=True, text=True, check=False
    )
    parts = probe.stdout.strip().split(".")[:3]
    if probe.returncode != 0 or not all(p.isdigit() for p in parts):
        return None
    return rscript if tuple(map(int, parts)) + (0,) * (3 - len(parts)) >= MIN_IRACE_VERSION else None


def _irace_tuner(**kwargs: object) -> ModelBasedTuner:
    rscript = _rscript()
    if rscript is None:
        pytest.skip("R with irace >= 4.4.4 is not available.")
    return ModelBasedTuner(task=_task(), backend="irace", irace_rscript=rscript, **kwargs)  # type: ignore[arg-type]


@pytest.mark.backends
@pytest.mark.slow
def test_irace_end_to_end_with_task_pairs(tmp_path: Path) -> None:
    calls: list[tuple[dict, str, int, int]] = []

    def evaluate(config: dict, ctx: EvalContext) -> float:
        calls.append((dict(config), ctx.instance.name, ctx.seed, ctx.budget))
        return _score(config, ctx)

    initial = {"x": 0.9, "kind": "b", "y": 1, "rate": 0.5}
    tuner = _irace_tuner(max_trials=50, seed=7, n_jobs=2, irace_work_dir=str(tmp_path), irace_initial_configs=[initial])
    best, history = tuner.run(evaluate)
    assert 0 < len(calls) <= 200
    assert {(name, seed) for _, name, seed, _ in calls} <= {("i1", 11), ("i1", 12), ("i2", 11), ("i2", 12)}
    assert {budget for *_, budget in calls} == {50}
    for config, *_ in calls:
        assert ("y" in config) == (config["kind"] == "b")
        assert ("rate" in config) == (config["kind"] == "a" or config["x"] > 0.5)
    by_id = {row.trial_id: row for row in history}
    assert by_id[1].config == initial
    elite = min((row for row in history if row.details["irace_elite_rank"]), key=lambda row: row.details["irace_elite_rank"])
    assert best == elite.config and elite.details["irace_elite_rank"] == 1
    for row in history:
        scores = [
            _score(config, EvalContext(instance=Instance(name, 3), seed=seed, budget=50))
            for config, name, seed, _ in calls
            if config == row.config
        ]
        assert row.score == pytest.approx(sum(scores) / len(scores)) and row.details["evaluations"] == len(scores)
    assert (tmp_path / "irace.Rdata").exists()
    with pytest.raises(FileExistsError):
        tuner.run(evaluate)


@pytest.mark.backends
@pytest.mark.slow
def test_irace_draws_seeds_in_irace_mode() -> None:
    seeds: set[int] = set()

    def evaluate(config: dict, ctx: EvalContext) -> float:
        seeds.add(ctx.seed)
        return _score(config, ctx)

    best, history = _irace_tuner(max_trials=50, seed=3, n_jobs=2, irace_instance_seeds="irace").run(evaluate)
    assert best in [row.config for row in history]
    assert seeds and not seeds & {11, 12}


@pytest.mark.backends
def test_irace_reraises_evaluation_errors() -> None:
    def evaluate(config: dict, ctx: EvalContext) -> float:
        raise ZeroDivisionError("broken candidate")

    with pytest.raises(RuntimeError, match="broken candidate") as info:
        _irace_tuner(max_trials=50).run(evaluate)
    assert isinstance(info.value.__cause__, ZeroDivisionError)


@pytest.mark.backends
def test_irace_failures_report_the_log_tail() -> None:
    with pytest.raises(RuntimeError, match="Insufficient budget"):
        _irace_tuner(max_trials=1).run(_score)
