"""``vamos profile``: warm-up and repeated timings, failing engines, and the hypervolume reference."""

from __future__ import annotations

import csv
import logging
from types import SimpleNamespace

import numpy as np
import pytest

import vamos
from vamos.experiment.profiler import runner
from vamos.experiment.profiler.cli import main
from vamos.foundation.core.hv_stop import near_nadir_reference
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.foundation.quality_indicators.hypervolume import hypervolume


class _FakeResult:
    def __init__(self) -> None:
        self.F = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
        self.data: dict[str, object] = {"F": self.F}

    def __len__(self) -> int:
        return len(self.F)


def _fake_engines(monkeypatch, *, first_call: float = 0.1, later_calls: float = 0.1, failing: tuple[str, ...] = ()):
    """Replace ``optimize`` and the runner's clock: each call advances the clock by its cost.

    The first call of each engine costs ``first_call`` seconds, later calls ``later_calls``.
    Returns the list of engines in call order.
    """
    clock = [0.0]
    calls: list[str] = []

    def fake_optimize(problem, *, engine, max_evaluations, seed, verbose):  # noqa: ANN001, ARG001
        if engine in failing:
            raise RuntimeError(f"{engine} kernel failed")
        clock[0] += first_call if engine not in calls else later_calls
        calls.append(engine)
        return _FakeResult()

    monkeypatch.setattr(runner, "optimize", fake_optimize)
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    return calls


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _expected_hv(problem: str, budget: int, seed: int = 42) -> float:
    front = np.loadtxt(resolve_reference_front_path(problem, None), delimiter=",")
    ref = near_nadir_reference(front)
    F = np.asarray(vamos.optimize(problem, engine="numpy", max_evaluations=budget, seed=seed, verbose=False).F)
    return hypervolume(F[np.all(F <= ref, axis=1)], ref)


def test_profile_times_after_warmup(monkeypatch):
    # The first run of an engine pays its compilation and imports; the median of the timed runs must not.
    _fake_engines(monkeypatch, first_call=5.0, later_calls=0.1)

    report = runner.run_profile("zdt1", engines=("numpy", "numba"), compute_hv=False)

    assert [result.time_seconds for result in report.results] == [pytest.approx(0.1), pytest.approx(0.1)]


def test_profile_reports_median_and_spread_of_repeats(monkeypatch):
    calls = _fake_engines(monkeypatch)

    report = runner.run_profile("zdt1", engines=("numpy",), compute_hv=False, repeats=4)

    assert calls == ["numpy"] * 5  # one warm-up run and four timed runs
    result = report.results[0]
    assert result.repeats == 4
    assert result.time_seconds == pytest.approx(0.1)
    assert result.time_min == pytest.approx(0.1) and result.time_max == pytest.approx(0.1)


def test_profile_unknown_engine_fails(monkeypatch, capsys):
    calls = _fake_engines(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        main(["--engines", "numpy,foo", "--no-hv"])

    assert excinfo.value.code == 2
    assert "foo" in capsys.readouterr().err
    assert calls == []


def test_profile_failed_engine_exits_nonzero_and_leaves_cells_empty(monkeypatch, tmp_path):
    _fake_engines(monkeypatch, failing=("numba",))
    output = tmp_path / "profile.csv"

    code = main(["--engines", "numpy,numba", "--no-hv", "--output", str(output)])

    assert code == 1
    rows = {row["engine"]: row for row in _read_csv(output)}
    assert float(rows["numpy"]["time_seconds"]) == pytest.approx(0.1)
    assert rows["numba"]["time_seconds"] == "" and rows["numba"]["n_solutions"] == ""
    assert "numba kernel failed" in rows["numba"]["error"]


def test_profile_hypervolume_uses_problem_reference():
    # ZDT3's front spans about [0, 0.85] x [-0.77, 1]: its reference is not [1.1, 1.1].
    report = runner.run_profile("zdt3", engines=("numpy",), budget=2000)

    hv = report.results[0].hypervolume
    assert hv is not None and hv > 0.0
    assert hv == pytest.approx(_expected_hv("zdt3", 2000))


def test_profile_hypervolume_supports_three_objectives():
    report = runner.run_profile("dtlz2", engines=("numpy",), budget=2000)

    hv = report.results[0].hypervolume
    assert hv is not None
    assert hv == pytest.approx(_expected_hv("dtlz2", 2000))
    assert hv > 0.0


def test_profile_explicit_hv_reference(monkeypatch):
    _fake_engines(monkeypatch)

    report = runner.run_profile("zdt1", engines=("numpy",), hv_reference=(0.6, 0.6))

    # Points outside the reference are left out: only (0.5, 0.5) counts.
    assert report.results[0].hypervolume == pytest.approx(0.01)
    with pytest.raises(ValueError, match="2 objectives"):
        runner.run_profile("zdt1", engines=("numpy",), hv_reference=(1.0, 1.0, 1.0))


def test_profile_csv_leaves_missing_hv_empty(tmp_path, caplog):
    output = tmp_path / "profile.csv"

    with caplog.at_level(logging.WARNING):
        code = main(["--problem", "lz09_f1", "--budget", "200", "--output", str(output)])

    assert code == 0
    (row,) = _read_csv(output)
    assert row["hypervolume"] == ""
    assert "--hv-reference" in caplog.text


def test_profile_reports_a_problem_that_cannot_be_built(monkeypatch):
    from vamos.foundation.problem.registry.selection import ProblemSelection

    def missing_dependency(self):  # type: ignore[no-untyped-def]
        raise ImportError("ml_tuning requires scikit-learn")

    monkeypatch.setattr(ProblemSelection, "instantiate", missing_dependency)

    report = runner.run_profile("ml_tuning", engines=("numpy",), budget=200, repeats=1)

    assert [result.engine for result in report.failed] == ["numpy"]
