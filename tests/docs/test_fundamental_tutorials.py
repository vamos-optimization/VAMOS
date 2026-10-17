from __future__ import annotations

import contextlib
import io
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vamos import OptimizationResult

ROOT = Path(__file__).resolve().parents[2]
SLUGS = ("t01-first-result", "t02-constrained-problem", "t03-pareto-decision")


def _load(lesson: int) -> dict:
    return runpy.run_path(str(ROOT / "docs/assets/tutorials" / f"{SLUGS[lesson - 1]}.py"))


def _run(lesson: int, **kwargs) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return _load(lesson)[f"run_t0{lesson}"](**kwargs)


def _assert_nondominated(front: np.ndarray) -> None:
    dominates = np.all(front[:, None] <= front[None, :], axis=2)
    dominates &= np.any(front[:, None] < front[None, :], axis=2)
    assert not dominates.any()


@pytest.mark.parametrize("slug", SLUGS)
def test_download_runs_standalone_without_writing_files(slug: str, tmp_path: Path) -> None:
    script = tmp_path / f"{slug}.py"
    script.write_bytes((ROOT / "docs/assets/tutorials" / script.name).read_bytes())
    completed = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30)
    assert completed.stdout.strip()
    assert list(tmp_path.iterdir()) == [script]


def test_t01_reports_real_objectives_and_aligned_front_rows() -> None:
    data = _run(1)
    result, indices = data["result"], data["front_indices"]
    assert result.X.shape == (40, 30)
    assert result.F.shape == (40, 2)
    assert result.data["evaluations"] == 400
    assert np.all((result.X >= 0) & (result.X <= 1))
    g = 1 + 9 * result.X[:, 1:].mean(axis=1)
    expected = np.column_stack([result.X[:, 0], g * (1 - np.sqrt(result.X[:, 0] / g))])
    np.testing.assert_allclose(result.F, expected)
    np.testing.assert_array_equal(data["front_F"], result.F[indices])
    _assert_nondominated(data["front_F"])


def test_t01_guided_budget_change_keeps_population_shape() -> None:
    result = _run(1, max_evaluations=800)["result"]
    assert result.data["evaluations"] == 800
    assert result.X.shape == (40, 30)


def test_t02_model_probes_feasibility_and_original_rows() -> None:
    module = _load(2)
    assert [module["process_constraints"](x)[0] for x in ([60, 2], [75, 6], [90, 8])] == [330, 0, -270]
    np.testing.assert_allclose(module["process_objectives"]([75, 6]), [0.290625, 23 / 24])
    data = _run(2)
    result, indices = data["result"], data["front_indices"]
    temperature, time = result.X.T
    assert np.all((temperature >= 60) & (temperature <= 100) & (time >= 2) & (time <= 10))
    expected = np.column_stack([((temperature - 60) / 40) ** 2 + time / 40, (100 - temperature) / 40 + 2 / time])
    np.testing.assert_allclose(result.F, expected)
    np.testing.assert_allclose(data["G"][:, 0], 450 - temperature * time)
    np.testing.assert_allclose(result.data["G"], data["G"])
    assert np.all(data["G"][indices] <= 0)
    np.testing.assert_array_equal(data["front_F"], result.F[indices])
    assert result.data["evaluations"] == 400
    _assert_nondominated(data["front_F"])


def test_infeasible_objective_winner_is_removed_before_pareto_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos

    synthetic = OptimizationResult(
        {
            "X": np.array([[60, 2], [75, 6], [90, 8]]),
            "F": np.array([[-10, -10], [0.290625, 23 / 24], [0.7625, 0.5]]),
        }
    )
    monkeypatch.setattr(vamos, "optimize", lambda *args, **kwargs: synthetic)
    data = _load(2)["solve_process"]()
    np.testing.assert_array_equal(data["front_indices"], [1, 2])
    np.testing.assert_array_equal(data["feasible"].X, synthetic.X[[1, 2]])


def test_no_feasible_rows_is_an_explicit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos

    monkeypatch.setattr(vamos, "optimize", lambda *args, **kwargs: OptimizationResult({"X": np.array([[60, 2]]), "F": np.zeros((1, 2))}))
    with pytest.raises(ValueError, match="No feasible candidates"):
        _load(2)["solve_process"]()


def test_t02_guided_constraint_change_updates_the_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load(2)
    constraint = lambda x: [600.0 - x[0] * x[1]]  # noqa: E731
    monkeypatch.setitem(module["solve_process"].__globals__, "process_constraints", constraint)
    assert [constraint(x)[0] for x in ([60, 2], [75, 6], [90, 8])] == [480, 150, -120]
    data = module["solve_process"]()
    np.testing.assert_allclose(data["G"][:, 0], 600 - np.prod(data["result"].X, axis=1))
    assert np.all(data["G"][data["front_indices"]] <= 0)


def test_t03_preferences_select_a_real_feasible_row_and_change_the_tradeoff() -> None:
    energy, shortfall = _run(3), _run(3, weights=(0.2, 0.8))
    for data in (energy, shortfall):
        front = data["front_F"]
        scaled = (front - front.min(axis=0)) / np.where(np.ptp(front, axis=0) > 0, np.ptp(front, axis=0), 1)
        scores = scaled @ data["weights"]
        assert data["selected_row"] == data["front_indices"][np.argmin(scores)]
        np.testing.assert_allclose(data["scores"], scores)
        assert data["G"][data["selected_row"], 0] <= 0
    np.testing.assert_array_equal(energy["result"].X, shortfall["result"].X)
    first = energy["result"].F[energy["selected_row"]]
    second = shortfall["result"].F[shortfall["selected_row"]]
    assert first[0] < second[0]
    assert first[1] > second[1]


@pytest.mark.parametrize("weights", [(0, 0), (-1, 2), (np.nan, 1), (np.inf, 1), (1,), (1, 2, 3)])
def test_invalid_preferences_are_rejected(weights: tuple) -> None:
    with pytest.raises(ValueError, match="weights"):
        _run(3, weights=weights)


def test_generated_sources_output_and_figures_are_synchronized() -> None:
    builder = runpy.run_path(str(ROOT / "tools/build_fundamental_tutorials.py"))
    builder["check"](ROOT)
