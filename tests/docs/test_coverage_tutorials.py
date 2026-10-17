from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/coverage"


@pytest.fixture(scope="module")
def lessons():
    return runpy.run_path(str(ROOT / "examples/tutorials/capability_examples.py"))


def test_generated_coverage_is_current():
    subprocess.run([sys.executable, "tools/build_coverage_tutorials.py", "--check"], cwd=ROOT, check=True)


@pytest.mark.parametrize("algorithm", ("smsemoa", "spea2", "ibea", "smpso", "agemoea", "rvea"))
def test_algorithms_honor_budget_and_return_aligned_fronts(lessons, algorithm):
    data = lessons[f"run_{algorithm}"](budget=80)
    result = data["result"]
    assert result.data["evaluations"] == 80
    assert len(result.X) == len(result.F) == (10 if algorithm == "rvea" else 40)
    assert np.isfinite(result.F).all()
    np.testing.assert_array_equal(data["front"], result.F[data["front_rows"]])


@pytest.mark.parametrize("encoding", ("binary", "integer", "permutation", "mixed"))
def test_encoding_domains_survive_variation(lessons, encoding):
    data = lessons["run_encoding"](encoding, budget=120)
    result, problem = data["result"], data["problem"]
    assert result.data["evaluations"] == 120
    assert (result.X >= problem.xl).all() and (result.X <= problem.xu).all()
    if encoding == "permutation":
        np.testing.assert_array_equal(np.sort(result.X, axis=1), np.tile(np.arange(problem.n_var), (40, 1)))
    elif encoding == "binary":
        assert np.isin(result.X, (0, 1)).all()
    else:
        indices = np.arange(problem.n_var) if encoding == "integer" else np.r_[problem.mixed_spec["int_idx"], problem.mixed_spec["cat_idx"]]
        np.testing.assert_array_equal(result.X[:, indices], result.X[:, indices].round())


def test_engineering_selection_is_feasible_and_preserves_decision_row(lessons):
    data = lessons["run_engineering"]()
    result, row = data["result"], data["selected_row"]
    assert data["problem"].n_constraints == 4
    assert result.data["G"].shape == (40, 4)
    assert (result.data["G"][row] <= 0).all()
    assert row in data["front_rows"]
    assert result.F[row, 0] == min(data["front"][:, 0])
    out = {"F": np.empty((1, 2)), "G": np.empty((1, 4))}
    data["problem"].evaluate(result.X[[row]], out)
    np.testing.assert_allclose(out["F"][0], result.F[row])
    np.testing.assert_allclose(out["G"][0], result.data["G"][row])


def test_missing_declared_constraints_cannot_be_reported_as_feasible(lessons, monkeypatch):
    import vamos
    from vamos import OptimizationResult
    from vamos.problems import WeldedBeamDesignProblem

    problem = WeldedBeamDesignProblem()
    problem.n_constraints = 4
    monkeypatch.setattr(vamos, "optimize", lambda *args, **kwargs: OptimizationResult({"X": np.zeros((1, 6)), "F": np.ones((1, 2))}))
    with pytest.raises(ValueError, match="constraints are missing"):
        lessons["execute_example"](problem, "nsgaii", None, 40)


def test_zdt1_figure_includes_analytical_reference_and_refuses_overwrite(lessons, tmp_path):
    pytest.importorskip("matplotlib")
    data = lessons["run_smsemoa"](budget=80)
    output = tmp_path / "front.svg"
    lessons["plot_example"](data, output)
    original = output.read_bytes()
    assert b"Analytical front" in original
    with pytest.raises(FileExistsError):
        lessons["plot_example"](data, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize("name", ("feature_selection", "model_tuning"))
def test_learning_cases_return_valid_decisions_and_objectives(lessons, name):
    pytest.importorskip("sklearn")
    data = lessons[f"run_{name}"](budget=32)
    result = data["result"]
    assert result.data["evaluations"] == 32
    assert np.isfinite(result.F).all() and np.all((result.F[:, 0] >= 0) & (result.F[:, 0] <= 1))
    if name == "feature_selection":
        assert np.isin(result.X, (0, 1)).all()
        nonempty = result.X.sum(axis=1) > 0
        np.testing.assert_array_equal(result.F[nonempty, 1], result.X[nonempty].sum(axis=1))
    else:
        assert np.isin(result.X[:, 2], (0, 1, 2)).all()
        assert np.isin(result.X[:, 3], (2, 3, 4, 5)).all()


def test_ablation_changes_only_mutation_and_metrics_match_verified_runs(tmp_path):
    moocore = pytest.importorskip("moocore")
    from vamos import load_study

    module = runpy.run_path(str(ROOT / "examples/tutorials/advanced_workflows.py"))
    data = module["run_ablation_example"](tmp_path / "ablation", budget=80, seeds=(11, 22))
    configs = [
        load_study(tmp_path / "ablation" / name).spec.as_intent_dict()["run_defaults"]["algorithm_configs"]
        for name in ("baseline", "without-mutation")
    ]
    assert configs[0] != configs[1]

    # The public serialized specs permit an independent recursive difference check.
    def differences(a, b, path=()):
        if isinstance(a, dict) and isinstance(b, dict):
            assert a.keys() == b.keys()
            return [d for key in a for d in differences(a[key], b[key], (*path, key))]
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            assert len(a) == len(b)
            return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, (*path, i))]
        return [] if a == b else [path]

    changed = differences(*configs)
    assert len(changed) == 1 and changed[0][-3:] == ("mutation", 1, "prob")
    assert len(data["rows"]) == 4
    for row in data["rows"]:
        assert row["state"] == "succeeded" and row["evaluations"] == 80
        assert row["hv"] == pytest.approx(moocore.hypervolume(row["front"], ref=[1.1, 11]))
        assert row["study_id"] and row["task_id"] and row["run_id"]


def test_recorded_tuning_splits_and_performance_samples_are_not_pooled():
    data = json.loads((ASSETS / "observations.json").read_text(encoding="utf-8"))
    tuning = data["tuning"]
    splits = list(map(set, tuning["seeds"].values()))
    assert all(not a & b for i, a in enumerate(splits) for b in splits[i + 1 :])
    assert len(tuning["trials"]) == 2 and tuning["backend"] == "random"
    performance = data["performance"]
    assert sum(r["evaluations"] for r in performance["rows"]) == 4800
    for backend in ("numpy", "numba", "moocore"):
        rows = [r for r in performance["rows"] if r["backend"] == backend]
        assert [r["seed"] for r in rows if r["phase"] == "subsequent"] == [0, 1, 2]
        assert len([r for r in rows if r["phase"] == "first"]) == 1
        assert all(np.isfinite(r["seconds"]) and r["seconds"] > 0 and np.isfinite(r["hv"]) for r in rows)


def test_existing_advanced_output_is_preserved(tmp_path):
    module = runpy.run_path(str(ROOT / "examples/tutorials/advanced_workflows.py"))
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module["new_output"](tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("script", ("smsemoa", "rvea", "permutation", "engineering", "tuning"))
def test_downloads_execute_outside_checkout(script, tmp_path):
    local = tmp_path / f"{script}.py"
    local.write_bytes((ASSETS / local.name).read_bytes())
    args = ["--output", str(tmp_path / "tuned")] if script == "tuning" else ["--budget", "400"]
    completed = subprocess.run([sys.executable, str(local), *args], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60)
    assert "Experimental tuner" in completed.stdout if script == "tuning" else "Evaluations: 400" in completed.stdout
