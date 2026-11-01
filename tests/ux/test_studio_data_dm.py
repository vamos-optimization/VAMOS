from pathlib import Path

import numpy as np

from vamos.ux.studio.data import RunRecord, build_fronts, normalize_objectives
from vamos.ux.studio.dm import build_decision_view, feasible_indices, filter_by_objective_ranges, rank_by_score
from vamos.ux.studio.export import export_solutions_to_csv, export_solutions_to_json


def test_build_fronts_and_decision_view(tmp_path: Path):
    runs = [RunRecord(None, "exp", "prob", "algo", 0, np.array([[0.1, 0.2]]), None)]
    runs.extend([RunRecord(None, "exp2", "prob", "algo", 1, np.array([[0.2, 0.1]]), None)])
    fronts = build_fronts(runs)
    assert len(fronts) == 1
    view = build_decision_view(fronts[0])
    assert view.normalized_F.shape == fronts[0].points_F.shape
    order = rank_by_score(view, "weighted_sum")
    assert order.shape[0] == fronts[0].points_F.shape[0]
    feasible = feasible_indices(view, max_violation=0.0)
    assert feasible.size == fronts[0].points_F.shape[0]
    filt = filter_by_objective_ranges(view, [(0.0, 0.15), (None, None)])
    assert filt.size >= 1


def test_export_helpers(tmp_path: Path):
    from vamos.ux.studio.data import FrontRecord

    front = FrontRecord(
        problem_name="prob",
        algorithm_name="algo",
        points_F=np.array([[0.1, 0.2], [0.2, 0.1]]),
        points_X=np.array([[1.0, 2.0], [3.0, 4.0]]),
        constraints=None,
        extra={},
    )
    view = build_decision_view(front)
    json_path = export_solutions_to_json(view, [0], tmp_path / "out.json")
    csv_path = export_solutions_to_csv(view, [0, 1], tmp_path / "out.csv")
    assert json_path.exists()
    assert csv_path.exists()
    assert json_path.read_text()
    assert csv_path.read_text()


def test_normalize_objectives():
    F = np.array([[1, 2], [3, 4]])
    norm = normalize_objectives(F)
    assert np.allclose(norm[0], [0.0, 0.0])
    assert np.allclose(norm[1], [1.0, 1.0])


def test_build_fronts_merges_seeds_without_dominated_points():
    # Seed 1's (0.5, 0.5) is dominated by seed 0's (0.4, 0.4); X rows name each point.
    runs = [
        RunRecord(None, "e0", "prob", "algo", 0, np.array([[0.2, 0.6], [0.4, 0.4]]), np.array([[20.0], [40.0]])),
        RunRecord(None, "e1", "prob", "algo", 1, np.array([[0.5, 0.5], [0.6, 0.1]]), np.array([[50.0], [60.0]])),
    ]

    (front,) = build_fronts(runs, merge_seeds=True)

    assert sorted(map(tuple, front.points_F)) == [(0.2, 0.6), (0.4, 0.4), (0.6, 0.1)]
    assert front.points_X is not None
    assert {tuple(f): float(x[0]) for f, x in zip(front.points_F, front.points_X, strict=True)} == {
        (0.2, 0.6): 20.0,
        (0.4, 0.4): 40.0,
        (0.6, 0.1): 60.0,
    }


def test_build_fronts_drops_x_when_a_run_lacks_it():
    runs = [
        RunRecord(None, "e0", "prob", "algo", 0, np.array([[0.2, 0.6]]), np.array([[20.0]])),
        RunRecord(None, "e1", "prob", "algo", 1, np.array([[0.6, 0.1]]), None),
    ]

    (front,) = build_fronts(runs, merge_seeds=True)

    assert front.points_X is None
    assert "decision variables" in front.extra.get("note", "")
