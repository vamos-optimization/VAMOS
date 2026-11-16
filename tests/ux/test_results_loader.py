import pytest

from vamos.experiment.runner import run_experiment
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.ux.analysis.results import aggregate_results, discover_runs, load_run_data


@pytest.mark.smoke
def test_discover_and_load_results(tmp_path):
    cfg = ExperimentConfig(output_root=str(tmp_path), population_size=6, max_evaluations=20, seed=2)
    run_experiment(
        algorithm="nsgaii",
        problem="zdt1",
        engine="numpy",
        config=cfg,
        selection_pressure=2,
    )

    runs = discover_runs(tmp_path)
    assert runs, "Expected to find at least one run"

    run = runs[0]
    data = load_run_data(run)
    assert data.metadata["resolved_spec"]["problem"]["component_id"] == "vamos.problem:zdt1@1"
    assert data.F is not None and data.F.shape[1] == 2

    agg = aggregate_results(runs)
    if isinstance(agg, list):
        assert any(rec.get("algorithm") == "nsgaii" for rec in agg)
    else:
        assert "algorithm" in agg.columns


def test_discover_runs_skips_replays(tmp_path):
    import vamos

    for seed in (1, 2):
        result = vamos.optimize("zdt1", algorithm="nsgaii", max_evaluations=40, pop_size=8, seed=seed, engine="numpy")
        vamos.save_result(result, tmp_path / f"seed-{seed}")
    replay = vamos.reproduce(tmp_path / "seed-1")
    assert replay.output_root.parent == tmp_path / "replays"  # Inside the tree that discover_runs searches.

    runs = discover_runs(tmp_path)

    assert sorted(run.seed for run in runs) == [1, 2]
    assert not any(run.replay for run in runs)
    everything = discover_runs(tmp_path, include_replays=True)
    assert sorted((run.seed, run.replay) for run in everything) == [(1, False), (1, True), (2, False)]
