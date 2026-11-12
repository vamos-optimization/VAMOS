from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import vamos
from vamos.engine.config.loader import load_experiment_spec

REPO = Path(__file__).resolve().parents[2]


def test_hv_archive_metrics_are_canonical(tmp_path: Path) -> None:
    source = REPO / "experiments" / "configs" / "hv_archive_validation_slice.yml"
    spec = load_experiment_spec(source)
    spec["defaults"]["output_root"] = str(tmp_path / "results")
    spec["defaults"]["population_size"] = 16
    spec["defaults"]["offspring_population_size"] = 16
    spec["defaults"]["max_evaluations"] = 64
    config = tmp_path / "hook-spec.json"
    config.write_text(json.dumps(spec), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(config)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout

    run_dirs = [path.parent for path in (tmp_path / "results").rglob("manifest.json")]
    assert len(run_dirs) == 1
    run = vamos.load_run(run_dirs[0], verify="all")
    assert {path.name for path in run.root.iterdir()} == {"manifest.json", "result.npz", "environment.json"}
    outcome = run.manifest["outcome"]
    hooks = outcome["metrics"]["hooks"]
    assert isinstance(hooks, Mapping)
    assert hooks["stopping"]["enabled"] is True
    assert hooks["archive"]["enabled"] is True
    assert hooks["stopping"]["trace"]
    assert hooks["archive"]["trace"]


def test_cli_genealogy_summary_is_in_canonical_metrics(tmp_path: Path) -> None:
    config = tmp_path / "genealogy.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {
                    "output_root": str(tmp_path / "results"),
                    "engine": "numpy",
                    "algorithm": "nsgaii",
                    "problem": "zdt1",
                    "population_size": 8,
                    "offspring_population_size": 8,
                    "max_evaluations": 16,
                    "seed": 0,
                    "track_genealogy": True,
                },
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(config)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout

    run_dirs = [path.parent for path in (tmp_path / "results").rglob("manifest.json")]
    assert len(run_dirs) == 1
    run = vamos.load_run(run_dirs[0], verify="all")
    genealogy = run.manifest["outcome"]["metrics"]["genealogy"]
    assert genealogy["operator_stats"]
    assert genealogy["generation_contributions"]


def _archive_hook(capacity: int | None = 20, *, stopping: bool = False, objective_tolerance: float = 0.0):
    from vamos.engine.archive import ExternalArchiveConfig
    from vamos.engine.hooks.hv_archive_hooks import HookManager, HookManagerConfig
    from vamos.engine.hooks.hv_convergence import HVConvergenceConfig

    return HookManager(
        HookManagerConfig(
            stopping_enabled=stopping,
            stop_cfg=HVConvergenceConfig(every_k=50),
            archive_enabled=True,
            archive_cfg=ExternalArchiveConfig(capacity=capacity, objective_tolerance=objective_tolerance),
            hv_ref_point=[2.0, 11.0],
        )
    )


def _counting_archive_rows(hook) -> list[int]:  # noqa: ANN001
    """Record the number of rows of every archive update of ``hook``."""
    offered: list[int] = []
    original = hook._ensure_archive

    def ensure(F):  # noqa: ANN001 - wraps the archive created on the first update
        original(F)
        archive = hook.archive
        if archive is not None and not getattr(archive, "_counted", False):
            update = archive.update

            def counted(X, F, G=None):  # noqa: ANN001
                offered.append(int(F.shape[0]))
                return update(X, F, G)

            archive.update = counted
            archive._counted = True

    hook._ensure_archive = ensure
    return offered


def test_archive_hook_cost_is_bounded_in_steady_state() -> None:
    from vamos.algorithms import MOEADConfig
    from vamos.engine.algorithm.moead import MOEAD
    from vamos.foundation.kernel.numpy_backend import NumPyKernel
    from vamos.foundation.problem.zdt1 import ZDT1Problem

    config = (
        MOEADConfig.builder()
        .pop_size(20)
        .neighbor_size(5)
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .build()
    )
    hook = _archive_hook()
    offered = _counting_archive_rows(hook)
    MOEAD(config.to_dict(), kernel=NumPyKernel()).run(ZDT1Problem(n_var=6), ("max_evaluations", 2000), seed=1, live_viz=hook)

    # MOEA/D notifies after every offspring with its whole population; the hook offers
    # each row once, so the archive sees at most one row per evaluation.
    assert 0 < sum(offered) <= 2000
    assert len(offered) <= 2000 - 20 + 1
    archive = hook.metadata_payload()["archive"]
    assert archive["total_inserted"] == sum(offered)
    assert len(archive["trace"]) == len(offered)


def test_archive_hook_offers_only_rows_it_has_not_seen() -> None:
    hook = _archive_hook()
    offered = _counting_archive_rows(hook)
    F = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    hook.on_generation(0, F=F, stats={"evals": 4})
    hook.on_generation(1, F=F, stats={"evals": 5})  # the child was rejected: nothing new
    hook.on_generation(2, F=np.vstack([F[:3], [[3.5, 0.5]]]), stats={"evals": 6})

    assert offered == [4, 1]
    trace = hook.metadata_payload()["archive"]["trace"]
    assert [(row["evals"], row["inserted"]) for row in trace] == [(4, 4), (6, 1)]


def test_archive_hook_with_an_objective_tolerance_reconsiders_earlier_near_duplicates() -> None:
    # With a tolerance, a row dropped as a near-duplicate of [1, 1] becomes eligible once
    # [0.7, 1] dominates [1, 1]; offering only new rows would never offer it again.
    hook = _archive_hook(capacity=None, objective_tolerance=0.2)
    hook.on_generation(0, F=np.array([[1.0, 1.0], [1.1, 0.9]]), stats={"evals": 2})
    hook.on_generation(1, F=np.array([[0.7, 1.0], [1.1, 0.9]]), stats={"evals": 3})

    assert sorted(map(tuple, hook._archive_F.tolist())) == [(0.7, 1.0), (1.1, 0.9)]


def test_archive_hook_counters_separate_dominated_rows_from_capacity_pruning() -> None:
    hook = _archive_hook(capacity=3)
    front = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    hook.on_generation(0, F=front, stats={"evals": 5})
    hook.on_generation(1, F=np.vstack([front[:4], [[5.0, 5.0]]]), stats={"evals": 10})

    archive = hook.metadata_payload()["archive"]
    first, second = archive["trace"]
    assert (first["inserted"], first["pruned_by_capacity"], first["dominated_or_duplicate"]) == (5, 2, 0)
    # Only [5, 5] is new in the second population, and it is dominated.
    assert (second["inserted"], second["pruned_by_capacity"], second["dominated_or_duplicate"]) == (1, 0, 1)
    for row in (first, second):
        assert row["pruned"] == row["dominated_or_duplicate"] + row["pruned_by_capacity"]
    assert (archive["total_inserted"], archive["total_pruned_by_capacity"], archive["total_dominated_or_duplicate"]) == (6, 2, 1)
