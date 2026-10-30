import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.registry import ALGORITHMS
from vamos.experiment import cli, runner
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.hv_stop import build_hv_stop_config
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.tsp import TSPProblem


def test_cli_hv_threshold_requires_reference_for_non_zdt(monkeypatch):
    default_cfg = ExperimentConfig()
    argv = ["prog", "--problem", "dtlz1", "--hv-threshold", "0.5"]
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # keep argparse deterministic
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        cli.parse_args(default_cfg)


def test_cli_hv_threshold_uses_builtin_reference_for_zdt1(monkeypatch):
    default_cfg = ExperimentConfig()
    argv = ["prog", "--problem", "zdt1", "--hv-threshold", "0.25"]
    monkeypatch.setattr(sys, "argv", argv)
    args = cli.parse_args(default_cfg)
    assert args.hv_reference_front
    assert "ZDT1" in args.hv_reference_front.upper()


def test_cli_hv_threshold_uses_builtin_reference_for_zcat(monkeypatch):
    default_cfg = ExperimentConfig()
    argv = ["prog", "--problem", "zcat1", "--n-obj", "3", "--hv-threshold", "0.25"]
    monkeypatch.setattr(sys, "argv", argv)
    args = cli.parse_args(default_cfg)
    assert args.hv_reference_front
    assert args.hv_reference_front.lower().endswith("zcat1.3d.csv")


def test_cli_external_archive_bounded_configuration(monkeypatch):
    default_cfg = ExperimentConfig()
    argv = [
        "prog",
        "--problem",
        "zdt1",
        "--external-archive-size",
        "64",
        "--external-archive-pruning",
        "knn",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = cli.parse_args(default_cfg)
    assert args.external_archive is not None
    assert args.external_archive.capacity == 64
    assert args.external_archive.pruning == "knn"


def test_build_hv_stop_config_uses_builtin_front():
    cfg = build_hv_stop_config(0.1, None, "zdt1")
    assert cfg["target_value"] > 0.0
    assert len(cfg["reference_point"]) == 2
    assert cfg["reference_front_path"].upper().endswith("ZDT1.CSV")


def test_permutation_problem_rejects_unsupported_algorithm():
    selection = SimpleNamespace(instantiate=lambda: TSPProblem(n_cities=6), spec=SimpleNamespace(key="tsp6"), n_var=6, n_obj=2)
    config = ExperimentConfig(population_size=4, offspring_population_size=4, max_evaluations=8, seed=1)
    with pytest.raises(ConfigurationError):
        runner.run_single(
            "numpy",
            "smpso",
            selection,
            config,
        )


def test_runner_rejects_genealogy_for_non_nsgaii():
    selection = make_problem_selection("zdt1", n_var=4)
    config = ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=12, seed=1)
    with pytest.raises(ValueError, match="NSGA-II"):
        runner.run_single("numpy", "ibea", selection, config, track_genealogy=True)


def test_runner_plugin_algorithm_uses_registry_path(tmp_path):
    algo_key = "_runner_mock_algo"

    def mock_algo_builder(cfg, kernel):
        class MockAlgo:
            def __init__(self, config_dict):
                self.config_dict = config_dict
                self.kernel = kernel

            def run(self, problem, termination, seed, eval_strategy=None, live_viz=None):
                return {
                    "X": np.zeros((3, problem.n_var)),
                    "F": np.full((3, problem.n_obj), 0.4, dtype=float),
                    "evaluations": 3,
                }

        return MockAlgo(dict(cfg))

    if algo_key not in ALGORITHMS:
        ALGORITHMS.register(algo_key, mock_algo_builder)

    @dataclass(frozen=True)
    class DummyConfig:
        pop_size: int = 5

        def to_dict(self) -> dict[str, object]:
            return {"pop_size": self.pop_size}

    selection = make_problem_selection("zdt1", n_var=4)
    config = ExperimentConfig(
        output_root=str(tmp_path),
        population_size=5,
        offspring_population_size=5,
        max_evaluations=10,
        seed=1,
    )

    result = runner.run_single("numpy", algo_key, selection, config, algorithm_config=DummyConfig())

    assert result["algorithm"] == algo_key
    assert result["F"].shape == (3, 2)


def test_cli_accepts_registered_plugin_algorithm(monkeypatch):
    algo_key = "_cli_mock_algo"
    if algo_key not in ALGORITHMS:
        ALGORITHMS.register(algo_key, lambda cfg, kernel: None)

    default_cfg = ExperimentConfig()
    argv = ["prog", "--problem", "zdt1", "--algorithm", algo_key]
    monkeypatch.setattr(sys, "argv", argv)
    args = cli.parse_args(default_cfg)

    assert args.algorithm == algo_key


@pytest.mark.parametrize("problem", ["re21", "rwa1", "zcat1"])
def test_cli_runs_problems_without_constraint_attribute(monkeypatch, tmp_path, problem):
    # RE, RWA and ZCAT problems do not declare ``n_constraints``; the CLI runner must treat that as unconstrained.
    from vamos.experiment.cli.main import main

    output_root = tmp_path / "results"
    argv = [
        "prog",
        "--problem",
        problem,
        "--algorithm",
        "nsgaii",
        "--engine",
        "numpy",
        "--population-size",
        "6",
        "--offspring-population-size",
        "6",
        "--max-evaluations",
        "12",
        "--seed",
        "1",
        "--output-root",
        str(output_root),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    manifests = sorted(output_root.rglob("manifest.json"))
    assert len(manifests) == 1
    stored = vamos.load_run(manifests[0].parent)
    assert stored.result.F is not None and stored.result.F.shape[0] > 0


def _run_cli(monkeypatch, argv):
    from vamos.experiment.cli.main import main

    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    main()


def test_cli_rvea_runs_with_default_population(monkeypatch, tmp_path):
    # The CLI default population (100) has no 12-partition lattice on two objectives; RVEA must derive the partitions.
    output_root = tmp_path / "results"
    _run_cli(
        monkeypatch,
        [
            "--problem",
            "zdt1",
            "--algorithm",
            "rvea",
            "--engine",
            "numpy",
            "--max-evaluations",
            "300",
            "--seed",
            "1",
            "--output-root",
            str(output_root),
        ],
    )

    manifests = sorted(output_root.rglob("manifest.json"))
    assert len(manifests) == 1
    stored = vamos.load_run(manifests[0].parent)
    config = stored.manifest.resolved_spec["algorithm"]["config"]
    assert config["pop_size"] == 100
    assert config["n_partitions"] == 99


def test_cli_rvea_population_without_lattice_fails_before_optimizing(monkeypatch, tmp_path, capsys):
    output_root = tmp_path / "results"
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(
            monkeypatch,
            [
                "--problem",
                "dtlz2",
                "--algorithm",
                "rvea",
                "--engine",
                "numpy",
                "--max-evaluations",
                "300",
                "--output-root",
                str(output_root),
            ],
        )

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "pop_size=100" in err
    assert "91" in err and "105" in err
    assert not output_root.exists()


def test_cli_both_continues_after_one_algorithm_fails(monkeypatch, tmp_path, caplog):
    from vamos.experiment import execution

    algo_key = "_cli_failing_algo"

    class _FailingAlgorithm:
        def run(self, problem, termination, seed, eval_strategy=None, live_viz=None):
            raise RuntimeError("deliberate failure")

    if algo_key not in ALGORITHMS:
        ALGORITHMS.register(algo_key, lambda cfg, kernel: _FailingAlgorithm())
    monkeypatch.setattr(execution, "ENABLED_ALGORITHMS", ("nsgaii", algo_key, "spea2"))
    caplog.set_level("INFO")
    output_root = tmp_path / "results"

    with pytest.raises(SystemExit) as excinfo:
        _run_cli(
            monkeypatch,
            [
                "--problem",
                "zdt1",
                "--algorithm",
                "both",
                "--engine",
                "numpy",
                "--population-size",
                "6",
                "--max-evaluations",
                "12",
                "--seed",
                "1",
                "--output-root",
                str(output_root),
            ],
        )

    assert excinfo.value.code != 0
    written = {path.parent.parent.parent.name for path in output_root.rglob("manifest.json")}
    assert written == {"nsgaii", "spea2"}
    assert "Experiment summary" in caplog.text
    assert algo_key in caplog.text and "deliberate failure" in caplog.text


def test_cli_problem_set_continues_after_one_problem_fails(monkeypatch, tmp_path, caplog):
    from vamos.foundation.problem import resolver

    # SMPSO rejects the permutation problem in the middle; the problems around it must still run.
    monkeypatch.setitem(resolver.PROBLEM_SET_PRESETS, "families", ("zdt1", "tsp6", "zdt2"))
    caplog.set_level("INFO")
    output_root = tmp_path / "results"

    with pytest.raises(SystemExit) as excinfo:
        _run_cli(
            monkeypatch,
            [
                "--problem-set",
                "families",
                "--algorithm",
                "smpso",
                "--engine",
                "numpy",
                "--population-size",
                "6",
                "--max-evaluations",
                "12",
                "--seed",
                "1",
                "--output-root",
                str(output_root),
            ],
        )

    assert excinfo.value.code != 0
    written = {path.parents[3].name for path in output_root.rglob("manifest.json")}
    assert len(written) == 2 and all(name.startswith("ZDT") for name in written)
    assert "tsp6 with smpso" in caplog.text
