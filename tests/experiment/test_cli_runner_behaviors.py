import json
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


def _run_small_cli(monkeypatch, output_root, *extra):
    from vamos.experiment.cli.main import main

    argv = ["prog", "--engine", "numpy", "--population-size", "6", "--max-evaluations", "12", "--seed", "1"]
    monkeypatch.setattr(sys, "argv", [*argv, "--output-root", str(output_root), *extra])
    main()


def _record_eval_strategies(monkeypatch):
    from vamos.experiment import runner_abstractions
    from vamos.foundation.eval.backends import SerialEvalBackend

    calls = []

    def _fake_resolve(name, **kwargs):
        calls.append((name, kwargs.get("dask_address")))
        return SerialEvalBackend()

    monkeypatch.setattr(runner_abstractions, "resolve_eval_strategy", _fake_resolve)
    return calls


def test_cli_dask_address_reaches_eval_backend(monkeypatch, tmp_path):
    calls = _record_eval_strategies(monkeypatch)
    output_root = tmp_path / "results"

    _run_small_cli(
        monkeypatch,
        output_root,
        *("--problem", "zdt1", "--algorithm", "nsgaii", "--eval-strategy", "dask", "--dask-address", "tcp://127.0.0.1:8786"),
    )

    assert calls == [("dask", "tcp://127.0.0.1:8786")]
    manifest = next(output_root.rglob("manifest.json"))
    # The scheduler address is infrastructure, not part of the run: only the strategy is recorded.
    assert "8786" not in manifest.read_text(encoding="utf-8")
    stored = vamos.load_run(manifest.parent)
    assert stored.manifest.resolved_spec["backend"]["evaluation"]["component_id"].startswith("vamos.evaluation:dask")


def test_cli_dask_address_runs_on_a_local_cluster(monkeypatch, tmp_path):
    distributed = pytest.importorskip("dask.distributed")
    output_root = tmp_path / "results"

    with distributed.LocalCluster(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None) as cluster:
        _run_small_cli(
            monkeypatch,
            output_root,
            *("--problem", "zdt1", "--algorithm", "nsgaii", "--eval-strategy", "dask", "--dask-address", cluster.scheduler_address),
        )

    stored = vamos.load_run(next(output_root.rglob("manifest.json")).parent)
    assert stored.result.F is not None and stored.result.F.shape == (6, 2)


def test_cli_problem_block_selects_dask_strategy_and_address(monkeypatch, tmp_path):
    calls = _record_eval_strategies(monkeypatch)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {"problem": "zdt1", "algorithm": "nsgaii"},
                "problems": {"zdt1": {"eval_strategy": "dask", "dask_address": "tcp://127.0.0.1:8787"}},
            }
        ),
        encoding="utf-8",
    )

    _run_small_cli(monkeypatch, tmp_path / "results", "--config", str(config_path))

    assert calls == [("dask", "tcp://127.0.0.1:8787")]


def test_hv_threshold_with_other_algorithm_warns(monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO")

    _run_small_cli(monkeypatch, tmp_path / "results", *("--problem", "zdt1", "--algorithm", "smsemoa", "--hv-threshold", "0.5"))

    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any("--hv-threshold" in message and "smsemoa" in message for message in warnings), warnings


def test_hv_threshold_with_nsgaii_does_not_warn(monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO")

    _run_small_cli(monkeypatch, tmp_path / "results", *("--problem", "zdt1", "--algorithm", "nsgaii", "--hv-threshold", "0.5"))

    assert not [record for record in caplog.records if record.levelname == "WARNING" and "--hv-threshold" in record.getMessage()]


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


class _CountingBackend:
    """Serial evaluation that records how many times the run closes it."""

    def __init__(self):
        from vamos.foundation.eval.backends import SerialEvalBackend

        self._serial = SerialEvalBackend()
        self.closed = 0

    def evaluate(self, X, problem):
        return self._serial.evaluate(X, problem)

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("wrap", ["instance", "spec", "resolver"], ids=["instance", "evaluator-spec", "custom-resolver"])
def test_run_keeps_caller_supplied_backends_open(wrap, tmp_path):
    from vamos.experiment.runner_abstractions import EvaluatorSpec

    backend = _CountingBackend()

    class _Resolver:
        def resolve(self, config):
            return backend

    evaluator = {"instance": backend, "spec": EvaluatorSpec(backend=backend), "resolver": _Resolver()}[wrap]
    selection = make_problem_selection("zdt1", n_var=6)
    for seed in (1, 2):
        config = ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=12, seed=seed, output_root=str(tmp_path))
        runner.run_single("numpy", "nsgaii", selection, config, evaluator=evaluator)

    assert backend.closed == 0


def test_run_closes_the_backend_it_creates(monkeypatch, tmp_path):
    from vamos.experiment import runner_abstractions
    from vamos.experiment.runner_abstractions import EvaluatorSpec

    created = []

    def _fake_resolve(name, **kwargs):
        created.append(_CountingBackend())
        return created[-1]

    monkeypatch.setattr(runner_abstractions, "resolve_eval_strategy", _fake_resolve)
    selection = make_problem_selection("zdt1", n_var=6)
    config = ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=12, seed=1, output_root=str(tmp_path))
    runner.run_single("numpy", "nsgaii", selection, config, evaluator=EvaluatorSpec(backend="serial", dask_address=None))

    assert [backend.closed for backend in created] == [1]
