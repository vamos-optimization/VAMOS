import json
import sys

import pytest

from vamos.foundation.exceptions import VAMOSError


def test_cli_validate_config_exits_cleanly(monkeypatch, tmp_path):
    config_file = tmp_path / "spec.json"
    config_file.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {
                    "algorithm": "nsgaii",
                    "engine": "numpy",
                    "population_size": 10,
                    "max_evaluations": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_file), "--validate-config"])
    from vamos.experiment.cli.main import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0


def test_cli_config_requires_version(monkeypatch, tmp_path):
    config_file = tmp_path / "spec.json"
    config_file.write_text(
        json.dumps(
            {
                "defaults": {
                    "algorithm": "nsgaii",
                    "engine": "numpy",
                    "population_size": 10,
                    "max_evaluations": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_file)])
    from vamos.experiment.cli.main import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2


def test_cli_vamos_error_barrier_suppresses_traceback(monkeypatch, capsys):
    from vamos.experiment.cli import main as cli_main

    def boom(argv=None):  # noqa: ANN001
        raise VAMOSError("bad config", suggestion="fix it")

    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(cli_main, "_dispatch_subcommand", lambda argv: False)
    monkeypatch.setattr(cli_main, "_config_only_path", lambda argv: None)
    monkeypatch.setattr(cli_main, "_run_standard_cli", boom)

    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "Error: bad config" in captured.err
    assert "Suggestion: fix it" in captured.err
    assert "Traceback" not in captured.err


def test_cli_traceback_flag_preserves_vamos_error(monkeypatch):
    from vamos.experiment.cli import main as cli_main

    def boom(argv=None):  # noqa: ANN001
        raise VAMOSError("bad config")

    monkeypatch.setattr(sys, "argv", ["prog", "--traceback"])
    monkeypatch.setattr(cli_main, "_dispatch_subcommand", lambda argv: False)
    monkeypatch.setattr(cli_main, "_config_only_path", lambda argv: None)
    monkeypatch.setattr(cli_main, "_run_standard_cli", boom)

    with pytest.raises(VAMOSError):
        cli_main.main()


# Algorithm name and the prefix of its variation flags (NSGA-III's flags are spelled --nsga3-*).
_VARIATION_FLAG_PREFIXES = [
    ("nsgaii", "nsgaii"),
    ("moead", "moead"),
    ("smsemoa", "smsemoa"),
    ("nsgaiii", "nsga3"),
    ("agemoea", "agemoea"),
    ("rvea", "rvea"),
]


def _parse(monkeypatch, *argv):
    from vamos.experiment import cli
    from vamos.foundation.core.experiment_config import ExperimentConfig

    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return cli.parse_args(ExperimentConfig())


def _run_and_load(monkeypatch, tmp_path, *argv):
    import vamos
    from vamos.experiment.cli.main import main

    output_root = tmp_path / "results"
    base = ["prog", "--problem", "zdt1", "--engine", "numpy", "--population-size", "13", "--max-evaluations", "26", "--seed", "1"]
    monkeypatch.setattr(sys, "argv", [*base, "--output-root", str(output_root), *argv])
    main()
    return vamos.load_run(next(output_root.rglob("manifest.json")).parent)


def _usage_error(monkeypatch, capsys, *argv):
    with pytest.raises(SystemExit) as excinfo:
        _parse(monkeypatch, *argv)
    err = capsys.readouterr().err
    assert excinfo.value.code == 2, err
    assert "Traceback" not in err
    return err


@pytest.mark.parametrize(("algorithm", "flag"), _VARIATION_FLAG_PREFIXES)
def test_nsga3_variation_options_reach_resolved_spec(monkeypatch, tmp_path, algorithm, flag):
    stored = _run_and_load(
        monkeypatch,
        tmp_path,
        *("--algorithm", algorithm, f"--{flag}-crossover", "sbx", f"--{flag}-crossover-prob", "0.5", f"--{flag}-crossover-eta", "5"),
        *(f"--{flag}-mutation", "polynomial", f"--{flag}-mutation-eta", "7"),
    )

    operators = stored.manifest.resolved_spec["operators"]
    assert dict(operators["crossover"]["config"]) == {"prob": 0.5, "eta": 5.0}
    assert operators["mutation"]["config"]["eta"] == 7.0


def test_nsga3_config_defaults_reach_resolved_spec(monkeypatch, tmp_path):
    config_path = tmp_path / "spec.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {
                    "algorithm": "nsgaiii",
                    "nsgaiii": {"crossover": {"method": "sbx", "prob": 0.5, "eta": 5}, "mutation": {"method": "polynomial", "eta": 7}},
                },
            }
        ),
        encoding="utf-8",
    )

    stored = _run_and_load(monkeypatch, tmp_path, "--config", str(config_path))

    operators = stored.manifest.resolved_spec["operators"]
    assert dict(operators["crossover"]["config"]) == {"prob": 0.5, "eta": 5.0}
    assert operators["mutation"]["config"]["eta"] == 7.0


def test_nsga3_probability_is_range_checked(monkeypatch, capsys):
    err = _usage_error(monkeypatch, capsys, "--algorithm", "nsgaiii", "--nsga3-crossover", "sbx", "--nsga3-crossover-prob", "7")
    assert "--nsga3-crossover-prob must be within [0, 1]" in err


def _run_error(monkeypatch, capsys, tmp_path, *argv):
    """Run the CLI and return stderr of a configuration error that stopped it before any run."""
    from vamos.experiment.cli.main import main

    output_root = tmp_path / "results"
    base = ["prog", "--problem", "zdt1", "--engine", "numpy", "--population-size", "13", "--max-evaluations", "26"]
    monkeypatch.setattr(sys, "argv", [*base, "--output-root", str(output_root), *argv])
    with pytest.raises(SystemExit) as excinfo:
        main()
    err = capsys.readouterr().err
    assert excinfo.value.code == 2, err
    assert "Traceback" not in err
    assert not output_root.exists()
    return err


@pytest.mark.parametrize(("algorithm", "flag"), _VARIATION_FLAG_PREFIXES)
def test_partial_operator_option_is_completed_or_rejected(monkeypatch, capsys, tmp_path, algorithm, flag):
    err = _run_error(monkeypatch, capsys, tmp_path, "--algorithm", algorithm, f"--{flag}-crossover-prob", "0.5")
    assert f"--{flag}-crossover-prob" in err
    assert f"--{flag}-crossover " in err


def test_problem_operator_override_completes_an_unnamed_cli_operator(monkeypatch, tmp_path):
    spec = {"problems": {"zdt1": {"nsgaii": {"crossover": {"method": "sbx", "prob": 0.8}}}}}
    config = tmp_path / "spec.json"
    config.write_text(json.dumps({"version": "1", **spec}), encoding="utf-8")

    stored = _run_and_load(monkeypatch, tmp_path, "--config", str(config), "--algorithm", "nsgaii", "--nsgaii-crossover-prob", "0.5")

    assert stored.manifest.resolved_spec["operators"]["crossover"]["config"]["prob"] == 0.8


def test_genealogy_is_checked_against_each_problems_algorithm(monkeypatch, capsys, tmp_path):
    spec = {"defaults": {"algorithm": "smsemoa", "track_genealogy": True}, "problems": {"zdt1": {"algorithm": "nsgaii"}}}
    config = tmp_path / "spec.json"
    config.write_text(json.dumps({"version": "1", **spec}), encoding="utf-8")

    stored = _run_and_load(monkeypatch, tmp_path, "--config", str(config))
    assert stored.result.F is not None

    err = _run_error(monkeypatch, capsys, tmp_path / "other", "--algorithm", "smsemoa", "--track-genealogy")
    assert "track_genealogy" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--seed", "-1"), "--seed"),
        (("--n-var", "0"), "--n-var"),
        (("--problem", "zdt1", "--n-obj", "3"), "fixed number of objectives"),
    ],
)
def test_invalid_cli_inputs_exit_with_usage_error(monkeypatch, capsys, argv, message):
    err = _usage_error(monkeypatch, capsys, *argv)
    assert message in err


def test_options_of_other_algorithms_warn(monkeypatch, caplog):
    caplog.set_level("WARNING")

    _parse(monkeypatch, "--algorithm", "smsemoa", "--nsga3-crossover", "sbx", "--external-archive-size", "10")

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "--nsga3-crossover" in messages
    assert "--external-archive-size" in messages


@pytest.mark.parametrize("algorithm", ["nsgaii", "spea2", "smpso"])
def test_external_archive_size_does_not_warn_for_archive_algorithms(monkeypatch, caplog, algorithm):
    caplog.set_level("WARNING")

    _parse(monkeypatch, "--algorithm", algorithm, "--external-archive-size", "10")

    assert "--external-archive-size" not in " ".join(record.getMessage() for record in caplog.records)


def test_config_blocks_of_other_algorithms_do_not_warn(monkeypatch, tmp_path, caplog):
    # A config may carry blocks for several algorithms; only flags given on the command line are "ignored".
    config_path = tmp_path / "spec.json"
    config_path.write_text(
        json.dumps(
            {"version": "1", "defaults": {"algorithm": "nsgaii", "moead": {"crossover": {"method": "sbx", "prob": 1.0, "eta": 20}}}}
        ),
        encoding="utf-8",
    )
    caplog.set_level("WARNING")

    _parse(monkeypatch, "--config", str(config_path))

    assert "Ignoring" not in " ".join(record.getMessage() for record in caplog.records)


def _write_spec(tmp_path, spec):
    config_file = tmp_path / "spec.json"
    config_file.write_text(json.dumps({"version": "1", **spec}), encoding="utf-8")
    return config_file


def test_selection_pressure_is_checked_against_each_problems_own_population(monkeypatch, tmp_path):
    from vamos.experiment.cli.main import main

    defaults = {"algorithm": "nsgaii", "engine": "numpy", "population_size": 20, "selection_pressure": 50, "max_evaluations": 200}
    spec = {"defaults": {**defaults, "output_root": str(tmp_path / "results")}, "problems": {"zdt1": {"population_size": 100}}}
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(_write_spec(tmp_path, spec)), "--problem", "zdt1"])

    main()

    assert list((tmp_path / "results").rglob("manifest.json"))


def test_selection_pressure_above_a_problems_population_stops_before_any_run(monkeypatch, tmp_path, capsys):
    from vamos.experiment.cli.main import main

    output_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--problem",
            "zdt1",
            "--population-size",
            "20",
            "--max-evaluations",
            "40",
            "--selection-pressure",
            "50",
            "--output-root",
            str(output_root),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    err = capsys.readouterr().err
    assert excinfo.value.code == 2, err
    assert "selection_pressure (50) cannot exceed population_size (20)" in err
    assert "Traceback" not in err
    assert not output_root.exists()


def test_problem_operator_named_with_name_key_completes_an_unnamed_cli_operator(monkeypatch, tmp_path):
    spec = {"problems": {"zdt1": {"nsgaii": {"crossover": {"name": "sbx", "prob": 0.8}}}}}
    config = tmp_path / "spec.json"
    config.write_text(json.dumps({"version": "1", **spec}), encoding="utf-8")

    stored = _run_and_load(monkeypatch, tmp_path, "--config", str(config), "--algorithm", "nsgaii", "--nsgaii-crossover-prob", "0.5")

    assert stored.manifest.resolved_spec["operators"]["crossover"]["config"]["prob"] == 0.8


def test_explicit_selection_operator_replaces_the_pressure_check(monkeypatch, tmp_path):
    from vamos.experiment.cli.main import main

    defaults = {"algorithm": "nsgaii", "engine": "numpy", "population_size": 20, "selection_pressure": 50, "max_evaluations": 40}
    spec = {
        "defaults": {**defaults, "output_root": str(tmp_path / "results")},
        "problems": {"zdt1": {"nsgaii": {"selection": {"method": "tournament", "size": 2}}}},
    }
    config = tmp_path / "spec.json"
    config.write_text(json.dumps({"version": "1", **spec}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config), "--problem", "zdt1"])

    main()

    assert list((tmp_path / "results").rglob("manifest.json"))


_YAML_ALGORITHM_FIELDS = [
    ("moead", {"neighbor_size": 7, "delta": 0.5, "replace_limit": 5, "batch_size": 2}),
    ("nsgaiii", {"reference_directions": {"divisions": 6}, "enforce_ref_dirs": False}),
    ("smsemoa", {"eliminate_duplicates": True, "reference_point": {"vector": None, "offset": 2.0, "adaptive": True}}),
    ("smpso", {"archive_size": 7}),
    ("nsgaii", {"result_mode": "non_dominated", "live_callback_mode": "population"}),
    ("spea2", {"constraint_mode": "none", "mutation_prob_factor": 2.0}),
    ("ibea", {"constraint_mode": "none", "result_mode": "population"}),
    ("agemoea", {"constraint_mode": "none"}),
    ("rvea", {"mutation_prob_factor": 2.0}),
]


@pytest.mark.parametrize(("algorithm", "fields"), _YAML_ALGORITHM_FIELDS, ids=[case[0] for case in _YAML_ALGORITHM_FIELDS])
def test_cli_yaml_algorithm_fields_reach_the_resolved_config(monkeypatch, tmp_path, algorithm, fields):
    config = _write_spec(tmp_path, {"defaults": {"algorithm": algorithm, algorithm: fields}})

    stored = _run_and_load(monkeypatch, tmp_path, "--config", str(config))

    resolved = stored.manifest.resolved_spec["algorithm"]["config"]
    for name, value in fields.items():
        if isinstance(value, dict):
            assert {key: resolved[name][key] for key in value} == value, name
        else:
            assert resolved[name] == value, name


def test_cli_problem_block_algorithm_fields_override_the_defaults_block(monkeypatch, tmp_path):
    spec = {"defaults": {"algorithm": "moead", "moead": {"delta": 0.5}}, "problems": {"zdt1": {"moead": {"delta": 0.7}}}}

    stored = _run_and_load(monkeypatch, tmp_path, "--config", str(_write_spec(tmp_path, spec)))

    assert stored.manifest.resolved_spec["algorithm"]["config"]["delta"] == 0.7


@pytest.mark.parametrize(
    ("algorithm", "field", "value", "hint"),
    [
        ("nsgaii", "pop_size", 50, "population_size"),
        ("nsgaii", "offspring_size", 10, "offspring_population_size"),
        ("spea2", "external_archive", {"capacity": 10}, "archive"),
        ("nsgaii", "track_genealogy", True, "track_genealogy"),
        ("nsgaii", "generation_callback", "module.fn", "Python API"),
        ("nsgaii", "parent_selection_filter", "module.fn", "Python API"),
        ("nsgaiii", "pop_size_auto", True, "population_size"),
    ],
)
def test_cli_yaml_rejects_algorithm_fields_the_cli_sets_elsewhere(monkeypatch, capsys, tmp_path, algorithm, field, value, hint):
    config = _write_spec(tmp_path, {"defaults": {"algorithm": algorithm, algorithm: {field: value}}})

    err = _usage_error(monkeypatch, capsys, "--config", str(config))

    assert f"{algorithm}.{field}" in err
    assert hint in err


def test_cli_yaml_out_of_domain_algorithm_field_stops_before_any_run(monkeypatch, capsys, tmp_path):
    from vamos.experiment.cli.main import main

    output_root = tmp_path / "results"
    spec = {"defaults": {"algorithm": "moead", "moead": {"neighbor_size": 50}}}
    base = ["prog", "--problem", "zdt1", "--population-size", "13", "--max-evaluations", "26", "--output-root", str(output_root)]
    monkeypatch.setattr(sys, "argv", [*base, "--config", str(_write_spec(tmp_path, spec))])

    with pytest.raises(SystemExit) as excinfo:
        main()

    err = capsys.readouterr().err
    assert excinfo.value.code == 2, err
    assert "neighbor_size" in err
    assert not output_root.exists()


def test_cli_yaml_initializer_reaches_the_resolved_operators(monkeypatch, tmp_path):
    config = _write_spec(tmp_path, {"defaults": {"algorithm": "ibea", "ibea": {"initializer": {"type": "lhs"}}}})

    stored = _run_and_load(monkeypatch, tmp_path, "--config", str(config))

    assert stored.manifest.resolved_spec["operators"]["initializer"]["component_id"] == "vamos.operator:lhs@1"


def test_cli_yaml_steady_state_runs_with_one_offspring(monkeypatch, capsys, tmp_path):
    spec = {"defaults": {"algorithm": "nsgaii", "offspring_population_size": 1, "nsgaii": {"steady_state": True}}}

    stored = _run_and_load(monkeypatch, tmp_path, "--config", str(_write_spec(tmp_path, spec)))

    assert stored.manifest.resolved_spec["algorithm"]["config"]["steady_state"] is True

    conflicting = _write_spec(tmp_path, {"defaults": {"algorithm": "nsgaii", "nsgaii": {"steady_state": True}}})
    with pytest.raises(SystemExit) as excinfo:
        _run_and_load(monkeypatch, tmp_path / "conflict", "--config", str(conflicting))
    err = capsys.readouterr().err
    assert excinfo.value.code == 2, err
    assert "steady_state" in err and "Traceback" not in err
