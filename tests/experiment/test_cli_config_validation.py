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


@pytest.mark.parametrize(("algorithm", "flag"), _VARIATION_FLAG_PREFIXES)
def test_partial_operator_option_is_completed_or_rejected(monkeypatch, capsys, algorithm, flag):
    err = _usage_error(monkeypatch, capsys, "--algorithm", algorithm, f"--{flag}-crossover-prob", "0.5")
    assert f"--{flag}-crossover-prob" in err
    assert f"--{flag}-crossover " in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--seed", "-1"), "--seed"),
        (("--n-var", "0"), "--n-var"),
        (("--problem", "zdt1", "--n-obj", "3"), "fixed number of objectives"),
        (("--population-size", "20", "--max-evaluations", "40", "--selection-pressure", "50"), "--selection-pressure"),
        (("--algorithm", "smsemoa", "--track-genealogy"), "--track-genealogy"),
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
