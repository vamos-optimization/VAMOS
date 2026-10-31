import json
import sys

import pytest


def test_cli_ablation_runs_and_writes_variants(monkeypatch, tmp_path):
    output_root = tmp_path / "ablation_results"
    config = {
        "algorithm": "moead",
        "engine": "numpy",
        "output_root": str(output_root),
        "default_max_evals": 12,
        "problems": ["zdt1"],
        "seeds": [1],
        "base_config": {"population_size": 6, "offspring_population_size": 6},
        "variants": [
            {"name": "baseline"},
            {
                "name": "tuned",
                "config_overrides": {"population_size": 8, "offspring_population_size": 8},
                "variations": {"moead": {"aggregation": {"method": "pbi", "theta": 5.0}}},
            },
        ],
    }
    config_path = tmp_path / "ablation.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    main()

    # Each variant writes under its own output root by default.
    baseline_dir = output_root / "baseline" / "study-0000"
    tuned_dir = output_root / "tuned" / "study-0001"
    assert (baseline_dir / "study-manifest.json").exists()
    assert (tuned_dir / "study-manifest.json").exists()
    assert len(list((baseline_dir / "runs").glob("*/manifest.json"))) == 1
    assert len(list((tuned_dir / "runs").glob("*/manifest.json"))) == 1

    summary_path = output_root / "summary" / "ablation_metrics.csv"
    assert summary_path.exists()


def test_cli_ablation_invalid_config_raises(monkeypatch, tmp_path):
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "default_max_evals": 10,
        "problems": ["zdt1"],
        "seeds": [1],
    }
    config_path = tmp_path / "ablation_invalid.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    with pytest.raises(ValueError):
        main()


def test_cli_ablation_schema_validation(monkeypatch, tmp_path):
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "default_max_evals": 10,
        "problems": "zdt1",
        "seeds": [1],
        "variants": [{"name": "baseline"}],
    }
    config_path = tmp_path / "ablation_invalid_schema.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    with pytest.raises(ValueError):
        main()


def test_cli_ablation_rejects_unknown_keys(monkeypatch, tmp_path):
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "default_max_evals": 10,
        "problems": ["zdt1"],
        "seeds": [1],
        "variants": [{"name": "baseline"}],
        "unexpected": 123,
    }
    config_path = tmp_path / "ablation_unknown.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    with pytest.raises(ValueError):
        main()


def test_cli_ablation_rejects_unknown_variation_keys(monkeypatch, tmp_path):
    config = {
        "algorithm": "moead",
        "engine": "numpy",
        "default_max_evals": 10,
        "problems": ["zdt1"],
        "seeds": [1],
        "variants": [
            {"name": "baseline", "variations": {"moead": {"unknown_key": 1}}},
        ],
    }
    config_path = tmp_path / "ablation_bad_variation.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    with pytest.raises(ValueError):
        main()


def _run_cli(monkeypatch, tmp_path, config, *extra):
    config_path = tmp_path / "ablation.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog", "ablation", "--config", str(config_path), *extra])

    from vamos.experiment.cli.main import main

    main()


def _resolved_operators(study_dir):
    from vamos import load_study

    return load_study(study_dir).plan.tasks[0].resolved_run_spec["operators"]


def test_cli_ablation_accepts_variation_mapping_form(monkeypatch, tmp_path):
    output_root = tmp_path / "out"
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "output_root": str(output_root),
        "default_max_evals": 40,
        "problems": ["zdt1"],
        "seeds": [1],
        "base_config": {"population_size": 10},
        "variants": [
            {"name": "baseline"},
            {"name": "sbx_map", "variations": {"nsgaii": {"crossover": {"method": "sbx", "prob": 0.5, "eta": 5}}}},
        ],
    }

    _run_cli(monkeypatch, tmp_path, config)

    operators = _resolved_operators(output_root / "sbx_map" / "study-0001")
    assert dict(operators["crossover"]["config"]) == {"prob": 0.5, "eta": 5}
    assert (output_root / "summary" / "ablation_metrics.csv").exists()


def test_cli_ablation_plans_every_variant_before_running(monkeypatch, tmp_path):
    output_root = tmp_path / "out"
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "output_root": str(output_root),
        "default_max_evals": 40,
        "problems": ["zdt1"],
        "seeds": [1],
        "base_config": {"population_size": 10},
        "variants": [
            {"name": "baseline"},
            # A population larger than the evaluation budget cannot be planned.
            {"name": "too_large", "config_overrides": {"population_size": 80}},
        ],
    }

    with pytest.raises(Exception, match="max_evaluations|INVALID_EVALUATION_BUDGET"):
        _run_cli(monkeypatch, tmp_path, config)

    assert not output_root.exists() or not any(output_root.rglob("study-manifest.json"))


@pytest.mark.parametrize("flat", [False, True], ids=["per_variant", "flat"])
def test_cli_ablation_writes_each_variant_under_its_output_root(monkeypatch, tmp_path, flat):
    output_root = tmp_path / "out"
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "output_root": str(output_root),
        "output_root_by_variant": {"tuned": str(tmp_path / "custom_tuned")},
        "default_max_evals": 40,
        "problems": ["zdt1"],
        "seeds": [1],
        "base_config": {"population_size": 10},
        "variants": [{"name": "baseline"}, {"name": "tuned", "config_overrides": {"pop_size": 12}}],
    }

    _run_cli(monkeypatch, tmp_path, config, *(["--flat-output"] if flat else []))

    studies = sorted(path.parent.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("study-manifest.json"))
    if flat:
        assert studies == ["custom_tuned/study-0001", "out/study-0000"]
    else:
        assert studies == ["custom_tuned/study-0001", "out/baseline/study-0000"]
