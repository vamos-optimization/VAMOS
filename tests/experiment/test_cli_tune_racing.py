"""`vamos tune --backend racing`: default race and multi-fidelity budget checks (review finding VR-205)."""

from __future__ import annotations

import json

import pytest

from vamos.experiment.cli import tune as tune_cli

# The "Split-based tuning" example of docs/topics/tuning.md with --backend racing:
# 6 instances split into 3 training instances, 5 training seeds and 20 initial configurations.
DOC_EXAMPLE = [
    "--instances",
    "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1",
    "--algorithm",
    "nsgaii",
    "--backend",
    "racing",
    "--split-strategy",
    "suite_stratified",
    "--budget",
    "5000",
    "--tune-budget",
    "200",
    "--n-jobs",
    "1",
    "--no-show-progress-bar",
    "--no-run-validation",
    "--no-run-statistical-finisher",
]


def _record_evaluations(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    seen: list[tuple[str, int, int]] = []

    def fake_make_evaluator(**_: object):
        def evaluate(config, ctx):
            seen.append((ctx.instance.name, int(ctx.seed), int(ctx.budget)))
            return float(config["crossover_prob"])

        return evaluate

    monkeypatch.setattr(tune_cli, "make_evaluator", fake_make_evaluator)
    return seen


def test_cli_racing_defaults_to_the_statistical_race(monkeypatch, tmp_path):
    seen = _record_evaluations(monkeypatch)

    assert tune_cli.main([*DOC_EXAMPLE, "--output-dir", str(tmp_path)]) == 0

    # The statistical race evaluates every block at --budget and stays within --tune-budget.
    assert 0 < len(seen) <= 200
    assert {budget for _, _, budget in seen} == {5000}
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.rglob("tuning_summary.json")]
    assert len(summaries) == 1 and summaries[0]["args"]["multi_fidelity"] is False


def test_cli_racing_rejects_a_first_level_over_the_tune_budget(monkeypatch, tmp_path):
    seen = _record_evaluations(monkeypatch)

    with pytest.raises(SystemExit, match=r"first fidelity level.*300 runs.*--tune-budget 200"):
        tune_cli.main([*DOC_EXAMPLE, "--multi-fidelity", "--output-dir", str(tmp_path)])

    assert seen == []
    assert not list(tmp_path.rglob("split_instances.csv")), "nothing may be written before the budget check"


def test_cli_multi_fidelity_racing_warns_about_unused_race_options(monkeypatch, tmp_path, caplog):
    _record_evaluations(monkeypatch)
    argv = [
        "--problem",
        "zdt1",
        "--backend",
        "racing",
        "--multi-fidelity",
        "--fidelity-levels",
        "100,200",
        "--initial-configs",
        "4",
        "--n-seeds",
        "2",
        "--tune-budget",
        "40",
        "--alpha",
        "0.1",
        "--no-use-statistical-tests",
        "--n-jobs",
        "1",
        "--no-show-progress-bar",
        "--no-run-validation",
        "--no-run-statistical-finisher",
        "--output-dir",
        str(tmp_path),
    ]

    with caplog.at_level("WARNING"):
        assert tune_cli.main(argv) == 0

    warnings = " ".join(record.getMessage() for record in caplog.records if record.levelname == "WARNING")
    assert "--alpha" in warnings and "--use-statistical-tests" in warnings
