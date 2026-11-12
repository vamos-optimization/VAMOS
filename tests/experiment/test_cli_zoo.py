"""`vamos zoo`: unknown names, encodings and the run summary (review finding VR-297)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vamos.experiment.zoo import cli as zoo_cli
from vamos.foundation.problem.registry_info import list_problems


def test_zoo_info_unknown_problem_fails(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert zoo_cli.main(["info", "zdt7"]) == 1
    assert "Problem 'zdt7' not found." in caplog.text


def test_zoo_list_names_one_category_per_encoding() -> None:
    categories = {category for info in list_problems() for category in info.categories}
    assert "continuous" not in categories
    assert "real" in categories


def test_zoo_run_does_not_report_a_missing_hypervolume(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    argv = ["run", "zdt1", "--budget", "40", "--pop-size", "8", "--output", str(tmp_path)]
    with caplog.at_level(logging.INFO):
        assert zoo_cli.main(argv) == 0
    finished = [record.getMessage() for record in caplog.records if record.getMessage().startswith("Run finished")]
    assert finished and "HV: None" not in finished[0], finished
