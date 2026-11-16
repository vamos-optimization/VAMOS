"""The pre-run checks of ``vamos`` report a missing engine package whatever the case of the engine's name."""

from __future__ import annotations

import argparse
import json
import logging

import pytest

from vamos.experiment.cli import preflight
from vamos.experiment.cli.parser import parse_args
from vamos.foundation.core.experiment_config import ExperimentConfig


@pytest.mark.parametrize("spelling", ["numba", "NUMBA", "Numba", "moocore", "MOOCORE", "MooCore"])
def test_a_missing_engine_package_is_reported_in_any_case(spelling, monkeypatch, caplog, tmp_path):
    # A config may spell the engine in any case: the config check and the run lower-case it.
    config = tmp_path / "spec.json"
    defaults = {"problem": "zdt1", "algorithm": "nsgaii", "engine": spelling}
    config.write_text(json.dumps({"version": "1", "defaults": defaults}), encoding="utf-8")
    args = parse_args(ExperimentConfig(), argv=["--config", str(config)])
    engine = spelling.lower()
    monkeypatch.setattr(preflight, "_has_module", lambda module: module != engine)

    with caplog.at_level(logging.WARNING, logger=preflight.__name__):
        preflight.run_preflight_checks(args)

    assert [record.getMessage() for record in caplog.records] == [
        f"Preflight: Engine '{engine}' selected but {engine} is not installed. Install with: pip install \"vamos-optimization[compute]\"."
    ]


@pytest.mark.parametrize("spelling", ["numpy", "NumPy", "auto", "AUTO"])
def test_engines_without_an_optional_package_need_no_warning(spelling, monkeypatch, caplog):
    monkeypatch.setattr(preflight, "_has_module", lambda module: False)

    with caplog.at_level(logging.WARNING, logger=preflight.__name__):
        preflight.run_preflight_checks(argparse.Namespace(engine=spelling))

    assert caplog.records == []
