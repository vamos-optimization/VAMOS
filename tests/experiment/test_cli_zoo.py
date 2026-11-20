"""`vamos zoo`: unknown names, encodings, the run summary and the download of MODAct fronts."""

from __future__ import annotations

import hashlib
import io
import logging
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from vamos.experiment.zoo import cli as zoo_cli
from vamos.foundation.problem import modact_fronts
from vamos.foundation.problem.registry_info import list_problems
from vamos.foundation.problem.resolver import resolve_reference_front_path


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


# VAMOS 1.5.1: `vamos zoo fetch-fronts` downloads MODAct fronts; these tests serve local files instead.
def _serve(monkeypatch: pytest.MonkeyPatch, rows: dict[str, str]) -> list[str]:
    urls: list[str] = []

    @contextmanager
    def fake_open(url: str) -> Iterator[io.BytesIO]:
        urls.append(url)
        name = url.split("/files/")[1].split("_PF.dat")[0]
        if name not in rows:
            raise OSError("connection refused")
        yield io.BytesIO(rows[name].encode("ascii"))

    for name, text in rows.items():
        monkeypatch.setitem(modact_fronts.ZENODO_FILES, name, (hashlib.md5(text.encode("ascii")).hexdigest(), len(text)))
    monkeypatch.setattr(modact_fronts, "_open_zenodo", fake_open)
    return urls


def test_zoo_fetch_fronts_writes_the_converted_fronts_where_vamos_finds_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("VAMOS_DATA_DIR", str(tmp_path))
    urls = _serve(monkeypatch, {f"cts{k}": "1.0e-01 -2.0e-01 3.0e+01\n" for k in range(1, 5)} | {"ctse4": "1.0 2.0 3.0 4.0\n"})

    with caplog.at_level(logging.INFO):
        assert zoo_cli.main(["fetch-fronts"]) == 0
        assert zoo_cli.main(["fetch-fronts", "modact_ctse4"]) == 0

    assert [url.split("/files/")[1] for url in urls] == [f"cts{k}_PF.dat/content" for k in range(1, 5)] + ["ctse4_PF.dat/content"]
    directory = tmp_path / "reference_fronts"
    assert (directory / "modact_cts1.csv").read_text(encoding="ascii") == "1.0e-01,2.0e-01,-3.0e+01\n"
    assert resolve_reference_front_path("modact_cts3", None) == str(directory / "modact_cts3.csv")
    assert resolve_reference_front_path("modact_ctse4", None) == str(directory / "modact_ctse4.csv")
    assert "CC BY 4.0" in (directory / "LICENSE-modact-fronts.txt").read_text(encoding="utf-8")
    assert str(directory) in caplog.text


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["fetch-fronts", "modact_cs1"], "packages the fronts of cs1"),
        (["fetch-fronts", "zdt1"], "No MODAct front"),
        (["fetch-fronts", "modact_ctsei1"], "Could not download"),
    ],
)
def test_zoo_fetch_fronts_fails_with_exit_code_one(
    argv: list[str], message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _serve(monkeypatch, {})
    with caplog.at_level(logging.INFO):
        assert zoo_cli.main([*argv, "--output", str(tmp_path / "out")]) == 1
    assert message in caplog.text
    assert not (tmp_path / "out" / "modact_ctsei1.csv").exists()


def test_zoo_fetch_fronts_help_reaches_no_network() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "zoo", "fetch-fronts", "--help"], capture_output=True, timeout=60
    )
    text = proc.stdout.decode("utf-8", errors="replace")
    assert proc.returncode == 0
    assert "zenodo.3824302" in text and "modact_cts1" in text
