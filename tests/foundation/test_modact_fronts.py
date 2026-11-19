"""MODAct's best-known fronts: the conversion, the checksums, the download helper and the lookup of downloaded fronts.

VAMOS 1.5.1. No test reaches the network: the downloads read tiny local files through a fake opener.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from vamos.foundation.problem import modact_fronts
from vamos.foundation.problem.modact_fronts import (
    DEFAULT_FETCH,
    FETCHABLE,
    NOTICE_FILE,
    PACKAGED,
    ZENODO_FILES,
    convert_front,
    convert_lines,
    fetch_fronts,
    notice,
)
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.resources import reference_front_path, user_reference_fronts_dir

# Rows as the Zenodo files write them, and the same rows in VAMOS's signs.
CTS_ROWS = (
    "1.582289891400155968e-01 -5.864055086698143349e-01 1.771332826183932241e+01\n"
    "\n"
    "1.481405094584629767e+00 1.992046735115843514e+00 4.834176693464925023e+01\n"
)
CTS_CSV = [
    "1.582289891400155968e-01,5.864055086698143349e-01,-1.771332826183932241e+01",
    "1.481405094584629767e+00,-1.992046735115843514e+00,-4.834176693464925023e+01",
]
CTSEI_ROW = "2.0e-01 -5.0e-01 3.0e+01 1.0e-01 1.2e+02"


def test_conversion_negates_the_maximized_columns_and_keeps_every_digit() -> None:
    assert list(convert_lines(CTS_ROWS.splitlines(), "cts1")) == CTS_CSV
    # CTSEI: torque excess, safety factor and efficiency are maximized; cost and transmission ratio are not.
    assert list(convert_lines([CTSEI_ROW], "modact_ctsei2")) == ["2.0e-01,5.0e-01,-3.0e+01,-1.0e-01,1.2e+02"]
    assert list(convert_lines(["2.4e-01 2.4e+01"], "cs1")) == ["2.4e-01,-2.4e+01"]
    # The values are those of the file, multiplied by the problem's objective signs.
    raw = np.loadtxt(io.StringIO(CTS_ROWS), ndmin=2)
    converted = np.loadtxt(io.StringIO("\n".join(CTS_CSV)), delimiter=",", ndmin=2)
    np.testing.assert_array_equal(converted, raw * [1.0, -1.0, -1.0])


@pytest.mark.parametrize("rows", [["1.0 2.0"], ["1.0 2.0 3.0 4.0"], ["1.0 abc 3.0"], ["abc 2.0 3.0"], ["nan 2.0 3.0"], ["1.0 2.0 inf"]])
def test_a_malformed_row_is_rejected(rows: list[str]) -> None:
    with pytest.raises(ValueError):
        list(convert_lines(rows, "cts1"))


def test_unknown_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="No MODAct front"):
        list(convert_lines(["1.0 2.0"], "cs5"))


def _fixture(tmp_path: Path, name: str, text: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A local stand-in for ``<name>_PF.dat`` whose checksum the record table then lists."""
    source = tmp_path / "zenodo" / f"{name}_PF.dat"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(text.encode("ascii"))
    monkeypatch.setitem(ZENODO_FILES, name, (hashlib.md5(text.encode("ascii")).hexdigest(), len(text)))
    return source


def test_convert_front_checks_the_checksum_and_writes_nothing_on_a_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _fixture(tmp_path, "cts1", CTS_ROWS, monkeypatch)
    out = tmp_path / "out"

    path = convert_front(source, out, "cts1")

    assert path == out / "modact_cts1.csv"
    assert path.read_text(encoding="ascii") == "".join(f"{row}\n" for row in CTS_CSV)
    source.write_bytes(CTS_ROWS.replace("1.58", "1.59").encode("ascii"))
    path.unlink()
    with pytest.raises(ValueError, match="MD5"):
        convert_front(source, out, "cts1")
    assert sorted(entry.name for entry in out.iterdir()) == []


def test_the_record_lists_every_problem_and_the_packaged_notice_is_the_shared_one() -> None:
    assert len(ZENODO_FILES) == 20 and set(PACKAGED) | set(FETCHABLE) == set(ZENODO_FILES)
    assert set(PACKAGED).isdisjoint(FETCHABLE) and DEFAULT_FETCH == ("cts1", "cts2", "cts3", "cts4")
    assert all(len(md5) == 32 and size > 0 for md5, size in ZENODO_FILES.values())
    packaged = resources.files("vamos.resources.reference_fronts").joinpath(NOTICE_FILE).read_text(encoding="utf-8")
    assert packaged == notice(PACKAGED, packaged=True)


class _FakeZenodo:
    """An opener that serves local files by the name at the end of the record's URL, and records the URLs."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.urls: list[str] = []

    @contextmanager
    def __call__(self, url: str) -> Iterator[io.BytesIO]:
        self.urls.append(url)
        name = url.split("/files/")[1].split("/")[0]
        if name not in self.files:
            raise OSError(f"404 for {url}")
        yield io.BytesIO(self.files[name])


def _rows(name: str) -> str:
    if name.startswith("ctsei"):
        return f"{CTSEI_ROW}\n"
    if name.startswith("ctse"):
        return "2.0e-01 -5.0e-01 3.0e+01 1.0e-01\n"
    return CTS_ROWS


def _fake_record(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> _FakeZenodo:
    files: dict[str, bytes] = {}
    for name in names:
        text = _rows(name)
        files[f"{name}_PF.dat"] = text.encode("ascii")
        monkeypatch.setitem(ZENODO_FILES, name, (hashlib.md5(text.encode("ascii")).hexdigest(), len(text)))
    return _FakeZenodo(files)


def test_fetch_downloads_checks_converts_and_writes_the_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_record(monkeypatch, ("cts1", "cts2", "cts3", "cts4", "ctse3", "ctsei4"))
    out = tmp_path / "fronts"

    written = fetch_fronts(directory=out, open_url=fake)

    assert written == [out / f"modact_cts{k}.csv" for k in range(1, 5)]
    assert fake.urls == [f"https://zenodo.org/api/records/3824302/files/cts{k}_PF.dat/content" for k in range(1, 5)]
    assert written[0].read_text(encoding="ascii") == "".join(f"{row}\n" for row in CTS_CSV)
    fetch_fronts(["modact_ctse3", "ctsei4"], out, open_url=fake)
    assert np.loadtxt(out / "modact_ctse3.csv", delimiter=",", ndmin=2).tolist() == [[0.2, 0.5, -30.0, -0.1]]
    assert np.loadtxt(out / "modact_ctsei4.csv", delimiter=",", ndmin=2).tolist() == [[0.2, 0.5, -30.0, -0.1, 120.0]]
    text = (out / NOTICE_FILE).read_text(encoding="utf-8")
    assert text == notice(["cts1", "cts2", "cts3", "cts4", "ctse3", "ctsei4"], packaged=False)
    for needed in ("Picard", "Schiffmann", "https://doi.org/10.5281/zenodo.3824302", "version 1.1.0", "CC BY 4.0", "Changes:"):
        assert needed in text
    assert "- modact_ctsei4.csv: ctsei4_PF.dat (MD5 " in text
    assert not list(out.glob(".*part"))


def test_fetch_keeps_a_front_that_is_there_unless_forced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_record(monkeypatch, ("cts2",))
    out = tmp_path / "fronts"
    fetch_fronts(["modact_cts2"], out, open_url=fake)
    (out / "modact_cts2.csv").write_text("kept\n", encoding="ascii")

    fetch_fronts(["modact_cts2"], out, open_url=fake)
    assert (out / "modact_cts2.csv").read_text(encoding="ascii") == "kept\n" and len(fake.urls) == 1

    fetch_fronts(["modact_cts2"], out, open_url=fake, force=True)
    assert (out / "modact_cts2.csv").read_text(encoding="ascii") == "".join(f"{row}\n" for row in CTS_CSV)


def test_fetch_rejects_packaged_and_unknown_fronts_and_keeps_nothing_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_record(monkeypatch, ("cts1",))
    out = tmp_path / "fronts"
    with pytest.raises(ValueError, match="packages the fronts of cs1"):
        fetch_fronts(["modact_cs1"], out, open_url=fake)
    with pytest.raises(ValueError, match="No MODAct front"):
        fetch_fronts(["modact_cts9"], out, open_url=fake)
    assert fake.urls == []
    # A file whose checksum is not the record's, and a failed download, leave neither a front nor a partial file.
    monkeypatch.setitem(ZENODO_FILES, "cts1", ("0" * 32, 1))
    with pytest.raises(ValueError, match="MD5"):
        fetch_fronts(["modact_cts1"], out, open_url=fake)
    with pytest.raises(OSError, match="404"):
        fetch_fronts(["modact_cts3"], out, open_url=fake)
    assert sorted(entry.name for entry in out.iterdir()) == []


def test_the_default_opener_is_the_one_fetch_uses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_record(monkeypatch, ("cts4",))
    monkeypatch.setattr(modact_fronts, "_open_zenodo", fake)
    fetch_fronts(["modact_cts4"], tmp_path)
    assert fake.urls and (tmp_path / "modact_cts4.csv").is_file()


def test_vamos_finds_a_downloaded_front_after_the_packaged_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAMOS_DATA_DIR", str(tmp_path))
    directory = user_reference_fronts_dir()
    assert directory == tmp_path / "reference_fronts"
    assert resolve_reference_front_path("modact_cts1", None) is None

    fetch_fronts(["modact_cts1"], open_url=_fake_record(monkeypatch, ("cts1",)))

    resolved = resolve_reference_front_path("modact_cts1", None)
    assert resolved is not None and Path(resolved) == directory / "modact_cts1.csv"
    assert resolve_reference_front_path("modact_cts1", None, n_obj=2) is None
    # A downloaded file never replaces a packaged front of the same name.
    (directory / "modact_cs1.csv").write_text("1.0,2.0\n", encoding="ascii")
    assert reference_front_path("modact_cs1") != directory / "modact_cs1.csv"
    assert np.loadtxt(reference_front_path("modact_cs1"), delimiter=",", ndmin=2).shape[0] == 1664


@pytest.mark.parametrize(
    ("platform", "variables", "expected"),
    [
        ("win32", {"LOCALAPPDATA": "D:/appdata/local"}, "D:/appdata/local/vamos/reference_fronts"),
        ("linux", {"XDG_DATA_HOME": "/srv/xdg-data"}, "/srv/xdg-data/vamos/reference_fronts"),
    ],
)
def test_the_user_directory_follows_the_platform(
    platform: str, variables: dict[str, str], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VAMOS_DATA_DIR", raising=False)
    monkeypatch.setattr("sys.platform", platform)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert user_reference_fronts_dir() == Path(expected)


def test_macos_and_unset_variables_use_the_home_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VAMOS_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("sys.platform", "darwin")
    assert user_reference_fronts_dir() == tmp_path / "Library" / "Application Support" / "vamos" / "reference_fronts"
    monkeypatch.setattr("sys.platform", "linux")
    assert user_reference_fronts_dir() == tmp_path / ".local" / "share" / "vamos" / "reference_fronts"
