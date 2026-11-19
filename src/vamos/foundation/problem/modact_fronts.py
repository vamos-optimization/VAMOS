"""MODAct's best-known Pareto fronts, the authors' files in VAMOS's signs.

VAMOS 1.5.1. Picard and Schiffmann published the best-known fronts of the 20 MODAct problems in the
Zenodo record https://doi.org/10.5281/zenodo.3824302, version 1.1.0, under CC BY 4.0. The files are in MODAct's
signs: the objectives it maximizes (torque excess, safety factor, efficiency) are larger when better.

VAMOS packages the fronts of CS1-CS4 and CT1-CT4. :func:`fetch_fronts` (``vamos zoo fetch-fronts``) downloads
the others, 1.4 MB to 193 MB each, into :func:`vamos.resources.user_reference_fronts_dir`, where the reference-front
lookup finds them. Both go through :func:`convert_front`, which ``experiments/scripts/generate_reference_fronts.py``
also uses for the packaged files:

- it checks the file's MD5 checksum against the one the Zenodo record publishes;
- it replaces the whitespace between columns by a comma, and changes the sign of the columns that MODAct
  maximizes at the text level, so each value keeps its digits and every column is minimized, as VAMOS's MODAct
  problems return them; no row is added or removed;
- it writes the CSV through a temporary file, so an interrupted conversion leaves no partial front.

:func:`write_notice` writes the attribution and change notice that the license asks for next to the files.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO, cast

from vamos.foundation.problem.modact import MAXIMIZED, OBJECTIVES
from vamos.foundation.version import __version__
from vamos.resources import user_reference_fronts_dir

ZENODO_DOI = "https://doi.org/10.5281/zenodo.3824302"
ZENODO_VERSION = "1.1.0"
_ZENODO_FILE_URL = "https://zenodo.org/api/records/3824302/files/{file}/content"
# MODAct's name of each problem: the MD5 checksum and the size in bytes of its <name>_PF.dat, as the record lists them.
ZENODO_FILES: dict[str, tuple[str, int]] = {
    "cs1": ("4990ad4b4fdb1e69bcd8c783194c4786", 83200),
    "cs2": ("3ee656a1337a8741f3563d0870ad7333", 106150),
    "cs3": ("dd70ee28cc5f72c2561dba9c6d235e89", 152800),
    "cs4": ("bb5dc7ac5b66453a01a4a435de5d099d", 89150),
    "ct1": ("6e4550df32881f98cb15af16635afcab", 55857),
    "ct2": ("797cd321c1f3712811914d6fc3faef61", 49901),
    "ct3": ("6331fe3b000d5d3f3487b13998b3cacc", 52709),
    "ct4": ("791662a99be6222a8d7a4750dae84974", 82406),
    "cts1": ("37d31eeccfbe8359248d33b82f2d8123", 2302740),
    "cts2": ("04e2f7e3840ff8ae9295b6f56f9ea768", 2418036),
    "cts3": ("99a62cbf88a87c8713ef110aed4996f0", 1422500),
    "cts4": ("13f89d4a364862bf79a26f8ba41e1427", 2769808),
    "ctse1": ("8d47b3f6fba08c3942021a16998b1fbc", 35213253),
    "ctse2": ("b9317d36698d5ab1a9bc2b84086f08f7", 33023417),
    "ctse3": ("9d76d73ed708d613b2e605025b2dba10", 11868617),
    "ctse4": ("5b2b3025e9bd1eac581fd85d9cf17ed9", 10986526),
    "ctsei1": ("11e7bf9117fced7596c24935a16d924b", 186195656),
    "ctsei2": ("63c22c0216683def6eeede7881339e0b", 193309838),
    "ctsei3": ("201e27f8da7a38e62de3e47a72eaa707", 133373982),
    "ctsei4": ("ad5591756d6e2d1d683026f8c0f8ba32", 116346385),
}
# The fronts VAMOS packages, and those fetch_fronts downloads (by default the three-objective ones).
PACKAGED: tuple[str, ...] = ("cs1", "cs2", "cs3", "cs4", "ct1", "ct2", "ct3", "ct4")
FETCHABLE: tuple[str, ...] = (
    "cts1",
    "cts2",
    "cts3",
    "cts4",
    "ctse1",
    "ctse2",
    "ctse3",
    "ctse4",
    "ctsei1",
    "ctsei2",
    "ctsei3",
    "ctsei4",
)
DEFAULT_FETCH: tuple[str, ...] = ("cts1", "cts2", "cts3", "cts4")
NOTICE_FILE = "LICENSE-modact-fronts.txt"

OpenURL = Callable[[str], AbstractContextManager[IO[bytes]]]

_NOTICE = """Third-party reference fronts: MODAct best-known Pareto fronts

The files below are adapted from:

  Cyril Picard and Jürg Schiffmann, "Multi-Objective Design of Actuators:
  Pareto fronts", version 1.1.0, Zenodo, 2020.
  https://doi.org/10.5281/zenodo.3824302

They are the best-known Pareto fronts of the MODAct problems described in:

  C. Picard and J. Schiffmann, "Realistic constrained multiobjective
  optimization benchmark problems from design", IEEE Transactions on
  Evolutionary Computation 25(2) (2021) 234-246.
  https://doi.org/10.1109/TEVC.2020.3020046

License: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. This license covers these data
files only; VAMOS's code is under its own MIT license. VAMOS does not include
or redistribute the authors' modact package (GPL-3.0), which evaluates the
problems.

Changes: the whitespace between columns is replaced by a comma, and the
values of the objectives that MODAct maximizes (the torque excess, the safety
factor and the efficiency) have the opposite sign, so that every column is
minimized, as VAMOS's MODAct problems return them. Each value keeps its
digits, and no row is added or removed.

{origin}

{files}
"""
_PACKAGED_ORIGIN = """VAMOS packages the fronts of the two-objective problems, CS1-CS4 and CT1-CT4.
`vamos zoo fetch-fronts` downloads the others from the record and converts
them in the same way."""
_FETCHED_ORIGIN = """`vamos zoo fetch-fronts` downloaded these files from the record, checked
their MD5 checksums against the record's and converted them."""


def modact_name(key: str) -> str:
    """MODAct's name of a problem, from ``"modact_cts1"`` or ``"cts1"``; ``ValueError`` for another name."""
    name = str(key).strip().lower().removeprefix("modact_")
    if name not in ZENODO_FILES:
        raise ValueError(f"No MODAct front is named {key!r}; the problems are modact_cs1 to modact_ctsei4.")
    return name


def source_file(name: str) -> str:
    """The Zenodo file of a problem's front, ``<name>_PF.dat``."""
    return f"{modact_name(name)}_PF.dat"


def front_file(name: str) -> str:
    """The CSV that VAMOS's reference-front lookup finds for the problem, ``modact_<name>.csv``."""
    return f"modact_{modact_name(name)}.csv"


def _number(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{value!r} is not a number.") from None
    if number != number or abs(number) == float("inf"):
        raise ValueError(f"{value!r} is not a finite number.")
    return number


def _negated(value: str) -> str:
    """``value`` with the opposite sign and the same digits."""
    negated = value[1:] if value.startswith("-") else f"-{value}"
    if _number(negated) != -_number(value):
        raise ValueError(f"{value!r} is not a number whose sign the text can change.")
    return negated


def _kept(value: str) -> str:
    _number(value)
    return value


def convert_lines(lines: Iterable[str], name: str) -> Iterator[str]:
    """The rows of a Zenodo front as CSV rows in VAMOS's signs; blank lines are skipped.

    Each row must have one value per objective of the problem. The values of the objectives that MODAct maximizes
    change sign at the text level, so every other character of the file is kept.
    """
    objectives = OBJECTIVES[modact_name(name).rstrip("1234")]
    maximized = [objective in MAXIMIZED for objective in objectives]
    for number, line in enumerate(lines, start=1):
        values = line.split()
        if not values:
            continue
        if len(values) != len(maximized):
            raise ValueError(f"Row {number} of the {name} front has {len(values)} values, not {len(maximized)}.")
        yield ",".join(_negated(value) if flip else _kept(value) for value, flip in zip(values, maximized))


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_front(source: Path, directory: Path, name: str) -> Path:
    """Write ``modact_<name>.csv`` in ``directory`` from the Zenodo file ``source``, after checking its MD5 checksum.

    Raises ``ValueError``, and writes nothing, when the checksum is not the record's or a row is malformed.
    """
    name = modact_name(name)
    expected, _ = ZENODO_FILES[name]
    digest = file_md5(source)
    if digest != expected:
        raise ValueError(
            f"{source} is not {source_file(name)} of Zenodo record {ZENODO_DOI}, version {ZENODO_VERSION}: its MD5 is "
            f"{digest}, the record's {expected}."
        )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / front_file(name)
    partial = directory / f".{front_file(name)}.part"
    try:
        with source.open(encoding="ascii") as rows, partial.open("w", encoding="ascii", newline="\n") as out:
            for row in convert_lines(rows, name):
                out.write(f"{row}\n")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def notice(names: Iterable[str], *, packaged: bool) -> str:
    """The CC BY 4.0 attribution and change notice of the converted fronts ``names``."""
    files = "\n".join(f"- {front_file(name)}: {source_file(name)} (MD5 {ZENODO_FILES[modact_name(name)][0]})" for name in names)
    return _NOTICE.format(origin=_PACKAGED_ORIGIN if packaged else _FETCHED_ORIGIN, files=files)


def write_notice(directory: Path, names: Iterable[str], *, packaged: bool) -> Path:
    """Write :func:`notice` as ``LICENSE-modact-fronts.txt`` in ``directory``."""
    path = directory / NOTICE_FILE
    path.write_text(notice(names, packaged=packaged), encoding="utf-8", newline="\n")
    return path


def _open_zenodo(url: str) -> AbstractContextManager[IO[bytes]]:
    from urllib.request import Request, urlopen

    request = Request(url, headers={"User-Agent": f"vamos-optimization/{__version__}"})
    return cast(AbstractContextManager[IO[bytes]], urlopen(request, timeout=60))


def fetch_fronts(
    keys: Sequence[str] = DEFAULT_FETCH,
    directory: Path | None = None,
    *,
    force: bool = False,
    open_url: OpenURL | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Download the fronts ``keys`` from Zenodo and write them, converted, in ``directory``.

    ``keys`` are problem keys such as ``"modact_cts1"`` (or MODAct's names, ``"cts1"``) of the fronts VAMOS does not
    package: CTS1-CTS4 (the default), CTSE1-CTSE4 and CTSEI1-CTSEI4. ``directory`` defaults to
    :func:`vamos.resources.user_reference_fronts_dir`, which the reference-front lookup searches. A front already
    there is kept unless ``force``. Each download is checked against the record's MD5 before it is converted, and
    ``LICENSE-modact-fronts.txt`` in ``directory`` then lists every MODAct front the directory holds. Returns the
    paths of the requested fronts. ``open_url`` opens a URL for reading (``urllib`` by default), and ``progress``
    receives one message per front.
    """
    names = [modact_name(key) for key in keys]
    packaged = [name for name in names if name in PACKAGED]
    if packaged:
        raise ValueError(f"VAMOS packages the fronts of {', '.join(packaged)}; fetch_fronts downloads {', '.join(FETCHABLE)}.")
    directory = user_reference_fronts_dir() if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    opener = _open_zenodo if open_url is None else open_url
    report = progress if progress is not None else (lambda message: None)
    written: list[Path] = []
    for name in dict.fromkeys(names):
        target = directory / front_file(name)
        if target.is_file() and not force:
            report(f"{target} is already there; pass force=True (--force) to download it again.")
            written.append(target)
            continue
        size_mb = ZENODO_FILES[name][1] / 1e6
        report(f"Downloading {source_file(name)} ({size_mb:.1f} MB) from {ZENODO_DOI} ...")
        download = directory / f".{source_file(name)}.part"
        try:
            with opener(_ZENODO_FILE_URL.format(file=source_file(name))) as response, download.open("wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
            written.append(convert_front(download, directory, name))
        finally:
            download.unlink(missing_ok=True)
        report(f"Wrote {written[-1]}")
    present = [name for name in FETCHABLE if (directory / front_file(name)).is_file()]
    write_notice(directory, present, packaged=False)
    return written


__all__ = [
    "DEFAULT_FETCH",
    "FETCHABLE",
    "NOTICE_FILE",
    "PACKAGED",
    "ZENODO_DOI",
    "ZENODO_FILES",
    "ZENODO_VERSION",
    "convert_front",
    "convert_lines",
    "fetch_fronts",
    "file_md5",
    "front_file",
    "modact_name",
    "notice",
    "source_file",
    "write_notice",
]
