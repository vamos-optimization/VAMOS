"""jMetal's reference fronts of ZDT and DTLZ, packaged as an optional source.

VAMOS's own samples stay the default. jMetal's files (``resources/referenceFronts`` at commit
``1e4b83cae6``, MIT, the files Evolver uses) agree with them on the hypervolume but not on IGD-like
indicators, because their points lie elsewhere along the front; with ``source="jmetal"`` the indicators
can be compared with jMetal's and Evolver's.
"""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from vamos.foundation.core.hv_stop import build_hv_stop_config, near_nadir_reference
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.foundation.quality_indicators.hypervolume import hypervolume

# The problems and objective counts with a jMetal front, and the jMetal file of each.
JMETAL = {
    **{(f"zdt{k}", 2): f"ZDT{k}" for k in (1, 2, 3, 4, 6)},
    **{(f"dtlz{k}", m): f"DTLZ{k}.{m}D" for k in range(1, 8) for m in (2, 3)},
}
CASES = sorted(JMETAL)


def _packaged_name(problem: str, n_obj: int) -> str:
    if problem.startswith("zdt"):
        return f"jmetal-{problem}.csv"
    # jMetal's DTLZ3 and DTLZ4 files are its DTLZ2 file, byte for byte: it is packaged once.
    return f"jmetal-{'dtlz2' if problem in ('dtlz3', 'dtlz4') else problem}.{n_obj}d.csv"


def _front(path: str | None) -> np.ndarray:
    assert path is not None
    return np.loadtxt(path, delimiter=",", ndmin=2)


@pytest.mark.parametrize(("problem", "n_obj"), CASES)
def test_the_jmetal_source_resolves_jmetals_front(problem: str, n_obj: int) -> None:
    path = resolve_reference_front_path(problem, None, n_obj=n_obj, source="jmetal")

    assert path is not None
    assert Path(path).name == _packaged_name(problem, n_obj)
    assert _front(path).shape[1] == n_obj


@pytest.mark.parametrize(("problem", "n_obj"), CASES)
def test_jmetals_and_vamoss_fronts_agree_on_the_hypervolume(problem: str, n_obj: int) -> None:
    vamos_path = resolve_reference_front_path(problem, None, n_obj=n_obj)
    if vamos_path is None:
        pytest.skip(f"VAMOS packages no {n_obj}-objective front of {problem}")
    ours, theirs = _front(vamos_path), _front(resolve_reference_front_path(problem, None, n_obj=n_obj, source="jmetal"))
    reference = near_nadir_reference(np.vstack([ours, theirs]))

    assert hypervolume(theirs, reference) == pytest.approx(hypervolume(ours, reference), rel=2e-2)


@pytest.mark.parametrize("n_obj", [2, 3])
def test_jmetals_dtlz3_and_dtlz4_fronts_are_its_dtlz2_front(n_obj: int) -> None:
    dtlz2 = resolve_reference_front_path("dtlz2", None, n_obj=n_obj, source="jmetal")

    assert resolve_reference_front_path("dtlz3", None, n_obj=n_obj, source="jmetal") == dtlz2
    assert resolve_reference_front_path("dtlz4", None, n_obj=n_obj, source="jmetal") == dtlz2
    assert resolve_reference_front_path(f"dtlz4.{n_obj}d", None, source="jmetal") == dtlz2
    packaged = {path.name for path in resources.files("vamos.resources.reference_fronts").iterdir()}
    assert not {f"jmetal-dtlz3.{n_obj}d.csv", f"jmetal-dtlz4.{n_obj}d.csv"} & packaged


def test_the_default_source_is_vamos() -> None:
    assert Path(str(resolve_reference_front_path("zdt1", None))).name.lower() == "zdt1.csv"
    assert resolve_reference_front_path("zdt1", None, source="vamos") == resolve_reference_front_path("zdt1", None)


def test_the_jmetal_source_never_falls_back_to_vamoss_fronts() -> None:
    assert resolve_reference_front_path("zdt5", None, source="jmetal") is None
    assert resolve_reference_front_path("wfg4", None, n_obj=2, source="jmetal") is None
    # VAMOS 1.5.0: VAMOS's samples of DTLZ with 4, 6 and 8 objectives are not jMetal's source.
    assert resolve_reference_front_path("dtlz2", None, n_obj=4) is not None
    assert resolve_reference_front_path("dtlz2", None, n_obj=4, source="jmetal") is None
    assert resolve_reference_front_path("dtlz3", None, n_obj=6, source="jmetal") is None


def test_an_explicit_path_wins_over_the_source() -> None:
    assert resolve_reference_front_path("zdt1", "my_front.csv", source="jmetal") == "my_front.csv"


def test_an_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="'vamos', 'jmetal'"):
        resolve_reference_front_path("zdt1", None, source="pymoo")


def test_the_jmetal_notice_lists_the_optional_fronts() -> None:
    text = resources.files("vamos.resources.reference_fronts").joinpath("LICENSE-jmetal.txt").read_text(encoding="utf-8")

    assert "jmetal-zdt1.csv" in text and "jmetal-dtlz7.3d.csv" in text


@pytest.mark.parametrize(
    ("front", "source", "name"),
    [(None, "vamos", "zdt1.csv"), ("jmetal", "jmetal", "jmetal-zdt1.csv"), ("vamos", "vamos", "zdt1.csv")],
)
def test_the_hv_stop_records_the_source_of_its_front(front: str | None, source: str, name: str) -> None:
    config = build_hv_stop_config(0.5, front, "zdt1")

    assert config is not None
    assert config["reference_front_source"] == source
    assert Path(str(config["reference_front_path"])).name.lower() == name


def test_the_hv_stop_records_a_file_as_such(tmp_path: Path) -> None:
    path = tmp_path / "front.csv"
    np.savetxt(path, [[0.0, 1.0], [1.0, 0.0]], delimiter=",")

    config = build_hv_stop_config(0.5, str(path), "zdt1")

    assert config is not None
    assert (config["reference_front_source"], config["reference_front_path"]) == ("file", str(path))


def test_the_cli_records_jmetals_front_in_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.experiment.cli.main import main

    argv = ["--problem", "zdt1", "--algorithm", "nsgaii", "--max-evaluations", "200", "--hv-threshold", "0.5"]
    monkeypatch.setattr(sys, "argv", ["vamos", *argv, "--hv-reference-front", "jmetal", "--output-root", str(tmp_path)])
    main()

    (manifest,) = tmp_path.rglob("manifest.json")
    termination = json.loads(manifest.read_text(encoding="utf-8"))["resolved_spec"]["termination"]["config"]
    assert termination["reference_front_source"] == "jmetal"
    assert Path(termination["reference_front_path"]).name == "jmetal-zdt1.csv"


def test_the_cli_records_the_default_front_as_vamos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Without --hv-reference-front the run uses VAMOS's packaged front, the same as --hv-reference-front vamos.
    from vamos.experiment.cli.main import main

    argv = ["--problem", "zdt1", "--algorithm", "nsgaii", "--max-evaluations", "200", "--hv-threshold", "0.5"]
    monkeypatch.setattr(sys, "argv", ["vamos", *argv, "--output-root", str(tmp_path)])
    main()

    (manifest,) = tmp_path.rglob("manifest.json")
    termination = json.loads(manifest.read_text(encoding="utf-8"))["resolved_spec"]["termination"]["config"]
    assert termination["reference_front_source"] == "vamos"
    # The packaged file is ZDT1.csv; a case-insensitive file system can report it in lower case.
    assert Path(termination["reference_front_path"]).name.lower() == "zdt1.csv"
