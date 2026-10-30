from importlib import resources

import vamos.resources as data_module
from vamos.experiment.execution import _default_weight_path
from vamos.foundation.core.hv_stop import build_hv_stop_config
from vamos.foundation.problem.tsplib import load_tsplib_coords
from vamos.resources import reference_front_path, weight_path


def test_reference_front_packaged_and_accessible():
    path = reference_front_path("zdt1")
    assert path.is_file()
    data = path.read_text().splitlines()
    assert len(data) > 0


def test_reference_front_zdt5_packaged_and_accessible():
    path = reference_front_path("zdt5")
    assert path.is_file()
    data = path.read_text().splitlines()
    assert len(data) == 31


def test_reference_front_wfg4_packaged_and_accessible():
    packaged = resources.files("vamos.resources.reference_fronts").joinpath("wfg4.csv")
    assert packaged.is_file()
    path = reference_front_path("wfg4")
    assert path.is_file()


def test_reference_front_zcat_bundle_packaged_and_accessible():
    path_3d = reference_front_path("zcat1.3d")
    assert path_3d.is_file()
    rows_3d = path_3d.read_text(encoding="utf-8").splitlines()
    assert len(rows_3d) > 0

    path_alias = reference_front_path("zcat1")
    assert path_alias.is_file()


def test_third_party_reference_fronts_ship_their_license():
    # Tanabe and Ishibuchi's approximated fronts are MIT-licensed: their notice must travel with the data.
    notice = resources.files("vamos.resources.reference_fronts").joinpath("LICENSE-reproblems.txt")
    assert notice.is_file()
    text = notice.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Copyright (c) 2026 Ryoji Tanabe" in text
    assert "- re24.csv: reference_points_RE24.dat" in text


def test_reference_front_prefers_override_data_over_packaged(monkeypatch, tmp_path):
    override_data = tmp_path / "override_fronts"
    pkg_data = tmp_path / "pkg_fronts"
    override_data.mkdir()
    pkg_data.mkdir()

    override_file = override_data / "zdt1.csv"
    pkg_file = pkg_data / "zdt1.csv"
    override_file.write_text("0.1,0.2\n", encoding="utf-8")
    pkg_file.write_text("0.9,0.8\n", encoding="utf-8")

    monkeypatch.setattr(data_module, "_reference_front_locations", lambda: [override_data, pkg_data])

    resolved = data_module.reference_front_path("zdt1")
    assert resolved == override_file


def test_weight_file_packaged_and_accessible():
    path = weight_path("zdt1problem_2obj_pop100.csv")
    assert path.is_file()
    assert path.stat().st_size > 0
    resolved = _default_weight_path("zdt1problem", 2, 100)
    assert resolved.endswith("zdt1problem_2obj_pop100.csv")


def test_hv_stop_uses_packaged_reference_front():
    cfg = build_hv_stop_config(0.1, None, "zdt1")
    assert cfg is not None
    ref_path = cfg["reference_front_path"]
    assert "ZDT1" in ref_path.upper()
    assert resources.files("vamos.resources.reference_fronts").joinpath("ZDT1.csv").is_file()


def test_hv_stop_uses_dimensioned_zcat_reference_front():
    cfg = build_hv_stop_config(0.1, None, "zcat1", n_obj=3)
    assert cfg is not None
    ref_path = str(cfg["reference_front_path"]).lower()
    assert ref_path.endswith("zcat1.3d.csv")


def test_tsplib_packaged_and_accessible():
    assert resources.files("vamos.resources.tsplib").joinpath("kroA100.tsp").is_file()
    coords = load_tsplib_coords("kroA100")
    assert coords.ndim == 2
    assert coords.shape[1] == 2
