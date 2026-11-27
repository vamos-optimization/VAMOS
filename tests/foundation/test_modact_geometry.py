"""VAMOS's stand-ins for the trimesh geometry of the MODAct problems give modact's values bit for bit.

Development after 1.5.2. ``MODActProblem(name, geometry="fast")`` evaluates a design with modact's own code and
:mod:`vamos.foundation.problem.modact_geometry`'s cylinders, union and hull in place of trimesh's objects;
``geometry="modact"`` keeps trimesh's. With modact 1.0.1, trimesh 5.2.2, python-fcl 0.7.0.11, SciPy 1.18.1 and
NumPy 2.5.5 on Windows, their ``F`` and ``G`` were bit-identical on the 134 478 final designs of a pilot campaign on
CS1-CTS4, on one perturbation of each by a relative step of 1e-9 to 1e-3, on 20 000 random designs per problem, at
202 contacts between parts located by bisection, and on 3000 designs per CTSE and CTSEI problem. These tests compare
them on a fixed sample, and compare each stand-in with the trimesh object it replaces. They need the modact extra,
which CI does not install.
"""

from __future__ import annotations

import os
import tempfile
import threading
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.foundation.problem import modact_geometry
from vamos.foundation.problem.modact import GEOMETRIES, MODACT, MODActProblem

Array = np.ndarray[Any, Any]


def _require_supported_modact() -> None:
    pytest.importorskip("modact.problems", reason="the modact extra is not installed", exc_type=ImportError)
    reason = modact_geometry.unsupported_reason()
    if reason is not None:
        pytest.skip(f"the stand-ins do not apply: {reason}")


def _bits(values: Array) -> Array:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def _sample(problem: MODActProblem, n: int) -> Array:
    rng = np.random.default_rng(zlib.crc32(problem.modact_name.encode()))
    X = problem.xl + rng.random((n, problem.n_var)) * (problem.xu - problem.xl)
    # Compact designs: the same ones pushed to small gears, little displacement and the motor's axis.
    compact = X.copy()
    compact[:, 4::6] = 0.3 + 0.05 * rng.random((n, 3))
    compact[:, 6::6] = 0.01 * rng.uniform(-1.0, 1.0, (n, 3))
    return np.vstack([X, compact])


@pytest.mark.parametrize("name", list(MODACT))
def test_the_fast_geometry_gives_modacts_bits(name: str) -> None:
    _require_supported_modact()
    fast = MODActProblem(name)
    reference = MODActProblem(name, geometry="modact")
    assert (fast.geometry, reference.geometry) == ("fast", "modact")
    X = _sample(fast, 4)
    X_before = X.copy()

    ours: dict[str, Array] = {}
    theirs: dict[str, Array] = {}
    fast.evaluate(X, ours)
    reference.evaluate(X, theirs)

    np.testing.assert_array_equal(X, X_before)
    for key in ("F", "G"):
        assert np.array_equal(_bits(ours[key]), _bits(theirs[key])), key
    # Each row is also modact's own call, with the documented signs.
    raw = [reference._problem(x.copy()) for x in X]
    np.testing.assert_array_equal(ours["F"], np.array([f for f, _ in raw], dtype=float) * fast.objective_signs)
    np.testing.assert_array_equal(ours["G"], np.array([g for _, g in raw], dtype=float) * fast.constraint_signs)


@pytest.mark.parametrize(
    ("radius", "height", "angle", "translation"),
    [
        (11.0, 16.8, 0.0, (0.0, 0.0, 8.4)),
        (1.3, 1.5, 0.7, (0.0, 0.0, -0.75)),
        (40.2, 15.0, -3.1, (55.0, -12.0, 30.0)),
        (2.5, 9.0, np.pi, (0.0, 0.0, 0.0)),
        (7.0, 0.001, 1.0, (1e-9, 0.0, 0.0)),
    ],
)
def test_a_cylinder_has_trimeshs_vertices_and_faces(radius: float, height: float, angle: float, translation: tuple[float, ...]) -> None:
    _require_supported_modact()
    transformations = pytest.importorskip("trimesh.transformations")
    primitives = pytest.importorskip("trimesh.primitives")
    transform = transformations.translation_matrix(translation).dot(transformations.rotation_matrix(angle, [0, 0, 1]))

    ours = modact_geometry.Cylinder(radius=radius, height=height, transform=transform.copy())
    theirs = primitives.Cylinder(radius=radius, height=height, transform=transform.copy())

    assert np.array_equal(_bits(ours.vertices), _bits(theirs.vertices))
    np.testing.assert_array_equal(ours.faces, theirs.faces)
    assert ours.is_convex and theirs.is_convex
    np.testing.assert_array_equal(ours.primitive.transform, theirs.primitive.transform)


def test_the_identity_transform_takes_trimeshs_shortcut() -> None:
    _require_supported_modact()
    primitives = pytest.importorskip("trimesh.primitives")
    ours = modact_geometry.Cylinder(radius=3.0, height=2.0, transform=np.eye(4))
    theirs = primitives.Cylinder(radius=3.0, height=2.0, transform=np.eye(4))
    assert np.array_equal(_bits(ours.vertices), _bits(theirs.vertices))


@pytest.mark.parametrize("radius", [0.0, -1.0, 1e-4, float("nan")])
def test_a_degenerate_cylinder_is_left_to_trimesh(radius: float) -> None:
    with pytest.raises(modact_geometry.GeometryFallback):
        modact_geometry.Cylinder(radius=radius, height=1.0, transform=np.eye(4))


def _actuators(name: str, n: int) -> list[Any]:
    problem = MODActProblem(name, geometry="modact")
    return [problem._problem.prepare(x.copy())[0] for x in _sample(problem, n)]


def test_the_union_hull_and_box_are_trimeshs() -> None:
    _require_supported_modact()
    for actuator in _actuators("cs3", 4):
        meshes, _, space = actuator.mesh
        ours = modact_geometry.merge_meshes(
            [
                modact_geometry.Cylinder(radius=m.primitive.radius, height=m.primitive.height, transform=m.primitive.transform)
                for m in meshes
            ]
        )
        assert np.array_equal(_bits(ours.vertices), _bits(space.vertices))
        assert len(ours.faces) == len(space.faces)
        assert np.array_equal(_bits(ours.bounding_box.extents), _bits(space.bounding_box.extents))
        area = space.convex_hull.area
        assert _bits(ours.convex_hull.area) == _bits(area)
        # With SciPy's temporary file for Qhull's messages, too.
        modact_geometry.REDIRECT_QHULL_MESSAGES = False
        try:
            assert _bits(modact_geometry.hull_area(ours.vertices)) == _bits(area)
        finally:
            modact_geometry.REDIRECT_QHULL_MESSAGES = True


def test_modacts_names_and_scipys_message_stream_are_restored() -> None:
    _require_supported_modact()
    import scipy._lib.messagestream as stream
    from modact import actuator, meshutils
    from modact.models import gears, motors
    from trimesh.primitives import Cylinder

    problem = MODActProblem("ct2")
    problem.evaluate(_sample(problem, 2), {})

    assert gears.Cylinder is Cylinder and motors.Cylinder is Cylinder
    assert actuator.merge_meshes is meshutils.merge_meshes
    assert stream.tempfile is tempfile and stream.os is os


def test_other_threads_keep_trimeshs_cylinders_meanwhile() -> None:
    _require_supported_modact()
    from modact.models import gears
    from trimesh.primitives import Cylinder

    found: list[type] = []
    with modact_geometry._stand_ins(modact_geometry._load()):
        here = type(gears.Cylinder(radius=2.0, height=1.0, transform=np.eye(4)))
        other = threading.Thread(target=lambda: found.append(type(gears.Cylinder(radius=2.0, height=1.0, transform=np.eye(4)))))
        other.start()
        other.join()

    assert here is modact_geometry.Cylinder and found == [Cylinder]
    assert gears.Cylinder is Cylinder


def test_only_the_hull_thread_writes_qhulls_messages_to_devnull(tmp_path: Any) -> None:
    here = modact_geometry._MessageTempfile(tempfile, threading.get_ident())
    fd, name = here.mkstemp(dir=tmp_path)
    os.close(fd)
    assert name == modact_geometry._DEVNULL_NAME and not list(tmp_path.iterdir())
    modact_geometry._MessageOs(os).remove(name)

    elsewhere = modact_geometry._MessageTempfile(tempfile, threading.get_ident() + 1)
    fd, name = elsewhere.mkstemp(dir=tmp_path)
    os.close(fd)
    assert Path(name).parent == tmp_path
    modact_geometry._MessageOs(os).remove(name)
    assert not list(tmp_path.iterdir())


def test_a_design_the_stand_ins_do_not_cover_is_evaluated_by_modact(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_supported_modact()
    fast = MODActProblem("cs2")
    reference = MODActProblem("cs2", geometry="modact")
    X = _sample(fast, 2)

    def reject(points: Array) -> float:
        raise modact_geometry.GeometryFallback("a test")

    monkeypatch.setattr(modact_geometry, "hull_area", reject)
    ours: dict[str, Array] = {}
    theirs: dict[str, Array] = {}
    fast.evaluate(X, ours)
    reference.evaluate(X, theirs)

    evaluator = fast._evaluate_design
    assert isinstance(evaluator, modact_geometry.ExactGeometry) and evaluator.fallbacks == len(X)
    for key in ("F", "G"):
        assert np.array_equal(_bits(ours[key]), _bits(theirs[key]))


def test_the_geometry_option_is_checked() -> None:
    assert GEOMETRIES == ("auto", "fast", "modact")
    with pytest.raises(ValueError, match="geometry must be one of"):
        MODActProblem("cs1", geometry="trimesh")


def test_other_versions_use_modacts_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modact.problems", reason="the modact extra is not installed", exc_type=ImportError)
    monkeypatch.setitem(modact_geometry.REPRODUCED_VERSIONS, "trimesh", "0.0")

    assert MODActProblem("cs1").geometry == "modact"
    with pytest.raises(ImportError, match="trimesh 0.0"):
        MODActProblem("cs1", geometry="fast")
