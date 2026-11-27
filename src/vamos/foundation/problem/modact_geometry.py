"""The actuator geometry of the MODAct problems, computed as trimesh computes it, without trimesh's mesh objects.

modact 1.0.1 builds an actuator's geometry with trimesh 5.2.2: one ``trimesh.primitives.Cylinder`` per motor and
gear, which modact places with its own translation and rotation matrices, the union of their meshes
(``modact.meshutils.merge_meshes``), the area of its convex hull (the cost of the housing), its bounding box
(constraint level 3) and python-fcl's collision checks between the parts (levels 2 to 4). trimesh's mesh objects
spend most of an evaluation hashing and caching arrays and checking that each cylinder is convex, and SciPy opens a
temporary file for Qhull's messages each time the hull is computed (on Windows, where it cannot keep them in memory).

:class:`ExactGeometry` evaluates a design with modact's own code, models and placement included, and replaces
only three names of modact while it runs, for its own thread: the ``Cylinder`` of ``modact.models.gears`` and
``modact.models.motors``, and the ``merge_meshes`` of ``modact.actuator`` (other threads keep trimesh's objects).
Their stand-ins compute the same arrays with the same NumPy, SciPy and python-fcl calls as trimesh 5.2.2, so ``F``
and ``G`` are the same bits:

- a cylinder's vertices are those of ``trimesh.creation.revolve``, with the same products, the same matrix product
  by the cylinder's transform and the same merging of the duplicated centres of its end faces, and its faces are
  revolve's; trimesh finds every such cylinder convex, so trimesh's collision manager gives python-fcl the same
  ``fcl.Convex``;
- the union stacks the cylinders' vertices in modact's order, so its bounding box is trimesh's;
- the hull area is ``trimesh.convex.convex_hull(...).area``: the same ``scipy.spatial.ConvexHull`` call on the
  same points, then trimesh's removal of zero-area triangles, outward winding, vertex merging and area sum. Qhull
  writes its messages to ``os.devnull`` instead of a temporary file; they only ever reach an error message.

A design outside these cases (a degenerate cylinder, a hull that Qhull rejects, a merged hull vertex, an
inconsistent hull winding) raises :class:`GeometryFallback` inside the stand-ins, and modact evaluates it with
trimesh's own objects. The stand-ins reproduce trimesh 5.2.2 and modact 1.0.1, which :func:`unsupported_reason`
checks; trimesh is MIT-licensed, and none of modact's (GPL-3.0) code is copied here.
"""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from importlib import metadata
from types import ModuleType
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]

# The versions whose code the stand-ins reproduce.
REPRODUCED_VERSIONS: dict[str, str] = {"modact": "1.0.1", "trimesh": "5.2.2"}

# trimesh.primitives.Cylinder's default number of sections, and the four points of the profile that
# trimesh.creation.cylinder revolves: the centre of the bottom face, the bottom and top rims and the top centre.
_SECTIONS = 32
_PER = 4
# trimesh's tolerances: tol.merge, the size of the grid on which merge_vertices merges vertices.
_MERGE_DIGITS = 8
# A smaller cylinder could lose revolve's triangles to tol.merge or merge rim vertices; none is that small.
_MIN_SIZE = 1e-3


class GeometryFallback(Exception):
    """A design whose geometry the stand-ins do not reproduce; modact evaluates it with trimesh's objects."""


def unsupported_reason() -> str | None:
    """Why the stand-ins cannot evaluate the installed modact, or ``None`` when they can."""
    for distribution, version in REPRODUCED_VERSIONS.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return f"{distribution} is not installed"
        if installed != version:
            return f"they reproduce {distribution} {version}, and {distribution} {installed} is installed"
    try:
        modules = _load()
    except ImportError as exc:
        return f"modact's geometry modules cannot be imported ({exc})"
    expected = (modules.trimesh_cylinder, modules.trimesh_cylinder, modules.merge_meshes)
    if _patched_names(modules) != expected:
        return "modact's Cylinder or merge_meshes is not trimesh's and modact's own"
    return None


@dataclass(frozen=True)
class _Modules:
    # modact's modules, whose names _stand_ins replaces.
    gears: Any
    motors: Any
    actuator: Any
    problems: Any
    trimesh_cylinder: Any
    merge_meshes: Any
    triangles: ModuleType
    util: ModuleType
    convex_hull: Any
    qhull_error: type[Exception]
    qhull_options: str


@cache
def _load() -> _Modules:
    spatial = importlib.import_module("scipy.spatial")
    return _Modules(
        gears=importlib.import_module("modact.models.gears"),
        motors=importlib.import_module("modact.models.motors"),
        actuator=importlib.import_module("modact.actuator"),
        problems=importlib.import_module("modact.problems"),
        trimesh_cylinder=importlib.import_module("trimesh.primitives").Cylinder,
        merge_meshes=importlib.import_module("modact.meshutils").merge_meshes,
        triangles=importlib.import_module("trimesh.triangles"),
        util=importlib.import_module("trimesh.util"),
        convex_hull=spatial.ConvexHull,
        qhull_error=spatial.QhullError,
        qhull_options=str(importlib.import_module("trimesh.convex").QHULL_DEFAULT),
    )


_STATE: dict[str, Any] = {}


def _lock() -> Any:
    """The lock under which a thread replaces modact's or SciPy's names (``dict.setdefault`` is atomic)."""
    lock = _STATE.get("lock")
    return lock if lock is not None else _STATE.setdefault("lock", threading.RLock())


def _patched_names(modules: _Modules) -> tuple[Any, Any, Any]:
    return (modules.gears.Cylinder, modules.motors.Cylinder, modules.actuator.merge_meshes)


# ---------------------------------------------------------------------------------------------------------
# Cylinders: trimesh.creation.revolve of the profile [[0, -h/2], [r, -h/2], [r, h/2], [0, h/2]]
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Topology:
    cos: Array  # revolve's unit circle, turned to each vertex
    sin: Array
    on_rim: Array  # 1.0 for a rim vertex, 0.0 for a centre
    on_top: Array  # 1.0 for a top vertex, -1.0 for a bottom one
    keep: Array  # the vertices that merge_vertices keeps
    faces: Array  # revolve's faces, numbered after merge_vertices
    identity: Array


@cache
def _topology() -> _Topology:
    """revolve's cylinder of 32 sections, before its radius, height and transform."""
    theta = np.linspace(0, np.pi * 2.0, _SECTIONS + 1)
    circle = np.column_stack((np.cos(theta), np.sin(theta)))
    n_vertices = _SECTIONS * _PER
    # Vertex PER * j + k is profile point k turned by theta[j]; the last section repeats the first and is dropped.
    turned = np.tile(circle, (1, _PER)).reshape((-1, 2))[:n_vertices]
    on_rim = np.tile([0.0, 1.0, 1.0, 0.0], _SECTIONS)
    on_top = np.tile([-1.0, -1.0, 1.0, 1.0], _SECTIONS)
    # revolve's faces: a quad per profile segment and section, less the two triangles that join a centre to
    # itself, which have zero area.
    quad = np.array([0, _PER, 1, 1, _PER, _PER + 1])
    single = np.tile(quad, _PER - 1).reshape((-1, 3))
    single += np.tile(np.arange(_PER - 1), (2, 1)).T.reshape((-1, 1))
    single = single[[1, 2, 3, 4]]
    offset = np.tile(np.arange(_SECTIONS) * _PER, (len(single), 1)).T.reshape((-1, 1))
    faces = (np.tile(single.ravel(), _SECTIONS).reshape((-1, 3)) + offset) % n_vertices
    # merge_vertices keeps the first occurrence of each vertex: the centres of section 0 and both rim points of
    # every section, in order.
    keep = [0, 1, 2, 3] + [_PER * j + k for j in range(1, _SECTIONS) for k in (1, 2)]
    inverse = np.zeros(n_vertices, dtype=np.int64)
    inverse[keep] = np.arange(len(keep))
    inverse[0::_PER] = 0
    inverse[3::_PER] = 3
    merged_faces = inverse[faces.reshape(-1)].reshape((-1, 3))
    merged_faces.flags.writeable = False
    cos, sin = np.ascontiguousarray(turned[:, 0]), np.ascontiguousarray(turned[:, 1])
    return _Topology(cos, sin, on_rim, on_top, np.array(keep), merged_faces, np.eye(4))


class _Attributes:
    """The ``primitive`` of a cylinder: the values that modact reads (the transform, for constraint level 4)."""

    __slots__ = ("height", "radius", "sections", "transform")

    def __init__(self, radius: float, height: float, transform: Array) -> None:
        self.radius = radius
        self.height = height
        self.transform = transform
        self.sections = _SECTIONS


class Cylinder:
    """Stands in for ``trimesh.primitives.Cylinder(radius, height, transform)`` with the same vertices and faces.

    ``is_convex`` is trimesh's verdict on every cylinder whose radius and height are at least ``_MIN_SIZE``:
    the cylinder is watertight and one body, and each pair of adjacent faces projects at most rounding errors
    above the other's plane, far below trimesh's threshold of ``1e-5`` times its size.
    """

    is_convex = True

    def __init__(self, radius: float = 1.0, height: float = 1.0, transform: Array | None = None, sections: int = _SECTIONS) -> None:
        if transform is None or sections != _SECTIONS:
            raise GeometryFallback("a cylinder without a transform or with other sections")
        radius = float(radius)
        height = float(height)
        matrix = np.asanyarray(transform, dtype=np.float64)
        if not (radius >= _MIN_SIZE and abs(height) >= _MIN_SIZE and np.isfinite(matrix).all() and matrix.shape == (4, 4)):
            raise GeometryFallback(f"a cylinder of radius {radius!r} and height {height!r}")
        self.primitive = _Attributes(radius, height, matrix)
        topology = _topology()
        self.faces = topology.faces
        self.vertices = _cylinder_vertices(topology, radius, height, matrix)


def _cylinder_vertices(topology: _Topology, radius: float, height: float, matrix: Array) -> Array:
    """The merged vertices of revolve's cylinder, moved by ``matrix`` as ``trimesh.transformations.transform_points``."""
    half = abs(height) / 2.0
    # revolve multiplies the turned circle by each profile point's radius, r or 0.0, and takes its height, +-h/2.
    radii = topology.on_rim * radius
    stack = np.empty((len(topology.cos), 4))
    stack[:, 0] = topology.cos * radii
    stack[:, 1] = topology.sin * radii
    stack[:, 2] = topology.on_top * half
    stack[:, 3] = 1.0
    if np.abs(matrix - topology.identity).max() < 1e-8:
        vertices = stack[:, :3]
    else:
        # A reflection would flip revolve's faces; modact only translates and turns about z.
        if _det3(matrix) < 0.5:
            raise GeometryFallback("a cylinder transform that is not a rotation")
        vertices = np.dot(matrix, stack.T).T[:, :3]
    # The centres of every section are one vertex each, merged onto section 0's when their rounded coordinates
    # agree; equal coordinates always do.
    if not ((vertices[0::_PER] == vertices[0]).all() and (vertices[3::_PER] == vertices[3]).all()):
        raise GeometryFallback("the centres of a cylinder's end faces differ")
    merged: Array = vertices[topology.keep]
    return merged


def _det3(m: Array) -> float:
    return float(
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


# ---------------------------------------------------------------------------------------------------------
# The union of the cylinders, its bounding box and its convex hull
# ---------------------------------------------------------------------------------------------------------


class _Extents:
    __slots__ = ("extents",)

    def __init__(self, extents: Array) -> None:
        self.extents = extents


class _Area:
    __slots__ = ("area",)

    def __init__(self, area: float) -> None:
        self.area = area


class Space:
    """Stands in for ``modact.meshutils.merge_meshes(meshes)``: the cylinders' vertices and faces, stacked."""

    def __init__(self, meshes: Sequence[Cylinder]) -> None:
        if not all(isinstance(mesh, Cylinder) for mesh in meshes):
            raise GeometryFallback("a part that is not a cylinder")
        self.vertices = np.vstack([mesh.vertices for mesh in meshes])
        counts = np.cumsum([0] + [len(mesh.vertices) for mesh in meshes[:-1]])
        self.faces = np.vstack([mesh.faces + count for mesh, count in zip(meshes, counts, strict=True)])
        self._hull: _Area | None = None

    @property
    def bounding_box(self) -> _Extents:
        """trimesh's axis-aligned box: ``np.ptp`` of the bounds of the referenced vertices, which are all of them."""
        bounds = np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])
        return _Extents(np.ptp(bounds, axis=0))

    @property
    def convex_hull(self) -> _Area:
        if self._hull is None:
            self._hull = _Area(hull_area(self.vertices))
        return self._hull


def merge_meshes(meshes: Sequence[Cylinder]) -> Space:
    return Space(meshes)


def hull_area(points: Array) -> float:
    """``trimesh.convex.convex_hull(points).area``, step by step, for a hull that trimesh does not repair further."""
    modules = _load()
    triangles, util = modules.triangles, modules.util
    try:
        hull = _qhull(points)
    except modules.qhull_error as exc:
        # trimesh retries with "QJ"; modact's own objects take that path.
        raise GeometryFallback("Qhull rejected the points") from exc
    vid = np.sort(hull.vertices)
    mask = np.zeros(len(hull.points), dtype=np.int64)
    mask[vid] = np.arange(len(vid))
    faces = mask[hull.simplices].copy()
    vertices = hull.points[vid].copy()
    crosses = _cross(vertices, faces)
    normals, valid = util.unitize(crosses, check_valid=True)
    faces = faces[valid]
    crosses = crosses[valid]
    if len(faces) == 0 or not np.isfinite(vertices).all():
        raise GeometryFallback("an empty or non-finite hull")
    triangles_area = triangles.area(crosses=crosses)
    triangles_center = vertices[faces].mean(axis=1)
    try:
        centroid = np.average(triangles_center, weights=triangles_area, axis=0)
    except ZeroDivisionError as exc:
        raise GeometryFallback("a hull of zero area") from exc
    backwards = util.diagonal_dot(normals, triangles_center - centroid) < 0.0
    faces[backwards] = np.fliplr(faces[backwards])
    # trimesh.Trimesh(process=True) merges the vertices that round to the same point of tol.merge's grid. Without
    # two such vertices, it only renumbers them, and the triangles keep their coordinates.
    referenced = np.zeros(len(vertices), dtype=bool)
    referenced[faces] = True
    rounded = (vertices[referenced] * (10**_MERGE_DIGITS)).round().astype(np.int64)
    ordered = rounded[np.lexsort(rounded.T)]
    if (ordered[1:] == ordered[:-1]).all(axis=1).any():
        raise GeometryFallback("two hull vertices merge")
    # fix_normals: fix_winding returns at once when the winding is consistent, and fix_inversion inverts a
    # watertight hull of negative volume. trimesh.graph.is_watertight finds the hull watertight and consistently
    # wound when every edge appears once in each direction.
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    directed = np.sort(edges[:, 0] * len(vertices) + edges[:, 1])
    reversed_ = np.sort(edges[:, 1] * len(vertices) + edges[:, 0])
    if not ((directed[1:] != directed[:-1]).all() and np.array_equal(directed, reversed_)):
        raise GeometryFallback("a hull that is not closed and consistently wound")
    crosses = _cross(vertices, faces)
    # The sign of trimesh's volume, the sum of these terms divided by 6, here summed in another order.
    terms = crosses[:, 0] * (vertices[faces[:, 0], 0] + vertices[faces[:, 1], 0] + vertices[faces[:, 2], 0])
    if not terms.sum() > 1e-9 * np.abs(terms).sum():
        raise GeometryFallback("a hull of negative or vanishing volume")
    area: float = triangles.area(crosses=crosses).sum()
    return area


def _cross(vertices: Array, faces: Array) -> Array:
    """``trimesh.triangles.cross(vertices[faces])``: ``np.cross`` of ``t1 - t0`` and ``t2 - t1``, term by term."""
    t0, t1, t2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    a = t1 - t0
    b = t2 - t1
    cross = np.empty_like(a)
    cross[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    cross[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    cross[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return cross


# ---------------------------------------------------------------------------------------------------------
# Qhull's messages
# ---------------------------------------------------------------------------------------------------------

# Whether hull_area sends Qhull's messages to os.devnull (see _qhull).
REDIRECT_QHULL_MESSAGES = True
_DEVNULL_NAME = b"vamos-modact-qhull-messages"


class _MessageTempfile:
    """``tempfile`` for SciPy's message stream: ``os.devnull`` in the thread that computes a hull."""

    def __init__(self, tempfile: Any, owner: int) -> None:
        self._tempfile = tempfile
        self._owner = owner

    def mkstemp(self, *args: Any, **kwargs: Any) -> tuple[int, Any]:
        if threading.get_ident() != self._owner:
            return self._tempfile.mkstemp(*args, **kwargs)  # type: ignore[no-any-return]
        return os.open(os.devnull, os.O_RDWR | getattr(os, "O_BINARY", 0)), _DEVNULL_NAME

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tempfile, name)


class _MessageOs:
    """``os`` for SciPy's message stream: nothing to remove for ``os.devnull``."""

    def __init__(self, os_module: Any) -> None:
        self._os = os_module

    def remove(self, path: Any, *args: Any, **kwargs: Any) -> None:
        if path != _DEVNULL_NAME:
            self._os.remove(path, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._os, name)


def _qhull(points: Array) -> Any:
    """``scipy.spatial.ConvexHull(points, qhull_options="Qt QbB Pp")``, trimesh's call.

    Where SciPy cannot keep Qhull's messages in memory (on Windows), ``scipy._lib.messagestream.MessageStream``
    creates and deletes a temporary file for each call, which takes longer than the hull. While this thread
    computes the hull, the stream's ``tempfile.mkstemp`` returns ``os.devnull`` instead; other threads keep the
    temporary file. Qhull only reads its input, so the hull is the same; SciPy reads the messages only to word
    an error, and an error sends the design to modact's own path.
    """
    modules = _load()
    if not REDIRECT_QHULL_MESSAGES:
        return modules.convex_hull(points, qhull_options=modules.qhull_options)
    with _lock(), _messages_to_devnull():
        return modules.convex_hull(points, qhull_options=modules.qhull_options)


@contextmanager
def _messages_to_devnull() -> Iterator[None]:
    try:
        stream: Any = importlib.import_module("scipy._lib.messagestream")
    except ImportError:
        yield
        return
    real_tempfile, real_os = getattr(stream, "tempfile", None), getattr(stream, "os", None)
    if not (isinstance(real_tempfile, ModuleType) and isinstance(real_os, ModuleType)):
        yield
        return
    stream.tempfile = _MessageTempfile(real_tempfile, threading.get_ident())
    stream.os = _MessageOs(real_os)
    try:
        yield
    finally:
        stream.tempfile = real_tempfile
        stream.os = real_os


# ---------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------

# The thread whose design the stand-ins evaluate; modact's names reach trimesh's objects from any other.
_owner: int | None = None


def _for_owner(ours: Callable[..., Any], theirs: Callable[..., Any]) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        return (ours if threading.get_ident() == _owner else theirs)(*args, **kwargs)

    return call


@contextmanager
def _stand_ins(modules: _Modules) -> Iterator[None]:
    """Give modact this module's ``Cylinder`` and ``merge_meshes`` while this thread evaluates one design."""
    global _owner
    with _lock():
        saved = _patched_names(modules)
        if saved != (modules.trimesh_cylinder, modules.trimesh_cylinder, modules.merge_meshes):
            raise RuntimeError("modact's Cylinder or merge_meshes changed while VAMOS evaluated a MODAct problem.")
        cylinder = _for_owner(Cylinder, modules.trimesh_cylinder)
        modules.gears.Cylinder = cylinder
        modules.motors.Cylinder = cylinder
        modules.actuator.merge_meshes = _for_owner(merge_meshes, modules.merge_meshes)
        _owner = threading.get_ident()
        try:
            yield
        finally:
            _owner = None
            modules.gears.Cylinder, modules.motors.Cylinder, modules.actuator.merge_meshes = saved


class ExactGeometry:
    """Evaluates a modact problem, ``modact.problems.get_problem(name)``, with the stand-ins of this module.

    Calling it with a design returns modact's ``(objectives, constraints)``. ``fallbacks`` counts the designs
    that modact evaluated with trimesh's objects instead (see :class:`GeometryFallback`).
    """

    def __init__(self, problem: Any) -> None:
        reason = unsupported_reason()
        if reason is not None:
            raise ImportError(f"VAMOS's MODAct geometry is unavailable: {reason}.")
        self._modules = _load()
        if type(problem) is not self._modules.problems.Problem:
            raise TypeError(f"Expected a modact.problems.Problem; got {type(problem).__name__}.")
        self._problem: Callable[[Array], tuple[Any, Any]] = problem
        self.fallbacks = 0

    def __call__(self, x: Array) -> tuple[Any, Any]:
        try:
            with _stand_ins(self._modules):
                return self._problem(x.copy())
        except GeometryFallback:
            self.fallbacks += 1
        return self._problem(x.copy())


__all__ = ["REPRODUCED_VERSIONS", "Cylinder", "ExactGeometry", "GeometryFallback", "Space", "hull_area", "unsupported_reason"]
