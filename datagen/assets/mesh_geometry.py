"""Turning a scanned mesh into a solid the labeller and the renderer can both intersect.

A procedural asset is a box, a cylinder or a sphere, and every question the grasp reference asks
(where does this line enter and leave, which way does the surface face, is this point inside) has a
closed-form answer. A scanned mesh has none of that for free, which is why this module exists rather
than a one-line ``trimesh.load``.

Scanned meshes are mostly not watertight, and trimesh's merge plus ``fill_holes`` does not repair
them. Watertightness is the property that decides everything: a ray span and a nearest-surface
normal are exact on any triangle soup, but "is this point inside the object" is only defined for a
closed surface, and that is the question the finger-collision test asks thousands of times per
object.

Almost every broken mesh fails on open boundaries alone: no non-manifold edges, and every boundary
vertex of degree exactly 2, so the boundary edge set is a disjoint union of simple loops. These are
scans missing the face the object was standing on. Capping each loop with a fan to its centroid
closes them exactly, keeps every original triangle, and fabricates surface only where no camera ever
looks. A ``merge_vertices()`` call after the fan welds the new ring into non-manifold edges and
reopens the mesh, so the repair would undo itself; `close_mesh` does not make that call.

Why open3d alongside trimesh. trimesh reads the files and does the topology; its ray engine is pure
Python and needs ``rtree``, which is not shipped here. open3d is already a dependency and its
``RaycastingScene`` is Embree-backed, so the exact path costs no new dependency and no meaningful
time. ``compute_occupancy`` and ``signed_distance < 0`` agree, which is the consistency check saying
the closed mesh really does bound a solid.

Meshes whose boundaries are not simple loops, and meshes that close to a sheet, are refused and
excluded from the bank: a visible loss rather than a silently wrong shape.

Conventions are the repo's: the mesh arrives from the file in metres and leaves here in millimetres,
centred on its axis-aligned bounds centre. The renderer places the rigid body at the placement
position, so a mesh whose file origin is not its centre would otherwise stand somewhere other than
the labels say.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, MESH_GEOMETRY_LOG_FILE

__all__ = [
    "MeshShape",
    "close_mesh",
    "load_mesh_shape",
]

logger = create_logger("MeshGeometry", MESH_GEOMETRY_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

_M_TO_MM: Final[float] = 1000.0

#: Below this a "solid" is a closed sheet rather than a body: capping a flat scan yields a surface
#: with essentially no interior, and every containment answer inside it is noise. The ratio is
#: closed volume over convex-hull volume, and the low tail is refused rather than labelled.
_MIN_VOLUME_RATIO: Final[float] = 0.02

#: Below this a closed surface encloses nothing at all: a scan that was a single sheet. In
#: cubic metres, because that is the unit a mesh arrives in; 1e-9 m^3 is a cubic millimetre.
_MIN_VOLUME_M3: Final[float] = 1e-9


@dataclass(frozen=True, slots=True, eq=False)
class MeshShape:
    """A closed mesh in body millimetres, centred on its bounds, with an Embree scene beside it.

    Immutable and cached per file: building it costs about 200 ms (read, close, orient) and a
    dataset run draws the same object into many scenes. ``eq=False`` so the shape hashes by
    identity: the arrays inside it are not comparable, and identity is the right key for the BVH
    cache below.
    """

    #: (V, 3) vertices in body millimetres, centred on the axis-aligned bounds centre.
    vertices_mm: np.ndarray
    #: (F, 3) triangle indices.
    faces: np.ndarray
    #: (F, 3) outward unit face normals, consistent with the closed orientation.
    face_normals: np.ndarray
    #: Full axis-aligned extent in body millimetres: the same number ``MeshAsset.extent_mm`` reports.
    extent_mm: tuple[float, float, float]
    #: Where the file's own origin sits relative to the bounds centre, in body mm. The renderer applies
    #: exactly this shift when authoring, or the drawn object and the labelled one stand apart.
    origin_offset_mm: tuple[float, float, float]
    source_path: str = ""

    # ------------------------------------------------------------------ derived

    @property
    def half_extent_mm(self) -> np.ndarray:
        return np.asarray(self.extent_mm, dtype=np.float64) / 2.0

    @property
    def bounding_radius_mm(self) -> float:
        return float(np.linalg.norm(self.vertices_mm, axis=1).max())

    def scene(self) -> Any:
        """The Embree BVH over this mesh, built on first use and cached for the process."""
        return _scene_for(self)

    # ------------------------------------------------------------------ queries
    #
    # All of these take body-frame millimetres. `Solid` does the world-to-body transform, exactly as
    # it does for the primitives, so the mesh path shares every frame convention with them.

    def signed_distance_mm(self, points_body_mm: np.ndarray) -> np.ndarray:
        """open3d's convention: positive outside, negative inside. Exact for a closed surface."""
        import open3d as o3d  # noqa: PLC0415 (heavy; only the mesh path needs it)

        points = np.ascontiguousarray(np.atleast_2d(points_body_mm), dtype=np.float32)
        query = o3d.core.Tensor(points)
        return self.scene().compute_signed_distance(query).numpy().astype(np.float64)

    def first_hit_mm(self, origins_body_mm: np.ndarray, directions_body: np.ndarray) -> np.ndarray:
        """Distance along each unit direction to the first surface, ``inf`` where the ray misses."""
        import open3d as o3d  # noqa: PLC0415

        rays = np.hstack([
            np.ascontiguousarray(np.atleast_2d(origins_body_mm), dtype=np.float32),
            np.ascontiguousarray(np.atleast_2d(directions_body), dtype=np.float32),
        ])
        result = self.scene().cast_rays(o3d.core.Tensor(rays))
        return result["t_hit"].numpy().astype(np.float64)

    def usd_arrays(self) -> tuple[np.ndarray, list[int], list[int], np.ndarray, np.ndarray]:
        """This mesh as USD wants it: ``(points_m, face_counts, face_indices, lower_m, upper_m)``.

        Here rather than at the two authoring sites: the renderer draws the scene and the shake cell
        rebuilds it for physics, and a mesh authored two slightly different ways is the failure this
        package exists to prevent. One conversion, one metre boundary, one set of triangles.
        """
        points = self.vertices_mm * (1.0 / _M_TO_MM)
        counts = [3] * len(self.faces)
        indices = [int(i) for i in np.asarray(self.faces).reshape(-1)]
        return points, counts, indices, points.min(axis=0), points.max(axis=0)

    def closest_face(self, point_body_mm: np.ndarray) -> int:
        """Index of the triangle nearest a point; a contact normal is read from this."""
        import open3d as o3d  # noqa: PLC0415

        query = o3d.core.Tensor(
            np.ascontiguousarray(np.atleast_2d(point_body_mm), dtype=np.float32))
        found = self.scene().compute_closest_points(query)
        return int(found["primitive_ids"].numpy()[0])


# Keyed by the shape itself, which hashes by identity (`eq=False`): an open3d scene is neither hashable
# nor picklable, so it cannot live on a frozen dataclass, and `load_mesh_shape` caches the shapes so
# their identities are stable for the life of a process.
@lru_cache(maxsize=512)
def _scene_for(shape: MeshShape) -> Any:
    import open3d as o3d  # noqa: PLC0415

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.ascontiguousarray(shape.vertices_mm, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(shape.faces, dtype=np.uint32)),
    )
    return scene


def _boundary_loops(mesh: Any) -> list[list[int]] | None:
    """The mesh's open boundaries as vertex rings, or ``None`` if they are not clean loops.

    ``None`` rather than a best effort: a boundary vertex shared by more than two boundary edges
    means two openings meet, and capping those by guessing produces a closed mesh that is not the
    object. Such a mesh is refused here and excluded from the bank, which is a visible loss rather
    than a silently wrong shape.
    """
    edges = np.sort(mesh.edges_sorted, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if not len(boundary):
        return []

    adjacency: dict[int, list[int]] = collections.defaultdict(list)
    for first, second in boundary:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    if any(len(neighbours) != 2 for neighbours in adjacency.values()):
        return None

    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        ring: list[int] = [start]
        previous, current = None, start
        visited.add(start)
        while True:
            following = next((v for v in adjacency[current] if v != previous), None)
            if following is None or following == start:
                break
            ring.append(following)
            visited.add(following)
            previous, current = current, following
        if len(ring) >= 3:
            loops.append(ring)
    return loops


def close_mesh(mesh: Any) -> Any | None:
    """Cap every open boundary so the mesh encloses a volume, or ``None`` when it cannot be closed.

    Does not call ``merge_vertices`` after fanning: that welds the new ring and turns the freshly
    closed mesh back into a non-manifold one.
    """
    import trimesh  # noqa: PLC0415

    closed = mesh.copy()
    closed.merge_vertices()
    closed.update_faces(closed.unique_faces())

    # Both exits, not just the watertight one. A mesh that arrived watertight must not leave here
    # refused, and this is the exit it actually takes: `_boundary_loops` returns None on a
    # non-manifold edge, and `merge_vertices` above is what makes it non-manifold by welding a seam
    # that was already closed. The watertight check further down never runs for these meshes, so a
    # guard placed only there is inert.
    #
    # A fallback rather than an early return, deliberately: returning the raw mesh up front would
    # also change every mesh the current path accepts, and those sit in shipped corpora whose
    # manifest hash the evaluation refuses to see move. This changes only what currently fails.
    raw_is_watertight = bool(getattr(mesh, "is_watertight", False))
    loops = _boundary_loops(closed)
    if loops is None:
        if not raw_is_watertight:
            return None
        # Fall back to the surface as it arrived and carry on: the volume, orientation and area checks
        # below still have to pass, so this rescues a mesh rather than waving one through.
        closed, loops = mesh.copy(), []
    if loops:
        vertices = closed.vertices.tolist()
        faces = closed.faces.tolist()
        for ring in loops:
            centre = np.asarray([vertices[i] for i in ring], dtype=np.float64).mean(axis=0)
            centre_index = len(vertices)
            vertices.append(centre.tolist())
            for i in range(len(ring)):
                faces.append([ring[i], ring[(i + 1) % len(ring)], centre_index])
        closed = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
    if not closed.is_watertight:
        # A mesh that arrived watertight must not leave here refused. `merge_vertices` welds
        # coincident vertices, and on a scan whose two sides meet at a seam that welding produces
        # non-manifold edges out of a surface that was already closed. Falling back to the raw mesh
        # only where the current path fails is additive: nothing that works today changes, and what
        # was silently dropped comes back.
        if raw_is_watertight:
            closed = mesh.copy()
        else:
            return None
    # Volume before orientation, and the order matters twice over. `fix_normals` orients the surface
    # against the volume it encloses, so on a closed sheet it divides by zero and answers nothing.
    # And a surface enclosing nothing has no inside for the labeller to ask about either, so
    # refusing here rather than later is the same refusal, earlier. (`is_watertight` is about edge
    # manifoldness, not winding, so checking it first is safe.)
    #
    # `errstate` because asking is the point: trimesh derives the volume through a centre-of-mass
    # division, which warns on a surface that encloses nothing. That answer may legitimately be
    # nothing, and a warning per refused mesh would be noise in every run. The refusal below is the
    # handling.
    with np.errstate(invalid="ignore", divide="ignore"):
        volume = float(closed.volume)
    if not np.isfinite(volume) or abs(volume) <= _MIN_VOLUME_M3:
        return None
    trimesh.repair.fix_normals(closed)
    return closed


@lru_cache(maxsize=512)
def load_mesh_shape(path: str) -> MeshShape | None:
    """Read one mesh file into a closed, centred `MeshShape`, or ``None`` if it is unusable.

    ``None`` for every refusal (unreadable file, boundaries that are not simple loops, a closed
    shell with no meaningful interior), because one bad object must cost that object and not the
    run. Callers count the refusals, so a silently shrinking bank stays visible.
    """
    import trimesh  # noqa: PLC0415

    try:
        raw = trimesh.load(path, force="mesh")
    except Exception:  # noqa: BLE001 (a corrupt file skips the asset, never kills the dataset)
        logger.warning("mesh unreadable, skipped: %s", path)
        return None
    if raw is None or len(getattr(raw, "faces", ())) == 0:
        logger.warning("mesh has no faces, skipped: %s", path)
        return None

    closed = close_mesh(raw)
    if closed is None:
        logger.info("mesh could not be closed (boundaries are not simple loops), skipped: %s", path)
        return None

    # The hull of the closed mesh, which is the same hull the raw one has: every vertex the capping
    # added is the centroid of a boundary ring, i.e. a convex combination of vertices already present.
    hull_volume = float(closed.convex_hull.volume)
    volume = float(closed.volume)
    if hull_volume <= 0.0 or volume / hull_volume < _MIN_VOLUME_RATIO:
        logger.info("mesh closes to a sheet (volume %.4g of hull %.4g), skipped: %s",
                    volume, hull_volume, path)
        return None

    vertices_mm = np.asarray(closed.vertices, dtype=np.float64) * _M_TO_MM
    lower, upper = vertices_mm.min(axis=0), vertices_mm.max(axis=0)
    centre = (lower + upper) / 2.0

    extent = tuple(float(v) for v in (upper - lower))
    offset = tuple(float(-v) for v in centre)
    return MeshShape(
        vertices_mm=vertices_mm - centre,
        faces=np.asarray(closed.faces, dtype=np.int64),
        face_normals=np.asarray(closed.face_normals, dtype=np.float64),
        extent_mm=extent,      # type: ignore[arg-type]
        origin_offset_mm=offset,  # type: ignore[arg-type]
        source_path=str(path),
    )
