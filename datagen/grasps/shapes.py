"""Exact solid geometry for the shapes the renderer authors: the reference's foundation.

Everything downstream in :mod:`datagen.grasps` is built on closed-form answers to two questions:
where does this line enter and leave this solid, and which way does the surface face there. Both are
exact for a box, a cylinder and a sphere, which is why the grasp labels are a reference rather than
a second opinion: they never touch the rendered depth, the point cloud, or any code that
:mod:`src.robot.grasping` also uses.

A fourth kind, ``mesh``. A scanned object answers the same two questions exactly, since a
ray/triangle intersection and a nearest-triangle normal are exact operations rather than
approximations, but not in closed form, and only once the surface is closed.
:mod:`datagen.assets.mesh_geometry` does the closing and owns the measurements behind it; here a
mesh solid is one more branch that every existing caller gets for free, because the verdict
machinery only ever asks through the methods below.

What changes for a mesh: the queries stay exact, but the enumeration built on them in
:mod:`datagen.grasps.labels` stops being complete. A box admits exactly three closing axes and they
can be listed; a mesh admits a continuum and it is sampled. So for meshes the label set is sound but
not complete: every label is a real grasp, while the absence of one no longer proves the absence of
a grasp.

The solid comes from the asset's ``extent_mm`` and the scene's settled pose, so it is the shape
physics left there, not the one the layout asked for. The table from kind to solid is imported from
:mod:`datagen.assets.procedural`, the same one the renderer switches on.

Conventions are the repo's: millimetres, XYZW quaternions, BASE frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datagen.assets.mesh_geometry import MeshShape
from datagen.assets.procedural import primitive_for_kind

__all__ = [
    "Solid", "TABLE_Z_MM", "quat_to_matrix", "solid_from_mesh", "solid_from_scene",
]

#: The support plane. Every family in this generator settles onto z = 0 in BASE.
TABLE_Z_MM = 0.0

_EPS = 1e-9


def quat_to_matrix(quat_xyzw) -> np.ndarray:
    """XYZW quaternion to a 3x3 body-to-world rotation, normalised first: settled poses drift."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < _EPS:
        raise ValueError("quaternion cannot be zero")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Solid:
    """One rigid primitive at a pose. ``half_extent_mm`` is half of the authored ``extent_mm``.

    For a cylinder the radius is ``half_extent_mm[0]`` and the half-height ``half_extent_mm[2]``,
    with the axis along body +z, which is how :mod:`datagen.render.isaac` authors
    ``UsdGeom.Cylinder``. For a sphere only ``half_extent_mm[0]`` is read.
    """

    kind: str                      # "box" | "cylinder" | "sphere" | "mesh"
    half_extent_mm: np.ndarray     # (3,)
    rotation: np.ndarray           # (3, 3), body to world
    centre_mm: np.ndarray          # (3,)
    instance_id: int = -1
    asset_id: str = ""
    mass_kg: float = 0.0
    #: Set for ``kind == "mesh"`` and ``None`` for everything else. Every method below takes the
    #: mesh branch first when it is present, so a primitive solid never reaches the mesh code.
    mesh: MeshShape | None = None

    # ---------------------------------------------------------------- frames

    def to_body(self, points_world: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
        return (p - self.centre_mm) @ self.rotation

    def to_world(self, points_body: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_body, dtype=np.float64))
        return p @ self.rotation.T + self.centre_mm

    def direction_to_body(self, direction_world: np.ndarray) -> np.ndarray:
        return np.asarray(direction_world, dtype=np.float64) @ self.rotation

    # ------------------------------------------------------------- geometry

    def line_span(self, origin_world, direction_world, *, margin_mm: float = 0.0):
        """Where the infinite line enters and leaves, as ``(t_enter, t_exit)``, or ``None`` if it misses.

        ``t`` is in millimetres along a unit ``direction``. ``margin_mm`` inflates the solid, which is
        how a finger's own thickness is accounted for without modelling the finger as a mesh.
        """
        o = self.to_body(origin_world)[0]
        d = self.direction_to_body(direction_world)
        d_norm = float(np.linalg.norm(d))
        if d_norm < _EPS:
            return None
        d = d / d_norm
        h = np.asarray(self.half_extent_mm, dtype=np.float64)

        if self.mesh is not None:
            return _ray_mesh(self.mesh, o, d, margin_mm)
        if self.kind == "sphere":
            radius = float(h[0]) + margin_mm
            return _ray_sphere(o, d, radius)
        if self.kind == "cylinder":
            radius, half_h = float(h[0]) + margin_mm, float(h[2]) + margin_mm
            return _ray_cylinder(o, d, radius, half_h)
        return _ray_box(o, d, h + margin_mm)

    def contains(self, point_world, *, margin_mm: float = 0.0) -> bool:
        return bool(self.contains_many(np.atleast_2d(point_world), margin_mm=margin_mm)[0])

    def contains_many(self, points_world, *, margin_mm: float = 0.0) -> np.ndarray:
        """Vectorised :meth:`contains`. The collision tests run thousands of points at a time."""
        p = self.to_body(points_world)
        h = np.asarray(self.half_extent_mm, dtype=np.float64)
        if self.mesh is not None:
            # Exactly the inflated solid, not an approximation of it: a point is inside the mesh grown
            # by `margin_mm` precisely when its signed distance to the surface is at most that margin.
            # open3d signs this positive outside, so the comparison reads the way it looks.
            return self.mesh.signed_distance_mm(p) <= margin_mm
        if self.kind == "sphere":
            return np.linalg.norm(p, axis=1) <= float(h[0]) + margin_mm
        if self.kind == "cylinder":
            return (np.hypot(p[:, 0], p[:, 1]) <= float(h[0]) + margin_mm) & (
                np.abs(p[:, 2]) <= float(h[2]) + margin_mm)
        return np.all(np.abs(p) <= h + margin_mm, axis=1)

    def normal_at(self, point_world) -> np.ndarray:
        """Outward unit surface normal at (or nearest to) ``point_world``, in world coordinates."""
        p = self.to_body(point_world)[0]
        h = np.asarray(self.half_extent_mm, dtype=np.float64)
        if self.mesh is not None:
            # The triangle's own normal, not an interpolated vertex normal: the antipodal test asks
            # what the surface does at the contact, and a smoothed normal would describe the shading.
            return self.rotation @ self.mesh.face_normals[self.mesh.closest_face(p)]
        if self.kind == "sphere":
            n = p
        elif self.kind == "cylinder":
            radial = np.array([p[0], p[1], 0.0])
            radial_norm = float(np.linalg.norm(radial))
            # Whichever surface the point is closer to: the barrel or one of the two caps.
            if float(h[2]) - abs(p[2]) < float(h[0]) - radial_norm or radial_norm < _EPS:
                n = np.array([0.0, 0.0, np.sign(p[2]) or 1.0])
            else:
                n = radial
        else:
            # The face whose plane the point is nearest, measured in units of that half-extent so a
            # thin plate's broad face wins over its edge.
            slack = np.abs(np.abs(p) - h)
            axis = int(np.argmin(slack))
            n = np.zeros(3)
            n[axis] = np.sign(p[axis]) or 1.0
        norm = float(np.linalg.norm(n))
        if norm < _EPS:
            return np.array([0.0, 0.0, 1.0])
        return self.rotation @ (n / norm)

    def support_width_mm(self, direction_world) -> float:
        """Total extent along ``direction``: what a jaw closing along it would have to span."""
        d = self.direction_to_body(direction_world)
        d_norm = float(np.linalg.norm(d))
        if d_norm < _EPS:
            return 0.0
        d = d / d_norm
        h = np.asarray(self.half_extent_mm, dtype=np.float64)
        if self.mesh is not None:
            projected = self.mesh.vertices_mm @ d
            return float(projected.max() - projected.min())
        if self.kind == "sphere":
            return 2.0 * float(h[0])
        if self.kind == "cylinder":
            axial = abs(float(d[2]))
            return 2.0 * (float(h[2]) * axial + float(h[0]) * float(np.sqrt(max(0.0, 1.0 - axial ** 2))))
        return 2.0 * float(np.abs(d) @ h)

    def top_z_mm(self) -> float:
        """Highest point of the solid in BASE: the depth a top-down approach starts clear of."""
        if self.mesh is not None:
            # Not centre + half the support width. That identity holds only for a shape symmetric about
            # its centre, which every primitive here is and a scanned object is not: a mug's centre of
            # bounds sits in its hollow, and half the z-extent above it would miss the rim.
            return float(self.centre_mm[2] + (self.mesh.vertices_mm @ self.rotation[2]).max())
        return float(self.centre_mm[2]) + 0.5 * self.support_width_mm(np.array([0.0, 0.0, 1.0]))

    def bounding_radius_mm(self) -> float:
        if self.mesh is not None:
            return self.mesh.bounding_radius_mm
        h = np.asarray(self.half_extent_mm, dtype=np.float64)
        if self.kind == "sphere":
            return float(h[0])
        if self.kind == "cylinder":
            return float(np.hypot(h[0], h[2]))
        return float(np.linalg.norm(h))


def solid_from_scene(
    asset_kind: str, extent_mm, position_mm, orientation_xyzw, *,
    instance_id: int = -1, asset_id: str = "", mass_kg: float = 0.0,
) -> Solid:
    """Build the solid the renderer authored for ``asset_kind`` at a settled pose."""
    extent = np.asarray(extent_mm, dtype=np.float64)
    return Solid(
        kind=primitive_for_kind(asset_kind),
        half_extent_mm=extent / 2.0,
        rotation=quat_to_matrix(orientation_xyzw),
        centre_mm=np.asarray(position_mm, dtype=np.float64),
        instance_id=instance_id,
        asset_id=asset_id,
        mass_kg=float(mass_kg),
    )


def solid_from_mesh(
    shape: MeshShape, position_mm, orientation_xyzw, *,
    instance_id: int = -1, asset_id: str = "", mass_kg: float = 0.0,
) -> Solid:
    """A real mesh at a settled pose, as one solid the rest of the reference cannot tell apart.

    ``half_extent_mm`` carries the axis-aligned body extent, which is what the anchor spreading and
    the broad-phase reach read; every question about the actual surface goes through the mesh.
    """
    return Solid(
        kind="mesh",
        half_extent_mm=shape.half_extent_mm,
        rotation=quat_to_matrix(orientation_xyzw),
        centre_mm=np.asarray(position_mm, dtype=np.float64),
        instance_id=instance_id,
        asset_id=asset_id,
        mass_kg=float(mass_kg),
        mesh=shape,
    )


def solids_from_parts(
    parts, position_mm, orientation_xyzw, *,
    instance_id: int = -1, asset_id: str = "", mass_kg: float = 0.0,
) -> list[Solid]:
    """Every solid of a composite object at a settled pose, the body first.

    A part is authored in the object's own frame, so its world pose is the object's rotation applied
    to its local offset and yaw. This mirrors what the renderer does, one rigid body carrying T*R
    with each part a local transform below it, because the labeller must intersect the shape that
    was rendered, not a nearby one.

    Mass is split by volume rather than shared: the analytic verdict reads it per solid, and giving
    every part the whole object's mass would make each of them independently too heavy to lift.
    """
    R = quat_to_matrix(orientation_xyzw)
    centre = np.asarray(position_mm, dtype=np.float64)
    volumes = [float(np.prod(np.asarray(p.extent_mm, dtype=np.float64))) for p in parts]
    total = float(sum(volumes)) or 1.0

    solids: list[Solid] = []
    for part, volume in zip(parts, volumes):
        local_R = np.eye(3)
        yaw = float(getattr(part, "yaw_deg", 0.0))
        if abs(yaw) > 1e-9:
            angle = np.radians(yaw)
            cos, sin = float(np.cos(angle)), float(np.sin(angle))
            local_R = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        offset = np.asarray(part.offset_mm, dtype=np.float64)
        solids.append(Solid(
            kind=part.primitive,
            half_extent_mm=np.asarray(part.extent_mm, dtype=np.float64) / 2.0,
            rotation=R @ local_R,
            centre_mm=centre + R @ offset,
            instance_id=instance_id,
            asset_id=asset_id,
            mass_kg=mass_kg * (volume / total),
        ))
    return solids


def wall_solid(centre_mm, half_extents_mm, *, name: str = "wall") -> Solid:
    """A bin wall: an axis-aligned box, so the same machinery covers it."""
    return Solid(
        kind="box",
        half_extent_mm=np.asarray(half_extents_mm, dtype=np.float64),
        rotation=np.eye(3),
        centre_mm=np.asarray(centre_mm, dtype=np.float64),
        asset_id=name,
    )


# --------------------------------------------------------------------- rays


def _ray_mesh(shape: MeshShape, o: np.ndarray, d: np.ndarray, margin_mm: float):
    """Outermost entry and exit of an infinite line through a closed mesh.

    Two rays rather than a march. Casting forward from a point guaranteed to be outside the mesh
    gives the first surface the line meets; casting backward from the mirror point gives the last.
    That is the span a parallel jaw sees, because a jaw closing along the axis touches the outermost
    surfaces first: for a concave object the hollow between them is the object's business, not the
    jaw's, and the primitive branches return the same thing.
    """
    if abs(margin_mm) > _EPS:
        # No caller passes a margin, and it is refused rather than approximated: inflating a mesh
        # is a Minkowski sum, and faking it by pushing the hit outward along the ray would report a
        # span that no surface has.
        raise NotImplementedError(
            "line_span with a margin is not defined for a mesh solid: inflating a mesh is a "
            "Minkowski sum, not a ray offset. Use contains_many(margin_mm=...), which is exact."
        )
    reach = shape.bounding_radius_mm + float(np.linalg.norm(o)) + 1.0
    starts = np.array([o - reach * d, o + reach * d])
    directions = np.array([d, -d])
    hits = shape.first_hit_mm(starts, directions)
    if not np.isfinite(hits).all():
        return None
    # Quantised, and this is load-bearing rather than cosmetic. `_pad_contacts` probes nine pad
    # positions and keeps whichever reports the extreme span; on a flat face all nine reach the same
    # plane, and the winner then decides where the contact is, which is what the finger samples, the
    # table check and the collision check are built from. Embree answers in float32, so nine
    # identical surfaces come back differing by about a micron, and that micron can move the contact
    # by up to 38 mm along the pad, far enough to pass a table check the closed-form path rejects.
    #
    # Rounding to a micrometre makes a float32 engine's spans deterministic in themselves: two pad
    # positions on one plane come back bitwise equal rather than a micron apart. `_pad_contacts`
    # also compares with a tolerance, and the two are complementary rather than redundant: that
    # tolerance decides who wins a tie, this decides whether there is a tie to win, and only the
    # second survives being written to a file.
    return (round(float(-reach + hits[0]), 3), round(float(reach - hits[1]), 3))


def _ray_box(o: np.ndarray, d: np.ndarray, h: np.ndarray):
    """Slab method. ``o``/``d`` are already in the box's own frame."""
    t_enter, t_exit = -np.inf, np.inf
    for axis in range(3):
        if abs(d[axis]) < _EPS:
            if abs(o[axis]) > h[axis]:
                return None
            continue
        t1 = (-h[axis] - o[axis]) / d[axis]
        t2 = (h[axis] - o[axis]) / d[axis]
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        t_enter, t_exit = max(t_enter, lo), min(t_exit, hi)
        if t_enter > t_exit:
            return None
    return (float(t_enter), float(t_exit))


def _ray_sphere(o: np.ndarray, d: np.ndarray, radius: float):
    b = float(o @ d)
    c = float(o @ o) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return None
    root = float(np.sqrt(disc))
    return (-b - root, -b + root)


def _ray_cylinder(o: np.ndarray, d: np.ndarray, radius: float, half_h: float):
    """Infinite circular barrel about body +z, clipped by the two end caps."""
    a = float(d[0] * d[0] + d[1] * d[1])
    if a < _EPS:                                   # parallel to the axis: only the caps can bound it
        if float(o[0] ** 2 + o[1] ** 2) > radius * radius:
            return None
        t_enter, t_exit = -np.inf, np.inf
    else:
        b = float(o[0] * d[0] + o[1] * d[1])
        c = float(o[0] ** 2 + o[1] ** 2) - radius * radius
        disc = b * b - a * c
        if disc < 0.0:
            return None
        root = float(np.sqrt(disc))
        t_enter, t_exit = (-b - root) / a, (-b + root) / a

    if abs(d[2]) < _EPS:
        if abs(o[2]) > half_h:
            return None
    else:
        t1 = (-half_h - o[2]) / d[2]
        t2 = (half_h - o[2]) / d[2]
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        t_enter, t_exit = max(t_enter, lo), min(t_exit, hi)
    if t_enter > t_exit:
        return None
    return (float(t_enter), float(t_exit))
