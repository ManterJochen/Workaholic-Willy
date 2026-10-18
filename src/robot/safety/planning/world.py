"""The static geometry of the cell, in the shape the trajectory planner accepts.

The planner takes three kinds of geometry. Boxes are what an operator writes down and
the only kind a refusal can name. A mesh is how a container keeps its hollow, so a cell
can reach into a tote rather than treat it as a solid block. A distance field over a grid
carries a whole scene at once, and that one is built in `perceived.py` from what the
cameras see rather than declared here.

This module is the single place that conversion happens, and it is pure: config in, a
list of wire dictionaries out, with no client, no process and no I/O. What a cell
declares to the planner is its bench and its meshes, wherever it declares the bench
(:func:`planning_world_on`); the guard's own fixtures stay the guard's, and the cameras
find what the parts stand on (the owner's cell since 2026-10-01, every cell since the
owner's cleanup of 2026-10-17).

The units flip at this boundary: the stack is millimetres and XYZW, the planner is
metres and WXYZ.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover (typing only, keeps the safety layer free of a config import)
    from src.config.schema.robot import PlanningWorldConfig

#: Collision slots the sidecar reserves at boot, which is ``CUBOID_CACHE`` in
#: ``curobo_planner_server.py``, and the fewest a reservation asks for: the camera world's
#: boxes come on top (``reservation.PlannerReservation``).
DEFAULT_MAX_CUBOIDS = 16

#: Identity rotation in the planner's WXYZ order.
_IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)


class PlanningWorldError(ValueError):
    """The declared world cannot be registered as it stands."""


def planning_world_on(config: "PlanningWorldConfig | None") -> bool:
    """Whether a cell's planner is told its world: wherever the cell declares the bench it stands on
    (``safety.planning_world.support_plane``).

    Registering a world replaces the planner's own, boot table included, so a cell that declares no
    bench registers none and its planner keeps the world it booted with. It was the key
    ``planning_world.enabled`` until 2026-10-17, which a load refused without a bench.
    """
    return config is not None and getattr(config, "support_plane", None) is not None


def planner_cuboid(
    name: str, centre_mm: Sequence[float], dims_mm: Sequence[float], *, yaw_rad: float = 0.0,
    rotation: "Sequence[float] | None" = None,
) -> dict[str, Any]:
    """One box in the planner wire format: metres, WXYZ, base frame.

    Declared geometry is written axis-aligned, because an operator measuring a bench with
    a rule writes down a bench that is square with the robot. Geometry fitted to a point
    cloud is not: a part lies where it fell. The wire has carried a quaternion all along,
    so a turned box costs nothing here and stops an axis-aligned hull around a diagonal
    part from blocking the space beside it.

    Everything in a cell stands on something, so the turn is about base Z, by ``yaw_rad``. The one
    box that tilts is the solid of a surface the parts stand on, which follows the surface as the
    camera read it (``support_surfaces``): ``rotation`` is its whole turn, nine numbers row-major
    whose columns are the box's axes in BASE, and the quaternion is written from it, ``yaw_rad``
    unread. The sidecar takes the pose's quaternion as it is.
    """
    if rotation is not None:
        quaternion = _quaternion_wxyz(rotation)
    else:
        half = float(yaw_rad) / 2.0
        quaternion = (math.cos(half), 0.0, 0.0, math.sin(half))
    return {
        "name": name,
        "dims_m": [float(d) / 1000.0 for d in dims_mm],
        "pose": [float(c) / 1000.0 for c in centre_mm] + list(quaternion),
    }


def _quaternion_wxyz(rotation: Sequence[float]) -> tuple[float, float, float, float]:
    """The unit quaternion, WXYZ with W not negative, of a row-major 3x3 rotation (Shepperd's method).

    Refuses a matrix that is not a rotation, because a box written from one is a box nobody can say where it is."""
    m = [float(v) for v in rotation]
    if len(m) != 9 or not all(math.isfinite(v) for v in m):
        raise PlanningWorldError(f"a box's rotation is nine finite numbers, got {list(rotation)!r}")
    r = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-6:
                raise PlanningWorldError(f"a box's rotation must be orthonormal, got {m!r}")
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    if det < 0.0:
        raise PlanningWorldError(f"a box's rotation must keep its handedness, got determinant {det:.6f}")
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2])
        w, x, y, z = (r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2])
        w, x, y, z = (r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1])
        w, x, y, z = (r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    sign = -1.0 if w < 0.0 else 1.0
    return (sign * w / norm, sign * x / norm, sign * y / norm, sign * z / norm)


def merge_planner_worlds(
    base: Sequence[dict[str, Any]], extra: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The declared world first, then whatever of `extra` does not collide with it by name.

    Registration replaces the planner world rather than extending it, so anything that
    sends its own boxes deletes the bench, the bin and every fixture unless it sends those
    too. This is that rule, in one place, because it used to be written into one driver
    and absent from the other: the arm that runs in simulation could not delete its bench
    and the arm that runs on real hardware could.

    The declared side wins a name collision. A perceived box is named with a prefix no
    fixture can carry, so the only way to collide is to declare a fixture with that
    prefix, and then the thing an operator wrote down is the one that survives.
    """
    merged = [dict(box) for box in base]
    known = {box.get("name") for box in merged}
    merged.extend(dict(box) for box in extra if box.get("name") not in known)
    return merged


def build_planner_meshes(config: "PlanningWorldConfig | None") -> list[dict[str, Any]]:
    """The declared meshes in the planner wire format: a path, a pose, a scale.

    Separate from the cuboids because they travel differently. A mesh is read by the
    planning sidecar from the path in this dictionary, in its own process, rather than
    sent through the pipe: a tote is tens of thousands of triangles and the protocol is
    one JSON object per line.

    Returns ``[]`` where the cell declares no bench (:func:`planning_world_on`).

    Nothing here reaches the guards. They work on boxes, so a shape declared as a mesh is
    known to the planner and to nothing else, and the schema says so where an operator
    will read it.
    """
    if config is None or not planning_world_on(config):
        return []
    out: list[dict[str, Any]] = []
    for mesh in getattr(config, "meshes", ()) or ():
        yaw = math.radians(float(getattr(mesh, "yaw_deg", 0.0)))
        half = yaw / 2.0
        scale = float(getattr(mesh, "scale", 1.0))
        out.append(
            {
                "name": str(mesh.name),
                "file_path": str(mesh.path),
                "pose": [float(c) / 1000.0 for c in mesh.center_mm]
                + [math.cos(half), 0.0, 0.0, math.sin(half)],
                "scale": [scale, scale, scale],
            }
        )
    return out


def build_planner_cuboids(config: "PlanningWorldConfig | None") -> list[dict[str, Any]]:
    """The declared bench in the planner wire format: one box, or none.

    It returns ``[]`` where the cell declares no bench (:func:`planning_world_on`): no
    registration happens and the planner keeps the world it booted with.

    The support plane's ``height_mm`` is the top surface, so the slab is built downwards by
    ``thickness_mm``. Raising the thickness therefore never moves the surface the arm has to
    stay above, which is the one number an operator can measure at the bench. Its
    ``center_mm`` places it in the base plane, defaulting to the base, because a robot at the
    head of a bench has no floor under the half that matters.
    """
    if config is None or not planning_world_on(config):
        return []
    plane = config.support_plane
    assert plane is not None  # noqa: S101 (planning_world_on said so)
    thickness = float(plane.thickness_mm)
    extent_x, extent_y = (float(e) for e in plane.extent_mm)
    # `center_mm` is the slab position in the base plane and defaults to the base itself, so a
    # configuration written before the field existed builds the same box it always did. Read with a
    # getattr because this module takes any object shaped like the config, and the sim harness hands
    # it hand-built stand-ins.
    centre_x, centre_y = (float(c) for c in getattr(plane, "center_mm", None) or (0.0, 0.0))
    return [planner_cuboid("support_plane", (centre_x, centre_y, float(plane.height_mm) - thickness / 2.0),
                           (extent_x, extent_y, thickness))]


def describe_planner_world(cuboids: Sequence[dict[str, Any]]) -> str:
    """One line per box, in millimetres, for an operator reading a boot banner or a refusal."""
    if not cuboids:
        return "no planning world declared: the planner keeps the world it booted with"
    rows = []
    for cuboid in cuboids:
        px, py, pz = (1000.0 * v for v in cuboid["pose"][:3])
        dx, dy, dz = (1000.0 * v for v in cuboid["dims_m"])
        rows.append(
            f"  {cuboid['name']:<20} centre ({px:8.1f}, {py:8.1f}, {pz:8.1f}) mm  "
            f"size ({dx:7.1f} x {dy:7.1f} x {dz:7.1f}) mm"
        )
    return f"{len(cuboids)} planner obstacle(s):\n" + "\n".join(rows)
