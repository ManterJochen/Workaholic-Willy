"""The static geometry of the cell, in the shape the trajectory planner accepts.

The planner takes three kinds of geometry. Boxes are what an operator writes down and
the only kind a refusal can name. A mesh is how a container keeps its hollow, so a cell
can reach into a tote rather than treat it as a solid block. A distance field over a grid
carries a whole scene at once, and that one is built in `perceived.py` from what the
cameras see rather than declared here.

This module is the single place that conversion happens, and it is pure: config in, a
list of wire dictionaries out, with no client, no process and no I/O. That matters
because the two things it protects against are both silent. A duplicate name collapses
two obstacles into one, because the planner side keys its world by name. And more boxes
than the sidecar reserved slots for is a registration that partly succeeds, which reads
as success at every layer above.

The units flip at this boundary: the stack is millimetres and XYZW, the planner is
metres and WXYZ.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover (typing only, keeps the safety layer free of a config import)
    from src.config.schema.robot import FixtureBoxConfig, PlanningWorldConfig

#: Collision slots the sidecar reserves at boot, which is ``CUBOID_CACHE`` in
#: ``curobo_planner_server.py``. A world larger than this cannot be registered in full,
#: and a partial registration is exactly the failure this module makes loud.
DEFAULT_MAX_CUBOIDS = 16

#: Identity rotation in the planner's WXYZ order.
_IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)


class PlanningWorldError(ValueError):
    """The declared world cannot be registered as it stands."""


def planner_cuboid(
    name: str, centre_mm: Sequence[float], dims_mm: Sequence[float], *, yaw_rad: float = 0.0
) -> dict[str, Any]:
    """One box in the planner wire format: metres, WXYZ, base frame.

    Declared geometry is written axis-aligned, because an operator measuring a bench with
    a rule writes down a bench that is square with the robot. Geometry fitted to a point
    cloud is not: a part lies where it fell. The wire has carried a quaternion all along,
    so a turned box costs nothing here and stops an axis-aligned hull around a diagonal
    part from blocking the space beside it.

    The turn is about base Z only. Everything in a cell stands on something.
    """
    half = float(yaw_rad) / 2.0
    rotation = (math.cos(half), 0.0, 0.0, math.sin(half))
    return {
        "name": name,
        "dims_m": [float(d) / 1000.0 for d in dims_mm],
        "pose": [float(c) / 1000.0 for c in centre_mm] + list(rotation),
    }


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

    Returns ``[]`` where the block is absent or disabled, which leaves behaviour
    unchanged.

    Nothing here reaches the guards. They work on boxes, so a shape declared as a mesh is
    known to the planner and to nothing else, and the schema says so where an operator
    will read it.
    """
    if config is None or not bool(getattr(config, "enabled", False)):
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


def _cuboid(name: str, centre_mm: Sequence[float], dims_mm: Sequence[float]) -> dict[str, Any]:
    return planner_cuboid(name, centre_mm, dims_mm)


def build_planner_cuboids(
    config: "PlanningWorldConfig | None",
    fixtures: "Sequence[FixtureBoxConfig] | None" = None,
    *,
    max_cuboids: int = DEFAULT_MAX_CUBOIDS,
) -> list[dict[str, Any]]:
    """Turn the declared cell geometry into the planner wire format.

    It returns ``[]`` where the block is absent or disabled, which leaves behaviour
    unchanged: no registration happens and the planner keeps the world it booted with.

    The support plane is emitted first, and its ``height_mm`` is the top surface, so the
    slab is built downwards by ``thickness_mm``. Raising the thickness therefore never
    moves the surface the arm has to stay above, which is the one number an operator can
    measure at the bench. Its ``center_mm`` places it in the base plane, defaulting to the
    base, because a robot at the head of a bench has no floor under the half that matters.

    A fixture with a zero extent on any axis is skipped. The schema documents an
    all-zero fixture as the way to disable one without deleting it, and a zero-volume
    box is nothing to a planner.

    Raises
    ------
    PlanningWorldError
        If two boxes would carry the same name, or the world does not fit the reserved
        slots.
    """
    if config is None or not bool(getattr(config, "enabled", False)):
        return []

    cuboids: list[dict[str, Any]] = []

    plane = getattr(config, "support_plane", None)
    if plane is not None:
        thickness = float(plane.thickness_mm)
        extent_x, extent_y = (float(e) for e in plane.extent_mm)
        # `center_mm` is the slab position in the base plane and defaults to the base
        # itself, so a configuration written before the field existed builds the same box
        # it always did. Read with a getattr because this module takes any object shaped
        # like the config, and the sim harness hands it hand-built stand-ins.
        centre_x, centre_y = (
            float(c) for c in getattr(plane, "center_mm", None) or (0.0, 0.0)
        )
        cuboids.append(
            _cuboid(
                "support_plane",
                (centre_x, centre_y, float(plane.height_mm) - thickness / 2.0),
                (extent_x, extent_y, thickness),
            )
        )

    if bool(getattr(config, "include_fixtures", True)):
        for fixture in fixtures or ():
            half = tuple(float(h) for h in fixture.half_extents_mm)
            if min(half) <= 0.0:
                continue
            cuboids.append(
                _cuboid(
                    str(fixture.name),
                    tuple(float(c) for c in fixture.center_mm),
                    tuple(2.0 * h for h in half),
                )
            )

    names = [c["name"] for c in cuboids]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise PlanningWorldError(
            f"the planning world declares {duplicates} more than once. The planner keys its world by "
            "name, so duplicates collapse into one obstacle and the others are silently absent. Give "
            "each fixture its own `name`."
        )
    if len(cuboids) > max_cuboids:
        raise PlanningWorldError(
            f"the planning world has {len(cuboids)} boxes but the planner reserves {max_cuboids} "
            "collision slots, so it cannot be registered in full. Merge adjacent boxes, drop the ones "
            "the arm cannot reach, or raise the sidecar's reserved count."
        )
    return cuboids


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
