"""The static geometry of the cell, in the shape the trajectory planner accepts.

The planner collision world is axis-aligned boxes and nothing else: no mesh, no point
cloud, no voxel channel. So everything the arm must not drive through arrives here as a
box, and anything that is not box-shaped is enclosed by one.

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


def _cuboid(name: str, centre_mm: Sequence[float], dims_mm: Sequence[float]) -> dict[str, Any]:
    return {
        "name": name,
        "dims_m": [float(d) / 1000.0 for d in dims_mm],
        "pose": [float(c) / 1000.0 for c in centre_mm] + list(_IDENTITY_WXYZ),
    }


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
    measure at the bench.

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
        cuboids.append(
            _cuboid(
                "support_plane",
                (0.0, 0.0, float(plane.height_mm) - thickness / 2.0),
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
