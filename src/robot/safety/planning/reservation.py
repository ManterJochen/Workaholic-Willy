"""What the planner allocates when it starts, decided once from the cell's config.

cuRobo allocates its collision storage exactly once, when the sidecar starts: so many box slots, so
many mesh slots, a voxel grid of a fixed size, and spheres for a carried part. Anything sent later
that does not fit is refused, so the numbers have to be settled before the first plan and they have
to describe the world the cell will actually send. A client started with a fixed 16 boxes, no mesh and
no grid refuses a tote declared as a mesh, or a live scene, at the first motion, because the planner
has nowhere to put either.

The counts are derived rather than configured: as many boxes as the cell declares plus the perceived
boxes it allows, never fewer than 16; one mesh slot per declared mesh; a grid only where a live scene
asks for one; payload spheres only where a carried part's length is declared, with the planning world
on or off. A profile that declares none of these starts the sidecar with 16 boxes and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.robot.safety.planning.perceived import WorldBuildLimits, voxel_grid_extent
from src.robot.safety.planning.world import DEFAULT_MAX_CUBOIDS, build_planner_cuboids

if TYPE_CHECKING:  # pragma: no cover (typing only, keeps the safety layer free of a config import)
    from src.config.schema.robot import RobotConfig

__all__ = ["PlannerReservation", "planner_world_limits"]

#: Bytes one grid value takes on the wire and in the planner's storage.
_FLOAT16_BYTES = 2

#: Only bounds the count below: the declared world is counted, not capped, because the reservation
#: is what the cap is read from.
_COUNT_EVERYTHING = 1_000_000


def planner_world_limits(robot_cfg: "RobotConfig") -> "WorldBuildLimits | None":
    """The workspace box, the bench and its clearance a perceived world is built against.

    The box is the one the workspace guard reads, and it bounds the TCP alone. It sets the grid a
    distance field is cut to, which the planner reserves when it starts. It is not where an obstacle
    stops mattering: the links, the hand and a wrist camera swing past it, so the live world also
    keeps what the robot's own body can reach (``SelfEnvelope.reach``), which it knows at every
    refresh and this config does not. The support plane comes from the planner block, because a
    bench registered as a hundred small obstacles fills the planner slots and duplicates a box the
    operator already wrote down. The plane's height is the slab's top, so a slab sunk below the bench
    needs ``perceived.plane_clearance_mm`` raised by the sink, or the whole bench comes back as an
    obstacle. The raised clearance is the bench band's alone: a declared fixture keeps its own band
    (``perceived.DECLARED_SURFACE_MM``). `None` when the cell declares no plane, which is also a cell
    that cannot have a perceived world.
    """
    world_cfg = getattr(robot_cfg.safety, "planning_world", None)
    plane = getattr(world_cfg, "support_plane", None) if world_cfg is not None else None
    if plane is None:
        return None
    workspace = robot_cfg.workspace_limits
    perceived = getattr(world_cfg, "perceived", None)
    return WorldBuildLimits(
        x_mm=(float(workspace.x_min), float(workspace.x_max)),
        y_mm=(float(workspace.y_min), float(workspace.y_max)),
        z_mm=(float(workspace.z_min), float(workspace.z_max)),
        support_plane_top_mm=float(plane.height_mm),
        plane_clearance_mm=float(getattr(perceived, "plane_clearance_mm", 5.0)),
    )


@dataclass(frozen=True, slots=True)
class PlannerReservation:
    """What one planner sidecar allocates when it starts.

    ``voxel_grid`` is ``x,y,z,voxel`` in metres to four decimals, or empty for no grid, in exactly
    the form the sidecar reads, and exactly the grid :func:`voxel_grid_extent` cuts a field into.
    """

    cuboid_slots: int
    mesh_slots: int
    voxel_grid: str
    sphere_slots: int

    @classmethod
    def from_parts(
        cls,
        *,
        declared_cuboids: int,
        perceived_boxes: int,
        meshes: int,
        grid_extent: "tuple[tuple[float, float, float], float] | None",
        sphere_slots: int,
    ) -> "PlannerReservation":
        """The reservation for a world of these sizes: the counts a cell will send, not guesses."""
        grid = "" if grid_extent is None else ",".join(
            f"{v / 1000.0:.4f}" for v in (*grid_extent[0], grid_extent[1])
        )
        return cls(
            cuboid_slots=max(DEFAULT_MAX_CUBOIDS, int(declared_cuboids) + int(perceived_boxes)),
            mesh_slots=max(0, int(meshes)),
            voxel_grid=grid,
            sphere_slots=max(0, int(sphere_slots)),
        )

    @classmethod
    def from_config(cls, *, robot_cfg: "RobotConfig") -> "PlannerReservation":
        """The reservation this cell's config asks for.

        A disabled planning world asks for the defaults and the payload spheres a declared carried
        part asks for.

        Raises
        ------
        PlanningWorldError
            When the declared world names an obstacle twice, the same refusal registration would meet.
        """
        world_cfg = getattr(robot_cfg.safety, "planning_world", None)
        # Payload spheres from one rule, world on or off: an enabled payload with a declared length.
        # Every start reads this number, the planner reserving them and the evidence lookup naming
        # them, so the two agree.
        spheres = 0
        payload = getattr(world_cfg, "payload", None) if world_cfg is not None else None
        if (payload is not None and bool(getattr(payload, "enabled", False))
                and payload.length_mm is not None):
            spheres = int(payload.sphere_slots)
        if world_cfg is None or not bool(getattr(world_cfg, "enabled", False)):
            return cls.from_parts(
                declared_cuboids=0, perceived_boxes=0, meshes=0, grid_extent=None, sphere_slots=spheres
            )
        declared = len(
            build_planner_cuboids(
                world_cfg, robot_cfg.safety.self_collision.fixtures, max_cuboids=_COUNT_EVERYTHING
            )
        )
        perceived_boxes = 0
        extent: "tuple[tuple[float, float, float], float] | None" = None
        perceived = getattr(world_cfg, "perceived", None)
        if perceived is not None and bool(getattr(perceived, "enabled", False)):
            perceived_boxes = int(perceived.max_boxes)
            limits = planner_world_limits(robot_cfg)
            if limits is not None:
                extent = voxel_grid_extent(limits, float(perceived.voxel_field_mm))
        return cls.from_parts(
            declared_cuboids=declared,
            perceived_boxes=perceived_boxes,
            meshes=len(getattr(world_cfg, "meshes", ()) or ()),
            grid_extent=extent,
            sphere_slots=spheres,
        )

    @property
    def grid_shape(self) -> "tuple[int, int, int] | None":
        """Voxels along x, y and z, or `None` for no grid."""
        if not self.voxel_grid:
            return None
        x, y, z, voxel = (float(v) for v in self.voxel_grid.split(","))
        return int(round(x / voxel)), int(round(y / voxel)), int(round(z / voxel))

    @property
    def grid_cells(self) -> int:
        shape = self.grid_shape
        return 0 if shape is None else shape[0] * shape[1] * shape[2]

    @property
    def grid_bytes(self) -> int:
        """What the grid costs as float16, which is how the sidecar stores it."""
        return self.grid_cells * _FLOAT16_BYTES

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        parts = [f"{self.cuboid_slots} box slot(s)", f"{self.mesh_slots} mesh slot(s)"]
        if self.voxel_grid:
            x, y, z, voxel = (float(v) * 1000.0 for v in self.voxel_grid.split(","))
            parts.append(
                f"a live scene grid of {x:.0f} x {y:.0f} x {z:.0f} mm at {voxel:.0f} mm "
                f"({self.grid_cells} cells, {self.grid_bytes / 1e6:.2f} MB as float16)"
            )
        else:
            parts.append("no live scene grid")
        parts.append(
            f"{self.sphere_slots} payload sphere slot(s)" if self.sphere_slots else "no payload spheres"
        )
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {
            "cuboid_slots": self.cuboid_slots,
            "mesh_slots": self.mesh_slots,
            "voxel_grid": self.voxel_grid,
            "sphere_slots": self.sphere_slots,
            "grid_cells": self.grid_cells,
            "grid_bytes": self.grid_bytes,
        }
