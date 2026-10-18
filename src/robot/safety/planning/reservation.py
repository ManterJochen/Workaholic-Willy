"""What the planner allocates when it starts, decided once from the cell's config.

cuRobo allocates its collision storage exactly once, when the sidecar starts: so many box slots, so
many mesh slots and a voxel grid of a fixed size. Anything sent later that does not fit is refused, so
the numbers have to be settled before the first plan and they have to describe the world the cell will
actually send. A client started with a fixed 16 boxes, no mesh and no grid refuses a tote declared as a
mesh, or a live scene, at the first motion, because the planner has nowhere to put either.

The counts are derived rather than configured: wherever a cell declares its bench, the bench plus the
boxes every cell's camera world may send, never fewer than 16, one mesh slot per declared mesh, and the
camera world's distance field with its fine grid about a goal. A profile that declares no bench starts
the sidecar with 16 boxes and nothing else. No carried part is modelled since 2026-10-17, so no spheres
are reserved for one.

Two switches ride along, because they too are fixed when the sidecar starts and the reservation is
what every driver hands its client before that: the time-optimal passes of a joint plan and the
planning model. Each defaults to how the sidecar ran before them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.robot.safety.planning.fine_world import FINE_REACH_MM
from src.robot.safety.planning.perceived import (
    CAMERA_BENCH_BAND_MM,
    CAMERA_FIELD_VOXEL_MM,
    CAMERA_MAX_BOXES,
    WorldBuildLimits,
    near_grid_extent,
    voxel_grid_extent,
)
from src.robot.safety.planning.world import DEFAULT_MAX_CUBOIDS, build_planner_cuboids, planning_world_on

if TYPE_CHECKING:  # pragma: no cover (typing only, keeps the safety layer free of a config import)
    from src.config.schema.robot import RobotConfig

__all__ = ["PlannerReservation", "planner_world_limits"]

#: Bytes one grid value takes on the wire and in the planner's storage.
_FLOAT16_BYTES = 2


def planner_world_limits(robot_cfg: "RobotConfig") -> "WorldBuildLimits | None":
    """The workspace box, the bench and its clearance a perceived world is built against.

    The box is the one the workspace guard reads, and it bounds the TCP alone. It sets the grid a
    distance field is cut to, which the planner reserves when it starts. It is not where an obstacle
    stops mattering: the links, the hand and a wrist camera swing past it, so the live world also
    keeps what the robot's own body can reach (``SelfEnvelope.reach``), which it knows at every
    refresh and this config does not. The support plane comes from the planner block, because a
    bench registered as a hundred small obstacles fills the planner slots and duplicates a box the
    operator already wrote down. The plane's height is the slab's top, and the bench band stands
    ``perceived.CAMERA_BENCH_BAND_MM`` over it, so the slab's top is the bench: a slab sunk below the
    bench brings the whole bench back as an obstacle. A declared fixture keeps its own band
    (``perceived.DECLARED_SURFACE_MM``). `None` when the cell declares no plane, which is also a cell
    that cannot have a perceived world.
    """
    world_cfg = getattr(robot_cfg.safety, "planning_world", None)
    plane = getattr(world_cfg, "support_plane", None) if world_cfg is not None else None
    if plane is None:
        return None
    workspace = robot_cfg.motion.workspace
    return WorldBuildLimits(
        x_mm=(float(workspace.x_min), float(workspace.x_max)),
        y_mm=(float(workspace.y_min), float(workspace.y_max)),
        z_mm=(float(workspace.z_min), float(workspace.z_max)),
        support_plane_top_mm=float(plane.height_mm),
        plane_clearance_mm=CAMERA_BENCH_BAND_MM,
    )


@dataclass(frozen=True, slots=True)
class PlannerReservation:
    """What one planner sidecar allocates when it starts, and how it is started.

    ``voxel_grid`` is ``x,y,z,voxel`` in metres to four decimals, or empty for no grid, in exactly
    the form the sidecar reads, and exactly the grid :func:`voxel_grid_extent` cuts a field into;
    after a ``;`` the largest fine grid about a goal the field sends beside it (:func:`near_grid_extent`).
    How a joint plan runs and the robot cuRobo plans with are every cell's since 2026-10-17
    (``_curobo_cspace.JOINT_FINETUNE_PASSES``, ``planning_model.LEAN``), so nothing more is reserved.
    """

    cuboid_slots: int
    mesh_slots: int
    voxel_grid: str

    @classmethod
    def from_parts(
        cls,
        *,
        declared_cuboids: int,
        perceived_boxes: int,
        meshes: int,
        grid_extent: "tuple[tuple[float, float, float], float] | None",
        near_extent: "tuple[tuple[float, float, float], float] | None" = None,
    ) -> "PlannerReservation":
        """The reservation for a world of these sizes: the counts a cell will send, not guesses."""
        # The fine grid about a goal goes beside a field's own grid, never without it.
        extents = [] if grid_extent is None else [extent for extent in (grid_extent, near_extent) if extent is not None]
        grid = ";".join(",".join(f"{v / 1000.0:.4f}" for v in (*extent[0], extent[1])) for extent in extents)
        return cls(
            cuboid_slots=max(DEFAULT_MAX_CUBOIDS, int(declared_cuboids) + int(perceived_boxes)),
            mesh_slots=max(0, int(meshes)),
            voxel_grid=grid,
        )

    @classmethod
    def from_config(cls, *, robot_cfg: "RobotConfig") -> "PlannerReservation":
        """The reservation this cell's config asks for: the defaults where it declares no bench.

        Returns:
            PlannerReservation: The slots and the grid the cell's planner starts with.
        """
        world_cfg = getattr(robot_cfg.safety, "planning_world", None)
        if not planning_world_on(world_cfg):
            return cls.from_parts(declared_cuboids=0, perceived_boxes=0, meshes=0, grid_extent=None)
        declared = len(build_planner_cuboids(world_cfg))
        # The camera world every cell builds (``camera_world_wiring``): its boxes, its distance field over the
        # workspace, and the field's fine grid about every goal, where the fine world is built.
        extent: "tuple[tuple[float, float, float], float] | None" = None
        near: "tuple[tuple[float, float, float], float] | None" = None
        limits = planner_world_limits(robot_cfg)
        if limits is not None:
            extent = voxel_grid_extent(limits, CAMERA_FIELD_VOXEL_MM)
            near = near_grid_extent(limits, CAMERA_FIELD_VOXEL_MM, FINE_REACH_MM)
        return cls.from_parts(
            declared_cuboids=declared,
            perceived_boxes=CAMERA_MAX_BOXES,
            meshes=len(getattr(world_cfg, "meshes", ()) or ()),
            grid_extent=extent,
            near_extent=near,
        )

    @property
    def _grids_mm(self) -> "list[tuple[float, float, float, float]]":
        """Each grid as ``x, y, z, voxel`` in millimetres: the field's own, then its fine grid about the goal."""
        return [tuple(float(v) * 1000.0 for v in part.split(",")) for part in self.voxel_grid.split(";") if part]  # type: ignore[misc]

    @property
    def grid_shape(self) -> "tuple[int, int, int] | None":
        """Voxels along x, y and z of the field's own grid, or `None` for no grid."""
        if not self.voxel_grid:
            return None
        x, y, z, voxel = self._grids_mm[0]
        return int(round(x / voxel)), int(round(y / voxel)), int(round(z / voxel))

    @property
    def grid_cells(self) -> int:
        """The values the sidecar's grid storage holds: a layer a grid, each as large as the largest grid."""
        cells = [round(x / voxel) * round(y / voxel) * round(z / voxel) for x, y, z, voxel in self._grids_mm]
        return len(cells) * max(cells, default=0)

    @property
    def grid_bytes(self) -> int:
        """What the grid costs as float16, which is how the sidecar stores it."""
        return self.grid_cells * _FLOAT16_BYTES

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        parts = [f"{self.cuboid_slots} box slot(s)", f"{self.mesh_slots} mesh slot(s)"]
        if self.voxel_grid:
            grids = [f"{x:.0f} x {y:.0f} x {z:.0f} mm at {voxel:.0f} mm" for x, y, z, voxel in self._grids_mm]
            parts.append(
                f"a live scene grid of {' and one about the goal of '.join(grids)} "
                f"({self.grid_cells} cells, {self.grid_bytes / 1e6:.2f} MB as float16)"
            )
        else:
            parts.append("no live scene grid")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {
            "cuboid_slots": self.cuboid_slots,
            "mesh_slots": self.mesh_slots,
            "voxel_grid": self.voxel_grid,
            "grid_cells": self.grid_cells,
            "grid_bytes": self.grid_bytes,
        }
