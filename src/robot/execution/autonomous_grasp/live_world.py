"""Assemble the thing that tells the planner what the cell looks like right now.

Everything this needs already exists and lives in three different places on purpose: a camera the
composition root opened, a CAMERA to BASE transform the calibration produced, and the declared world
the config states. Nothing below may reach across those, because the dependency stack runs downward
and a driver that imported perception would invert the one edge the layering rests on. So the
assembly happens here, in the layer that is allowed to know about all three, and what the arm gets
handed is a thing it can ask one question of.

Two adapters and a builder, and the adapters are the point:

  * A camera rig speaks colour and depth frames. The world source wants depth and a capture time,
    and wants them without a detector or a segmenter running, because a world refresh sits in front
    of every motion and a perception cycle costs hundreds of milliseconds where a depth grab costs
    tens.
  * A frame resolver answers with a transform for a frame. The world source wants a matrix.

What this deliberately does NOT do is decide anything. Whether a cell gets a live world at all is a
config question, and the answer is in one place: `planning_world.enabled` with a `perceived` block
that does not turn itself off.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from src.robot.safety.planning.live_world import (
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
)
from src.robot.safety.planning.perceived import (
    WorldBuildLimits,
    WorldBuildTuning,
    voxel_grid_extent,
)
from src.robot.safety.planning.world import build_planner_cuboids, build_planner_meshes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.schema.robot import RobotConfig

__all__ = [
    "LiveWorldUnavailable",
    "RigDepthSource",
    "build_live_planner_world",
    "planner_voxel_reservation",
    "static_camera_to_base_mm",
]


class LiveWorldUnavailable(ValueError):
    """This cell asked for a live planner world and cannot have one as it stands."""


class RigDepthSource:
    """A camera rig, asked for depth and nothing else.

    Deliberately narrower than the perception source the pick loop drives. That one runs a detector
    and a segmenter to produce named objects, which is the expensive part of a cell and exactly what
    a world refresh does not need: a planner wants to know where geometry is, not what it is called.

    Returns `None` rather than raising, for every reason a camera can fail to answer. The caller is
    about to decide whether a motion may happen, and a decision needs a verdict rather than an
    exception: the world source turns a `None` into a refusal that names the camera.
    """

    __slots__ = ("_handle", "_name")

    def __init__(self, handle: Any, *, name: str = "") -> None:
        self._handle = handle
        self._name = name or getattr(handle, "rig_id", "camera")

    def __repr__(self) -> str:
        return f"RigDepthSource({self._name!r})"

    @property
    def name(self) -> str:
        return self._name

    def grab_surface_depth(self) -> "DepthSnapshot | None":
        """One depth reading with the time its shutter opened, or `None` when this rig cannot answer.

        The stamp is taken before the grab rather than after. Age is measured to decide whether the
        world can still be planned against, and a stamp taken after a slow read reports a frame as
        fresher than it is, which is the one direction that matters.
        """
        stamped = time.time()
        try:
            frame = self._handle.grab()
            intrinsics = self._handle.get_intrinsics()
        except Exception:  # noqa: BLE001 - a camera that cannot answer is a refusal, not a crash
            return None

        depth = getattr(frame, "depth", None)
        if depth is None or intrinsics is None:
            # A stereo rig has no depth of its own and an uncalibrated one has no matrix. Both are
            # states a cell can be in, and neither is something to guess a way around.
            return None
        array = np.asarray(depth, dtype=np.float64)
        if array.ndim != 2 or array.size == 0:
            return None
        return DepthSnapshot(
            depth_mm=array,
            intrinsics=np.asarray(intrinsics, dtype=np.float64),
            timestamp=stamped,
        )


def planner_voxel_reservation(robot_cfg: "RobotConfig") -> str:
    """The grid the planner must allocate at boot, as ``x,y,z,voxel`` in metres, or empty.

    The planner allocates its voxel storage when it starts and never again, so this has to be
    settled before the sidecar spawns and then not change. The same numbers are what the field is
    built into, from :func:`voxel_grid_extent`, so a reservation and a field cannot describe
    different grids: one that did would register without an error and put the geometry somewhere the
    cell is not.
    """
    limits = _world_limits(robot_cfg)
    perceived = getattr(getattr(robot_cfg.safety, "planning_world", None), "perceived", None)
    if limits is None or perceived is None or not bool(getattr(perceived, "enabled", False)):
        return ""
    extent = voxel_grid_extent(limits, float(getattr(perceived, "voxel_field_mm", 0.0)))
    if extent is None:
        return ""
    dims, voxel = extent
    return ",".join(f"{v / 1000.0:.4f}" for v in (*dims, voxel))


def build_live_planner_world(
    robot_cfg: "RobotConfig",
    cameras: "Sequence[tuple[str, Any, np.ndarray]]",
) -> "LivePlannerWorld | None":
    """The world source for this cell, or `None` when this cell has not asked for one.

    `cameras` is one entry per camera: a name for a refusal to use, something that answers
    `grab_surface_depth`, and its CAMERA to BASE transform as a 4x4 in millimetres. A cell with two
    cameras passes two, and the points are combined before anything is clustered, so a part one
    camera sees the front of and the other sees the side of is one obstacle.

    Returns `None` rather than an empty source in three cases, and all three mean the same thing:
    this cell did not ask for a live world. The block is disabled, the `perceived` sub-block is
    turned off, or no camera was handed in. A cell that asked and cannot have one is a different
    matter entirely, and the refusal for that belongs at the moment of the motion, not here.
    """
    world_cfg = getattr(robot_cfg.safety, "planning_world", None)
    if world_cfg is None or not bool(getattr(world_cfg, "enabled", False)):
        return None
    perceived = getattr(world_cfg, "perceived", None)
    if perceived is None or not bool(getattr(perceived, "enabled", False)) or not cameras:
        return None
    limits = _world_limits(robot_cfg)
    if limits is None:
        return None

    if float(perceived.voxel_field_mm) > 0.0:
        # Pay the distance-transform import here, where a cell is being built and a tenth of a
        # second costs nothing. Measured against the real planner: the first field a cell ever
        # builds took 139.6 ms and every one after it 6.3 ms, and the difference was this import
        # landing on the first motion instead of on the build.
        import scipy.ndimage  # noqa: F401, PLC0415 - imported for its cost, not for its names

    return LivePlannerWorld(
        cameras=tuple(
            CameraView(name=name, depth_source=source, camera_to_base=np.asarray(transform, float))
            for name, source, transform in cameras
        ),
        declared=tuple(
            build_planner_cuboids(world_cfg, robot_cfg.safety.self_collision.fixtures)
        ),
        limits=limits,
        declared_meshes=tuple(build_planner_meshes(world_cfg)),
        tuning=WorldBuildTuning(
            pixel_stride=int(perceived.pixel_stride),
            voxel_size_mm=float(perceived.voxel_size_mm),
            cluster_voxel_mm=float(perceived.cluster_voxel_mm),
            min_points=int(perceived.min_points),
            margin_mm=float(perceived.margin_mm),
            max_boxes=int(perceived.max_boxes),
            voxel_field_mm=float(perceived.voxel_field_mm),
            floor_to_plane=bool(perceived.floor_to_plane),
        ),
        max_age_ms=float(perceived.max_age_ms),
        self_radius_mm=float(perceived.self_radius_mm),
        tool_radius_mm=float(perceived.tool_radius_mm),
    )


def _world_limits(robot_cfg: "RobotConfig") -> "WorldBuildLimits | None":
    """Where an obstacle may be, from the same declaration the workspace guard reads.

    One source for the reach, so a perceived box can never appear somewhere the arm was already
    forbidden to go, and the support plane from the planner block, because a bench registered as a
    hundred small obstacles is a bench that fills the planner slots and duplicates a box the
    operator already wrote down.
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


def static_camera_to_base_mm(resolver: Any) -> np.ndarray:
    """The fixed CAMERA to BASE matrix of an eye-to-hand resolver, in millimetres.

    Fixed cameras only, and deliberately so. An eye-in-hand rig has a transform that changes with
    every joint the arm moves, and the one that matters is the transform at the SHUTTER of the depth
    frame, not the one at the moment the boxes are built. The two differ by however far the arm
    travelled in between, which is exactly the distance an obstacle would appear away from where it
    is. Registering that as a world is worse than registering nothing, so a wrist camera is refused
    here rather than approximated.

    Raises
    ------
    LiveWorldUnavailable
        For a resolver that is not a fixed transform, naming what would have to change.
    """
    transform = getattr(resolver, "transform", None)
    matrix = getattr(transform, "to_matrix", None)
    if transform is None or not callable(matrix):
        raise LiveWorldUnavailable(
            f"{type(resolver).__name__} does not carry a fixed CAMERA to BASE transform, so this "
            "camera cannot feed the planner world. A wrist camera moves between the shutter and the "
            "moment the geometry is built, and the world would be displaced by exactly that travel. "
            "Use a fixed camera for the planner world, or stamp the tool pose onto the depth frame "
            "and compose it here."
        )
    return np.asarray(matrix(), dtype=np.float64)
