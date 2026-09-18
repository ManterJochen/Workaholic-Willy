"""Which cameras feed a cell's live planner world, and the world they feed.

Every enabled RGB-D rig that declares its calibration feeds the world of a cell that enables it, the
primary first. A rig's own `enabled` decides, not `grasping.fusion.cameras`, which names what fusion
takes. A fixed camera is placed by its CAMERA to BASE, and a camera on the wrist frame by frame, by its
CAMERA to TOOL and the tool pose each frame is stamped with.

It lives in `execution` rather than in `autonomous_grasp` so that `Robot` can be handed the same world:
`Robot`'s import guard forbids that package, the camera package and the calibration package. Rigs,
cameras and calibrations are therefore taken by duck typing, and nothing here imports them.

Every camera in a world has to answer. One camera that stays blind, silent or stale refuses the whole
world, is asked `fresh_frame_attempts` more times, and then every refreshing motion raises
`CameraWorldUnavailable` naming it. A cell that enables the world and calibrates a second rig therefore
stops every planned motion when that rig fails.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.contracts import UNSET, Maybe, chosen
from src.robot.safety.planning.depth_source import RigDepthSource
from src.robot.safety.planning.live_world import CameraView, LivePlannerWorld
from src.robot.safety.planning.perceived import WorldBuildTuning
from src.robot.safety.planning.reservation import PlannerReservation, planner_world_limits
from src.robot.safety.planning.world import build_planner_cuboids, build_planner_meshes

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

__all__ = ["CameraWorldPlan", "CameraWorldRequired", "CameraWorldWiring", "OpenedCameras"]


class CameraWorldRequired(ValueError):
    """A cell that plans with cuRobo holds a calibrated camera and gets no live world from it.

    Once a rig's calibration is declared on the rig the world is mandatory: this refuses a calibrated
    cell that would decline every motion forever.
    """


def _rig_key(rig_id: str) -> str:
    return f"camera.cameras.rigs[{rig_id!r}].extrinsics"


@dataclass(frozen=True, slots=True)
class CameraWorldPlan:
    """Which rigs feed a cell's live planner world, primary first, and why every other rig does not.

    Pure: it reads config and opens nothing, so a cell that did not ask for a world opens no camera
    for one.
    """

    #: The rigs the world is built from, the primary first and then in `camera.cameras.rigs` order. Empty
    #: for no world.
    rig_ids: tuple[str, ...]
    #: Every other rig, with why it is not a world camera.
    left_out: tuple[tuple[str, str], ...] = ()
    #: Why there is no world, empty when there is one.
    reason: str = ""
    #: Whether the cell asked for a world: the block and its `perceived` half on, and a support plane
    #: declared. A cell that did not ask is told nothing; one that asked and gets no world is told why.
    asked: bool = False
    #: Every enabled RGB-D rig that declares its calibration, whether or not the world takes it. A
    #: cuRobo cell with one and no world is refused (:meth:`refusal`).
    calibrated: tuple[str, ...] = ()
    #: The primary rig's id, for the refusal sentence.
    primary_rig_id: str = ""

    @classmethod
    def from_config(cls, robot_cfg: "RobotConfig", rigs: Sequence[Any], *, primary_rig_id: str) -> CameraWorldPlan:
        calibrated = tuple(
            str(rig.rig_id) for rig in rigs
            if bool(getattr(rig, "enabled", False)) and getattr(rig, "source", None) == "rgbd"
            and getattr(rig, "extrinsics", None) is not None
        )
        primary = str(primary_rig_id)
        world_cfg = getattr(getattr(robot_cfg, "safety", None), "planning_world", None)
        if world_cfg is None or not bool(getattr(world_cfg, "enabled", False)):
            return cls((), reason="safety.planning_world.enabled is false, so this cell has no live planner world",
                       calibrated=calibrated, primary_rig_id=primary)
        perceived = getattr(world_cfg, "perceived", None)
        if perceived is None or not bool(getattr(perceived, "enabled", False)):
            return cls((), reason=(
                "safety.planning_world.perceived.enabled is false, so no camera feeds the planner world"),
                calibrated=calibrated, primary_rig_id=primary)
        if planner_world_limits(robot_cfg) is None:
            return cls((), reason=(
                "safety.planning_world declares no support_plane, so no perceived world can be built"),
                calibrated=calibrated, primary_rig_id=primary)

        wired: list[str] = []
        left_out: list[tuple[str, str]] = []
        # The primary first. The sort is stable, so every other rig keeps its place in the camera section.
        for rig in sorted(rigs, key=lambda each: str(each.rig_id) != primary_rig_id):
            rig_id = str(rig.rig_id)
            source = getattr(rig, "source", None)
            if not bool(getattr(rig, "enabled", False)):
                left_out.append((rig_id, "enabled: false"))
            elif source != "rgbd":
                left_out.append((rig_id, f"a {source} rig, which carries no depth of its own"))
            elif getattr(rig, "extrinsics", None) is None:
                left_out.append((rig_id, f"not calibrated, {_rig_key(rig_id)} is not declared"))
            else:
                wired.append(rig_id)

        if not wired:
            return cls((), tuple(left_out), asked=True, reason=(
                "no enabled RGB-D rig declares its calibration, so no camera can feed the planner world"),
                calibrated=calibrated, primary_rig_id=primary)
        if wired[0] != primary_rig_id:
            why = dict(left_out).get(primary_rig_id, "not a rig in camera.cameras.rigs")
            return cls((), tuple(left_out), asked=True, reason=(
                f"the primary rig {primary_rig_id!r} is not a world camera ({why}), and the pick loop offers its "
                "masks under the primary's name, which a world that does not hold that camera refuses"),
                calibrated=calibrated, primary_rig_id=primary)
        return cls(tuple(wired), tuple(left_out), asked=True, calibrated=calibrated, primary_rig_id=primary)

    def refusal(self) -> str | None:
        """Why a cell planning with cuRobo may not build on this plan, or ``None``.

        A calibrated enabled rig that yields no world is refused: the calibration is on the rig, so
        the world is mandatory, and a cell built anyway would refuse every motion nothing declined. An
        uncalibrated cell builds with no world, and every planned motion then needs a decline. The
        caller decides whether the cell plans with cuRobo.
        """
        if self.rig_ids or not self.calibrated:
            return None
        return (
            f"this cell plans with cuRobo and calibrates {', '.join(repr(rig) for rig in self.calibrated)}, and no "
            f"live camera world comes of it: {self.reason}. Once a rig is calibrated its world is mandatory: enable "
            "safety.planning_world with a support_plane and perceived.enabled, with the primary rig "
            f"{self.primary_rig_id!r} among the calibrated ones, or remove the rig's extrinsics while it is not used"
        )

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if self.rig_ids:
            text = "camera world from " + ", ".join(repr(rig_id) for rig_id in self.rig_ids) + ", primary first"
        else:
            text = f"no camera world: {self.reason}"
        if self.left_out:
            text += "; left out: " + "; ".join(f"{rig_id!r} ({why})" for rig_id, why in self.left_out)
        return text

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {"rig_ids": list(self.rig_ids), "left_out": dict(self.left_out), "reason": self.reason,
                "asked": self.asked, "calibrated": list(self.calibrated), "refusal": self.refusal()}


@dataclass(frozen=True, slots=True)
class CameraWorldWiring:
    """The world a plan's cameras build, or why they build none."""

    #: The world to hand the arm, or `None`.
    world: LivePlannerWorld | None
    #: The cameras it is built from, in plan order.
    cameras: tuple[str, ...] = ()
    #: Why there is no world, empty when there is one.
    reason: str = ""
    #: `robot.ur.motion_planner`, because on a UR cell only the cuRobo planner reads the world.
    planner: str | None = None

    @classmethod
    def from_cameras(
        cls,
        robot_cfg: "RobotConfig",
        *,
        plan: CameraWorldPlan,
        cameras: Mapping[str, Any],
        tool_pose: Maybe[Callable[[], Any]] = UNSET,
    ) -> CameraWorldWiring:
        """The world built from ``cameras``, the open owners by rig id.

        Each owner answers `rig_id`, `handle()` and `calibration()`. ``tool_pose`` is the arm's
        `get_tcp_pose`, which a camera on the wrist needs and a fixed one does not. A calibration that
        does not load raises what the loader raised, naming the rig's key.
        """
        planner = getattr(getattr(robot_cfg, "ur", None), "motion_planner", None)
        if not plan.rig_ids:
            return cls(None, reason=plan.reason, planner=planner)
        views: list[CameraView] = []
        for rig_id in plan.rig_ids:
            owner = cameras.get(rig_id)
            if owner is None:
                raise ValueError(f"the camera world plan names {rig_id!r} and no open camera was handed in for it")
            handle = owner.handle()
            if _answers_with_a_stereo_pair(handle):
                return cls(None, planner=planner, reason=(
                    f"camera {rig_id!r} answers with a stereo pair, which carries no depth of its own, so a world "
                    "built on it would stop every planned motion. Wire an RGB-D camera, or turn "
                    "safety.planning_world.perceived off"))
            calibration = owner.calibration()
            if str(calibration.mounting_mode) != "eye_in_hand":
                views.append(CameraView(name=rig_id, depth_source=RigDepthSource(handle),
                                        camera_to_base=calibration.camera_to_base().to_matrix()))
                continue
            if chosen(tool_pose):
                source = RigDepthSource(handle, tool_pose=tool_pose, motion_tolerance=(
                    float(calibration.shutter_motion_tolerance_mm), float(calibration.shutter_motion_tolerance_deg)))
                views.append(CameraView(name=rig_id, depth_source=source,
                                        camera_to_tool=calibration.camera_to_tool().to_matrix()))
                continue
            return cls(None, planner=planner, reason=(
                f"camera {rig_id!r} is on the wrist and no reader of the arm's TCP was handed in, so where it stood "
                "at each shutter would be unknown"))
        return cls(_live_planner_world(robot_cfg, views), plan.rig_ids, planner=planner)

    @property
    def wired(self) -> bool:
        return self.world is not None

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if self.world is None:
            return f"no live planner world for this cell: {self.reason}"
        named = ", ".join(repr(camera) for camera in self.cameras)
        if self.planner != "curobo":
            # The UR arm reads this world only inside its cuRobo planner, so on any other planner the
            # world is handed over and never consulted, and the sentence below would announce a
            # protection that does not run.
            return (f"live planner world wired: {len(self.cameras)} camera(s) ({named}), but "
                    f"robot.ur.motion_planner is {self.planner!r}, so no planner runs on this cell and nothing "
                    "reads it")
        return f"live planner world wired: {len(self.cameras)} camera(s) ({named}), refreshed before every plan"

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {"wired": self.wired, "cameras": list(self.cameras), "reason": self.reason, "planner": self.planner}


@dataclass(frozen=True, slots=True)
class OpenedCameras:
    """The cameras a build opened for the world alone, given back with the rest of the cell's cameras.

    `release_perception` walks it beside the pick camera and the fused cameras, so a console rebuild
    meets no camera still held by the build before it. Each `release` is the owner's, which is
    idempotent and never raises.
    """

    cameras: tuple[Any, ...] = ()

    def close(self) -> None:
        for camera in self.cameras:
            camera.release()


def _answers_with_a_stereo_pair(handle: Any) -> bool:
    """Whether one grab from this rig is a stereo pair: two images and no depth. A failed grab says nothing.

    A rig that cannot answer while the cell is built is judged at the motion, where the refresh asks it
    again and raises if it stays silent. Only a rig that did answer, with two images and no depth, is
    known here never to be able to carry a world.
    """
    try:
        frame = handle.grab()
    except Exception:  # noqa: BLE001 (judged at the motion instead, see above)
        return False
    return getattr(frame, "depth", None) is None and hasattr(frame, "left") and hasattr(frame, "right")


def _live_planner_world(robot_cfg: "RobotConfig", views: Sequence[CameraView]) -> LivePlannerWorld:
    """The world source over ``views``, tuned by the cell's `safety.planning_world`.

    The plan has already checked that the block is on.
    """
    world_cfg = robot_cfg.safety.planning_world
    perceived = world_cfg.perceived
    limits = planner_world_limits(robot_cfg)
    if limits is None:
        raise ValueError("safety.planning_world declares no support_plane, so no perceived world can be built")
    if float(perceived.voxel_field_mm) > 0.0:
        # Pay the distance-transform import here, where a cell is being built and a tenth of a second
        # costs nothing. Measured against the real planner: the first field a cell builds took 139.6 ms
        # and every one after it 6.3 ms, and the difference is this import, which would otherwise land
        # on the first motion instead of on the build.
        import scipy.ndimage  # noqa: F401, PLC0415 (imported for its cost, not for its names)
    return LivePlannerWorld(
        cameras=tuple(views),
        declared=tuple(build_planner_cuboids(
            world_cfg, robot_cfg.safety.self_collision.fixtures,
            max_cuboids=PlannerReservation.from_config(robot_cfg=robot_cfg).cuboid_slots,
        )),
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
        fresh_frame_attempts=int(perceived.fresh_frame_attempts),
        require_registration=bool(getattr(world_cfg, "require_registration", True)),
    )
