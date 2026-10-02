"""Two of the owner's recorded looks of 2026-10-01, depth only, for the tests of the cell fixes.

The owner's cell: a UR10, a Robotiq Hand-E, a D415 on the wrist, a foam mat about 55 mm thick on the aluminium table,
and the Zollstock in a pile of parts on the mat. Both looks were taken at LOOK[0] by example 12, 13:43 and 13:53:

* ``P1``, ``pick-63e1b3dac29f``, the first Zollstock pick;
* ``P5``, ``pick-44c3e7e060ac``, the last.

The owner allowed the depth to ship, not the colour images (2026-10-01), so no colour array is here. What is:

* the surface depth at every second pixel, as the cell's world reads it (``pixel_stride`` 2), in whole millimetres.
  The recording holds ``depth_0`` and ``surface_depth_0`` bit-identical, so one array is kept. :func:`look` hands it
  back at the frame's own size, each read pixel repeated over the three beside it: a world read at stride 2 (or 4)
  reads exactly the recorded pixels, and a frame judged whole holds a depth where the recording did;
* the intrinsics, CAMERA to BASE and TOOL to BASE the recording carries, and the look's joints;
* the target: the mask the analysis recovered for it (``E2``) and the full-resolution depth at its pixels, from which
  the recorded target cloud comes back to within 0.0001 mm, in its own order.

The workspace box grown by 200 mm fills the whole frame at this look (rows 0-716, columns 76-1280), so the frame is
kept whole rather than windowed; each file stays under 150 kB.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.robot.safety.planning.perceived import DepthView, WorldBuildLimits, WorldBuildTuning

_DATA = Path(__file__).resolve().parent / "data" / "cell_2026_10_01"
_FILES = {"P1": "p1_pick-63e1b3dac29f.npz", "P5": "p5_pick-44c3e7e060ac.npz"}

#: The names :func:`look` answers.
LOOKS = tuple(_FILES)

#: The owner's workspace box (the cell's ``robot.workspace_limits``), the bench declared at 0 with the 5 mm band.
OWNER_LIMITS = WorldBuildLimits(
    x_mm=(-420.0, 400.0), y_mm=(-900.0, -270.0), z_mm=(16.0, 800.0), support_plane_top_mm=0.0,
    plane_clearance_mm=5.0,
)


@dataclass(frozen=True)
class CellLook:
    """One recorded look, depth only."""

    #: ``P1`` or ``P5``.
    name: str
    #: The recording's pick id, ``pick-...``.
    pick: str
    #: The recording this came from, by its file name.
    source: str
    #: The camera's depth, CAMERA millimetres, at the frame's size: every second pixel the camera's, the three beside
    #: it repeating it. ``depth_mm`` and ``surface_depth_mm`` are the same array, as in the recording.
    surface_depth_mm: np.ndarray
    #: The intrinsics of the frame, 3x3.
    intrinsics: np.ndarray
    #: CAMERA to BASE, 4x4, millimetres.
    camera_to_base: np.ndarray
    #: TOOL to BASE when the shutter opened, 4x4, millimetres.
    tool_to_base: np.ndarray
    #: The look's joints, degrees.
    joints_deg: tuple[float, float, float, float, float, float]
    #: The camera and the joints, as the recording names the view.
    view: str
    #: The target's mask in the frame, as recovered for the analysis.
    target_mask: np.ndarray
    #: The recorded target cloud, BASE millimetres, ``(N, 3)``.
    target_points_base_mm: np.ndarray

    @property
    def depth_mm(self) -> np.ndarray:
        """The raw depth: identical to the surface depth in the recording."""
        return self.surface_depth_mm

    @property
    def camera_to_tool(self) -> np.ndarray:
        """CAMERA to TOOL, as the recording's two transforms give it."""
        return np.linalg.inv(self.tool_to_base) @ self.camera_to_base


@lru_cache(maxsize=None)
def look(name: str) -> CellLook:
    """The recorded look ``name`` (:data:`LOOKS`)."""
    with np.load(_DATA / _FILES[name]) as data:
        stride = int(data["stride"])
        shape = tuple(int(v) for v in data["frame_shape"])
        strided = data["surface_depth_strided_mm"].astype(np.float64)
        frame = np.repeat(np.repeat(strided, stride, axis=0), stride, axis=1)[: shape[0], : shape[1]]
        intrinsics = data["intrinsics"].astype(np.float64)
        camera_to_base = data["camera_to_base"].astype(np.float64)
        mask = data["target_mask"].astype(bool)
        rows, cols = np.nonzero(mask)
        z = data["target_depth_mm"].astype(np.float64)
        camera = np.column_stack(((cols - intrinsics[0, 2]) * z / intrinsics[0, 0],
                                  (rows - intrinsics[1, 2]) * z / intrinsics[1, 1], z))
        cloud = (camera @ camera_to_base[:3, :3].T + camera_to_base[:3, 3])[data["target_kept"].astype(bool)]
        joints = tuple(float(v) for v in data["joints_deg"])
        frame.setflags(write=False)
        mask.setflags(write=False)
        cloud.setflags(write=False)
        return CellLook(
            name=name, pick=_FILES[name].split("_", 1)[1][: -len(".npz")], source=str(data["source"]),
            surface_depth_mm=frame, intrinsics=intrinsics, camera_to_base=camera_to_base,
            tool_to_base=data["tool_to_base"].astype(np.float64), joints_deg=joints,  # type: ignore[arg-type]
            view=str(data["view"]), target_mask=mask, target_points_base_mm=cloud,
        )


def owner_tuning(**changes: object) -> WorldBuildTuning:
    """The owner's ``planning_world.perceived`` tuning as a :class:`WorldBuildTuning`, ``changes`` on top."""
    values: dict[str, object] = dict(pixel_stride=2, voxel_size_mm=10.0, cluster_voxel_mm=25.0, min_points=12,
                                     margin_mm=15.0, max_boxes=64, voxel_field_mm=0.0, floor_to_plane=True)
    values.update(changes)
    return WorldBuildTuning(**values)  # type: ignore[arg-type]


def depth_view(recorded: CellLook, **changes: object) -> DepthView:
    """The look as one :class:`DepthView`, named as the cell names its camera, ``changes`` on top."""
    values: dict[str, object] = dict(
        surface_depth_mm=recorded.surface_depth_mm, intrinsics=recorded.intrinsics,
        camera_to_base=recorded.camera_to_base, name="EIH_Cam", timestamp=0.0,
    )
    values.update(changes)
    return DepthView(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------------------------
# The owner's cell as far as the camera world and the exact guard read it, built offline: no camera, no controller.
# ---------------------------------------------------------------------------------------------------------------------

#: The joints of LOOK[0], where both recorded looks were taken, degrees.
LOOK_0_DEG = (-87.50, -64.51, -85.79, -129.82, 90.27, -108.84)


def owner_robot(**safety_world: object) -> dict[str, Any]:
    """The owner's ``robot`` block as far as these tests need it: a UR10 on cuRobo, the Hand-E on a 23 mm plate with
    its PolyScope tool frame, the cell's workspace box and 5 mm distances, and the bench declared under the work area as
    the Monday config moves it. ``safety_world`` goes into ``safety.planning_world`` on top."""
    world: dict[str, Any] = {
        "enabled": True, "include_fixtures": False,
        "support_plane": {"height_mm": 0.0, "extent_mm": [900.0, 700.0], "thickness_mm": 50.0,
                          "center_mm": [-10.0, -585.0]},
        "payload": {"enabled": False},
    }
    world.update(safety_world)
    return {
        "vendor": "ur",
        "ur": {"model": "ur10", "motion_planner": "curobo"},
        "workspace_limits": {"x_min": -420.0, "x_max": 400.0, "y_min": -900.0, "y_max": -270.0, "z_min": 16.0,
                             "z_max": 800.0},
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "hand_adapter", "thickness_mm": 23.0}],
                    "tool_frame": {"source": "polyscope", "offset_mm": [0.0, 0.0, 157.0],
                                   "rotation_quat_xyzw": [0.0, 0.0, -0.7071068, 0.7071068]}},
        "safety": {"payload": {"enforce": False},
                   "self_collision": {"backend": "fcl", "min_distance_mm": 5.0, "perceived_min_distance_mm": 5.0,
                                      "kinematics_model": "ur10", "planner_margin_mm": 4.0, "tool_model": "finger"},
                   "planning_world": world},
    }


class OwnerCell:
    """The owner's arm, its exact guard and its camera world, offline."""

    def __init__(self, robot: "dict[str, Any] | None" = None) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.ur.arm import URRobotArm
        from src.robot.drivers.ur.tool_frame import tool_frame_matrix

        self.config = RobotConfig.model_validate(robot if robot is not None else owner_robot())
        self.arm = URRobotArm(self.config)
        self.preflight = self.arm._preflight  # noqa: SLF001
        self.guard = self.preflight._path_authority(self.arm)  # noqa: SLF001
        offset, turn = self.config.gripper.tool_frame.offset_mm, self.config.gripper.tool_frame.rotation_quat_xyzw
        self.tool = np.asarray(tool_frame_matrix((float(offset[0]), float(offset[1]), float(offset[2])),
                                                 (float(turn[0]), float(turn[1]), float(turn[2]), float(turn[3]))),
                               dtype=np.float64)

    def has_the_engine(self) -> bool:
        return self.guard is not None and self.guard.exact_mesh_engine(self.arm) is not None

    def tcp(self, joints_rad: Any) -> np.ndarray:
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        frames = ur_link_transforms_mm("ur10", np.asarray(joints_rad, dtype=np.float64))
        assert frames is not None
        return np.asarray(frames[-1]) @ self.tool

    def ik_tcp(self, goal: np.ndarray, near_rad: Any) -> np.ndarray:
        """Joints that put the TCP at ``goal``, nearest ``near_rad``, solved numerically."""
        from scipy.optimize import least_squares

        target = np.asarray(goal, dtype=np.float64)

        def residual(q: np.ndarray) -> np.ndarray:
            t = self.tcp(q)
            turn = t[:3, :3].T @ target[:3, :3]
            angle = 0.5 * np.array([turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1]])
            return np.concatenate((t[:3, 3] - target[:3, 3], 200.0 * angle))

        return np.asarray(least_squares(residual, np.asarray(near_rad, dtype=np.float64), xtol=1e-12, ftol=1e-12,
                                        gtol=1e-12).x)

    def envelope(self, joints_rad: Any) -> Any:
        from src.robot.safety.planning.self_envelope import self_envelope

        envelope = self_envelope(self.preflight, self.arm, list(joints_rad))
        assert envelope is not None
        return envelope

    def goal_keep_out(self, tcp: np.ndarray) -> Any:
        from src.robot.safety.planning.self_envelope import goal_keep_out

        return goal_keep_out(self.preflight, self.arm, tcp)

    def world(self, recorded: CellLook, depth_mm: "np.ndarray | None" = None, config: Any = None,
              stamp: "float | None" = None) -> Any:
        """The cell's live camera world over the wrist camera, answering with ``depth_mm`` (the look's own by default),
        built the way the cell builds it (``camera_world_wiring._live_planner_world``) from ``config`` (this cell's by
        default). Each frame is stamped ``stamp``, or the time it is grabbed."""
        from src.robot.execution.camera_world_wiring import _live_planner_world
        from src.robot.safety.planning.live_world import CameraView, DepthSnapshot

        depth = recorded.surface_depth_mm if depth_mm is None else depth_mm

        class _Wrist:
            def grab_surface_depth(self) -> DepthSnapshot:
                return DepthSnapshot(depth_mm=depth, intrinsics=recorded.intrinsics,
                                     timestamp=time.time() if stamp is None else float(stamp),
                                     tool_to_base_mm=recorded.tool_to_base)

        return _live_planner_world(self.config if config is None else config,
                                   [CameraView(name="EIH_Cam", depth_source=_Wrist(),
                                               camera_to_tool=recorded.camera_to_tool)])

    def refused(self, boxes: Any, joints_rad: Any) -> str:
        """What the exact guard says of ``joints_rad`` against the guard boxes ``boxes``: empty where it accepts."""
        from src.robot.core import JointPositions, MotionCommand
        from src.robot.safety.guard import SafetyContext

        assert self.guard is not None
        self.guard.set_perceived_fixtures(tuple(boxes))
        decision = self.guard.evaluate(SafetyContext(command=MotionCommand.MOVE_JOINTS, arm=self.arm,  # type: ignore[arg-type]
                                                     target_joints=JointPositions(tuple(float(v) for v in joints_rad))))
        return str(decision.message) if decision.rejected else ""

    def distance_mm(self, joints_rad: Any, boxes: Any) -> float:
        """The exact meshes' least distance at ``joints_rad`` to the axis-aligned or turned ``boxes``, millimetres, by
        bisection on the guard's own backend; 0 where one is met."""
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        assert self.guard is not None
        backend: Any = self.guard._exact_mesh_backend("ur10")  # noqa: SLF001
        transforms = ur_link_transforms_mm("ur10", np.asarray(joints_rad, dtype=np.float64))
        if backend.evaluate(transforms, 0.0, tuple(boxes), 1e-6, arm_pairs=False) is not None:
            return 0.0
        low, high = 0.0, 400.0
        for _ in range(36):
            middle = (low + high) / 2.0
            if backend.evaluate(transforms, 0.0, tuple(boxes), middle, arm_pairs=False) is None:
                low = middle
            else:
                high = middle
        return low

    def line(self, near_rad: Any, grasp: np.ndarray, length_mm: float, step_mm: float = 2.0) -> list[np.ndarray]:
        """Joints along the straight TCP line from ``length_mm`` back along the grasp's approach to the grasp."""
        approach = grasp[:3, 2]
        out, q = [], np.asarray(near_rad, dtype=np.float64)
        for k in range(int(round(length_mm / step_mm)) + 1):
            goal = grasp.copy()
            goal[:3, 3] = grasp[:3, 3] - approach * (length_mm - k * step_mm)
            q = self.ik_tcp(goal, q)
            out.append(q)
        return out


@lru_cache(maxsize=1)
def owner_cell() -> OwnerCell:
    """One owner cell per test run: building the arm and its guard costs about a second."""
    return OwnerCell()


def rays(recorded: CellLook) -> "tuple[np.ndarray, np.ndarray]":
    """Every pixel's ray in BASE: its origin, and its direction with unit camera depth."""
    rows, cols = recorded.surface_depth_mm.shape
    v, u = np.mgrid[0:rows, 0:cols]
    k = recorded.intrinsics
    camera = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones(u.shape)], -1).reshape(-1, 3)
    turn, origin = recorded.camera_to_base[:3, :3], recorded.camera_to_base[:3, 3]
    return np.broadcast_to(origin, (camera.shape[0], 3)), camera @ turn.T


def with_solids(recorded: CellLook, *, cylinders: Any = (), boxes: Any = (), depth_mm: "np.ndarray | None" = None,
                noise_mm: float = 1.0, seed: int = 5) -> np.ndarray:
    """The look's depth with upright cylinders ``(x, y, radius, z_low, z_high)`` and boxes ``(centre, size, rotation)``
    drawn in where they stand nearer than what was recorded, the new pixels with Gaussian noise."""
    depth = np.array(recorded.surface_depth_mm if depth_mm is None else depth_mm, dtype=np.float64, copy=True)
    origin, direction = rays(recorded)
    nearest = np.full(origin.shape[0], np.inf)
    for x, y, radius, z0, z1 in cylinders:
        ox, oy = origin[:, 0] - x, origin[:, 1] - y
        a = direction[:, 0] ** 2 + direction[:, 1] ** 2
        b = 2.0 * (ox * direction[:, 0] + oy * direction[:, 1])
        c = ox * ox + oy * oy - radius * radius
        disc = b * b - 4.0 * a * c
        ok = (disc >= 0.0) & (a > 1e-12)
        t0 = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * np.where(a > 1e-12, a, 1.0))
        z = origin[:, 2] + t0 * direction[:, 2]
        side = np.where(ok & (t0 > 0.0) & (z >= z0) & (z <= z1), t0, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            lid = (z1 - origin[:, 2]) / direction[:, 2]
        on_lid = (lid > 0.0) & ((ox + lid * direction[:, 0]) ** 2 + (oy + lid * direction[:, 1]) ** 2 <= radius ** 2)
        nearest = np.minimum(nearest, np.minimum(side, np.where(on_lid, lid, np.inf)))
    for centre, size, turn in boxes:
        rotation = np.eye(3) if turn is None else np.asarray(turn, dtype=np.float64)
        local_o = (origin - np.asarray(centre, dtype=np.float64)) @ rotation
        local_d = direction @ rotation
        half = np.asarray(size, dtype=np.float64) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - local_o) / local_d, (half - local_o) / local_d
        near = np.nanmax(np.minimum(t1, t2), axis=1)
        far = np.nanmin(np.maximum(t1, t2), axis=1)
        nearest = np.minimum(nearest, np.where((far >= near) & (far > 0.0), np.where(near > 0.0, near, far), np.inf))
    drawn = nearest.reshape(depth.shape)
    nearer = np.isfinite(drawn) & (~(depth > 0.0) | (drawn < depth))
    noise = np.random.default_rng(seed).normal(0.0, noise_mm, int(nearer.sum()))
    depth[nearer] = drawn[nearer] + noise
    return depth


def level_table(recorded: CellLook, z_mm: float = 0.0, *, noise_mm: float = 1.3, seed: int = 7) -> np.ndarray:
    """The look's camera seeing nothing but a level bare table at ``z_mm``, with Gaussian noise: the owner's "the mat is
    gone tomorrow", with a camera that reads the table right (or ``z_mm`` off)."""
    origin, direction = rays(recorded)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_mm - origin[:, 2]) / direction[:, 2]
    t = t.reshape(recorded.surface_depth_mm.shape)
    noisy = t + np.random.default_rng(seed).normal(0.0, noise_mm, t.shape)
    return np.where(np.isfinite(t) & (t > 0.0), noisy, 0.0)


def local_reading(recorded: CellLook, x: float, y: float, inner_mm: float = 25.0, outer_mm: float = 45.0) -> float:
    """The median height the look reads on a ring about (x, y) between the two radii, at the mat's level (35-75 mm)."""
    rows, cols = recorded.surface_depth_mm.shape
    v, u = np.mgrid[0:rows:2, 0:cols:2]
    z = recorded.surface_depth_mm[::2, ::2]
    ok = z > 0.0
    k, t = recorded.intrinsics, recorded.camera_to_base
    camera = np.column_stack(((u[ok] - k[0, 2]) * z[ok] / k[0, 0], (v[ok] - k[1, 2]) * z[ok] / k[1, 1], z[ok]))
    points = camera @ t[:3, :3].T + t[:3, 3]
    radius = np.hypot(points[:, 0] - x, points[:, 1] - y)
    ring = (radius > inner_mm) & (radius < outer_mm) & (points[:, 2] > 35.0) & (points[:, 2] < 75.0)
    return float(np.median(points[ring, 2])) if ring.sum() > 20 else math.nan
