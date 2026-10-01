"""Probe, on the GPU box, that the planner holds the camera world's turned boxes as the exact guard holds them (F2), that
a pose in the planner's cushion band beside them is decided as the band admission (F1) decides it, and that a bin the
camera saw beside the base is the exact guard's where only its boxes refuse the planner's world (Option 1).

Run from the repository root with the repository's own interpreter; the sidecar is spawned with the cuRobo
environment's, as a cell spawns it::

    python scripts/curobo/probe_turned_boxes.py
    python scripts/curobo/probe_turned_boxes.py --curobo-python <cuRobo env python> --coal-prefix <coal env>

Why it exists. The camera world is a height map (``planning/height_map.py``): an open bin comes back as its walls with
a free inside, a turned object as a turned box, each from the bench to its top + 15 mm. cuRobo takes every box turned,
and since 2026-09-30 the exact guard judges the same turned box, at ``perceived_min_distance_mm`` (5 mm) rather than the
10 mm a declared fixture and the arm itself keep. The planner reserves 1 + declared + ``max_boxes`` (64) box slots when
it starts. The CPU suite holds the pipeline, both box lists and the guard with stand-in planners; this holds the real
sidecar's cuboid cache, its world term on turned boxes, its judgement with the camera's boxes set aside, a plan in a
full world, and the driver's own decision on them.

cuRobo judges its world with its sphere cover, which reaches past the meshes by design: 25 to 29 mm past the UR10's
shoulder housing (CPU replica, 2026-09-30). Alone, that kept a bin the camera saw about 50 mm from the housing for a
plan and 60 for a straight line, where the exact guard takes about 26. The owner's Option 1: where only the camera's
boxes refuse the planner's world, the driver asks the planner again with them set aside (``check_js`` with
``ignore_perceived``), and the exact guard, holding the same boxes, decides them. Step 5 holds that at 30, 40 and 47.7
mm of real clearance, and records 54.3 and 60.8, at two poses: the check joints, which lie in the planner's cushion band
too, and the same housing with the wrist out of the band, the ordinary pose beside a bin, where the planner refuses on
its world alone (verifier W, 2026-10-01).

What it does, in order, on an owner-like cell (UR10, Hand-E on a 20 mm plate, approach +Z and closing +X on the flange,
planner_margin_mm 4.0, the bench declared at z 0, max_boxes 64, a home configuration for the joint-limit guard):

  1. the reservation the driver asks for, 1 bench + 64 = 65 box slots, and the planner started through the driver's own
     start, which refuses a sidecar whose composed_sha256 is not the one the committed evidence names;
  2. a camera 1100 mm over the bench looking straight down at a 300 x 200 mm open bin (40 mm rim) turned 30 deg beside
     the shoulder housing, 60.8 mm from the arm's meshes at the check joints, and 48 small parts turned at random beside
     it, more than the 25 mm cluster distance away so the bin stays an object of its own (four walls turned 30 deg), and
     enough of them to fill the slots; the driver's own refresh (``CuroboUrPlanner.refresh_world``): the boxes merge to
     fit the 64 slots, the planner confirms all 65 it is sent, and the guard is handed the same boxes, turned;
  3. at the investigation's joints (0, -60, 80, -110, -90, 0) deg, the screen, then the driver's own decision (the
     plain check, the planner's report and ``URRobotArm._exact_guard_decides``) with the turned boxes held by both:
     VALID. The pose lies in the planner's cushion band (forearm|wrist_2): the planner's padded spheres refuse it on that
     pair, the exact guard decides the pair, the planner's world term is clear of the turned walls (held on its own,
     since the camera's boxes alone would now be the exact guard's too), and the exact guard accepts them at 5 mm;
  4. the same boxes sent to both as their axis-aligned enclosures, by hand: the planner's world term meets the
     enclosure of the bin's near wall where it clears the turned wall, which is the turn reaching the kernel; the exact
     guard, 5 mm from a seen box, still accepts the enclosures there; and the driver's refusal stands on the planner's
     world, because a world set by hand is no refresh's and its camera boxes are never set aside;
  5. the bin 30, 40 and 47.7 mm from the arm's meshes (its own solids, measured with the exact engine), each refreshed
     through the driver, at the check joints and at (0, -60, 80, -95, -90, 0) deg, the same housing with the wrist out
     of the band: the planner's world at no clearance and at the line clearance (a straight line), with the camera's
     boxes and with them set aside, its whole report before and after, the exact guard, and the driver's decision: the
     world set aside clears it, the exact guard accepts it, the driver admits it, and off the band the planner refused
     it on its world alone and typed the refusal WORLD. 54.3 and 60.8 mm are recorded alike. At 30 mm: every camera box
     comes back (one configuration inside each, its verdict and depth the same before and after a set-aside); a
     straight joint line through the driver's own move (``move_to_joints_on_the_line``, a recording controller) runs as
     one moveJ at the check joints and off the band; with the hand's count closed and nothing attached (a plain close,
     or a grasp on a cell that models no part) the line out of the off-band pose is held at its first sample, and runs
     once the hand reads open again, because the camera's boxes are set aside only for a hand known empty and open; the
     grasp's lift there, a moveL 100 mm straight up judged as if the jaws held a part 40 mm across
     (``carried_line_refusal``), is refused at its first sample with nothing sent and nothing left attached, while the
     same lift with the hand empty runs as one moveL (the owner, 2026-10-01: a grasp judges its lift before it closes);
     the screen says the off-band pose is beside the camera's boxes, no ERROR; and a planned move out of the check joints
     is recorded: cuRobo plans in its own world, the camera's boxes included;
  6. a square bin declared at its measured place, 20 to 45 mm from the arm's meshes, its walls and floor sent to the
     planner by hand: where the planner's own spheres meet a declared wall at no clearance and at the line clearance,
     recorded at both poses;
  7. a planned move from those joints to (-25, -70, 90, -110, -90, 0) deg in the turned world, the bin at 60.8 mm
     again (``URRobotArm._planned``, then ``_judged_waypoints``, as a move runs them): out of the band by a straight
     escape leg of at most ``band.BAND_LEG_MAX_DEG`` (20) per joint where the planner refuses the start, cuRobo's plan
     behind it, the route judged whole by both authorities, and its time. The CPU replica plans nothing, and says so;
  8. once that planner stopped, a cell that models the carried part (``planning_world.payload.length_mm`` 60, as a
     cuRobo UR cell must declare; its own composition and evidence, with the attach spheres) and a planner of its own,
     the bin 30 mm away, a recording controller: the grasp's lift at the off-band pose, judged before the jaws close at
     the width a toggle's attach carries (39 mm for a 40 mm part), hands the part to the sidecar, is refused at its
     first sample and takes the part back, nothing sent; the same lift empty-handed runs as one moveL; the whole grasp
     (``GraspExecutionPolicy``) there backs out with the jaws open; and with the shoulder turned away it closes once,
     the sidecar holds the part the check judged, the same width, and the lift runs. Each attach is timed. The CPU
     replica models no carried part, so step 8 holds nothing there (verifier W, 2026-10-01).

The probe's arm carries a hand that reads as the owner's toggle once a person said its jaws stand open: the camera's
boxes are set aside for no other hand. It prints one line per step and writes ``logs/curobo/turned_boxes_probe.json``. Exit code 0 when every expectation held,
1 otherwise. The planner is stopped on every way out. On the GPU it spawns a sidecar of its own, so stop the console, the
API and every other planner first: one sidecar on the GPU at a time. The cell is built here, in code, and reads nothing
of the cell's tree: it carries no wrist camera, so the robobrain.eye housing is not in it.

``--cpu-replica <cuRobo content root>`` judges with the CPU replica of the sidecar that ``probe_band_admission.py``
defines (the same composed robot, cuRobo's spheres by numpy forward kinematics, the kernel's rules; its world term is
sphere against turned box at the clearance asked), for a box where an application-control policy refuses cuRobo's
kernels. Every line it prints then starts with ``[replica]``, every log record it makes carries the same tag, its
``planner under test`` line says CPU REPLICA and its JSON names the replica, so no line of it passes for a GPU line: it
proves the driver's decision on the real geometry, not the GPU kernel's. It registers every box it is sent and plans
nothing, so step 7 says so and holds nothing, and its world refusals carry no depth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_band_admission import CpuReplicaClient, say, tag_replica_output  # noqa: E402

#: The investigation's joints (2026-09-30): the shoulder housing beside the bin. The pose lies in the planner's cushion
#: band (forearm|wrist_2) too, so beside the bin the planner refuses it on the robot itself as well.
CHECK_DEG = (0.0, -60.0, 80.0, -110.0, -90.0, 0.0)
#: The same with the wrist turned on: the housing stays where it is, beside the bin, along the whole line to it.
LINE_DEG = (0.0, -60.0, 80.0, -105.0, -90.0, 0.0)
#: The same housing with the wrist turned out of the band: the ordinary pose beside a bin, which the planner refuses on
#: its world alone and types WORLD (verifier W, 2026-10-01). OFF_LINE_DEG turns the wrist on further, still out of it.
OFF_DEG = (0.0, -60.0, 80.0, -95.0, -90.0, 0.0)
OFF_LINE_DEG = (0.0, -60.0, 80.0, -90.0, -90.0, 0.0)
#: A goal beside it, which a planned move reaches in the turned world.
GOAL_DEG = (-25.0, -70.0, 90.0, -110.0, -90.0, 0.0)
#: Where the joint-limit guard centres its half turn (``safety.joint_limits.within_half_turn_of_home``).
HOME_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)
#: The point the slot budget keeps its boxes nearest, in the controller's base, millimetres.
NEAR_POINT_MM = (600.0, 0.0, 300.0)
#: How far the turned bin is moved out along base -y from where the investigation drew it (its corner 48.3 mm from the
#: housing by that ruler), millimetres, and the exact distance of the arm's meshes to the bin's own solids then, at the
#: check joints (measured on the CPU, 2026-09-30).
SHIFTS_MM = {0.0: 47.7, 10.0: 54.3, 20.0: 60.8}
#: The shift of steps 2 to 4 and 7: past where the planner's cover meets the bin at the check joints.
BIN_SHIFT_MM = 20.0
#: The real clearances of step 5, millimetres: where the driver has to admit the bin (Option 1), and where it is
#: recorded. The exact guard keeps 5 mm to the camera's boxes, which reach 15 mm past a face and 21 across an edge, so it
#: accepts this bin from about 26 mm.
ADMITTED_MM = (30.0, 40.0, 47.7)
RECORDED_MM = (54.3, 60.8)
#: A square bin declared at its measured place (its walls and floor as fixtures, never set aside), recorded at these real
#: clearances from the arm's meshes at the check joints: where the planner's own spheres clear a declared wall for a
#: plan's clearance and for a line's (verifier W, 2026-10-01: the 50 to 60 mm said of a declared bin was not measured).
DECLARED_MM = (20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
#: A 640 x 480 pinhole with a 500 px focal length, and a camera 1100 mm over the bench looking straight down.
WIDTH, HEIGHT, FOCAL = 640, 480, 500.0
K = np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
CAMERA_XYZ_MM = (150.0, -430.0, 1100.0)


def _rad(values: "tuple[float, ...]") -> list[float]:
    return [math.radians(v) for v in values]


def _config(payload_length_mm: "float | None" = None) -> Any:
    """The owner-like cell; ``payload_length_mm`` declares the carried part (``planning_world.payload``), as a cuRobo UR
    cell must, which reserves the attach spheres and names the evidence of that composition (step 8)."""
    from src.config.schema.robot import RobotConfig

    world: dict[str, Any] = {"enabled": True, "support_plane": {"height_mm": 0.0, "extent_mm": [3000.0, 3000.0]},
                             "perceived": {"max_boxes": 64, "pixel_stride": 2}}
    if payload_length_mm is not None:
        world["payload"] = {"enabled": True, "length_mm": float(payload_length_mm)}
    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10", "motion_planner": "curobo"},
        "home_joint_positions_deg": list(HOME_DEG),
        "workspace_limits": {"x_min": -1500.0, "x_max": 1500.0, "y_min": -1500.0, "y_max": 1500.0,
                             "z_min": -50.0, "z_max": 1500.0},
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": {"payload": {"enforce": False},
                   "self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "kinematics_model": "ur10",
                                      "planner_margin_mm": 4.0},
                   "planning_world": world},
    })


# The scene: upright solids turned about BASE Z on a bench at z = 0, seen by a pinhole looking straight down.

def _solid(centre: "tuple[float, float, float]", half: "tuple[float, float, float]", yaw_deg: float) -> dict:
    return {"centre": centre, "half": half, "yaw_deg": yaw_deg}


def _open_bin(centre_xy: "tuple[float, float]", size_xy: "tuple[float, float]", rim_mm: float,
              yaw_deg: float) -> list[dict]:
    """A bin with 5 mm walls up to ``rim_mm`` and a 3 mm floor, as five solids."""
    half_x, half_y, wall, rim = size_xy[0] / 2.0, size_xy[1] / 2.0, 2.5, rim_mm / 2.0
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def at(u: float, v: float, z: float) -> "tuple[float, float, float]":
        return (centre_xy[0] + c * u - s * v, centre_xy[1] + s * u + c * v, z)

    return [_solid(at(-half_x + wall, 0.0, rim), (wall, half_y, rim), yaw_deg),
            _solid(at(half_x - wall, 0.0, rim), (wall, half_y, rim), yaw_deg),
            _solid(at(0.0, -half_y + wall, rim), (half_x, wall, rim), yaw_deg),
            _solid(at(0.0, half_y - wall, rim), (half_x, wall, rim), yaw_deg),
            _solid(at(0.0, 0.0, 1.5), (half_x, half_y, 1.5), yaw_deg)]


def _scene(shift_mm: float) -> list[dict]:
    """The investigation's bin, turned 30 deg and moved ``shift_mm`` further out along base -y, and 48 parts.

    The investigation drew the bin's corner 48.3 mm from the housing with its centre at y = -260 - 161.6 mm. The parts
    start 140 mm past the bin's far corner in x: at 220, as the first draft of this probe had them, they lay within the
    25 mm cluster distance of the bin, the bin and the parts were one object turned 13 deg, and no box was the bin's.
    """
    reach = 150.0 * math.sin(math.radians(30.0)) + 100.0 * math.cos(math.radians(30.0))
    solids = _open_bin((0.0, -260.0 - reach - shift_mm), (300.0, 200.0), 40.0, 30.0)
    rng = np.random.default_rng(4)
    for ix in range(6):
        for iy in range(8):
            height, yaw = float(rng.uniform(20.0, 80.0)), float(rng.uniform(-60.0, 60.0))
            solids.append(_solid((320.0 + 75.0 * ix, -780.0 + 90.0 * iy, height / 2.0), (18.0, 25.0, height / 2.0),
                                 yaw))
    return solids


def _bin_solids(shift_mm: float) -> list[dict]:
    """The bin's own five solids, as :func:`_scene` places them, without the parts."""
    return _scene(shift_mm)[:5]


def _square_bin(shift_mm: float) -> list[dict]:
    """The same bin square to base X/Y, its near wall ``shift_mm`` past where the investigation drew the turned one's
    corner: the bin a person declares, walls and floor as boxes square to the base."""
    return _open_bin((0.0, -260.0 - 100.0 - shift_mm), (300.0, 200.0), 40.0, 0.0)


def _exact_mm(arm: Any, joints_deg: "tuple[float, ...]", solids: "list[dict]") -> float:
    """The exact meshes' distance at ``joints_deg`` to ``solids``, millimetres: the guard's own engine, by bisection."""
    from src.robot.safety._capsule import AxisAlignedBox, TurnedBox
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm

    backend = arm.safety_preflight._path_authority(arm)._exact_mesh_backend("ur10")
    transforms = ur_link_transforms_mm("ur10", np.asarray(_rad(joints_deg), dtype=np.float64))
    boxes = tuple(AxisAlignedBox(center_mm=np.asarray(s["centre"], dtype=np.float64),
                                 half_extents_mm=np.asarray(s["half"], dtype=np.float64), name=f"solid_{i}",
                                 turned=TurnedBox(half_extents_mm=np.asarray(s["half"], dtype=np.float64),
                                                  yaw_rad=math.radians(s["yaw_deg"])))
                  for i, s in enumerate(solids))
    if backend.evaluate(transforms, 0.0, boxes, 1e-6, arm_pairs=False) is not None:
        return 0.0
    low, high = 0.0, 400.0
    for _ in range(40):
        middle = (low + high) / 2.0
        if backend.evaluate(transforms, 0.0, boxes, middle, arm_pairs=False) is None:
            low = middle
        else:
            high = middle
    return low


def _shift_for(arm: Any, real_mm: float, solids: "Any" = None) -> float:
    """The shift along base -y that puts the bin's own solids (``solids(shift)``, the turned bin's by default)
    ``real_mm`` from the arm's meshes at the check joints."""
    place = solids or _bin_solids
    near, far = -120.0, 60.0
    for _ in range(50):
        middle = (near + far) / 2.0
        if _exact_mm(arm, CHECK_DEG, place(middle)) > real_mm:
            far = middle
        else:
            near = middle
    return (near + far) / 2.0


def _in_box_configs(arm: Any, planner: Any, seen: "tuple[Any, ...]") -> "list[tuple[str, list[float]]]":
    """Per box the camera saw, one configuration whose tool centre stands in it, pointing down, and which the planner
    refuses on its world alone: the configurations whose verdicts say how deep each box reaches into the planner's world,
    so a box that does not come back changes one of them."""
    from src.robot.safety._ur_ik import ur_flange_ik

    found: list[tuple[str, list[float]]] = []
    for box in seen:
        flange = np.eye(4)
        flange[:3, :3] = np.diag([1.0, -1.0, -1.0])
        flange[:3, 3] = np.asarray(box.center_mm, dtype=np.float64) + np.array([0.0, 0.0, 155.75])
        solutions = [list(q) for q in (ur_flange_ik("ur10", flange, q6_if_singular=0.0) or ())]
        if not solutions:
            continue
        report = planner.judge_joint_path(solutions, clearance_mm=0.0, name_pairs=False)
        rows = {row.index: row for row in report.refused}
        for index, q in enumerate(solutions):
            row = rows.get(index)
            if row is not None and row.bound_ok and row.self_ok and not row.world_ok:
                found.append((str(box.name), [float(v) for v in q]))
                break
    return found


def _camera_to_base() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = CAMERA_XYZ_MM
    return transform


def _render(camera_to_base: np.ndarray, solids: "list[dict]") -> np.ndarray:
    """Depth along each pixel's ray to the nearest solid or the bench (the slab method in each solid's own frame)."""
    cols, rows = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = np.stack([(cols - K[0, 2]) / FOCAL, (rows - K[1, 2]) / FOCAL, np.ones_like(cols)], axis=-1)
    rays = rays @ camera_to_base[:3, :3].T
    origin = camera_to_base[:3, 3]
    down = rays[..., 2] < 0.0
    depth = np.where(down, -origin[2] / np.where(down, rays[..., 2], -1.0), np.inf)
    for solid in solids:
        c, s = math.cos(math.radians(solid["yaw_deg"])), math.sin(math.radians(solid["yaw_deg"]))
        start = origin - np.asarray(solid["centre"], dtype=np.float64)
        start = np.array([c * start[0] + s * start[1], -s * start[0] + c * start[1], start[2]])
        way = np.stack([c * rays[..., 0] + s * rays[..., 1], -s * rays[..., 0] + c * rays[..., 1], rays[..., 2]],
                       axis=-1)
        half = np.asarray(solid["half"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            low, high = (-half - start) / way, (half - start) / way
        near = np.nanmax(np.minimum(low, high), axis=-1)
        far = np.nanmin(np.maximum(low, high), axis=-1)
        depth = np.where((far >= near) & (near > 0.0) & (near < depth), near, depth)
    return np.where(np.isfinite(depth), depth, 0.0)


class _StillCamera:
    """A camera that answers with the frame it was last drawn, stamped at each grab."""

    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth

    def grab_surface_depth(self) -> Any:
        from src.robot.safety.planning.live_world import DepthSnapshot

        return DepthSnapshot(depth_mm=self.depth, intrinsics=K, timestamp=time.time())


class _Hand:
    """The hand the probe's arm carries, read as the owner's toggle reads: connected, its count standing, open unless
    ``jaws_closed``. The camera's boxes are set aside only while it reads open (the owner, 2026-10-01). A grasp closes it
    with one change (``set_closed``), counted in ``changes``."""

    toggles_without_sensor = True
    edge_unknown = False
    is_connected = True
    min_width_mm = 5.0
    max_width_mm = 49.99

    def __init__(self) -> None:
        self.jaws_closed = False
        self.changes = 0

    def why_jaws_unknown(self) -> str:
        return ""

    def jaws_open_for_a_pick(self) -> str:
        return ""

    def set_closed(self, closed: bool) -> None:
        if bool(closed) is not self.jaws_closed:
            self.changes += 1
            self.jaws_closed = bool(closed)


def _closed_form_ik(arm: Any) -> Any:
    """The controller's inverse kinematics, which the probe's recording controller cannot give: the closed form, the
    solution nearest the seed, each joint on the full turn nearest it."""
    from src.robot.core import JointPositions
    from src.robot.safety._ur_ik import nearest_turn, ur_flange_ik

    def ik(pose: Any, *, seed: Any = None) -> Any:
        flange = np.asarray(arm._pose_to_flange(pose).to_matrix(), dtype=np.float64)
        near = [float(v) for v in (seed.tolist() if seed is not None else arm._conn.get_joint_positions())]
        best: "list[float] | None" = None
        for solution in ur_flange_ik("ur10", flange, q6_if_singular=near[5]) or ():
            turned = [nearest_turn(float(v), near=n, lower=-2.0 * math.pi, upper=2.0 * math.pi)
                      for v, n in zip(solution, near)]
            if any(v is None for v in turned):
                continue
            q = [float(v) for v in turned if v is not None]
            if best is None or max(abs(a - b) for a, b in zip(q, near)) < max(abs(a - b) for a, b in zip(best, near)):
                best = q
        if best is None:
            raise RuntimeError("no closed-form solution for a sample of the probe's line")
        return JointPositions(tuple(best))

    return ik


class _Controller:
    """A recording controller with a state, for step 8: it reads a controller that can move, a moveL lands where it was
    told (the closed form nearest the joints it stands at), and a planned move is a stub that lands on its goal, since
    step 8 holds the judgement of lines, not a plan. Every motion is recorded in ``sent``."""

    def __init__(self, arm: Any, joints_deg: "tuple[float, ...]") -> None:
        self.arm = arm
        self.joints = _rad(joints_deg)
        self.sent: list[str] = []
        conn = arm._conn
        conn.get_joint_positions.side_effect = lambda: list(self.joints)
        conn.get_robot_mode.return_value = 7  # RUNNING
        conn.get_safety_mode.return_value = 1  # NORMAL
        conn.is_protective_stopped.return_value = False
        conn.is_emergency_stopped.return_value = False
        conn.dashboard_safety_status.return_value = ""
        motion = MagicMock()
        motion.get_current_pose.side_effect = self._current
        motion.move_to.side_effect = self._move_to
        arm._motion = motion
        arm._drive_curobo = self._planned
        arm.ik = _closed_form_ik(arm)

    def stand(self, joints_deg: "tuple[float, ...]") -> None:
        self.joints = _rad(joints_deg)
        self.sent.clear()

    def _flange(self) -> Any:
        from src.geometry import Frame, Pose
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        frames = ur_link_transforms_mm("ur10", np.asarray(self.joints, dtype=np.float64))
        assert frames is not None  # noqa: S101 (the UR10 has a DH chain)
        return Pose.from_matrix(np.asarray(frames[-1]), frame=Frame.BASE)

    def _current(self) -> Any:
        from src.robot.drivers.ur.pose_adapter import pose_to_urpose

        return pose_to_urpose(self._flange())

    def _land(self, flange: Any) -> None:
        from src.robot.core import JointPositions

        solved = self.arm.ik(self.arm._pose_from_flange(flange), seed=JointPositions(tuple(self.joints)))
        self.joints = [float(v) for v in solved.tolist()]

    def _move_to(self, urpose: Any, *args: Any, **kwargs: Any) -> bool:
        from src.geometry import Frame
        from src.robot.drivers.ur.pose_adapter import urpose_to_pose

        self._land(urpose_to_pose(urpose, frame=Frame.BASE))
        self.sent.append("moveL" if kwargs.get("linear") else "move")
        return True

    def _planned(self, pose: Any, **_: Any) -> Any:
        from src.robot.core import MotionCommand, MotionResult

        self._land(self.arm._pose_to_flange(pose))
        self.sent.append("planned (stub)")
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def grasp_here(self) -> Any:
        """A part 40 mm across right where the tool stands, approached along the tool's own axis."""
        from src.geometry.quaternion import to_rotation_matrix
        from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

        part = self.arm.get_tcp_pose()
        turned = to_rotation_matrix(np.asarray(part.quaternion_xyzw, dtype=np.float64))
        return GraspPoint(position=np.asarray(part.position_mm, dtype=np.float64), approach=turned[:, 2],
                          axis=turned[:, 0], grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label="the part")


class Probe:
    def __init__(self, arm: Any, hand: "_Hand | None" = None) -> None:
        self.arm = arm
        self.hand = hand if hand is not None else _Hand()
        self.planner: Any = None
        self.failed: list[str] = []
        self.cases: list[dict[str, Any]] = []

    def expect(self, name: str, ok: bool, said: str) -> None:
        say(f"  [{'ok' if ok else 'FAIL'}] {name}: {said}")
        if not ok:
            self.failed.append(name)

    def decide(self, name: str, degrees: "tuple[float, ...]", *, admitted: bool, guard_accepts: bool,
               stands_on: "tuple[str, ...]" = (), world_ok: "bool | None" = None) -> dict:
        """The plain check, the planner's report and the driver's decision at one configuration, the exact guard's own
        verdict beside them; ``stands_on`` are phrases one of which the reason a refusal stands on has to hold, and
        ``world_ok`` what the planner's own world term has to say of it in the report, where given: a decision that
        admits it says nothing of the world term, because the camera's boxes alone are now the exact guard's."""
        from src.robot.core import JointPositions, MotionCommand

        config = _rad(degrees)
        verdict = self.planner.check_joint_path([config], refresh=False)
        report = self.planner.judge_joint_path([config], clearance_mm=0.0)
        standing = (self.arm._exact_guard_decides(self.planner, [config], verdict, clearance_mm=0.0,
                                                  command=MotionCommand.MOVE_JOINTS)
                    if not verdict.valid else None)
        valid = verdict.valid or standing is None
        guard = self.arm.safety_preflight.gate_joint_target(JointPositions(tuple(config)), arm=self.arm)
        terms = "; ".join(f"bound_ok={r.bound_ok} self_ok={r.self_ok} world_ok={r.world_ok}"
                          + (" (" + ", ".join(f"{p.link_a}|{p.link_b} {p.depth_mm:.2f} mm" for p in r.pairs) + ")"
                             if r.pairs else "") for r in report.refused)
        why = standing.reason if standing is not None else ""
        world_clear = all(row.world_ok for row in report.refused)
        result = {"case": name, "joints_deg": list(degrees), "plain_valid": verdict.valid, "report": terms,
                  "world_ok": world_clear, "exact_guard": "accepts" if guard is None else guard.message,
                  "admitted": valid, "stands_because": why}
        self.cases.append(result)
        self.expect(name, valid is admitted and (not stands_on or any(phrase in why for phrase in stands_on)),
                    f"{'VALID' if valid else 'INVALID'} (plain check {'valid' if verdict.valid else 'invalid'}; "
                    f"{terms or 'no refused sample'}" + (f"; stands: {why}" if why else "") + ")")
        self.expect(f"{name}: the exact guard {'accepts' if guard_accepts else 'refuses'} it", (guard is None)
                    is guard_accepts, str(result["exact_guard"]))
        if world_ok is not None:
            self.expect(f"{name}: the planner's own world term {'clears' if world_ok else 'meets'} it",
                        world_clear is world_ok, f"world_ok={world_clear} in the planner's report")
        return result

    def beside(self, real_mm: float, measured_mm: "dict[str, float]", line_mm: float, *, expect: bool) -> dict:
        """A bin ``real_mm`` from the arm's meshes, at the check joints (in the band too) and at the off-band joints
        (refused on the world alone): the planner's world at a plan's clearance and a line's, with the camera's boxes
        and with them set aside, its whole report before and after, the exact guard, and the driver's decision. Every
        bin holds the world put back exactly; ``expect`` holds Option 1 to it as well: the world set aside clears it,
        the exact guard accepts it, the driver admits it, and off the band the planner typed its refusal WORLD and
        refused nothing of the robot itself."""
        from src.robot.core import JointPositions, MotionCommand

        preflight = self.arm.safety_preflight
        names = tuple(sorted(str(box.name) for box in preflight.perceived_obstacles(self.arm)))

        def world(judged: Any) -> str:
            return "meets it" if any(not row.world_ok for row in judged.refused) else "clears it"

        said: dict[str, Any] = {"bin_mm": real_mm, "measured_mm": {k: round(v, 2) for k, v in measured_mm.items()},
                                "seen_boxes": len(names)}
        for pose, degrees in (("check", CHECK_DEG), ("off", OFF_DEG)):
            config = [_rad(degrees)]
            guard = preflight.gate_joint_target(JointPositions(tuple(config[0])), arm=self.arm)
            said[pose] = {"exact_guard": "accepts" if guard is None else guard.message}
            for what, clearance in (("plan", 0.0), ("line", line_mm)):
                plain = self.planner.judge_joint_path(config, clearance_mm=clearance)
                aside = self.planner.judge_joint_path(config, clearance_mm=clearance, name_pairs=False,
                                                      ignore_perceived=names)
                back = self.planner.judge_joint_path(config, clearance_mm=clearance)
                self.expect(f"bin {real_mm:g} mm, {pose}, {what}: the planner's whole report is the same after the "
                            "camera's boxes were set aside", back.refused == plain.refused,
                            f"before {world(plain)}, after {world(back)}; every term and pair depth compared")
                verdict = self.planner.check_joint_path(config, refresh=False, clearance_mm=clearance)
                kind = getattr(getattr(verdict, "refusal", None), "kind", None)
                standing = (self.arm._exact_guard_decides(self.planner, config, verdict, clearance_mm=clearance,
                                                          command=MotionCommand.MOVE_JOINTS)
                            if not verdict.valid else None)
                rows = [(r.bound_ok, r.self_ok, r.world_ok) for r in plain.refused]
                said[pose][what] = {"clearance_mm": clearance, "world": world(plain),
                                    "world_set_aside": world(aside), "set_aside": list(aside.perceived_ignored),
                                    "typed": None if kind is None else str(getattr(kind, "value", kind)),
                                    "terms": rows, "admitted": verdict.valid or standing is None,
                                    "stands": None if standing is None else standing.reason}
                case = said[pose][what]
                say(f"  [{'exp' if expect else 'rec'}] bin {real_mm:g} mm (measured {measured_mm[pose]:.2f}) at the "
                    f"{'check' if pose == 'check' else 'off-band'} joints, {clearance:g} mm "
                    f"({'a plan' if what == 'plan' else 'a straight line'}): the planner's world {case['world']}"
                    f" (typed {case['typed']}), with the camera's {len(names)} box(es) set aside it "
                    f"{case['world_set_aside']}; the exact guard {said[pose]['exact_guard']}; the driver "
                    + ("admits it" if case["admitted"] else f"refuses it: {case['stands']}"))
                if expect:
                    self.expect(f"bin {real_mm:g} mm, {pose}, {what}: the world set aside clears it",
                                case["world_set_aside"] == "clears it", case["world_set_aside"])
                    self.expect(f"bin {real_mm:g} mm, {pose}, {what}: the driver admits it", bool(case["admitted"]),
                                str(case["stands"] or "admitted"))
                    if pose == "off":
                        self.expect(f"bin {real_mm:g} mm, off, {what}: refused on the planner's world alone, typed "
                                    "WORLD", case["typed"] == "world" and rows == [(True, True, False)],
                                    f"typed {case['typed']}, terms (bound_ok, self_ok, world_ok) {rows}")
            if expect:
                self.expect(f"bin {real_mm:g} mm, {pose}: the exact guard accepts it", guard is None,
                            str(said[pose]["exact_guard"]))
        return said

    def world_put_back(self, seen: "tuple[Any, ...]") -> dict:
        """⭐ The whole world comes back: one configuration in each box the camera saw, refused on the world alone, each
        judged on its own before and after a judgement with every camera box set aside. Its verdict carries the depth
        its box reaches into the planner's world, so one box left out changes one of them, whatever the others say."""
        configs = _in_box_configs(self.arm, self.planner, seen)
        names = tuple(sorted(str(box.name) for box in seen))

        def verdicts() -> list[tuple]:
            out = []
            for _, q in configs:
                v = self.planner.check_joint_path([q], refresh=False)
                r = getattr(v, "refusal", None)
                out.append((v.valid, v.first_invalid, None if r is None else str(r.kind),
                            None if r is None else repr(r.depth_mm), None if r is None else repr(r.clearance_mm)))
            return out

        before = verdicts()
        aside = self.planner.judge_joint_path([q for _, q in configs], clearance_mm=0.0, name_pairs=False,
                                              ignore_perceived=names)
        after = verdicts()
        cleared = len(configs) - len([row for row in aside.refused if not row.world_ok])
        said = {"boxes": len(seen), "configurations": len(configs), "cleared_set_aside": cleared,
                "same_before_and_after": before == after}
        self.expect("the camera's boxes all come back: every in-box verdict the same before and after",
                    before == after and len(configs) >= len(seen) // 2,
                    f"{len(configs)} configuration(s) inside {len(seen)} box(es), each refused on the world alone with "
                    f"its depth; {cleared} of them clear with the camera's boxes set aside; before == after: "
                    f"{before == after}")
        return said

    def line_beside(self, start: "tuple[float, ...]", goal: "tuple[float, ...]", name: str, *,
                    runs: bool = True, stands_on: "tuple[str, ...]" = ()) -> dict:
        """A straight joint line beside the bin through the driver's own move, with a recording controller: it runs as
        one moveJ where ``runs``, else it is refused naming every one of ``stands_on`` and nothing is sent."""
        from src.robot.core import JointPositions

        self.arm._conn.get_joint_positions.return_value = _rad(start)
        before = self.arm._conn.moveJ.call_count
        started = time.perf_counter()
        try:
            result = self.arm.move_to_joints_on_the_line(JointPositions(tuple(_rad(goal))))
        finally:
            self.arm._conn.get_joint_positions.return_value = _rad(CHECK_DEG)
        sent = self.arm._conn.moveJ.call_count - before
        said = {"ok": result.ok, "status": result.status.value, "message": result.message, "moveJ": sent,
                "seconds": round(time.perf_counter() - started, 2)}
        held = (result.ok and sent == 1) if runs else (not result.ok and sent == 0 and all(
            phrase in (result.message or "") for phrase in stands_on))
        self.expect(name, held, f"{result.status.value}: {result.message or 'executed'}; {sent} moveJ in "
                                f"{said['seconds']} s")
        return said

    def carried_beside(self) -> dict:
        """⭐ The camera's boxes are set aside only for a hand known empty and open (the owner, 2026-10-01): with the
        hand's count closed and nothing attached, as a plain close or a grasp on a cell that models no part leaves it,
        the line out of the off-band pose beside the bin is refused at its first sample; open again, the same line runs.
        Then the grasp's lift there, judged before the jaws close (:meth:`lift_beside`)."""
        self.hand.jaws_closed = True
        try:
            closed = self.line_beside(OFF_DEG, OFF_LINE_DEG,
                                      "the hand's count says closed, nothing attached: the line out of the off-band "
                                      "pose is held", runs=False,
                                      stands_on=("at sample 0 of", "the hand is not known to be empty and open"))
        finally:
            self.hand.jaws_closed = False
        opened = self.line_beside(OFF_DEG, OFF_LINE_DEG, "the hand reads open again: the same line runs")
        return {"closed": closed, "opened": opened, "lift": self.lift_beside()}

    def lift_beside(self) -> dict:
        """⭐ A grasp judges its lift before the jaws close (the owner, 2026-10-01): at the off-band pose beside the bin,
        the lift a grasp makes, a moveL 100 mm straight up, judged as if the jaws held a part 40 mm across
        (``carried_line_refusal``), is refused at its first sample, nothing sent and nothing left attached; the same lift
        with the hand empty runs as one moveL through the driver's own move (a recording controller; the closed form
        stands in for the controller's inverse kinematics)."""
        from src.geometry import Frame, Pose
        from src.robot.core.arm_capabilities import PayloadModel
        from src.robot.drivers.ur.pose_adapter import pose_to_urpose
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        here = _rad(OFF_DEG)
        controller = MagicMock()
        frames = ur_link_transforms_mm("ur10", np.asarray(here, dtype=np.float64))
        assert frames is not None  # noqa: S101 (the UR10 has a DH chain)
        flange = np.asarray(frames[-1])
        controller.get_current_pose.return_value = pose_to_urpose(Pose.from_matrix(flange, frame=Frame.BASE))
        controller.move_to.return_value = True
        saved = self.arm._motion
        self.arm._motion = controller
        self.arm.ik = _closed_form_ik(self.arm)
        self.arm._conn.get_joint_positions.return_value = list(here)
        try:
            top = self.arm.get_tcp_pose()
            lift = Pose(position_mm=np.asarray(top.position_mm, dtype=np.float64) + np.array([0.0, 0.0, 100.0]),
                        quaternion_xyzw=np.asarray(top.quaternion_xyzw, dtype=np.float64), frame=Frame.BASE,
                        label="the grasp's lift")
            started = time.perf_counter()
            judged = self.arm.carried_line_refusal(lift, grip_width_mm=40.0)
            judged_s = round(time.perf_counter() - started, 2)
            held_after = bool(getattr(self.planner, "carries_part", True)) or (
                self.arm.payload_model() is not PayloadModel.NONE)
            sent_judging = controller.move_to.call_count
            moved = self.arm.move(lift, linear=True)
            sent = controller.move_to.call_count - sent_judging
        finally:
            self.arm._motion = saved
            del self.arm.ik
            self.arm._conn.get_joint_positions.return_value = _rad(CHECK_DEG)
        message = "" if judged is None else (judged.message or "")
        said = {"judged_refused": judged is not None, "judged": message, "seconds": judged_s,
                "sent_while_judging": sent_judging, "attached_after": held_after,
                "empty_ok": moved.ok, "empty_message": moved.message, "moveL": sent}
        self.expect("the grasp's lift beside the bin, judged as if the jaws held a part, is refused at its first sample",
                    judged is not None and "at sample 0 of" in message and "a part is carried" in message
                    and sent_judging == 0, (message or "it would run") + f"; {sent_judging} moveL sent; {judged_s} s")
        self.expect("nothing is left attached after the judgement", not held_after,
                    f"planner carries_part={getattr(self.planner, 'carries_part', None)}, "
                    f"arm payload {self.arm.payload_model().value}")
        self.expect("the same lift with the hand empty runs as one moveL", moved.ok and sent == 1,
                    f"{moved.status.value}: {moved.message or 'executed'}; {sent} moveL")
        return said

    def screen_off(self) -> dict:
        """The screen at the off-band pose beside the bin says what a move does there: beside the camera's boxes, no
        ERROR."""
        from src.robot.core import JointPositions

        screen = self.arm.screen_configuration(JointPositions(tuple(_rad(OFF_DEG))))
        line = screen.line("OFF")
        say(f"  {line}")
        self.expect("the screen at the off-band pose says it is beside the camera's boxes, no ERROR",
                    screen.verdict.value == "seen_boxes" and not screen.is_error, screen.verdict.value)
        return {"verdict": screen.verdict.value, "line": line}

    def declared(self, real_mm: float, measured_mm: float, line_mm: float, solids: "list[dict]") -> dict:
        """A square bin declared at its measured place, ``real_mm`` from the arm's meshes: its walls and floor sent to the
        planner as the declared boxes they would be, beside the bench and no camera box. Recorded: whether the planner's
        own world meets it at the check and the off-band joints, at a plan's clearance and a line's. The exact guard
        keeps a declared box 10 mm (min_distance_mm) and is not asked here."""
        from src.robot.safety.planning.world import planner_cuboid

        boxes = [planner_cuboid(f"declared_bin_{index}", [float(v) for v in solid["centre"]],
                                [2.0 * float(v) for v in solid["half"]]) for index, solid in enumerate(solids)]
        registered = self.planner.set_world(boxes)
        said: dict[str, Any] = {"bin_mm": real_mm, "measured_mm": round(measured_mm, 2), "registered": registered}
        for pose, degrees in (("check", CHECK_DEG), ("off", OFF_DEG)):
            for what, clearance in (("plan", 0.0), ("line", line_mm)):
                judged = self.planner.judge_joint_path([_rad(degrees)], clearance_mm=clearance, name_pairs=False)
                meets = any(not row.world_ok for row in judged.refused)
                said[f"{pose}_{what}"] = "meets it" if meets else "clears it"
        say(f"  [rec] a square bin declared {real_mm:g} mm from the arm's meshes (measured {measured_mm:.2f}): the "
            f"planner's world at no clearance (a plan) {said['check_plan']} at the check joints, {said['off_plan']} off "
            f"the band; at {line_mm:g} mm (a straight line) {said['check_line']}, {said['off_line']}")
        return said

    def plan_beside(self, replica: bool) -> dict:
        """A planned move out of the check joints beside the bin, recorded: what cuRobo makes of its own world."""
        from src.robot.core import JointPositions, MotionCommand

        if replica:
            say("  [info] the CPU replica plans nothing")
            return {"replica": True}
        goal = JointPositions(tuple(_rad(GOAL_DEG)))
        started = time.perf_counter()
        trajectory, how, refusal, legs = self.arm._planned(self.planner, _rad(GOAL_DEG), name="the probe's goal",
                                                           here=_rad(CHECK_DEG), command=MotionCommand.MOVE_JOINTS,
                                                           target_joints=goal)
        said = {"found": bool(trajectory), "how": how, "legs": list(legs),
                "refusal": None if refusal is None else str(getattr(refusal, "render", lambda: refusal)()),
                "seconds": round(time.perf_counter() - started, 2)}
        say("  [rec] a planned move out of the check joints beside the bin: "
            + (f"{len(trajectory or ())} waypoint(s); {how or 'no band leg'}" if trajectory else
               f"none; {how or 'no reason given'}" + (f"; {said['refusal']}" if said["refusal"] else "")))
        return said


#: The carried part step 8 declares, millimetres past the fingertips. A cuRobo UR cell has to declare one (the real-cell
#: checklist blocks otherwise), so the owner's cell runs the attach, check and detach a grasp's judgement makes; this
#: composition carries the attach spheres and names its own committed evidence.
PAYLOAD_LENGTH_MM = 60.0
#: Step 8's control: the off-band joints with the shoulder turned 30 deg away from the bin, 65 mm clear of its boxes.
AWAY_DEG = (-30.0, -60.0, 80.0, -95.0, -90.0, 0.0)
#: The width a toggle's attach carries for a part 40 mm across: what its jaws are told, 1 mm under (``close_squeeze_mm``).
TOLD_MM = 39.0


def modelled_part(probe: Probe, replica: bool, stderr_log: str = "") -> dict:
    """⭐ Step 8, a cell that models the carried part (``planning_world.payload.length_mm``), with a planner of its own
    started once the first one stopped: the grasp's lift beside the 30 mm bin, judged before the jaws close, hands the
    part to the sidecar, is refused at its first sample, and takes the part back, nothing sent; the same lift
    empty-handed runs as one moveL; the whole grasp there backs out with the jaws open; and away from the bin it closes
    once, hands the sidecar the box the check judged and lifts. Each attach is a sphere fit of its own, timed. The CPU
    replica models no carried part, so it holds nothing here. ``stderr_log`` takes this sidecar's stderr, beside the
    first one's."""
    if replica:
        say("  [info] the CPU replica models no carried part, so step 8 holds nothing")
        return {"replica": True}
    from src.geometry import Frame, Pose
    from src.robot.core.arm_capabilities import PayloadModel
    from src.robot.drivers.ur.arm import URRobotArm
    from src.robot.execution.camera_world_wiring import _live_planner_world
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.safety.planning.live_world import CameraView

    if stderr_log:
        os.environ["WILLY_CUROBO_STDERR"] = stderr_log
    config = _config(PAYLOAD_LENGTH_MM)
    arm = URRobotArm(config)
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.moveJ.return_value = True
    hand = _Hand()
    arm.set_hand(hand)
    jaws: Any = hand  # the owner's toggle as the policy reads it: one change to close, nothing measured
    controller = _Controller(arm, OFF_DEG)
    camera_to_base = _camera_to_base()
    camera = _StillCamera(_render(camera_to_base, _scene(BIN_SHIFT_MM)))
    arm.set_live_planner_world(_live_planner_world(config, (CameraView(
        name="probe_modelled_part", depth_source=camera, camera_to_base=camera_to_base),)))
    out: dict[str, Any] = {"payload_length_mm": PAYLOAD_LENGTH_MM}
    attaches: list[dict[str, Any]] = []
    detaches: list[bool] = []
    try:
        declined = arm.payload_declined_reason()
        probe.expect("step 8: the cell models the carried part", declined is None,
                     declined or f"planning_world.payload.length_mm {PAYLOAD_LENGTH_MM:g}")
        started = time.perf_counter()
        identity = arm.start_planner()
        out["start_s"] = round(time.perf_counter() - started, 1)
        out["composed_sha256"] = identity.composed_sha256
        say(f"  planner started in {out['start_s']} s, composed_sha256 {identity.composed_sha256} (with the attach "
            "spheres; the start refuses any other than the committed evidence's)")
        planner = arm._curobo_ur
        assert planner is not None  # noqa: S101 (the start built it)
        attach, detach = planner.attach_payload, planner.detach_payload

        def timed_attach(joints: Any, dims_mm: Any, centre_mm: Any) -> bool:
            began = time.perf_counter()
            ok = bool(attach(joints, dims_mm, centre_mm))
            attaches.append({"dims_mm": [round(float(v), 1) for v in dims_mm], "ok": ok,
                             "seconds": round(time.perf_counter() - began, 2)})
            return ok

        def counted_detach() -> bool:
            ok = bool(detach())
            detaches.append(ok)
            return ok

        setattr(planner, "attach_payload", timed_attach)  # noqa: B010 (the probe times the glue's own call)
        setattr(planner, "detach_payload", counted_detach)  # noqa: B010
        camera.depth = _render(camera_to_base, _scene(_shift_for(arm, ADMITTED_MM[0])))
        planner.refresh_world(near_point_mm=NEAR_POINT_MM)
        refresh = planner.last_world_refresh
        say(f"  the bin {ADMITTED_MM[0]:g} mm from the arm's meshes: "
            f"{refresh.render() if refresh is not None else 'no refresh'}")

        controller.stand(OFF_DEG)
        top = arm.get_tcp_pose()
        lift = Pose(position_mm=np.asarray(top.position_mm, dtype=np.float64) + np.array([0.0, 0.0, 100.0]),
                    quaternion_xyzw=np.asarray(top.quaternion_xyzw, dtype=np.float64), frame=Frame.BASE,
                    label="the grasp's lift")
        began = time.perf_counter()
        judged = arm.carried_line_refusal(lift, grip_width_mm=TOLD_MM)
        out["judged_s"] = round(time.perf_counter() - began, 2)
        message = "" if judged is None else (judged.message or "")
        out["judged"] = message
        probe.expect("step 8: the lift beside the bin, judged with the part in the sidecar, is refused at its first "
                     "sample, nothing sent", judged is not None and "at sample 0 of" in message
                     and "a part is carried" in message and not controller.sent,
                     f"{message or 'it would run'}; {len(controller.sent)} motion(s) sent; {out['judged_s']} s")
        probe.expect("step 8: the part was handed to the sidecar for the judgement and taken back",
                     [a["ok"] for a in attaches] == [True] and detaches == [True],
                     f"attach {attaches}, detach {detaches}")
        probe.expect("step 8: nothing is left attached after the judgement",
                     planner.carries_part is False and arm.payload_model() is PayloadModel.NONE
                     and not arm._closed_on_part,
                     f"carries_part={planner.carries_part}, payload {arm.payload_model().value}")
        moved = arm.move(lift, linear=True)
        probe.expect("step 8: the same lift with the hand empty runs as one moveL",
                     moved.ok and controller.sent == ["moveL"], f"{moved.status.value}: {moved.message or 'executed'}; "
                                                                 f"{controller.sent}")

        controller.stand(OFF_DEG)
        changes = hand.changes
        began = time.perf_counter()
        report = GraspExecutionPolicy(arm=arm, gripper=jaws).execute(controller.grasp_here())
        out["beside"] = {"outcome": report.outcome.value, "message": report.motion_message, "sent": list(controller.sent),
                         "jaw_changes": hand.changes - changes, "seconds": round(time.perf_counter() - began, 2)}
        probe.expect("step 8: the grasp beside the bin backs out with the jaws open, nothing attached",
                     report.outcome.value == "carried_retreat_refused" and hand.changes == changes
                     and controller.sent == ["planned (stub)", "moveL", "moveL"] and planner.carries_part is False
                     and not hand.jaws_closed,
                     f"{report.outcome.value}; {hand.changes - changes} jaw change(s); {controller.sent}; "
                     f"{report.motion_message}")

        controller.stand(AWAY_DEG)
        changes, before = hand.changes, len(attaches)
        began = time.perf_counter()
        report = GraspExecutionPolicy(arm=arm, gripper=jaws).execute(controller.grasp_here())
        out["away"] = {"outcome": report.outcome.value, "message": report.motion_message, "sent": list(controller.sent),
                       "jaw_changes": hand.changes - changes, "seconds": round(time.perf_counter() - began, 2),
                       "attaches": attaches[before:]}
        probe.expect("step 8: away from the bin the grasp closes once, the sidecar holds the part, and the lift runs",
                     report.outcome.value == "executed" and hand.changes == changes + 1
                     and controller.sent == ["planned (stub)", "moveL", "moveL"] and planner.carries_part is True,
                     f"{report.outcome.value}; {hand.changes - changes} jaw change(s); {controller.sent}; "
                     f"{report.motion_message or 'executed'}")
        handed = attaches[before:]
        probe.expect("step 8: the check and the attach after the close hand the sidecar the same part",
                     len(handed) == 2 and all(a["ok"] for a in handed) and handed[0]["dims_mm"] == handed[1]["dims_mm"],
                     "; ".join(f"{a['dims_mm']} mm in {a['seconds']} s" for a in handed))
        out["released"] = arm.detach_payload()
        hand.set_closed(False)
    except Exception as exc:  # noqa: BLE001 (reported with its trace, and the planner still stops)
        probe.failed.append(f"step 8: {type(exc).__name__}: {exc}")
        out["error"] = traceback.format_exc(limit=12)
        say(out["error"])
    finally:
        arm.stop_planner()
    out["attaches"] = attaches
    out["detaches"] = detaches
    return out


def run(args: argparse.Namespace) -> int:
    if args.cpu_replica:
        tag_replica_output()
    if args.curobo_python:
        os.environ["WILLY_CUROBO_PYTHON"] = args.curobo_python
    if args.coal_prefix:
        os.environ["WILLY_COAL_PREFIX"] = args.coal_prefix
    if args.stderr_log:
        os.environ["WILLY_CUROBO_STDERR"] = args.stderr_log
    from src.robot.core import JointPositions, MotionCommand
    from src.robot.drivers.ur.arm import URRobotArm
    from src.robot.execution.camera_world_wiring import _live_planner_world
    from src.robot.safety._capsule import AxisAlignedBox
    from src.robot.safety.planning.band import BAND_LEG_MAX_DEG
    from src.robot.safety.planning.environment import resolve_collision_engine
    from src.robot.safety.planning.live_world import CameraView
    from src.robot.safety.planning.reservation import PlannerReservation
    from src.robot.safety.planning.world import planner_cuboid

    config = _config()
    arm = URRobotArm(config)
    replica: "CpuReplicaClient | None" = None
    if args.cpu_replica:
        replica = CpuReplicaClient(arm, Path(args.cpu_replica))
        setattr(arm, "_curobo_client_factory", lambda: replica)  # noqa: B010 (the probe's own stand-in, by name)
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = _rad(CHECK_DEG)
    arm._conn.moveJ.return_value = True
    # The hand a cell's connect hands the arm: the camera's boxes are set aside only while it reads empty and open.
    hand = _Hand()
    arm.set_hand(hand)
    camera_to_base = _camera_to_base()
    camera = _StillCamera(_render(camera_to_base, _scene(BIN_SHIFT_MM)))
    arm.set_live_planner_world(_live_planner_world(config, (CameraView(
        name="probe_over_the_bin", depth_source=camera, camera_to_base=camera_to_base),)))
    line_mm = float(config.safety.planned_motion.line_clearance_mm)
    preflight = arm.safety_preflight
    assert preflight is not None  # noqa: S101 (a UR arm always wires one)
    probe = Probe(arm, hand)
    out: dict[str, Any] = {"engine": resolve_collision_engine().backend,
                           "planner": "CPU REPLICA, not the GPU sidecar" if replica is not None else "GPU sidecar"}
    say(f"planner under test: {out['planner']}")
    try:
        reservation = PlannerReservation.from_config(robot_cfg=config)
        out["cuboid_slots"] = reservation.cuboid_slots
        probe.expect("the reservation", reservation.cuboid_slots == 1 + 64,
                     f"{reservation.render()} (1 bench + max_boxes 64)")
        started = time.perf_counter()
        identity = arm.start_planner()
        out["start_s"] = round(time.perf_counter() - started, 1)
        out["composed_sha256"] = identity.composed_sha256
        say(f"planner started in {out['start_s']} s, composed_sha256 {identity.composed_sha256} (the start refuses "
            f"any other than the committed evidence's), exact engine {out['engine']}")
        probe.planner = arm._curobo_ur

        say(f"the camera world, the bin {SHIFTS_MM[BIN_SHIFT_MM]:g} mm from the arm's meshes, through the driver's own "
            "refresh:")
        started = time.perf_counter()
        probe.planner.refresh_world(near_point_mm=NEAR_POINT_MM)
        refresh = probe.planner.last_world_refresh
        seen = tuple(refresh.guard_boxes)
        turns = sorted({round(math.degrees(box.turned.yaw_rad) % 90.0, 1) for box in seen if box.turned is not None})
        out["refresh"] = {"ok": refresh.ok, "sent": refresh.sent, "registered": refresh.registered,
                          "seen_boxes": len(seen), "turns_deg_mod_90": turns,
                          "seconds": round(time.perf_counter() - started, 2), "render": refresh.render()}
        say(f"  {refresh.render()}")
        probe.expect("every slot filled and every box sent registered",
                     refresh.ok and refresh.registered == refresh.sent == 1 + len(seen) == reservation.cuboid_slots,
                     f"{refresh.registered} of {refresh.sent} (the bench and {len(seen)} seen box(es))"
                     + (" (the replica registers whatever it is sent)" if replica is not None else ""))
        probe.expect("the guard holds them turned", bool(seen) and all(box.turned is not None for box in seen)
                     and 30.0 in turns, f"{len(seen)} seen box(es), turns mod 90: {turns} deg")

        say("at the investigation's joints, the turned boxes held by both:")
        screen = arm.screen_configuration(JointPositions(tuple(_rad(CHECK_DEG))))
        out["screen"] = {"verdict": screen.verdict.value, "line": screen.line("CHECK")}
        say(f"  {screen.line('CHECK')}")
        # F2's claim, held on its own: the planner's world term clears the turned walls here; the pose is admitted on
        # the robot's own pairs alone, with nothing of the camera's boxes set aside.
        probe.decide("turned boxes", CHECK_DEG, admitted=True, guard_accepts=True, world_ok=True)

        say("the same boxes as their axis-aligned enclosures, held by both:")
        enclosures = [AxisAlignedBox(center_mm=np.asarray(box.center_mm, dtype=np.float64),
                                     half_extents_mm=np.asarray(box.half_extents_mm, dtype=np.float64), name=box.name)
                      for box in seen]
        probe.planner.set_world([planner_cuboid(box.name, [float(v) for v in box.center_mm],
                                                [2.0 * float(v) for v in box.half_extents_mm]) for box in enclosures])
        preflight.set_perceived_obstacles(enclosures)
        # The planner's world refuses it: the report's world term, the turn reaching the kernel. A world set by hand is
        # no refresh's, so its camera boxes are never set aside, and the refusal stands on the planner's world.
        probe.decide("enclosures", CHECK_DEG, admitted=False, guard_accepts=True,
                     stands_on=("no refresh it confirmed",))

        say(f"bins beside the housing: the planner at no clearance and at the line clearance ({line_mm:g} mm, a "
            "straight line), at the check joints (in the band too) and off the band (the world alone), with the "
            "camera's boxes and with them set aside, the exact guard, and the driver's decision (Option 1):")
        beside = []
        for real_mm in (*ADMITTED_MM, *RECORDED_MM):
            shift_mm = _shift_for(arm, real_mm)
            measured = {pose: _exact_mm(arm, degrees, _bin_solids(shift_mm))
                        for pose, degrees in (("check", CHECK_DEG), ("off", OFF_DEG))}
            camera.depth = _render(camera_to_base, _scene(shift_mm))
            probe.planner.refresh_world(near_point_mm=NEAR_POINT_MM)
            row = probe.beside(real_mm, measured, line_mm, expect=real_mm in ADMITTED_MM)
            row["shift_mm"] = round(shift_mm, 2)
            if real_mm == ADMITTED_MM[0]:
                row["world_put_back"] = probe.world_put_back(tuple(probe.planner.last_world_refresh.guard_boxes))
                row["straight_line"] = probe.line_beside(CHECK_DEG, LINE_DEG,
                                                         "a straight joint line beside the bin runs (in the band too)")
                row["straight_line_off"] = probe.line_beside(
                    OFF_DEG, OFF_LINE_DEG, "a straight joint line beside the bin runs off the band (the world alone)")
                row["carried"] = probe.carried_beside()
                row["screen_off"] = probe.screen_off()
                row["planned_move"] = probe.plan_beside(replica is not None)
            beside.append(row)
        out["beside"] = beside

        say("a square bin declared at its measured place, its walls and floor sent to the planner by hand, no camera "
            "box: where the planner's own spheres clear it (recorded):")
        declared = []
        for real_mm in DECLARED_MM:
            shift_mm = _shift_for(arm, real_mm, _square_bin)
            solids = _square_bin(shift_mm)
            row = probe.declared(real_mm, _exact_mm(arm, CHECK_DEG, solids), line_mm, solids)
            row["shift_mm"] = round(shift_mm, 2)
            declared.append(row)
        out["declared"] = declared
        # Step 7 plans in the world of steps 2 to 4 again: the bin 60.8 mm from the meshes.
        camera.depth = _render(camera_to_base, _scene(BIN_SHIFT_MM))
        probe.planner.refresh_world(near_point_mm=NEAR_POINT_MM)

        say("a planned move in the turned world:")
        goal = JointPositions(tuple(_rad(GOAL_DEG)))
        started = time.perf_counter()
        trajectory, how, refusal, legs = arm._planned(probe.planner, _rad(GOAL_DEG), name="the probe's goal",
                                                      here=_rad(CHECK_DEG), command=MotionCommand.MOVE_JOINTS,
                                                      target_joints=goal)
        plan_s = round(time.perf_counter() - started, 2)
        if replica is not None:
            say(f"  [info] the CPU replica plans nothing: {how or 'no plan'}")
            out["plan"] = {"replica": True, "said": how}
        else:
            running, refused = ((), None) if not trajectory else arm._judged_waypoints(
                probe.planner, trajectory, command=MotionCommand.MOVE_JOINTS, target_joints=goal, legs=legs)
            escape = ([round(math.degrees(a - b), 2) for a, b in zip(trajectory[1], _rad(CHECK_DEG))]
                      if trajectory and legs[0] else None)
            out["plan"] = {"found": bool(trajectory), "how": how, "legs": list(legs), "seconds": plan_s,
                           "waypoints": len(trajectory or ()), "running": len(running), "escape_turn_deg": escape,
                           "refused": None if refused is None else refused.message}
            probe.expect("cuRobo plans in the turned world", bool(trajectory) and refused is None,
                         f"{len(trajectory or ())} waypoint(s) in {plan_s} s, {len(running)} moveJ(s) after both "
                         f"authorities judged the route; {how or 'no band leg'}"
                         + (f"; refused: {refused.message}" if refused is not None else "")
                         + (f"; {refusal}" if refusal is not None else ""))
            if escape is not None:
                probe.expect("its escape leg", max(abs(v) for v in escape) <= BAND_LEG_MAX_DEG + 1e-9,
                             f"turns {escape} deg")
    except Exception as exc:  # noqa: BLE001 (reported with its trace, and the planner still stops)
        probe.failed.append(f"{type(exc).__name__}: {exc}")
        out["error"] = traceback.format_exc(limit=12)
        say(out["error"])
    finally:
        arm.stop_planner()
    say(f"a cell that models the carried part ({PAYLOAD_LENGTH_MM:g} mm past the fingertips), with a planner of its own "
        "once the first one stopped: the grasp's lift judged before the jaws close, the part in the sidecar:")
    beside_log = (str(Path(args.stderr_log).with_name(f"{Path(args.stderr_log).stem}_modelled_part"
                                                     f"{Path(args.stderr_log).suffix}")) if args.stderr_log else "")
    out["modelled_part"] = modelled_part(probe, replica is not None, beside_log)
    out["cases"] = probe.cases
    out["failed"] = probe.failed
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    say(f"{'ALL EXPECTATIONS HELD' if not probe.failed else 'FAILED: ' + '; '.join(probe.failed)}; wrote {target}")
    return 0 if not probe.failed else 1


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--curobo-python", default="", help="the cuRobo environment's python (WILLY_CUROBO_PYTHON)")
    parser.add_argument("--coal-prefix", default="", help="the Coal environment (WILLY_COAL_PREFIX)")
    parser.add_argument("--stderr-log", default="", help="where the sidecar's stderr goes (WILLY_CUROBO_STDERR)")
    parser.add_argument("--cpu-replica", default="", metavar="CONTENT_ROOT",
                        help="judge with a CPU replica of the sidecar instead of the GPU one, on the descriptor and URDF "
                             "under this cuRobo content root (ext_deps/curobo/curobo/content): for a box whose cuRobo "
                             "kernels an application-control policy blocks. Every line it prints then starts with "
                             "[replica]")
    parser.add_argument("--out", default=str(REPO / "logs" / "curobo" / "turned_boxes_probe.json"))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
