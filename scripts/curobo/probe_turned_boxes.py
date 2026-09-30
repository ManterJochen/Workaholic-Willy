"""Probe, on the GPU box, that the planner holds the camera world's turned boxes as the exact guard holds them (F2), and
that a pose in the planner's cushion band beside them is decided as the band admission (F1) decides it.

Run from the repository root with the repository's own interpreter; the sidecar is spawned with the cuRobo
environment's, as a cell spawns it::

    python scripts/curobo/probe_turned_boxes.py
    python scripts/curobo/probe_turned_boxes.py --curobo-python <cuRobo env python> --coal-prefix <coal env>

Why it exists. The camera world is a height map (``planning/height_map.py``): an open bin comes back as its walls with
a free inside, a turned object as a turned box, each from the bench to its top + 15 mm. cuRobo takes every box turned,
and since 2026-09-30 the exact guard judges the same turned box, at ``perceived_min_distance_mm`` (5 mm) rather than the
10 mm a declared fixture and the arm itself keep. The planner reserves 1 + declared + ``max_boxes`` (64) box slots when
it starts. The CPU suite holds the pipeline, both box lists and the guard with stand-in planners; this holds the real
sidecar's cuboid cache, its world term on turned boxes, a plan in a full world, and the driver's own decision on them.

The world stays the planner's (the owner, 2026-09-30): cuRobo judges it with its sphere cover, which reaches past the
meshes by design. Beside the UR10's shoulder housing it reaches 25 to 29 mm past them (CPU replica, 2026-09-30), so the
planner keeps a seen bin further from the housing than the exact guard does: at the investigation's joints, with the
30 deg bin 47.7 mm from the arm's meshes, the guard keeps 26.6 mm to the grown box and the planner's spheres reach 2 mm
into it. Step 5 records where each authority clears the bin.

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
     pair, the exact guard decides the pair, the planner's world term is clear of the turned walls, and the exact guard
     accepts them at 5 mm;
  4. the same boxes sent to both as their axis-aligned enclosures: INVALID, the planner's world term meets the
     enclosure of the bin's near wall where it clears the turned wall, which is the turn reaching the kernel; the exact
     guard, 5 mm from a seen box, still accepts the enclosures there;
  5. recorded, not expected either way: the bin 47.7, 54.3 and 60.8 mm from the arm's meshes, and at each the planner's
     world term at no clearance (a plan) and at the line clearance (a straight line), and the exact guard;
  6. a planned move from those joints to (-25, -70, 90, -110, -90, 0) deg in the turned world (``URRobotArm._planned``,
     then ``_judged_waypoints``, as a move runs them): out of the band by a straight escape leg of at most
     ``band.BAND_LEG_MAX_DEG`` (20) per joint where the planner refuses the start, cuRobo's plan behind it, the route
     judged whole by both authorities, and its time. The CPU replica plans nothing, and says so.

It prints one line per step and writes ``logs/curobo/turned_boxes_probe.json``. Exit code 0 when every expectation held,
1 otherwise. The planner is stopped on every way out. On the GPU it spawns a sidecar of its own, so stop the console, the
API and every other planner first: one sidecar on the GPU at a time. The cell is built here, in code, and reads nothing
of the cell's tree: it carries no wrist camera, so the robobrain.eye housing is not in it.

``--cpu-replica <cuRobo content root>`` judges with the CPU replica of the sidecar that ``probe_band_admission.py``
defines (the same composed robot, cuRobo's spheres by numpy forward kinematics, the kernel's rules; its world term is
sphere against turned box at the clearance asked), for a box where an application-control policy refuses cuRobo's
kernels. Every line it prints then starts with ``[replica]``, every log record it makes carries the same tag, its
``planner under test`` line says CPU REPLICA and its JSON names the replica, so no line of it passes for a GPU line: it
proves the driver's decision on the real geometry, not the GPU kernel's. It registers every box it is sent and plans
nothing, so step 6 says so and holds nothing.
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

#: The investigation's joints (2026-09-30): the shoulder housing beside the bin.
CHECK_DEG = (0.0, -60.0, 80.0, -110.0, -90.0, 0.0)
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
#: The shift of steps 2 to 4 and 6: past where the planner's cover meets the bin at the check joints.
BIN_SHIFT_MM = 20.0
#: A 640 x 480 pinhole with a 500 px focal length, and a camera 1100 mm over the bench looking straight down.
WIDTH, HEIGHT, FOCAL = 640, 480, 500.0
K = np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
CAMERA_XYZ_MM = (150.0, -430.0, 1100.0)


def _rad(values: "tuple[float, ...]") -> list[float]:
    return [math.radians(v) for v in values]


def _config() -> Any:
    from src.config.schema.robot import RobotConfig

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
                   "planning_world": {"enabled": True,
                                      "support_plane": {"height_mm": 0.0, "extent_mm": [3000.0, 3000.0]},
                                      "perceived": {"max_boxes": 64, "pixel_stride": 2}}},
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


class Probe:
    def __init__(self, arm: Any) -> None:
        self.arm = arm
        self.planner: Any = None
        self.failed: list[str] = []
        self.cases: list[dict[str, Any]] = []

    def expect(self, name: str, ok: bool, said: str) -> None:
        say(f"  [{'ok' if ok else 'FAIL'}] {name}: {said}")
        if not ok:
            self.failed.append(name)

    def decide(self, name: str, degrees: "tuple[float, ...]", *, admitted: bool, guard_accepts: bool,
               stands_on: "tuple[str, ...]" = ()) -> dict:
        """The plain check, the planner's report and the driver's decision at one configuration, the exact guard's own
        verdict beside them; ``stands_on`` are phrases one of which the reason a refusal stands on has to hold."""
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
        result = {"case": name, "joints_deg": list(degrees), "plain_valid": verdict.valid, "report": terms,
                  "exact_guard": "accepts" if guard is None else guard.message, "admitted": valid,
                  "stands_because": why}
        self.cases.append(result)
        self.expect(name, valid is admitted and (not stands_on or any(phrase in why for phrase in stands_on)),
                    f"{'VALID' if valid else 'INVALID'} (plain check {'valid' if verdict.valid else 'invalid'}; "
                    f"{terms or 'no refused sample'}" + (f"; stands: {why}" if why else "") + ")")
        self.expect(f"{name}: the exact guard {'accepts' if guard_accepts else 'refuses'} it", (guard is None)
                    is guard_accepts, str(result["exact_guard"]))
        return result


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
    camera_to_base = _camera_to_base()
    camera = _StillCamera(_render(camera_to_base, _scene(BIN_SHIFT_MM)))
    arm.set_live_planner_world(_live_planner_world(config, (CameraView(
        name="probe_over_the_bin", depth_source=camera, camera_to_base=camera_to_base),)))
    line_mm = float(config.safety.planned_motion.line_clearance_mm)
    preflight = arm.safety_preflight
    assert preflight is not None  # noqa: S101 (a UR arm always wires one)
    probe = Probe(arm)
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
        probe.decide("turned boxes", CHECK_DEG, admitted=True, guard_accepts=True)

        say("the same boxes as their axis-aligned enclosures, held by both:")
        enclosures = [AxisAlignedBox(center_mm=np.asarray(box.center_mm, dtype=np.float64),
                                     half_extents_mm=np.asarray(box.half_extents_mm, dtype=np.float64), name=box.name)
                      for box in seen]
        probe.planner.set_world([planner_cuboid(box.name, [float(v) for v in box.center_mm],
                                                [2.0 * float(v) for v in box.half_extents_mm]) for box in enclosures])
        preflight.set_perceived_obstacles(enclosures)
        # The planner's world refuses it: the report's world term, or a refusal the sidecar typed as its world.
        probe.decide("enclosures", CHECK_DEG, admitted=False, guard_accepts=True,
                     stands_on=("planner's world", "not a self collision"))

        say(f"recorded: the bin nearer the housing, the planner at no clearance and at the line clearance "
            f"({line_mm:g} mm), and the exact guard:")
        recorded = []
        for shift_mm, bin_mm in SHIFTS_MM.items():
            camera.depth = _render(camera_to_base, _scene(shift_mm))
            probe.planner.refresh_world(near_point_mm=NEAR_POINT_MM)
            config_rad = [_rad(CHECK_DEG)]
            plan = probe.planner.judge_joint_path(config_rad, clearance_mm=0.0)
            line = probe.planner.judge_joint_path(config_rad, clearance_mm=line_mm)
            guard = preflight.gate_joint_target(JointPositions(tuple(config_rad[0])), arm=arm)

            def world(judged: Any) -> str:
                return "meets it" if any(not row.world_ok for row in judged.refused) else "clears it"

            said = {"bin_mm": bin_mm, "planner_plan": world(plan), "planner_line": world(line),
                    "exact_guard": "accepts" if guard is None else guard.message}
            recorded.append(said)
            say(f"  [rec] bin {bin_mm:g} mm from the arm's meshes: the planner's world {said['planner_plan']} at no "
                f"clearance and {said['planner_line']} at {line_mm:g} mm; the exact guard {said['exact_guard']}")
        out["recorded"] = recorded

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
